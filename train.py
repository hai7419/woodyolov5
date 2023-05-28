import torch
import argparse
from pathlib import Path
import yaml
from general import (increment_path, check_dataset,labels_to_class_weights,EarlyStopping,fitness,EarlyStopping,ModelEMA,
                     init_seeds)
from yolov5net import Net
from smart_optimizer import smart_optimizer
from torch.optim import lr_scheduler
from dataloads import create_dataloader
import numpy as np
import time
from loss import ComputeLoss
from tqdm import  tqdm
from val import validate
from copy import deepcopy
from datetime import datetime
from general import LOGGER
import wandb
import os
from PIL import Image,ImageDraw,ExifTags,ImageOps
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default=ROOT/'hyp.scratch.yaml')
    parser.add_argument('--project',type=str,default=ROOT/'run/train',help='save to project name')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--epochs',type=int,default=500,help='total training epochs')
    parser.add_argument('--imgsz',type=int,default=640,help='train val image size pixels')
    parser.add_argument('--batch_size',type=int,default=2,help='total batch size')
    parser.add_argument('--workers',type=int,default=1,help='max dataloader workers')
    parser.add_argument('--exist_ok',action='store_true',help='existing project name ok,do not increment')
    parser.add_argument('--data',type=str,default=ROOT/'data/coco128.yaml',help='dataset.yaml path')
    parser.add_argument('--optimizer', type=str,choices=['SGD','Adam','AdamW'],default='SGD',help='optimizer')
    parser.add_argument('--rect',action='store_true',help='rectangular training')
    parser.add_argument('--label-smoothing',type=float,default=0.0,help='label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=50, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    
    return parser.parse_args()

def train(hyp, opt, device):
    save_dir = Path(opt.save_dir)
    epochs = opt.epochs
    batch_size = opt.batch_size
    # resume = opt.resume
    # workers = opt.workers
    

    w = save_dir/'weights'
    w.mkdir(parents=True,exist_ok=True)
    last,best = w/'last.pt',w/'best.pt'
    opt.hyp = hyp
    # cuda = device.type != 'cpu'
    init_seeds(0)


    data_dict = check_dataset(ROOT)
    
    train_path,val_path = data_dict['train'],data_dict['val']
    LOGGER.info(f'train_path is {str(train_path)}')
    nc = data_dict['nc']
    names = data_dict['names']

    model = Net(ch=3,nc=nc,anchors=None).to(device)


    # gs = 32
    imgsz = 640

    nbs = 4    #nominal batch size
    accumulate = max(round(nbs/batch_size),1)
    hyp['weight_decay'] *= batch_size*accumulate/nbs
    optimizer = smart_optimizer(model,opt.optimizer,hyp['lr0'],hyp['momentum'],hyp['weight_decay'])

    lf = lambda x:(1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model)

    best_fitness, start_epoch = 0.0, 0

    train_loader,dataset = create_dataloader(
        train_path,
        imgsz=imgsz,
        batch_size=batch_size,
        stride=32,
        hyp=hyp,
        augment=True,
        # cache=None if opt.cache == 'val' else opt.cache,
        pad=0,
        rect=False,
        # workers=1,
        # image_weights = opt.image_weights,
        shuffle=True,
        seed=0
    )
    labels = np.concatenate(dataset.labels,0)

    val_loader = create_dataloader(
        path=val_path,
        imgsz=imgsz,
        batch_size=batch_size,
        stride=32,
        hyp=hyp,
        augment=False,
        pad=0.5,
        rect=True,
        shuffle=False,
        seed=0
    )[0]

    train_val_loader = create_dataloader(
        path=train_path,
        imgsz=imgsz,
        batch_size=batch_size,
        stride=32,
        hyp=hyp,
        augment=False,
        pad=0.5,
        rect=True,
        shuffle=False,
        seed=0
    )[0]

    model.half().float()
    nl = 3     # number of detection layers (to scale hyps)
    hyp['box'] *= 3/nl
    hyp['cls'] *= nc/80*3/nl
    hyp['obj'] *=(imgsz/640)**2*3/nl
    hyp['label_smoothing'] = opt.label_smoothing
    
    model.nc = nc
    model.hyp = hyp
    model.names = names
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    t0 = time.time()
    nb = len(train_loader)    
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  #number of warmup 

    maps = np.zeros(nc)

    results = (0,0,0,0,0,0,0)   #P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    scheduler.last_epoch = start_epoch - 1

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    stopper,stop = EarlyStopping(patience=opt.patience), False
    computeloss = ComputeLoss(model,autobalance=False)
    last_opt_step = -1
    
    
    wandb.init(project = "yolov5-augment")
    
    for epoch in range(start_epoch,epochs):
        model.train()

        # if opt.image_weigts:
        #     pass
        
        mloss = torch.zeros(3,device=device)
        pbar = enumerate(train_loader)

        pbar = tqdm(pbar,total=nb,bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()
        for i,(imgs,targets,_,_) in pbar:
            
            

            ni = i+nb*epoch
            if ni<2:
                uploadiamge(imgs,targets)

            imgs = imgs.to(device,non_blocking=True).float()/255

            
            
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1,np.interp(ni,xi,[1,nbs/batch_size]).round())
                # for j,x in enumerate(optimizer.param_groups):
                #     pass


            # if epoch < 5:
            #     print(targets[0])
            #     print(targets[1])
            #     print(imgs[0,0,0])
            
            pred = model(imgs)
            
            # if epoch < 5:
            #     print(f'pred is{pred[0][0,0,0,5,:]}')

            loss, loss_items = computeloss(pred,targets.to(device))
            loss.backward()
            # if epoch < 3:
            #     print(f'loss is{loss}')
            # with torch.cuda.amp.autocast(False):
            #     pred = model(imgs)
            #     loss, loss_items = computeloss(pred,targets.to(device))

            # scaler.scale(loss).backward()
            if ni - last_opt_step  >= accumulate:
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()
                last_opt_step = ni
                if ema:
                    ema.update(model)
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
                # optimizer.zero_grad()
                # if ema:
                #     ema.update(model)
                # last_opt_step = ni
            # pbar.set_description()
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                


        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        ema.update_attr(model,include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        
        if epoch == 10:
            pass
        if epoch == 50:
            pass
        if epoch == 80:
            pass

        final_epoch = (epoch+1 ==epochs) or stopper.possible_stop
        with torch.inference_mode():
            val_results, _ = validate(
                data=data_dict,
                model=ema.ema,
                names=names,
                dataloader=val_loader,
                compute_loss=computeloss,
                half=False,
                epoch=epoch
            )

            train_results, _ = validate(
                data=data_dict,
                model=ema.ema,
                names=names,
                dataloader=train_val_loader,
                compute_loss=computeloss,
                half=False,
                epoch=epoch
            )
            

        fi = fitness(np.array(results).reshape(1,-1))
        stop = stopper(epoch=epoch,fitness=fi)

        if fi > best_fitness:
            best_fitness = fi
        

        
        ckpt = {
            'epoch':epoch,
            'best_finess':best_fitness,
            'model':deepcopy(model).float(),
            'optimizer':optimizer.state_dict,
            'opt':vars(opt),
            'date':datetime.now().isoformat()
        }
        if final_epoch:
            torch.save(ckpt,last)
        if best_fitness == fi:
            torch.save(ckpt,best)
        if opt.save_period > 0 and epoch % opt.save_period==0:
            torch.save(ckpt,w/f'epoch{epoch}.pt')
        del ckpt
        wandb.log({
            'epoch':epoch,
            'v_mp':val_results[0],
            'v_mr':val_results[1],
            'v_map50':val_results[2],
            'v_map':val_results[3],
            'v_lbox':val_results[4],
            'v_lobj':val_results[5],
            'v_lcls':val_results[6],
            'train_mp':train_results[0],
            'train_mr':train_results[1],
            'train_map50':train_results[2],
            'train_map':train_results[3],
            'train_lbox':train_results[4],
            'train_lobj':train_results[5],
            'train_lcls':train_results[6],



            't_lbox':mloss[0],
            't_lobj':mloss[1],
            't_lcls':mloss[2],
            'lr0':lr[0],
            'lr1':lr[1],
            'lr2':lr[2]

        })
    wandb.finish()
    torch.cuda.empty_cache()
    return results
    




class_name = ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']

def xywhn2xyxy(x,w=640,h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = w * (x[..., 2] - x[..., 4] / 2)   # top left x
    y[..., 3] = h * (x[..., 3] - x[..., 5] / 2)   # top left y
    y[..., 4] = w * (x[..., 2] + x[..., 4] / 2)   # bottom right x
    y[..., 5] = h * (x[..., 3] + x[..., 5] / 2)   # bottom right
    return y

def uploadiamge(img,labs):
    """
    input   img [batch channel h w ] type:tensor
            labs [class,xywh]  type:tensor
    output  draw image and target
    
    """    
    uploadimgs=deepcopy(img).to('cpu')
    uploadlabs=deepcopy(labs).to('cpu')
    
    shape = uploadimgs.shape
    labs = xywhn2xyxy(uploadlabs,shape[3],shape[2])

    for i in range(uploadimgs.shape[0]):
        # lb = torch.ones(labs.shape)
        j = labs[:,0]==i
        lb = labs[j]
        im = Image.fromarray(uploadimgs[i].numpy().transpose(1,2,0))
        draw = ImageDraw.Draw(im)
        for k in range(lb.shape[0]):

            draw.rectangle(lb[k,2:].numpy(),outline='red')
            # print(f'text lab is {tuple(lb[k,2:4].numpy().astype(np.uint))}')
            draw.text(tuple(lb[k,2:4].numpy().astype(np.uint)),str(lb[k,1].numpy().astype(np.uint)),fill='red')
            
        del draw
        # im.show()
        #print(im.mode)
        # im.save('D:\python\yolov5-mysely\ccc.jpg',format='png')
        wandb.log({"train images": wandb.Image(im)})

    



def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if opt.resume:
        pass
    else:
        
        #opt.save_dir = str(increment_path(Path(opt.project)/'exp',exist_ok=False,mkdir=True))
        opt.save_dir = str(increment_path(Path('/content/drive/MyDrive/trian/exp'),exist_ok=False,mkdir=True))

    train(hyp, opt, device)
    




path = r'D:\\python\\my_pcb_dataset\\train\\'

hyp = {
    
    'weight_decay':0.0005,  # optimizer weight decay
    'lr0':0.01,
    'momentum':0.937,
    'lrf':0.01,
    
    'box':0.05,         # box loss gain
    'cls':0.5,          # cls loss gain
    'obj':1.0,          # obj loss gain

    'degrees':0.0,
    'translate':0.1,
    'scale':0.5,
    'shear':0.0,
    'fliplr':0.5,
    'mosaic':1.0,
    'mixup':0.0,
    'copy_paste':0.0,
    'perspective':0.0,
    'anchors':[[10,13,16,30,33,23],[30,61,62,45,59,119],[116,90,156,198,373,326]],
    'warmup_epochs':3,
    'label_smoothing':0,
    'cls_pw':1.0,
    'obj_pw':1.0,
    'anchor_t':4.0


    }

os.environ["WANDB_API_KEY"] = "67c7d73851439a0047abc953e8608567ec36bbbe"

if __name__ == '__main__':
    wandb.login()
    opt = parse_opt()
    main(opt)


