import torch
import numpy as np
import contextlib
import time
from tqdm import tqdm
import torchvision
from general import LOGGER
from copy import deepcopy
from PIL import Image,ImageDraw

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
def validate(
      data,
      model = None,
      names = None,         # dict
      dataloader = None,
      compute_loss = None,
      augment = False,
      epoch = 0,
      half = False,
      max_det = 300,
      iou_thres = 0.6,
      conf_thres = 0.001  
):
    """
    
    """
    
    
    device, pt, jit, engine = next(model.parameters()).device, True, False, False
    half &= device.type != 'cpu'
    model.half() if half else model.float()
    
    model.eval()  #configure Dropout和BatchNorm
    cuda = device.type != 'cpu'
    nc = int(data['nc'])
    iouv = torch.linspace(0.5,0.95,10,device=device)
    niou = iouv.numel()

    seen = 0
    # confusion_matrix = confusionMatrix(nc=nc)
    names = names
    class_map = list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3,device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)

    for batch_i,(im,targets,shapes,aaa) in enumerate(pbar):
        if epoch<1 and batch_i<1:
            uploadiamge(im,targets)
            print(f'index is {aaa}')
            
        
        
        if cuda:
            im = im.to(device,non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()
        im /= 255
        nb,_,height,width = im.shape
        preds, train_out = model(im) #if compute_loss else(model(im),None)
        if compute_loss:
            loss += compute_loss(train_out,targets)[1] #lbox, lobj, lcls
        targets[:,2:] *= torch.tensor((width,height,width,height),device=device)
        # lb = [targets[targets[:,0] == i,1:] for i in range(nb)]
        preds = non_max_suppression(
            preds,
            conf_thres,
            iou_thres,
            # lables = lb,
            max_det = max_det
        
        )

        for si, pred in enumerate(preds):
            labels = targets[targets[:,0]==si, 1:]
            nl,npr = labels.shape[0],pred.shape[0]      # number of labels, predictions
            shape = shapes[si][0]
            correct = torch.zeros(npr,niou,dtype=torch.bool,device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct,*torch.zeros((2,0),device=device),labels[:,0]))
                continue
            predn = pred.clone()
            scale_boxes(im[si].shape[1:],predn[:,:4],shape,shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:,1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    stats = [torch.cat(x,0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        _, _, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
        ap50,ap=ap[:,0],ap.mean(1)
        mp,mr,map50,map = p.mean(),r.mean(),ap50.mean(),ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    model.float()
    maps = np.zeros(nc) + map
    
    for i,c in enumerate(ap_class):
        maps[c]=ap[i]
    return (mp,mr,map50,map,*(loss.cpu() / len(dataloader))),maps

def xywhn2xyxy(x,w=640,h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = w * (x[..., 2] - x[..., 4] / 2)   # top left x
    y[..., 3] = h * (x[..., 3] - x[..., 5] / 2)   # top left y
    y[..., 4] = w * (x[..., 2] + x[..., 4] / 2)   # bottom right x
    y[..., 5] = h * (x[..., 3] + x[..., 5] / 2)   # bottom right
    return y


import wandb
class_name = ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']
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
            print(f'text lab is {tuple(lb[k,2:4].numpy().astype(np.uint))}')
            draw.text(tuple(lb[k,2:4].numpy().astype(np.uint)),str(lb[k,1].numpy().astype(np.uint)),fill='red')
        del draw
        # im.show()
        #print(im.mode)
        # im.save('D:\python\yolov5-mysely\ccc.jpg',format='png')
        wandb.log({"val images": wandb.Image(im)})



def ap_per_class(tp, conf, pred_cls, target_cls,  names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)





# def ap_per_class(
#         tp, 
#         conf, 
#         pred_cls, 
#         target_cls,
#         names = None, 
#         eps=1e-16 
#         ):
#     """
#         Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

#         input   tp          []
#                 conf        []
#                 pred_cls    []
#                 target_cls  
#                 names
#                 eps             zuixiaozhi
#         output  tp
#                 fp
#                 p
#                 r
#                 f1
#                 ap


#     """
#     i = np.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i],pred_cls[i]

#     unique_classes,nt = np.unique(target_cls,return_counts=True)
#     nc = unique_classes.shape[0]

#     px, py = np.linspace(0,1,1000),[]
#     ap,p,r = np.zeros((nc,tp.shape[1])),np.zeros((nc,1000)),np.zeros((nc,1000))
#     for ci,c in enumerate(unique_classes):
#         i = pred_cls == c
#         n_l = nt[ci]
#         n_p = i.sum()
#         if n_p == 0 or n_l == 0 :
#             continue
#         fpc = (1-tp[i]).cumsum(0)
#         tpc = tp[i].cumsum(0)

#         recall = tpc / (n_l+eps)
#         r[ci] = np.interp(-px,-conf[i],recall[:,0],left=0)

#         precision = tpc / (tpc + fpc)
#         p[ci] = np.interp(-px,-conf[i],precision[:,0],left=1)

#         for j in range(tp.shape[1]):
#             ap[ci,j],_,_ = compute_ap(recall[:,j],precision[:,j])
#     f1 = 2*p*r/(p + r + eps)

#     i = smooth(f1.mean(0),0.1).argmax()
#     p,r,f1 = p[:,i], r[:,i],f1[:,i]
#     tp = (r*nt).round()
#     fp = (tp/(p+eps)-tp).round()
#     return tp,fp,p,r,f1,ap,unique_classes.astype(int)


# def smooth(y, f=0.05):
#     """
#     mean filtering
#     input   y[n] 
#             f ratio
#     output  after filter value  [n] 

    
#     """
#     nf = round(len(y) * f * 2) // 2 + 1 
#     p = np.ones(nf // 2)  
#     yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  
#     return np.convolve(yp, np.ones(nf) / nf, mode='valid')  



# def compute_ap(
#     recall, 
#     precision    
# ):
#     """
#     input   recall [ ] 
#             precision [ ]
#     output  ap      
#             mpr     
#             mrec    
    
#     """
#     mrec = np.concatenate(([0.0], recall, [1.0]))
#     mpre = np.concatenate(([1.0], precision, [0.0]))

#     mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
#     method = 'interp'  # methods: 'continuous', 'interp'
#     if method == 'interp':
#         x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
#         ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
#     else:  # 'continuous'
#         i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

#     return ap, mpre, mrec





# def process_batch(
#         detections, 
#         labels, 
#         iouv
#         ):
#     """
#     input   detections  [N,6]   x1 y1 x2 y2 conf class
#             labels      [M 5]   class x1 y1 x2 y2
#             iouv        [10]  0.5-0.95
#     output  correct     [ ,10]  0.5-0.95 bool
       
#     """
#     correct = np.zeros((detections.shape[0],iouv.shape[0])).astype(bool)
#     iou = box_iou(labels[:,1:],detections[:,:4])
#     correct_class = labels[:,0:1]==detections[:,5]
#     for i in range(len(iouv)):
#         x = torch.where((iou > iouv[i]) & correct_class) 
#         if x[0].shape[0]:
#             matches = torch.cat((torch.stack(x,1),iou[x[0],x[1]][:,None]),1).cpu().numpy()
#             if x[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#             correct[matches[:,1].astype(int),i] = True
#         return  torch.tensor(correct, dtype=torch.bool, device=iouv.device)  





def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        input   img1_shape  h w
                boxes       x1 y1 x2 y2
                img0_shape  h w
                ratio_pad
        output  boxes       x1 y1 x2 y2
    """
    
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    """
        Clip boxes (xyxy) to image shape (height, width)
        input   boxes
                shape
        output  clip boxes 
    """
    
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def non_max_suppression(
        predition,
        # labels=(),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300        
):
    """
    
        input predition[ ]  batch_size   n   85
        output [ ] n*6  x1 y1 x2 y2 conf class
    """
    device = predition.device
    bs = predition.shape[0]
   
    xc = predition[...,4]>conf_thres
    
    max_wh  = 7680  #(pixels) maximum box width and height
    max_nms = 30000 #maximum number of boxes into torchvision.ops.nms()'
    

    output = [torch.zeros((0,6),device=device)]*bs
    for xi ,x in enumerate(predition):
        x=x[xc[xi]]
        # if labels and len(labels[x]):
        #     lb = labels[xi]
        #     v = torch.zeros((len(lb), nc + 5), device=x.device)
        #     v[:, :4] = lb[:, 1:5]  # box
        #     v[:, 4] = 1.0  # conf
        #     v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
        #     x = torch.cat((x, v), 0)
        
        if not x.shape[0]:
            continue
        x[:,5:] *= x[:,4:5]  # conf = obj_conf * cls_conf

        box = xywh2xyxy(x[:,:4])

        i,j = (x[:,5:] > conf_thres).nonzero(as_tuple = False).T
        x = torch.cat((box[i],x[i,5+j,None],j[:,None].float()),1)
        
        n = x.shape[0]
        if not n:
            continue
        x = x[x[:,4].argsort(descending=True)[:max_nms]]

        c =x[:,5:6]*max_wh
        boxes,scores = x[:,:4] + c,x[:,4]
        i = torchvision.ops.nms(boxes=boxes,scores=scores,iou_threshold=iou_thres)
        i = i[:max_det]
        


        output[xi] = x[i]

    return output



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y









def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

            
            
class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()
        
   
    



