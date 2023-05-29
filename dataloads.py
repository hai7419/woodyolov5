import torch
from torch.utils.data import DataLoader, Dataset, distributed
import glob
import os
from pathlib import Path
import cv2
import math
import numpy as np
from augmentaton import letterBox,randon_perspective
from PIL import Image,ImageDraw,ExifTags,ImageOps
import random
import psutil
from tqdm import tqdm
import contextlib
from multiprocessing.pool import Pool,ThreadPool
import torch.distributed as dist
from general import LOGGER
from contextlib import contextmanager
import hashlib
from itertools import repeat


NUM_THREADS = min(4, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
class yolodateset(Dataset):
    def __init__(
            self,
            path,
            img_size=640,
            batch_size = 4,
            hyp = None,
            augment = False,
            rect = False,
            stride = 32,
            pad = 0
            
            ) -> None:
        super().__init__()
        self.path = path
        self.im_size = img_size
        self.im_files = sorted(glob.glob(os.path.join(self.path,'*.jpg')))
        self.im_labs =  img2label_paths(self.im_files)
        self.lab_files = []
        # self.img = []
        self.hyp = hyp
        self.rect = rect
        self.augment = augment
        self.mosaic =  augment and not rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        
        
        # for im_files in self.im_files:
        #         self.img.append(cv2.imread(im_files))
         
        

        # print(f'lab file is {self.im_labs[0]}')
        # print(f'lab file is {self.im_labs[1]}')
       
        # print(f'im file is {self.im_files[0]}')
        # print(f'im file is {self.im_files[1]}')
        
        # print(f'labels 0 is {self.lab_files[0]}')
        # print(f'labels 1 is {self.lab_files[1]}')


        cache_path = Path(self.im_labs[0]).parent.with_suffix('.cache')
        # try:

        #     cache,exists = np.load(cache_path,allow_pickle=True)
        # except:
        cache,exists = self.cache_labels(cache_path), False
        
        (nf,nm,ne,nc,n) = cache.pop('results')
        if exists:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
        
        [cache.pop('msgs')]
        labels,shapes = zip(*cache.values())
        

        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())

       

        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        if self.rect:
             # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.im_labs = [self.im_labs[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        for lb_file in self.im_labs:
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            self.lab_files.append(lb)
        cache_images = True
        if not self.check_cache_ram():
            cache_images = False
        self.ims = [None] * n
        # self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        # self.npy_files[None] *n


        if cache_images:
            # b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            # fcn = self.load_img
            # results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            # pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT)
            # print(f'n is {n}')
            for i in range(n):
                # print(f'n is {i},im file is {self.im_files[i]} lab file is {self.im_labs[i]}')
                self.ims[i], self.im_hw0[i], self.im_hw[i] = self.load_img(i)  # im, hw_orig, hw_resized = load_image(self, i)


                


    def __len__(self):
        return len(self.im_files)
    def __getitem__(self, index):
        
        hyp = self.hyp
        mosaic = self.mosaic #and random.random() < hyp['mosaic']
        shapes = None
        aaa = [index]
        if mosaic:
            img, labels = self.load_mosaic(index)
            shapes = None

        else:

            
            img, (h0,w0),(h, w) = self.load_img(index)
            shape = self.batch_shapes[self.batch[index]] if self.rect   else (self.im_size,self.im_size)
            img,r,dw,dh = letterBox(im=img,new_shape=shape,auto=False)

            shapes = (h0,w0),((h/h0,w/w0),(dw,dh))

            labels = self.lab_files[index].copy()
            
            if labels.size:
                labels[:,1:] = self.xywhn2xyxy(labels[:,1:],r*w,r*h,dw,dh)
                # labels[:,1:] = self.xyxy2xywhn(labels[:,1:],img.shape[1],img.shape[0])
            # if index <4:
            #     print(f'index {index} labels is {labels}')

            if self.augment:
                img,labels = randon_perspective(
                im=img,
                targets=labels,
                degrees=hyp['degrees'],
                translate=hyp['translate'],
                scale=hyp['scale'],
                shear=hyp['shear'],
                perspective=hyp['perspective']
                )

        num_lab = len(labels)
        labels_out = torch.zeros((num_lab, 6))
        if num_lab:
            labels[:, 1:5] = self.xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2,0,1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # if index <4:
        #         print(f'index {index} labels is {labels}')

        # if index <4:
        #         print(f'index {index} labels_out is {labels_out}')

        return torch.from_numpy(img),labels_out,shapes,aaa
    
    
    def check_cache_ram(self,safety_margin=0.1):
        #check image cacheing requirements available memory
        b = 0
        n = min(self.n,30)
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))
            ratio = self.im_size / max(im.shape[0],im.shape[1])
            b += im.nbytes*ratio*ratio

        mem_required = b*self.n/n
        mem = psutil.virtual_memory() #  total, available, percent, used  
        cache = mem_required*(1+safety_margin)<mem.available
        
        LOGGER.info(f"{'caching images ' if cache else 'not caching images '}")

        return cache
    def cache_labels(self,path=Path('./labels.cache')):
        """
        
        input   path  cache path
        output  x{}     'im_file':[lb, shape] 
                                        lb, [cxywh] Normalization
                                        shape,[h,w]image shape
                        'results':nf,nm,ne,nc,len(self.im_files),
                        'msgs':msgs
                        '

        """
        x = {}
        nm, nf, ne, nc, msgs = 0,0,0,0,[]  ## number missing, found, empty, corrupt, messages
        desc = f'Scanning{path.parent/path.stem}'
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label,zip(self.im_files,self.im_labs)),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=TQDM_BAR_FORMAT
                        )
            for im_file, lb, shape,nm_f,nf_f,ne_f,nc_f,msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'



            pbar.close()
        if nf == 0:
            pass

        x['results'] = nf,nm,ne,nc,len(self.im_files)
        x['msgs'] = msgs

        try:
            np.save(path , x)
            path.with_suffix('.cache.npy').rename(path)
        except:
            pass
        return x
    # def cache_images_to_disk(self, i):
    #         # Saves an image as an *.npy file for faster loading
    #         f = self.npy_files[i]
    #         if not f.exists():
    #             np.save(f.as_posix(), cv2.imread(self.im_files[i]))
    
    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        """
            output  img [hwc]   type:numpy
                    lables [class xyxy]  pixels
        """
        labels4 = []
        s = self.im_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_img(i=index) 

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.labels[index].copy()#, self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                # segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            # segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:]):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = randon_perspective(img4,
                                           labels4,
                                           degrees=self.hyp['degrees'],# degrees=self.hyp['degrees'],
                                           translate= self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=0,#self.hyp['perspective'],
                                           border=self.mosaic_border
                                           )  # border to remove

        return img4, labels4
    

    def load_img(self,i):
        """
                no pad
                input   index 
                output  im      [hwc]   type:numpy
                        (h0,w0)   image original h  w    
                        im.shape[:2]  image resize h w
        """
        
        im, f,  = self.ims[i], self.im_files[i]     #, self.npy_files[i],
        if im is None:  # not cached in RAM
            # if fn.exists():  # load npy
            #     im = np.load(fn)
            # else:  # read image
            im = cv2.imread(f)  # BGR
            assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.im_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized



       
        
    def xywhn2xyxy(self,x,w=640,h=640,dw=0,dh=0):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + dw  # top left x
        y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + dh  # top left y
        y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + dw  # bottom right x
        y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + dh  # bottom right
        return y
    def xyxy2xywhn(self,x,w=640, h=640,clip = True,eps=1E-3):
        if clip:
            
            if isinstance(x, torch.Tensor):  # faster individually
                x[..., 0].clamp_(0, w-eps)  # x1
                x[..., 1].clamp_(0, h-eps)  # y1
                x[..., 2].clamp_(0, w-eps)  # x2
                x[..., 3].clamp_(0, h-eps)  # y2
            else:  # np.array (faster grouped)
                x[..., [0, 2]] = x[..., [0, 2]].clip(0, w-eps)  # x1, x2
                x[..., [1, 3]] = x[..., [1, 3]].clip(0, h-eps)
        
        
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
        y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
        y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
        y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
        return y
    @staticmethod
    def collate_fn(data):
        im,lb,shapes,aaa = zip(*data)
        for i,lab in enumerate(lb):
            lab[:,0] = i
        return torch.stack(im,0) ,torch.cat(lb,0),shapes,aaa


        


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

class_name = ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']

def xywh2xyxy(x,w=640,h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = w * (x[..., 2] - x[..., 4] / 2)   # top left x
    y[..., 3] = h * (x[..., 3] - x[..., 5] / 2)   # top left y
    y[..., 4] = w * (x[..., 2] + x[..., 4] / 2)   # bottom right x
    y[..., 5] = h * (x[..., 3] + x[..., 5] / 2)   # bottom right
    return y

def draw_targets(img,labs):
    """
    input   img [batch channel h w ] type:tensor
            labs [class,xywh]  type:tensor
    output  draw image and target
    
    """    
    shape = img.shape
    labs = xywh2xyxy(labs,shape[2],shape[3])

    for i in range(img.shape[0]):
        # lb = torch.ones(labs.shape)
        j = labs[:,0]==i
        lb = labs[j]
        im = Image.fromarray(img[i].numpy().transpose(1,2,0))
        draw = ImageDraw.Draw(im)
        for k in range(lb.shape[0]):

            draw.rectangle(lb[k,2:].numpy(),outline='red')
            draw.text(lb[k,2:4].numpy().astype(np.uint)+[0,-8],class_name[lb[k,1].numpy().astype(np.uint)],fill='red')
        del draw
        im.show()
        im.save('D:\python\yolov5-mysely\ccc.jpg',format='png')

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s



def verify_image_label(args):
    """
        input       argss:im_fiel,lb_file
        output      im_file     im_fiel  
                    lb,         [cxywh] Normalization
                    shape,      [h,w]image shape
                    nm,         if nm = 1 label missing  
                    nf,         if nf = 1 label found
                    ne,         if ne = 1 label empty
                    nc          if nc = 1 file corrupt
                    msg 
    """
    
    im_file,lb_file=args
    nm, nf, ne, nc, msg, = 0, 0, 0, 0, ''
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        # assert (shape[0]>9 & shape[1]>9),f'image size{shape}<10 pixels'
        if im.format.lower() in ('jpg','jpeg'):
            with open(im_file,'rb') as f:
                f.seek(-2,2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'WARNING  {im_file}: corrupt JPEG restored and saved'

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)

            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 {lb.shape[1]} columns detected'
                assert (lb>=0).all,f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'

                _, i = np.unique(lb,axis=0,return_index=True)
                if len(i)<nl:
                    lb=lb[i]
            else:
                ne = 1   #label empty
                lb = np.zeros((0,5),dtype=np.float32)
        else:
            nm = 1  #label missing
            lb = np.zeros((0,5),dtype=np.float32)
        return  [im_file, lb, shape, nm, nf, ne, nc, msg]   

    except Exception as e:
        nc = 1
        msg = f'WARNING {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, nm, nf, ne, nc, msg]
        


# class InfiniteDataLoader(DataLoader):
#     """ Dataloader that reuses workers

#     Uses same syntax as vanilla DataLoader
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
#         self.iterator = super().__iter__()

#     def __len__(self):
#         return len(self.batch_sampler.sampler)

#     def __iter__(self):
#         for _ in range(len(self)):
#             yield next(self.iterator)


# class _RepeatSampler:
#     """ Sampler that repeats forever

#     Args:
#         sampler (Sampler)
#     """

#     def __init__(self, sampler):
#         self.sampler = sampler

#     def __iter__(self):
#         while True:
#             yield from iter(self.sampler)

# def seed_worker(worker_id):
#     # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def create_dataloader(
        path,
        imgsz,
        batch_size,
        stride,
        hyp=None,
        augment=False,
        pad=0.0,
        rect=False,
        shuffle=False,
        seed=0
    ):
    if rect and shuffle:
        LOGGER.warning('WARNING  --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(-1):
        dataset = yolodateset(
            path = path,
            img_size = imgsz,
            batch_size = batch_size,
            hyp=hyp,  # hyperparameters
            augment=augment,  # augmentation
            rect=rect,  # rectangular batches
            stride=stride,
            pad=pad
            )

    batch_size = min(batch_size, len(dataset))
    # nd = torch.cuda.device_count()  # number of CUDA devices
    # nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
     
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed -1)
    return DataLoader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle,
                  num_workers=1,
                  sampler=None,
                  pin_memory=PIN_MEMORY,
                  collate_fn= yolodateset.collate_fn,
                #   worker_init_fn=seed_worker,
                  generator=generator), dataset



@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

if __name__ == "__main__":
    
    
    path = r'D:\\python\\datasets\\pcb\\images\\train'

    hyp = {'box':0.05,
        'degrees':0,
        'translate':0,
        'scale':0,
        'shear':0.0,
        'fliplr':0.5,
        'mosaic':1.0,
        'mixup':0.0,
        'copy_paste':0.0,
        'perspective':0
        }
    
    
    
    
    yolov5dataloader, yolov5dataset = create_dataloader(
        path=Path(path),
        imgsz=640,
        batch_size=4,
        stride=32,
        hyp=hyp,
        augment=True,
        pad=0.0,
        rect=False,
        # workers=1,
        shuffle=False,
        seed=0
    )


    for im,lab,_,_ in yolov5dataloader:
        
        draw_targets(im,lab)
        
        # plt.figure()
        # plt.axis('off')
        # for i in range(im.shape[0]):
        #     plt.imshow(im[i].numpy().transpose(1,2,0))
        # plt.ion()
        # plt.pause(30)
        # plt.close()
        
        # for i in range(im.shape[0]): 
        #     image = Image.fromarray(im[i].numpy().transpose(1,2,0))
        #     draw = ImageDraw.Draw(image)
        #     draw.rectangle(lab[:,1])
        #     draw.text([100, 200], text="pcb", font=font, fill=(100, 185, 179))
        #     del draw
        #     image.show()

        # for i in range(im.shape[0]):
        #     imagee = im[i].numpy().transpose(1,2,0)
        #     image = cv2.rectangle(image, start_point, end_point, color, thickness)
        #     image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA) 
        #     cv2.imshow('pcb',imagee)


    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # def visualize_bbox(img, bbox, class_name, color, ax):
    #     """
    #     img:图片数据 (H, W, C)数据格式
    #     bbox:array或者tensor， 假定数据格式是 [x_mid, y_mid, width, height]
    #     classname:str 目标框对应的种类
    #     color:str
    #     thickness:目标框的宽度
    #     """
    #     x_mid, y_mid, width, height = bbox
    #     x_min = int(x_mid - width / 2)
    #     y_min = int(y_mid - height / 2)
    #     # 画目标检测框
    #     rect = patches.Rectangle((x_min, y_min), 
    #                                 width, 
    #                                 height, 
    #                                 linewidth=3,
    #                                 edgecolor=color,
    #                                 facecolor="none"
    #                                 )
    #     ax.imshow(img)
    #     ax.add_patch(rect)
    #     ax.text(x_min + 1, y_min - 3, class_name, fontSize=10, bbox={'facecolor':color, 'pad': 3, 'edgecolor':color})

    # def visualize(img, bboxes, category_ids, category_id_to_name, category_id_to_color):
    #     fig, ax = plt.subplots(1, figsize=(8, 8))
    #     ax.axis('off')
    #     for box, category in zip(bboxes, category_ids):
    #         class_name = category_id_to_name[category]
    #         color = category_id_to_color[category]
    #         visualize_bbox(img, box, class_name, color, ax)
    #     plt.show()