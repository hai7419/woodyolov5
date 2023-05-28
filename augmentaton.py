import albumentations as A
import random
import numpy as np
import cv2
import math







class Albumentations:
    """
    
    """
    
    def __init__(self,size=640) -> None:
        T = [
            # 
            A.RandomSizedCrop(height=size,width=size,scale=(0.8,1.0),ratio=(0.9,1.11),p=0.0),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            # Contrast Limited Adaptive Histogram Equalization
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75,p=0.0)
        ]
        self.transform = A.Compose(T,bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels']))

    def __call__(self, im, labels, p=1.0 ) :
        if self.transform and random.random()<p:
            new = self.transform(image=im,bboxes=labels[:,1:],class_labels=labels[:,0])
            im ,labels = new['image'],np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]) 
        return im,labels
    

def letterBox(im, new_shape=(640,640), color=(114,114,114),auto=True, scaleup = False, stride=32):
        """
        imput   im [h,w,c] 
                new_shape [h,w] mod stride
                auto=True,   minimum rectangle
                scaleup = True  only scale down, do not scale up
                stride    network stride
        output im   [h,w,c]shape is new_shape while auto false
                    [h,w,c]shape is minimum rectangle while auto True
                r   image resize ratio
                dw,dh The pad size of each border of the image

        """
        
        shape = im.shape[:2]

        r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
        if not scaleup: # only scale down, do not scale up
             r= min(r,1.0)
       

        new_nupad = int(round(shape[1]*r)),int(round(shape[0]*r))
        dh,dw = new_shape[0]-new_nupad[1],new_shape[1]-new_nupad[0]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)     
        
        
        dh /= 2
        dw /= 2

        if shape[::-1] != new_nupad:
            im = cv2.resize(im, new_nupad,interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        
        im = cv2.copyMakeBorder(im,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)
        return im,r,dw,dh


def mosaic():
     pass


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates



def randon_perspective(
                        im,
                        targets=(),
                        degrees=10,
                        translate=.1,
                        scale=.1,
                        shear=10,
                        perspective=0.0,
                        border=(0, 0)
                    ):
    """
    https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    input   im  [h w c]     no batch
            targets=(),     [class x y x y]
            degrees=10,     degree
            translate=.1,   
            scale=.1,
            shear=10,       degree
            perspective=0.0,
            border=(0, 0)
    output  im  [h w c]
            targets     [class x y x y]
    """
    height = im.shape[0] + border[0]*2
    width = im.shape[0] + border[0]*2
                            #1  0  x
                            #0  1  y
    C = np.eye(3)           #0  0  1            
    C[0,2] = -im.shape[1]/2     #  x translation (pixels)             
    C[1,2] = -im.shape[0]/2     #  y translation (pixels)   

                            #1  0  0
                            #0  1  0
    P = np.eye(3)           #x  y  1
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

                            #S  R  0
    # Rotation and Scale    #R  S  0
    R = np.eye(3)           #0  0  1
    a = random.uniform(-degrees,degrees)
    s = random.uniform(1-scale,1+scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a,center=(0,0),scale=s)

                            #1  x  0
                            #x  1  0
    # Shear                 #0  0  1
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

                            #1  0  x
                            #0  1  y
    # Translation           #0  0  1
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114,114,114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114,114,114))


    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        xy = np.ones((n * 4, 3)) 
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine



        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr= 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets






if __name__ == "__main__":
     a=[]