import torch
import math
import torch.nn as nn
from general import LOGGER


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
       

        m = model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch






# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
# ch = [128, 256, 512]

# def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
#     # return positive, negative label smoothing BCE targets
#     return 1.0 - 0.5 * eps, 0.5 * eps

# def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
#     # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

#     # Get the coordinates of bounding boxes
#     if xywh:  # transform from xywh to xyxy
#         (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
#         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#         b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#         b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     else:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
#         w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
#         w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

#     # Intersection area
#     inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
#             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

#     # Union Area
#     union = w1 * h1 + w2 * h2 - inter + eps

#     # IoU
#     iou = inter / union
#     if CIoU or DIoU or GIoU:
#         cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
#         ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
#         if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
#             if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps))
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU
#             return iou - rho2 / c2  # DIoU
#         c_area = cw * ch + eps  # convex area
#         return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#     return iou  # IoU


# class compute_loss:
#     sort_obj_iot = False
#     def __init__(self, model, autobalance=False,hyp=None) -> None:
#         # Define criteria
#         self.device = next(model.parameters()).device
#         self.anchors = model.anchors.to(self.device)
#         self.balance = [4.0, 1.0, 0.4]
#         self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=self.device))
#         self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=self.device))
#          # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=hyp.get('label_smoothing', 0.0))  # positive, negative BCE targets

        
#         self.hyp = hyp
#         self.nc = model.nc
#         LOGGER.info(f'anchors device is {self.anchors.device} model device is {self.device}')

#     def __call__(self, p,targets,*args, **kwdsy) :
#         lcls = torch.zeros(1,device=self.device)
#         lbox = torch.zeros(1,device=self.device)
#         lobj = torch.zeros(1,device=self.device)
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)
#         for i, pi in enumerate(p):
#             ima,a,gi,gj = indices[i]
#             tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

#             n = ima.shape[0]
#             if n:
#                 pxy, pwh, _, pcls = pi[ima, a, gj, gi].split((2, 2, 1, self.nc), 1) 
                
#                 pxy = pxy.sigmoid() * 2 - 0.5
#                 pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)

#                 iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
#                 lbox += (1.0 - iou).mean()  # iou loss
#                 iou = iou.detach().clamp(0).type(tobj.dtype)
#                 tobj[ima,a,gj,gi]=iou
#                 if self.nc>1:
#                     t = torch.full_like(pcls, self.cn, device=self.device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += self.BCEcls(pcls, t)

#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj']
#         lcls *= self.hyp['cls']
#         bs = tobj.shape[0]   #batch size

#         return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()



    # def build_targets(self,p,targets):
    #     na = 3                        #number of anchor
    #     nt = targets.shape[0]         #number of targets 
    #     tcls,tbox,indices,anch=[],[],[],[]
    #     gain = torch.ones(7,device=self.device)              #
    #     ai = torch.arange(na,device=self.device).float().view(na,1).repeat(1,nt)     #[na,nt]
    #     targets = torch.cat((targets.repeat(na,1,1),ai[...,None]),2)  # [na,nt,6] #[na,nt,1]->#[na,nt,7]
    #     g=0.5
    #     off=torch.tensor(
    #         [
    #         [0,0],
    #         [1,0],      #right
    #         [0,1],      #down
    #         [-1,0],     #left
    #         [0,-1],     #up
    #         ],
    #         device=device
    #     ).float()*g 
    #     for i in range(3):
    #         anchors, shape = self.anchors[i], p[i].shape
    #         gain[2:6] = torch.tensor(shape)[[3,2,3,2]]    #grid number [1,1,gn,gn,gn,gn,1]
    #         t = targets *gain     #
    #         ratio = t[...,4:6]/anchors[:,None]        #target w/anchor w
    #         j=torch.max(ratio,1/ratio).max(2)[0]<4      #[na,nt] bool
    #         t = t[j]
    #             #image target x y w h an
    #         gxy = t[:,2:4]          # grid x,y  target left up
    #         gxyi = gain[[2,3]]-gxy  # grid x,y target right down
    #         j, k = ((gxy % 1 < g) & (gxy > 1)).T
    #         l, m = ((gxyi % 1 < g) & (gxyi > 1)).T

    #         j=torch.stack((torch.ones_like(j), j, k, l, m))
    #         t=t.repeat((5, 1, 1))[j]
    #         offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

    #         ic, gxy, gwh, a = t.chunk(4, 1)  #(image class) grid xy grid wh anchor
    #         a = a.long().view(-1)
    #         ima,c= ic.long().T
    #         gij = (gxy - offsets).long()
    #         gi,gj=gij.T
    #         tcls.append(c)
    #         tbox.append(torch.cat((gxy-gij, gwh), 1))
    #         indices.append((ima,a, gi.clamp_(0, shape[3] - 1), gj.clamp_(0, shape[2] - 1)))
    #         anch.append(anchors[a])

    #     return tcls, tbox, indices, anch




