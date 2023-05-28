import torch
import torch.nn as nn
import math
import thop
from copy import deepcopy
from pathlib import Path
from general import LOGGER
import warnings






import platform
def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
import pkg_resources as pkg
def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result

class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid



def model_info(model, verbose=False, imgsz=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
    except Exception:
        fs = ''

    # for layer in model.modules():
    #     print(f'{layer}')


    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f'{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')



class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # def _profile_one_layer(self, m, x, dt):
    #     c = m == self.model[-1]  # is final layer, copy input as inplace fix
    #     o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
    #     t = time_sync()
    #     for _ in range(10):
    #         m(x.copy() if c else x)
    #     dt.append((time_sync() - t) * 100)
    #     if m == self.model[0]:
    #         LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
    #     LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
    #     if c:
    #         LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
    #     LOGGER.info('Fusing layers... ')
    #     for m in self.model.modules():
    #         if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
    #             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #             delattr(m, 'bn')  # remove batchnorm
    #             m.forward = m.forward_fuse  # update forward
    #     self.info()
    #     return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True







def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

import contextlib
def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        # LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Conv, Bottleneck, SPPF,  BottleneckCSP, C3, nn.ConvTranspose2d}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3,}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            # if m is Segment:
            #     args[3] = make_divisible(args[3] * gw, 8)
        # elif m is Contract:
        #     c2 = ch[f] * args[0] ** 2
        # elif m is Expand:
        #     c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)




class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # def _forward_augment(self, x):
    #     img_size = x.shape[-2:]  # height, width
    #     s = [1, 0.83, 0.67]  # scales
    #     f = [None, 3, None]  # flips (2-ud, 3-lr)
    #     y = []  # outputs
    #     for si, fi in zip(s, f):
    #         xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
    #         yi = self._forward_once(xi)[0]  # forward
    #         # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
    #         yi = self._descale_pred(yi, fi, si, img_size)
    #         y.append(yi)
    #     y = self._clip_augmented(y)  # clip augmented tails
    #     return torch.cat(y, 1), None  # augmented inference, train

    # def _descale_pred(self, p, flips, scale, img_size):
    #     # de-scale predictions following augmented inference (inverse operation)
    #     if self.inplace:
    #         p[..., :4] /= scale  # de-scale
    #         if flips == 2:
    #             p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
    #         elif flips == 3:
    #             p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
    #     else:
    #         x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
    #         if flips == 2:
    #             y = img_size[0] - y  # de-flip ud
    #         elif flips == 3:
    #             x = img_size[1] - x  # de-flip lr
    #         p = torch.cat((x, y, wh, p[..., 4:]), -1)
    #     return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Net = DetectionModel




# class Conv(nn.Module):
#     def __init__(self, c1,c2,k,s,p=0,*args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv = nn.Conv2d(c1,c2,k,s,p,bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU()

#     def forward(self,x):
#         return self.act(self.bn(self.conv(x)))
    






# class Bottleneck(nn.Module):
#     def __init__(self,c1,c2, shortcut=True,*args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         c_ = int(c2*1)
#         self.cv1 = Conv(c1,c_,1,1)
#         self.cv2 = Conv(c_,c2,3,1,1)
#         self.add = shortcut and c1==c2

#     def forward(self,x):
#         return x+self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# class C3(nn.Module):
#     def __init__(self, c1,c2,n,shortcut=True,*args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         c_ = int(c2*0.5)
#         self.cv1 = Conv(c1,c_,1,1)
#         self.cv2 = Conv(c1,c_,1,1)
#         self.cv3 = Conv(2*c_,c2,1,1)
#         self.m = nn.Sequential(*(Bottleneck(c_,c_,shortcut) for _ in range(n)))
#     def forward(self,x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)),self.cv2(x)),1))
    

# class SPPF(nn.Module):
#     def __init__(self, c1,c2,k=5,*args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         c_=c1//2
#         self.cv1=Conv(c1,c_,1,1)
#         self.cv2=Conv(c_*4,c2,1,1)
#         self.m=nn.MaxPool2d(kernel_size=k,stride=1,padding=k//2)
#     def forward(self,x):
#         x=self.cv1(x)
#         y1=self.m(x)
#         y2=self.m(y1)
#         y3=self.m(y2)
#         return self.cv2(torch.cat((x,y1,y2,y3),1))


# class concat(nn.Module):
#     def __init__(self, d,*args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.d = d;
#     def forward(self,x):
#         return torch.cat(x,self.d)
        
# class Detect(nn.Module):
#     export = False
#     def __init__(self,nc=80, anchors=(),ch=(),*args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.nc= nc
#         self.no = nc+5
#         self.nl = len(anchors)  #number of detection layers
#         self.na = len(anchors[0])//2 #number of anctors
#         self.grid = [torch.empty(0)  for _ in range(self.nl)]  #init grid
#         self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  #init anchor grid
#         self.register_buffer('anchors',torch.tensor(anchors).float().view(self.nl,-1,2)) # shape(nl,na,2)
#         self.m=nn.ModuleList(nn.Conv2d(x,self.no*self.na,1) for x in ch)  #output conv
#         self.inplace = True
#         self.stride = torch.zeros(3)
#     def forward(self,x):
#         z=[]
#         for i in range(self.nl):
#             x[i]=self.m[i](x[i])
#             bs,_,ny,nx=x[i].shape  #bx(bs,255,20,20) to x(bs,3,20,20,85)
            
#             x[i]=x[i].view(bs,self.na,self.no,ny,nx).permute(0,1,3,4,2).contiguous()
#             if not self.training:
#                 self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
#                 xy,wh,conf = x[i].sigmoid().split((2,2,self.nc+1),4)
#                 xy = (xy*2+self.grid[i])*self.stride[i]
#                 wh = (wh*2)**2*self.anchor_grid[i]
#                 y=torch.cat((xy,wh,conf),4)
#                 z.append(y.view(bs,self.na*nx*ny,self.no))
#         return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
    
#     def _make_grid(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         t = self.anchors[i].dtype
#         shape = 1, self.na, ny, nx, 2  # grid shape
#         y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
#         yv, xv = torch.meshgrid(y, x, indexing='ij') #if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
#         grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
#         anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
#         return grid, anchor_grid
    
# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         super().__init__()
#         self.d = dimension

#     def forward(self, x):
#         return torch.cat(x, self.d)
        
                
# class Net(nn.Module):
#     def __init__(self, ch = 3, nc= 6, anchors = None,*args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)                   #input 640*640
#         self.names = [str(i) for i in range(nc)]
#         self.inplace = True
#         self.model = nn.Sequential(
#             *[Conv(3,32,6,2,2),                         # 0  32@320*320
#             Conv(32,64,3,2,1),                          # 1  64@160*160
#             C3(64,64,1),                                # 2  64@160*160
#             Conv(64,128,3,2,1),                         # 3  128@80*80
#             C3(128,128,2),                              # 4  128@80*80
#             Conv(128,256,3,2,1),                        # 5  256@40*40
#             C3(256,256,3),                              # 6  256@40*40
#             Conv(256,512,3,2,1),                        # 7  512@20*20
#             C3(512,512,1),                              # 8  512@20*20
#             SPPF(512, 512,5),                           # 9  512@20*20
#             Conv(512,256,1,1),                          #10  256@20*20
#             nn.Upsample(None,2,'nearest'),              #11  256@40*40            #12 upsampling 
#             Concat(1),                                  #12  cat #11#6  512@40*40
#             C3(512,256,1,False),                        #13  256@40*40
#             Conv(256,128,1,1),                          #14  128@40*40   
#             nn.Upsample(None,2,'nearest'),              #15  128@80*80
#             Concat(1),                                  #16 cat#15#4 256*80*8-
#             C3(256,128,1,False),                        #17 128@80*80 
#             Conv(128,128,3,2,1),                        #18 128@40*40
#             Concat(1),                                  #19 #18#14 256@40*40
#             C3(256,256,1,False),                        #20 256@40*40
#             Conv(256,256,3,2,1),                        #21 256@20*20
#             Concat(1),                                  #22 cat#21#10 512@20*20
#             C3(512,512,1,False),                        #23 512@20*20 
#             Detect(nc=nc, anchors=anchors, ch=[128,256,512]) ]       

#         )
#         m=self.model[-1]
#         m.inplace = self.inplace
        
#         forward = lambda x:self.forward(x)
        
#         s = 640
#         m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1,ch,s,s))])
#         m.anchors /= m.stride.view(-1,1,1)
#         self.stride = m.stride
#         self.anchors = m.anchors
#         self.nc = nc
#         self.initialize_biases()
#         self.initialize_weights()
#         self.info()
                      
#     def forward(self,x):        
#         trace = []
#         for i, layer in enumerate(self.model):
#             if i==12:
#                 x = layer([x,trace[1]])
#             elif i==16:
#                 x = layer([x,trace[0]])
#             elif i==19:
#                 x = layer([x,trace[3]])
#             elif i==22:
#                 x = layer([x,trace[2]])
#             elif i==24:
#                 x = layer([trace[4],trace[5],trace[6]])
#             else:
#                 x = layer(x)
#             if i in [4,6,10,14,17,20,23]:
#                 trace.append(x)
#         return  x
#     def initialize_biases(self,cf=None):
        
#         m = self.model[-1]
#         for mi,s in zip(m.m,m.stride):
#             b = mi.bias.view(m.na,-1)
#             b.data[:,4] +=math.log(8/(640/s)**2)
#             b.data[:,5:5+m.nc] += math.log(0.6/(m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
#             mi.bias = torch.nn.Parameter(b.view(-1),requires_grad=True)
    
#     def initialize_weights(self):
#         for m in self.modules():
#             t=type(m)
#             if t is nn.Conv2d:
#                 pass
#             elif t is nn.BatchNorm2d:
#                 m.eps = 1e-3
#                 m.momentum = 0.03
#             elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
#                 m.inplace = True
#     def _apply(self, fn):
#         self = super()._apply(fn)
#         m = self.model[-1]
#         m.stride = fn(m.stride)
#         m.grid = list(map(fn,m.grid))
#         m.anchor_grid = list(map(fn,m.anchor_grid))
#         return self
#     def info(self,verbose=True,img_size=640):
#         model_info(self,verbose,img_size)

# def model_info(model,verbose,imgsz):
#     n_p = sum(x.numel() for x in model.parameters())  # number parameters
#     n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
#     if verbose:
#         print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
#         for i, (name, p) in enumerate(model.named_parameters()):
#             name = name.replace('module_list.', '')
#             # print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
#             #       (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    
#     try:  # FLOPs
#         p = next(model.parameters())
#         stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
#         im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
#         flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
#         imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
#         fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
#     except Exception:
#         fs = ''
#     # for layer in model.modules():
#     #     print(f'{layer}')
#     name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
#     LOGGER.info(f'{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')











        


        
        



if __name__ == "__main__":

    model =  Net()

    # compute_loss = loss（model）



    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()





    im = torch.rand(1, 3, 640, 640).to(device)
    
    
    output = model(im)
    # # print("output",output[0].shape())
    # for layer in mode.:
    #     im = layer(im)
    #     print(layer.__call__.__name__,'output shepe: \t', im.shape)

