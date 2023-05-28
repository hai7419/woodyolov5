import torch
import torch.nn as nn
import warnings


class Conv(nn.Module):
    def __init__(self, c1,c2,k,s,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=c1,out_channels=c2,kernel_size=k,stride=s,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self,x):
        return self.act(self.conv(x))



class Bottleneck(nn.Module):
    def __init__(self,c1,c2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        c_ = int(c2*0.5)
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c_,c2,3,1)
        #self.add = True

    def forward(self,x):
        return x+self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1,c2,n,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        c_ = int(c2*0.5)
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c1,c_,1,1)
        self.cv3 = Conv(2*c_,c2,1,1)
        self.m = nn.Sequential(*(Bottleneck(c_,c_) for _ in range(n)))

    def forward(self,x):
        return self.cv3(torch.cat((self.m(self.cv1(x)),self.cv2(x)),1))
    

class SPPF(nn.Module):
    def __init__(self, c1,c2,k=5,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        c_=c1//2
        self.cv1=Conv(c1,c_,1,1)
        self.cv2=Conv(c_*4,c2,1,1)
        self.m=nn.MaxPool2d(kernel_size=k,stride=1,padding=k//2)
    
    def forward(self,x):
        x=self.cv1(x)
        #with warnings.catch_warnings():
        y1=self.m(x)
        y2=self.m(y1)
        y3=self.m(y2)
        return self.cv2(torch.cat((x,y1,y2,y3),1))



    
    
    

    

