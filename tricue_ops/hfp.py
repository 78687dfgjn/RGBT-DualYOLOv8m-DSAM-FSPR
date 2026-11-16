import math
import torch
from torch import nn
from torch.nn import functional as F
def fp32_context(x): return torch.autocast(device_type=x.device.type, enabled=False)

class HFP(nn.Module):
    def __init__(self,channels,ratio=(.25,.25),patch=(8,8)):
        super().__init__();assert channels%32==0
        self.ratio,self.patch=ratio,patch
        self.channel1=nn.Conv2d(channels,channels,1,groups=32,bias=False)
        self.channel2=nn.Conv2d(channels,channels,1,groups=32,bias=False)
        self.out=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False),nn.Conv2d(channels,channels,1,bias=True),nn.GroupNorm(32,channels))
        self.register_buffer('_basis_h',torch.empty(0),persistent=False)
        self.register_buffer('_basis_w',torch.empty(0),persistent=False)

    @staticmethod
    def basis(size,device):
        k=torch.arange(size,device=device,dtype=torch.float32)[:,None]
        n=torch.arange(size,device=device,dtype=torch.float32)[None,:]
        matrix=torch.cos(math.pi/size*(n+.5)*k)*math.sqrt(2/size)
        matrix[0]*=1/math.sqrt(2)
        return matrix

    def high_pass(self,x):
        h,w=x.shape[-2:]
        if self._basis_h.shape!=(h,h) or self._basis_h.dtype!=torch.float32:
            self._basis_h=self.basis(h,x.device)
        if self._basis_w.shape!=(w,w) or self._basis_w.dtype!=torch.float32:
            self._basis_w=self.basis(w,x.device)
        a,b=self._basis_h,self._basis_w
        coeff=torch.matmul(torch.matmul(a,x),b.T)
        mask=torch.ones((h,w),device=x.device,dtype=torch.float32)
        mask[:int(h*self.ratio[0]),:int(w*self.ratio[1])]=0
        return torch.matmul(torch.matmul(a.T,coeff*mask),b)

    def forward(self,x):
        dtype=x.dtype
        with fp32_context(x):
            z=x.float();high=self.high_pass(z)
            maximum=F.adaptive_max_pool2d(high,self.patch).relu().sum((2,3),keepdim=True)
            average=F.adaptive_avg_pool2d(high,self.patch).relu().sum((2,3),keepdim=True)
            attention=torch.sigmoid(self.channel2(self.channel1(maximum)+self.channel1(average)))
            result=self.out(z*high+z*attention)
        return result.to(dtype)
