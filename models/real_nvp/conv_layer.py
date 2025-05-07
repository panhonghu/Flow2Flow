import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la
logabs = lambda x: torch.log(torch.abs(x))


class ActivationLayer(nn.Module):
    def __init__(self, scale_factor=5.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x, sldj=None, reverse=True):
        if not reverse:
            x = x / self.scale_factor
            out = (torch.exp(x)-torch.exp(-x)) / (torch.exp(x)+torch.exp(-x))
            derivative = (4*torch.exp(2*x)) / (torch.exp(2*x)+1)**2
            sldj += torch.sum(derivative)
            return out, sldj/self.scale_factor
        else:
            x = torch.clamp(x, -0.9999999, 0.9999999)
            out = torch.log((1+x) / (1-x)) / 2
            return out*self.scale_factor, sldj


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, x, sldj=None, reverse=True):
        _, _, height, width = x.shape
        if not reverse:
            out = F.conv2d(x, self.weight)
            sldj += (height * width * torch.slogdet(self.weight.squeeze().double())[1].float())
            return out, sldj
        else:
            out = F.conv2d(x, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            return out, sldj


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, x, sldj=None, reverse=True):
        _, _, height, width = x.shape
        if not reverse:
            weight = self.calc_weight()
            out = F.conv2d(x, weight)
            sldj += height * width * torch.sum(self.w_s)
            return out, sldj
        else:
            weight = self.calc_weight()
            out = F.conv2d(x, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            return out, sldj

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)


