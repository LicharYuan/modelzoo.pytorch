import torch
import torch.nn as nn
from models.basic import ConvBNReLU6, ConvBN

class FusedConv(nn.Module):
    """fused mb conv, replace expand conv to normal conv"""
    def __init__(self, inc, outc, kernel_size=3, stride=1, expansion=8, use_se=False, residual=True):
        super().__init__()
        self.use_se = use_se
        self.residual = residual
        assert expansion >= 1
        assert kernel_size % 2 == 1 # padding 
        hidden = expansion * inc
        padding = (kernel_size - 1) // 2 
        # fused operation
        self.expand = ConvBNReLU6(inc, hidden, kernel_size, stride=stride, padding=padding)
        if use_se:
            self.se = SELayer()
        self.proj = ConvBN(hidden, outc, 1)
        
    
    def forward(self, x):
        inputs = x
        x = self.expand(x)
        if self.use_se:
            x = self.se(x)
        x = self.proj(x)
        if self.residual:
            return x + inputs
        else:
            return x

class TuckerConv(nn.Module):
    """proposed on mobiledet <https://arxiv.org/pdf/2102.05610.pdf>"""
    def __init__(self, inc, outc, kernel_size=3, stride=1, in_ratio=0.25, out_ratio=0.25, residual=True):
        super().__init__()
        c1 = _scale_filters(inc, in_ratio)
        c2 = _scale_filters(c1, out_ratio)
        self.convbnrelu1 = ConvBNReLU6(inc, c1, 1)
        self.convbnrelu2 = ConvBNReLU6(c1, c2, kernel_size, stride=stride)
        self.proj = nn.Conv2d(c2, outc, 1, bias=False)
        self.convbn = ConvBN(c2, outc, 1)
        self.residual = residual

    def forward(self, x):
        inputs = x
        x = self.convbnrelu1(x)
        x = self.convbnrelu2(x)
        x = self.convbn(x)
        if self.residual:
            return x+inputs
        else:
            return x

class InvertedBottleNeck(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, expansion=8, use_se=False, residual=True):
        super().__init__()
        self._ks = kernel_size
        expand_dim = int(inc*expansion)
        self.residual = residual
        self.expand = ConvBNReLU6(inc, expand_dim, 1)
        self.dwconv = ConvBNReLU6(expand_dim, expand_dim, kernel_size, 
            stride=stride, groups=expand_dim)
        self.proj = ConvBN(expand_dim, outc, 1)
        self._reset_parameters()
    
    def _reset_parameters(self):
        # using diff initialize
        stddev = (2.0 / self._ks**2)**0.5 / .87962566103423978
        truncated_normal_(self.dwconv.weight, std=stddev)
        # nn.init.normal_(self.dwconv.weight, std=stddev)

    def forward(self, x):
        inputs = x
        x = self.expand(x)
        x = self.dwconv(x)
        x = self.proj(x)
        if self.residual:
            return x+inputs
        else:
            return x
