import torch
import torch.nn as nn
from functools import partial

class ConvBNActi(nn.Sequential):
    def __init__(self, in_planes, out_planes, 
                 kernel_size=3, stride=1, groups=1, acti_type='relu'):
        assert kernel_size % 2 == 1 
        padding = (kernel_size - 1) // 2
        super(ConvBNActi, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            acti_factory(acti_type, inplace=True)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, 
                 kernel_size=3, stride=1, groups=1):
        assert kernel_size % 2 == 1 
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
        )


def acti_factory(acti_type='relu', **kwargs):
    # case-insensitive    
    acti_type = acti_type.lower()
    if acti_type == 'relu':
        return  nn.ReLU(**kwargs)
    elif acti_type == 'lrelu':
        return nn.LeakyReLU(**kwargs)
    elif acti_type == 'prelu':
        return nn.PReLU(**kwargs)
    elif acti_type == 'relu6':
        return nn.ReLU6(**kwargs)
    elif acti_type == 'sigmoid':
        return nn.Sigmoid(**kwargs)
    else:
        raise NotImplementedError(acti_type)

def pool_factory(pool_type='max', ks=3, stride=1, padding=None, outsize=None):
    if padding is None:
        padding = (ks - 1)// 2
    if pool_type == 'max':
        return nn.MaxPool2d(ks, stride, padding)
    elif pool_type == 'avg':
        return nn.AvgPool2d(ks, stride, padding)
    elif pool_type == 'adaptive_max':
        return nn.AdaptiveMaxPool2d(outsize)
    elif pool_type == 'adpative_avg':
        return nn.AdaptiveAvgPool2d(outsize)
    else:
        raise NotImplementedError(pool_type)

def _scale_filters(filters, multiplier, base=8):
  """Scale the filters accordingly to (multiplier, base)."""
  round_half_up = int(int(filters) * multiplier / base + 0.5)
  result = int(round_half_up * base)
  return max(result, base)

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
# alias
ConvBNReLU = ConvBNActi
ConvBNReLU6 = partial(ConvBNActi, acti_type="relu6")



