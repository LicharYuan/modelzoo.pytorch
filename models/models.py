from .blocks import *
import torch.nn as nn
from .stages import *

class BaseNet(nn.Module):
    def __init__(self, op_list, width, strides, extras=None, inc=3):
        # op_list, width build body(including stem and stages)
        assert len(op_list) == len(width), "config error"
        super().__init__()
        self.op_list = block
        self.width = width
        self.strides = strides
        self._build_body(inc)
        self._with_extras = extras is not None
        if extras:
            self._build_extras(extras)
        
    def _build_body(self, inc):
        self.body = nn.ModuleList()
        for i, (blocks, outc, stride) in enumerate(zip(self.op_list, self.width, self.strides)):
            _blocks = []
            for j, block in enumerate(blocks):
                _blocks.append(block(inc, outc, stride=stride))
                inc = outc

            _blocks = nn.Sequential(*_blocks)
            self.body.append(_blocks)

    def _build_extras(self, extras):
        # TODO: using abc
        raise NotImplementedError

    def forward(self, x):
        for ele in self.body:
            x = ele(x)

        if self._with_extras:
            for extra in self.extras:
                x = extra(x)

        return x

class StageNet(nn.Module):
    def __init__(self, stage_list, width, depth, strides, stem_func, extras=None, stem_inc=3, stem_outc=32):
        assert len(stage_list) == len(width) == len(depth) == len(strides)
        super().__init__()
        self.width = width
        self.depth = depth
        self.strides = strides
        self.stem_func = stem_func
        self.stage_list = stage_list
        self._stem_inc = stem_inc
        self._stem_outc = stem_outc
        self._build_stem()
        self._build_stages(stem_outc)
        self._with_extras = extras is not None
        if extras:
            self._build_extras()
    
    def _build_stem(self):
        self.stem = self.stem_func(self._stem_inc, self._stem_outc)

    def _build_stages(self, inc):
        self.stages = nn.ModuleList()
        for i, (stage, outc, stride) in enumerate(zip(self.stage_list, self.width, self.strides)):
            _stages = []
            for j in range(self.depth[i]):
                stride = stride if j==0 else 1
                _stages.append(stage(inc, outc, stride=stride))
                inc = outc
            _stages = nn.Sequential(*_stages)
            self.stages.append(_stages)

    def _build_extras(self, extras):
        pass

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        
        if self._with_extras:
            for extra in self.extras:
                x = extra(x)
        return x


class ResNet(StageNet):
    STAGE = {
        "34": [ResNet18Stage3x3, ResNet18Stage3x3, ResNet18Stage3x3, ResNet18Stage3x3]
    }
    WIDTH = {
        "34": [32, 64, 128, 256]
    }
    DEPTH = {
        "34": [3, 4, 6, 3]
    }
    STRIDE = {
        "34": [1, 2, 2, 2]
    }

    def __init__(self, net_type, stem_func, extras=None):
        width = self.WIDTH[net_type]
        stage_list = self.STAGE[net_type]
        depth = self.DEPTH[net_type]
        stride = self.STRIDE[net_type]
        super().__init__(stage_list, width, depth, stride, stem_func, extras)

def resnet34(extras=None):
    return ResNet("34", ResNetStem3x3, extras)




        



