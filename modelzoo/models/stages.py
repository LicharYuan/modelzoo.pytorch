from .blocks import *
import torch.nn as nn


class BaseStage(nn.Module):
    def __init__(self,block_func, inc, outc, kernel_size=3, stride=1, depth=2):
        super().__init__()
        self.depth =  depth
        self.stride = stride
        self.inc = inc
        self.block_func = block_func
        self.outc = outc
        self.ks = kernel_size
        self._build_stages()

    def _build_stages(self):
        stage = []
        for i in range(self.depth):
            if i == 0:
                proj = self.inc != self.outc and self.stride!=1
                stage.append(self.block_func(self.inc, self.outc, self.ks, self.stride, proj=proj))
            else:
                stage.append(self.block_func(self.outc, self.outc, self.ks, 1, proj=False))

        self.stage = nn.Sequential(*stage)

    def forward(self, x):
        return self.stage(x)


class ResNet18Stage(BaseStage):
    def __init__(self, inc, outc, kernel_size=3, stride=1, depth=2):
        super().__init__(ResNetBasic, inc, outc, kernel_size, stride, depth)


# alias
ResNet18Stage3x3 = ResNet18Stage


        

        