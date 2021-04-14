import sys
import os
sys.path.append(os.getcwd())
from models.basic import *
from models.blocks import *


if __name__ == "__main__":
    a = ConvBNReLU(10, 20)
    b = FusedConv()