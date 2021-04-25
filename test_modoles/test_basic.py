import sys
import os
sys.path.append(os.getcwd())
from models.basic import *
from models.blocks import *
from models.models import *

if __name__ == "__main__":
    x = torch.ones(1, 3, 224, 224)
    net = resnet34()
    y = net(x)
    print(y.shape)
    
    
