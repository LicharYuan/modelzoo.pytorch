import sys
import os
sys.path.append(os.getcwd())
from models.basic import *
from models.blocks import *
from models.models import *
from utils.param_flops import get_model_complexity_info
if __name__ == "__main__":
    net = resnet34()
    print(get_model_complexity_info(net, (3,224,224)))
    # y = net(x)
    # print(y.shape)
    
    
