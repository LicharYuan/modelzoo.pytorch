import sys
import os
sys.path.append(os.getcwd())
from modelzoo.models.basic import *
from modelzoo.models.blocks import *
from modelzoo.models.models import *
from modelzoo.utils.param_flops import get_model_complexity_info
if __name__ == "__main__":
    net = resnet34()
    print(get_model_complexity_info(net, (3,224,224)))
    # y = net(x)
    # print(y.shape)
    
    
