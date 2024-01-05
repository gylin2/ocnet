import argparse
import os
import torch
import torch.nn as nn
# from torchsummary import summary
from model import RawNet
from deepspeed.profiling.flops_profiler import get_model_profile

def model_structure(model):
    blank = ' '
    print('-'*90)
    print('|'+' '*11+'weight name'+' '*10+'|' \
            +' '*15+'weight shape'+' '*15+'|' \
            +' '*3+'number'+' '*3+'|')
    print('-'*90)
    num_para = 0
    type_size = 1  
    
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30: 
            key = key + (30-len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40-len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10-len(str_num)) * blank
    
        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-'*90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-'*90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument("--feat_len", type=int, help="features length", default=128000)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNet(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    
    # summary(model, (args.feat_len,))
    
    flops, macs, params = get_model_profile(model, (1,args.feat_len,), as_string=False)
    print('FLOPs: {}'.format(flops))
    print('Parameters: {}'.format(params))
    print('FLOPs: %.2fG' % (flops / 1e9))
    print('Parameters: %.2fM' % (params / 1e6))
