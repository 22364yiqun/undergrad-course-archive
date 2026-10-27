import torch
import torch.nn as nn
import importlib
from model_init.axial_attention import my_own_model

def init_network(option):
    net = my_own_model(option)   

    if option['use_gpu']:
        net = nn.DataParallel(net).cuda()
    return net


def init_loss(**option):
    loss = importlib.import_module('loss.' + option['loss'])
    criterion = getattr(loss, option['loss'])(**option)
    if option['use_gpu']:
        criterion.cuda()
    
    return criterion

def init_optimizer(net_c, criterion, **options):
    params_list = [{'params': net_c.parameters()}, {'params': criterion.parameters()}]
    opt_c = torch.optim.Adam(params_list, lr=options['learning_rate'])

    return opt_c