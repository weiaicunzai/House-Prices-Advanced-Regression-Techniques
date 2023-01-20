import argparse
import os
import time
import re
import sys
from collections import OrderedDict
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import transforms
import utils
from conf import settings
from dataset.camvid import CamVid
#from lr_scheduler import PolyLR
from metric import eval_metrics

#from train import
from train import evaluate
#import train.evaluate as evaluate

import utils

#from utils import get_



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    #parser.add_argument('-lr', type=float, default=0.007,
                        #help='initial learning rate')
    #parser.add_argument('-e', type=int, default=120, help='training epoches')
    #parser.add_argument('-iter', type=int, default=40000, help='training epoches')
    #parser.add_argument('-eval_iter', type=int, default=2000, help='training epoches')
    #parser.add_argument('-wd', type=float, default=0, help='training epoches')
    #parser.add_argument('-resume', type=bool, default=False, help='if resume training')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-prefix', type=str, default='', help='checkpoint and runs folder prefix')
    parser.add_argument('-alpha', type=float, default=1, help='panalize parameter')
    parser.add_argument('-op', type=str, default='or', help='mask operation')
    #parser.add_argument('-download', action='store_true', default=False,
        #help='whether to download camvid dataset')
    parser.add_argument('-gpu', action='store_true', default=False, help='whether to use gpu')
    parser.add_argument('-baseline', action='store_true', default=False, help='base line')
    parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')
    parser.add_argument('-poly', action='store_true', default=False, help='poly decay')
    parser.add_argument('-min_lr', type=float, default=1e-4, help='min_lr for poly')
    parser.add_argument('-branch', type=str, default='hybird', help='dataset name')
    #parser.add_argument('-fp16', action='store_true', default=False, help='whether to use mixed precision training')
    parser.add_argument('-scale', type=float, default=1, help='min_lr for poly')
    parser.add_argument('-download', action='store_true', default=False,
        help='whether to download camvid dataset')
    args = parser.parse_args()

    dataloader = utils.data_loader(args, 'val')
    print(dataloader.dataset.class_num)

    net = utils.get_model(args.net, 3, dataloader.dataset.class_num)
    #path = '/data/hdd1/by/FullNet-varCE/experiments/GlaS/tg/checkpoints/checkpoint_best.pth.tar'
    #path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Thursday_19_January_2023_01h_27m_36s/best_total_F1_0.8694_iter_39999.pt'
    path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Friday_20_January_2023_02h_19m_12s/best_total_F1_0.8804_iter_5999.pt'
    #print(torch.load(path).keys())
    #state_dict = torch.load(path)['state_dict']
    state_dict = torch.load(path)
    net.load_state_dict(state_dict)

    #new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
    #    name = k[7:] #remove 'module'
    #    new_state_dict[name] = v
    #import sys; sys.exit()
    #net.load_state_dict(new_state_dict)



    net = net.cuda()
    total, testA, testB = evaluate(net, dataloader, args)
    print(total, testA, testB)
