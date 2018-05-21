import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs

import os
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from EDSR import train

#plot and save one
epoch_list = {'1':3,'2':3,'3':3,'4':3}
activate_list = {'1':[nn.ReLU(),'ReLU'],'2':[nn.ELU(),'ELU'],'3':[nn.SELU(),'SELU'],'4':[nn.LeakyReLU(),'LeakyReLU']}
for i in range(len(activate_list)):
    train(epoch_list[str(i+1)],activate_list[str(i+1)][0],activate_list[str(i+1)][1])

##plot all
status_type = ['train_loss','train_psnr','train_ssim']
activate_list = ['ReLU','ELU','SELU','LeakyReLU']
color_list = ['green','red','skyblue','blue']

status_list = []

for _status_type in status_type:
    fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel(_status_type[-4:])
    plt.title('train')
    plt.grid(True)
    for i,activate_name in enumerate(activate_list):
        with open('train_status\\'+'{}_{}.txt'.format(_status_type,activate_name),'r') as f:
            context = f.readline()
            for value in context.split(','):
                status_list.append(float(value))
        axis = np.linspace(0, len(status_list)-1, len(status_list))
        plt.plot(axis, status_list,color=color_list[i],label=activate_name)
        status_list.clear()
    plt.legend()
    plt.savefig('{}.pdf'.format(_status_type))
    plt.close(fig)