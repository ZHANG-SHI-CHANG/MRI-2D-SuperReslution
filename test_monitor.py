import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs

import os
import glob
import numpy as np
from scipy.interpolate import spline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from EDSR_test import test

#plot and save one
activate_list = {'1':[nn.ReLU(),'ReLU'],'2':[nn.ELU(),'ELU'],'3':[nn.SELU(),'SELU'],'4':[nn.LeakyReLU(),'LeakyReLU']}
for i in range(len(activate_list)):
    test(activate_list[str(i+1)][0],activate_list[str(i+1)][1])

##plot all
status_type = ['test_loss','test_psnr','test_ssim']
activate_list = ['ReLU','ELU','SELU','LeakyReLU']
color_list = ['green','red','skyblue','blue']

status_list = []

for _status_type in status_type:
    fig = plt.figure()
    plt.xlabel('image id')
    plt.ylabel(_status_type[-4:])
    plt.title('test')
    plt.grid(True)
    for i,activate_name in enumerate(activate_list):
        with open('test_status\\'+'{}_{}.txt'.format(_status_type,activate_name),'r') as f:
            context = f.readline()
            for value in context.split(','):
                status_list.append(float(value))
        
        axis = np.linspace(0, len(status_list)-1, len(status_list))
        axis_new = np.linspace(axis.min(),axis.max(),100000)
        status_array = spline(axis,np.array(status_list),axis_new)
        
        plt.plot(axis_new, status_array,color=color_list[i],label=activate_name)
        status_list.clear()
    plt.legend()
    plt.savefig('{}.pdf'.format(_status_type))
    plt.close(fig)