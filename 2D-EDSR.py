import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs

import numpy as np
import cv2
import nrrd
import skimage.io as sio
import skimage.color as sc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import glob
import random

root = os.getcwd()
if_restore = False 

def PaddingData(datas,scale=4):
    row,col,cha = datas.shape
    if row%scale==0:
        pass
    else:
        padding = np.zeros((scale-(row-scale*(row//scale)),col,cha),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=0)
    row,col,cha = datas.shape
    if col%scale==0:
        pass
    else:
        padding = np.zeros((row,scale-(col-scale*(col//scale)),cha),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=1)
    return datas
def calc_PSNR(input, target, set_name='None', rgb_range=255, scale=4):
    def quantize(img, rgb_range):
        return np.floor( np.clip(img*(255 / rgb_range)+0.5, 0, 255) )/255
    def rgb2ycbcrT(rgb):
        return sc.rgb2ycbcr(rgb) / 255
    
    test_Y = ['Set5', 'Set14', 'B100', 'Urban100']

    h, w, c = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[0:h, 0:w, :], rgb_range)
    diff = input - target
    if set_name in test_Y:
        shave = scale
        if c > 1:
            input_Y = rgb2ycbcrT(input)
            target_Y = rgb2ycbcrT(target)
            diff = np.reshape(input_Y - target_Y,( h, w, 3))
    else:
        shave = scale + 6

    diff = diff[shave:(h - shave), shave:(w - shave), :]
    mse = np.power(diff,2).mean()
    psnr = -10 * np.log10(mse)

    return psnr
def calc_SSIM(I1,I2):
    C1 = 6.5025
    C2 = 58.5225
    I1,I2 = I1.astype(np.float32),I2.astype(np.float32)
    I22 = I2*I2
    I11 = I1*I1
    I12 = I1*I2
    I_blur1 = cv2.GaussianBlur(I1,(11,11),1.5)
    I_blur2 = cv2.GaussianBlur(I2,(11,11),1.5)
    I_blur11 = I_blur1*I_blur1
    I_blur22 = I_blur2*I_blur2
    I_blur12 = I_blur1*I_blur2
    sigma11 = cv2.GaussianBlur(I11,(11,11),1.5) - I_blur11
    sigma22 = cv2.GaussianBlur(I22,(11,11),1.5) - I_blur22
    sigma12 = cv2.GaussianBlur(I12,(11,11),1.5) - I_blur12
    t1 = 2*I12 + C1
    t2 = 2*sigma12 + C2
    t3 = t1*t2
    t1 = I_blur11 + I_blur22 + C1
    t2 = sigma11 + sigma22 + C2
    t1 = t1*t2
    ssim_map = t3/t1
    return ssim_map.mean()
#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self,conv=default_conv,act=None):
        super(EDSR, self).__init__()
        n_feats = 64#64
        kernel_size = 3
        n_resblock = 16#16
        act = act
        res_scale = 1
        scale = 4
        
        self.head = nn.Sequential(conv(3,n_feats,kernel_size))
        
        modules_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblock)]
        self.body = nn.Sequential(*modules_body)
        
        modules_tail = [
            conv(n_feats,n_feats*16,1),
            nn.PixelShuffle(4),
            conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)
        
    def forward(self, x):
        x = x.contiguous()
        x = self.head(x) 
        
        res = self.body(x)
        res += x        
        
        x = self.tail(res)

        return x 
        
####################
activate_list = {'1':[nn.ReLU(),'ReLU'],'2':[nn.ELU(),'ELU'],'3':[nn.SELU(),'SELU'],'4':[nn.LeakyReLU(),'LeakyReLU']}
activate = activate_list['3'][0]
activate_name = activate_list['3'][1]
####################
edsr = EDSR(act=activate)
print(edsr)
#重载入模型
if if_restore:
    print('loading weight...')
    edsr.load_state_dict(torch.load(os.path.join(root,'model','model_lastest.pt')))
    print('load weight success')

optimizer = torch.optim.Adam(edsr.parameters(),lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
loss_function = nn.L1Loss()
#####################################################################
loss_list = []
psnr_list = []
ssim_list = []

for step in range(200):
    paths = glob.glob(os.path.join(root,'dataset_train','*.png'))
    random.shuffle(paths)#打乱
    
    avg_loss = 0
    avg_psnr = 0
    avg_ssim = 0
    
    for i,datas_path in enumerate(paths):
        original_datas = cv2.imread(datas_path)
        original_shape = original_datas.shape
        original_datas = PaddingData(original_datas)
        padding_shape = list(map(lambda x:x[0]-x[1],zip(original_datas.shape,original_shape)))
        
        result = np.zeros_like(original_datas,dtype=original_datas.dtype)
        
        _datas = cv2.resize(original_datas,(original_datas.shape[1]//4,original_datas.shape[0]//4))
        
        x = _datas.transpose((2,0,1))[np.newaxis,:,:,:]
        x =  Variable(torch.from_numpy(x)).float()
        target = original_datas.transpose((2,0,1))[np.newaxis,:,:,:]
        target = Variable(torch.from_numpy(target)).float()

        optimizer.zero_grad()
        out = edsr(x)
        loss = loss_function(out,target)
        loss.backward()
        optimizer.step()
        
        result = out.cpu().data.numpy()[0,:,:,:].transpose((1,2,0))
        psnr = calc_PSNR(original_datas,result)
        ssim = calc_SSIM(original_datas,result)
        
        avg_loss += loss.cpu().data[0]
        avg_psnr += psnr
        avg_ssim += ssim
        
        print('epoch:{} step:{} loss:{} psnr:{} ssim:{}'.format(step,i,loss.cpu().data[0],psnr,ssim))
    
    avg_loss = avg_loss/len(paths)
    avg_psnr = avg_psnr/len(paths)
    avg_ssim = avg_ssim/len(paths)
    print('\nstep:{},avg loss:{}, avg psnr:{}, avg ssim:{}\n'.format(step,avg_loss,avg_psnr,avg_ssim))
    loss_list.append(avg_loss)
    psnr_list.append(avg_psnr)
    ssim_list.append(avg_ssim)
    
    with open('train_loss_{}.txt'.format(activate_name),'w') as f:
        for i,loss in enumerate(loss_list):
            if i==len(loss_list)-1:
                f.write(str(loss))
            else:
                f.write(str(loss)+',')
    axis = np.linspace(0, step, step+1)
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('train activate:{}'.format(activate_name))
    plt.grid(True)
    plt.plot(axis, loss_list)
    plt.legend()
    plt.savefig('train_loss_{}.pdf'.format(activate_name))
    plt.close(fig)
    
    with open('train_psnr_{}.txt'.format(activate_name),'w') as f:
        for i,psnr in enumerate(psnr_list):
            if i==len(psnr_list)-1:
                f.write(str(psnr))
            else:
                f.write(str(psnr)+',')
    axis = np.linspace(0, step, step+1)
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('psnr')
    plt.title('train activate:{}'.format(activate_name))
    plt.grid(True)
    plt.plot(axis, psnr_list)
    plt.legend()
    plt.savefig('train_psnr_{}.pdf'.format(activate_name))
    plt.close(fig)
    
    with open('train_ssim_{}.txt'.format(activate_name),'w') as f:
        for i,ssim in enumerate(ssim_list):
            if i==len(ssim_list)-1:
                f.write(str(ssim))
            else:
                f.write(str(ssim)+',')
    axis = np.linspace(0, step, step+1)
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('ssim')
    plt.title('train activate:{}'.format(activate_name))
    plt.grid(True)
    plt.plot(axis, ssim_list)
    plt.legend()
    plt.savefig('train_ssim_{}.pdf'.format(activate_name))
    plt.close(fig)
    
    if step%99==0:
        if step%2==0:
            torch.save(edsr,'model/edsr_1.pt')
            torch.save(edsr.state_dict(),'model/edsr_2_params.pt')
        else:
            torch.save(edsr,'model/edsr_1.pt')
            torch.save(edsr.state_dict(),'model/edsr_1_params.pt')
