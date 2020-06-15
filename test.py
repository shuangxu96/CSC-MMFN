# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:32:55 2020

@author: BSawa
"""
import torch
import os
import xlwt

from networks import ModelA
from scipy.io import savemat
from kornia.losses import psnr_loss, ssim
from torch.utils.data import DataLoader 
from utils import MMSRDataset
from utils import mkdir

scale = 2
num_blocks = 4
act = 'sst'
save_reduced_data = True
net = ModelA(num_blocks, in_lr=31, in_guide=3, out_channels=64, act=act).cuda()
save_path = r'MMF_logs/05-10-10-47_bs16_epoch100_lr0.00010_nb4_sst_scale2_channels256'

net.load_state_dict(torch.load(os.path.join(save_path,'best_net.pth')))
testloader = DataLoader(MMSRDataset(r'MMF_data/scale2/test', scale),      
                              batch_size=1)

metrics = torch.zeros(2,testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (lr, guide, gt) in enumerate(testloader):
        if save_reduced_data:
            mkdir(r'MMF_data/scale2/test_reduced')
            savemat(os.path.join(r'MMF_data/scale2/test_reduced',
                         testloader.dataset.files[i].split('\\')[-1]), 
            {'LR': lr.squeeze(0).numpy(),
            'Guide': guide.squeeze(0).numpy(),
            'GT': gt.squeeze(0).numpy()}
            )
        lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
        imgf = torch.clamp(net(lr, guide), 0., 1.)
        metrics[0,i] = psnr_loss(imgf, gt, 1.)
        metrics[1,i] = 1-2*ssim(imgf, gt, 5, 'mean', 1.)
        savemat(os.path.join(save_path,testloader.dataset.files[i].split('\\')[-1]),
               {'HR':imgf.squeeze().detach().cpu().numpy()} )


f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
img_name = [i.split('\\')[-1].replace('.mat','') for i in testloader.dataset.files]
metric_name = ['PSNR','SSIM']
for i in range(len(metric_name)):
    sheet1.write(i+1,0,metric_name[i])
for j in range(len(img_name)):
   sheet1.write(0,j+1,img_name[j])  # 顺序为x行x列写入第x个元素
for i in range(len(metric_name)):
    for j in range(len(img_name)):
        sheet1.write(i+1,j+1,float(metrics[i,j]))
sheet1.write(0,len(img_name)+1,'Mean')
for i in range(len(metric_name)):
    sheet1.write(i+1,len(img_name)+1,float(metrics.mean(1)[i]))
f.save(os.path.join(save_path,'test_result.xls'))