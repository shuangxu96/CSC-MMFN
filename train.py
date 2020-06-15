# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:18:05 2020

@author: BSawa
"""
import torch
import torch.nn as nn
import datetime
import os
from networks import ModelA
from utils import H5Dataset, prepare_data, MMSRDataset
from torch.utils.data import DataLoader 
from tensorboardX import SummaryWriter
from kornia.losses import psnr_loss, ssim

# model
lr = 5e-4
num_epoch = 100
scale = 2
num_blocks = 4
channels = 256
act = 'sst'
net = ModelA(num_blocks, in_lr=31, in_guide=3, out_channels=channels, act=act).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

# loaders
prepare_data_flag = False # set it to False, if you have prepared dataset
if prepare_data_flag is True:
    prepare_data(data_path = 'MMF_data/scale2', 
                     patch_size=32, aug_times=4, stride=25, synthetic=True, scale=2,
                     file_name='cave_train.h5'
                     )
batch_size = 16
trainloader      = DataLoader(H5Dataset(r'cave_train.h5'),      
                              batch_size=batch_size, shuffle=True) #[N,C,K,H,W]
validationloader = DataLoader(MMSRDataset(r'MMF_data/scale2/validation',scale),      
                              batch_size=1)
loader = {'train':      trainloader,
          'validation': validationloader}

# logger
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join('MMF_logs',timestamp+'_bs%d_epoch%d_lr%.5f_nb%d_%s_scale%d_channels%d'%(batch_size,num_epoch,lr,num_blocks,act,scale,channels))
writer = SummaryWriter(save_path)

'''
Train
'''
step = 0
best_psnr_val,psnr_val = 0., 0.
best_ssim_val,ssim_val = 0., 0.
torch.backends.cudnn.benchmark = True

for epoch in range(num_epoch):
    ''' train '''
    for i, (lr, guide, gt) in enumerate(loader['train']):
        lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
        
        #1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        imgf = net(lr, guide)
        loss = nn.MSELoss()(gt, imgf)
        loss.backward()
        optimizer.step()
        
        #2.  print
        print("[%d,%d] Loss:%.4f, PSNR: %.4f, SSIM: %.4f" %
                (epoch+1, i+1, 
                 loss.item(),
                 psnr_val, ssim_val))
        #3. Log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        step+=1
    
    ''' validation ''' 
    psnr_val = 0.
    ssim_val = 0.
    with torch.no_grad():
        net.eval()
        for i, (lr, guide, gt) in enumerate(loader['validation']):
            lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
            imgf = torch.clamp(net(lr, guide), 0., 1.)
            psnr_val += psnr_loss(imgf, gt, 1.)
            ssim_val += ssim(imgf, gt, 5, 'mean', 1.)
        psnr_val = float(psnr_val/loader['validation'].__len__())
        ssim_val = 1-2*float(ssim_val/loader['validation'].__len__())
    writer.add_scalar('PSNR on validation data', psnr_val, epoch)
    writer.add_scalar('SSIM on validation data', ssim_val, epoch)

    
    ''' decay the learning rate '''
#    scheduler.step()
    
    ''' save model ''' 
    if best_psnr_val<psnr_val:
        best_psnr_val = psnr_val
        torch.save(net.state_dict(), os.path.join(save_path, 'best_net.pth'))
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch},
                os.path.join(save_path, 'last_net.pth'))


'''
Test
'''
from scipy.io import savemat
net.load_state_dict(torch.load(os.path.join(save_path,'best_net.pth')))
testloader = DataLoader(MMSRDataset('MMF_data/scale2/test', scale),      
                              batch_size=1)

metrics = torch.zeros(2,testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (lr, guide, gt) in enumerate(testloader):
        lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
        imgf = torch.clamp(net(lr, guide), 0., 1.)
        metrics[0,i] = psnr_loss(imgf, gt, 1.)
        metrics[1,i] = 1-2*ssim(imgf, gt, 5, 'mean', 1.)
        savemat(os.path.join(save_path,testloader.dataset.files[i].split('\\')[-1]),
               {'HR':imgf.squeeze().detach().cpu().numpy()} )

import xlwt
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