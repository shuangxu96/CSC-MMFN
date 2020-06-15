# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:18:41 2020

@author: BSawa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import blocks as B
from GuidedFilterLayer import FastGuidedFilter

def set_backbone(img_channels, out_channels, num_csc, act='sst'):
    backbone = nn.Sequential()
    backbone.add_module('rpad0', nn.ReflectionPad2d(1))
    backbone.add_module('conv0', nn.Conv2d(img_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False))
    #backbone.add_module('norm0', nn.BatchNorm2d(out_channels))
    backbone.add_module('relu0', B.get_activation(act, num_parameters=out_channels))
    for i in range(num_csc):
        backbone.add_module('csc'+str(i+1), B.DictConv2dBlock(img_channels, out_channels, out_channels, kernel_size=3, act=act))
    return backbone

def set_decoder(out_channels, in_channels, act_fun=None):
    decoder = nn.Sequential()
    decoder.add_module('rpad0', nn.ReflectionPad2d(1))
    decoder.add_module('conv0', nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=0, bias=False))
    if act_fun is not None:
        decoder.add_module('afun0', act_fun())
    return decoder

class ModelA(nn.Module):
    def __init__(self, num_csc, in_lr, in_guide, out_channels, radius=1, 
                 eps=1e-4, act='sst'):
        super(ModelA, self).__init__()
        self.backbone_guide = set_backbone(in_guide, out_channels, num_csc-1, act)
        self.backbone_lr  = set_backbone(in_lr , out_channels, num_csc-1, act)
        self.guided_filter = FastGuidedFilter(radius, eps)
        self.decoder = set_decoder(out_channels, in_lr)
        
    def block_forward(self, data, block):
        code = data
        for _,layer in enumerate(block):
            if type(layer).__name__.startswith('DictConv2d'):
                code = layer(data,code)
            else:
                code = layer(code)
        return code
    
    def upsample(self, input, h, w):
        return F.interpolate(input, (h,w), mode='bilinear', align_corners=True)
    
    def forward(self, lr, guide):
        feat_guide = self.block_forward(guide,self.backbone_guide)
        feat_lr  = self.block_forward(lr,self.backbone_lr)
        feat_lguide = self.upsample(feat_guide, feat_lr.shape[2],feat_lr.shape[3])
        feat_hlr = self.guided_filter(feat_lguide, feat_lr, feat_guide)
        hlr = self.decoder(feat_hlr) + self.upsample(lr,guide.shape[2],guide.shape[2])
        return hlr