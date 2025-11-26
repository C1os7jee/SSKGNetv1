import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 支持绝对导入
sys.path.insert(0, os.path.dirname(__file__))

from modules.encoder import build_encoder
from modules.cka_decoder import CKADecoder
from modules.strip_decoder import StripDecoder
from config import Config
from modules.gcn import GCNBranch

class GDRNet(nn.Module):
    def __init__(self):
        super(GDRNet, self).__init__()
        self.config = Config()
        
        # 1. 主干编码器
        self.encoder = build_encoder()
        
        # 2. 并行GCN分支
        self.gcn_branch = GCNBranch(in_channels=self.config.model_in_channels, out_channels=self.config.gcn_channels)

        # 3. 解码器
        if self.config.decoder_type == 'cka':
            self.decoder = CKADecoder(
                dims=self.config.decoder_dims, 
                gcn_channels=self.config.gcn_channels
            )
        else:
            self.decoder = StripDecoder(
                dims=self.config.decoder_dims, 
                gcn_channels=self.config.gcn_channels,
                use_dynamic=getattr(self.config, 'dynamic_strip', False),
                stage_k2s=getattr(self.config, 'strip_k2s', None),
                use_low_gcn=getattr(self.config, 'use_low_gcn', True),
            )

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_inputs = encoder_features[::-1]

        gcn_features = self.compute_gcn_features(x, decoder_inputs)

        # 2. 解码器融合所有特征并进行预测
        all_features = {
            'encoder': decoder_inputs,  
            'gcn': gcn_features
        }
        predictions = self.decoder(all_features)
        
        return predictions

    def _resolve_stage_sizes(self, decoder_inputs, input_hw):
        _, x3, x2, x1 = decoder_inputs
        stage_sizes = [
            x3.shape[2:],  # Stage 4 目标分辨率
            x2.shape[2:],  # Stage 3
            x1.shape[2:],  # Stage 2
            (x1.shape[2] * 2, x1.shape[3] * 2)  # Stage 1 (上采样一倍)
        ]
        input_h, input_w = input_hw
        stage_sizes[-1] = (
            min(stage_sizes[-1][0], input_h),
            min(stage_sizes[-1][1], input_w)
        )
        return stage_sizes

    def compute_gcn_features(self, x, decoder_inputs, gcn_branch=None):
        branch = gcn_branch if gcn_branch is not None else self.gcn_branch
        stage_sizes = self._resolve_stage_sizes(decoder_inputs, (x.shape[2], x.shape[3]))
        return [branch(x, target_size=size) for size in stage_sizes]
