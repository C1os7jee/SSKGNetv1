
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):

    def __init__(self, num_state, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): 节点特征, shape [B, num_state, num_node]
        """
        h = self.conv1(x)
        h = h - x
        h = self.relu(self.conv2(h))
        return h

class GCNBranch(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, num_nodes=16):
        super(GCNBranch, self).__init__()
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.grid_size = max(1, int(math.sqrt(num_nodes)))

        # 1. 初始卷积层，用于“Token化”图像，不再强制下采样
        self.stem = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)

        # 2. GCN层
        self.gcn = GCN(num_state=out_channels)

    def forward(self, x, target_size=None):
        """
        Args:
            x (Tensor): 输入图像
            target_size (tuple[int, int], optional): 期望输出的空间尺寸 (H, W)
        """
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        B, _, H, W = x.shape

        # 1. Token化，保持与目标尺寸一致
        x_tokenized = self.stem(x)  # -> [B, C, H, W]
        _, C, h, w = x_tokenized.shape

        # 2. 归纳代表性节点
        pool_h = max(1, min(self.grid_size, h))
        pool_w = max(1, min(self.grid_size, w))
        nodes = F.adaptive_avg_pool2d(x_tokenized, output_size=(pool_h, pool_w))
        nodes = nodes.view(B, C, -1)  # -> [B, C, num_nodes']

        # 3. GCN处理
        gcn_out = self.gcn(nodes)  # -> [B, C, num_nodes']

        # 4. 将GCN输出的节点信息广播回原始Token
        tokens = x_tokenized.view(B, C, -1).permute(0, 2, 1)  # -> [B, H*W, C]
        nodes_t = gcn_out.permute(0, 2, 1)  # -> [B, num_nodes', C]

        affinity = torch.matmul(tokens, nodes_t.transpose(1, 2))
        affinity = F.softmax(affinity, dim=-1)
        features_enhanced = torch.matmul(affinity, nodes_t)

        out = features_enhanced.transpose(1, 2).reshape(B, C, h, w)

        return out
