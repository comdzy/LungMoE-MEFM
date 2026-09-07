import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import math

from attn.WTConv2d import DepthwiseSeparableConvWithWTConv2d

def kernel_size(in_channel):
    k = int((math.log2(in_channel) + 1)) // 2
    return k + 1 if k % 2 == 0 else k

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConvWithWTConv2d(in_channels, out_channels, wt_type='db1')
        self.conv2 = DepthwiseSeparableConvWithWTConv2d(in_channels, out_channels, wt_type='bior1.1')
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        return out1 + out2

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()
        self.num_features = num_features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channels)
        
        self.channel_convs = nn.ModuleList([
            nn.Conv1d(2*num_features, 1, kernel_size=self.k, padding=self.k//2)
            for _ in range(num_features)
        ])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, *features):
        pooled = []
        for feat in features:
            pooled.append(self.avg_pool(feat))
            pooled.append(self.max_pool(feat))
        
        channel_pool = torch.cat(pooled, dim=2).squeeze(-1).transpose(1, 2)
        attentions = [conv(channel_pool) for conv in self.channel_convs]
        
        channel_stack = torch.stack(attentions, dim=0)
        channel_stack = self.softmax(channel_stack)
        return channel_stack.transpose(-1, -2).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(2*num_features, 1, kernel_size=7, padding=3)
            for _ in range(num_features)
        ])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, *features):
        spatial_pools = []
        for feat in features:
            spatial_pools.append(torch.mean(feat, dim=1, keepdim=True))
            spatial_pools.append(torch.max(feat, dim=1, keepdim=True)[0])
        
        spatial_pool = torch.cat(spatial_pools, dim=1)
        attentions = [conv(spatial_pool) for conv in self.spatial_convs]
        
        return self.softmax(torch.stack(attentions, dim=0))

class Merge(nn.Module):
    def __init__(self, in_channel, num_features):
        super().__init__()
        self.num_features = num_features
        self.channel_attention = ChannelAttention(in_channel, num_features)
        self.spatial_attention = SpatialAttention(num_features)

    def forward(self, *features):
        channel_stack = self.channel_attention(*features)
        spatial_stack = self.spatial_attention(*features)
        stack_attention = channel_stack + spatial_stack + 1
        return sum(w * feat for w, feat in zip(stack_attention, features))

"""
多专家特征融合模块 （Multi-Experts Features Fusion Module）MEFM：

初始化参数：
    in_channels - 输入张量的通道数
    num_features - 特征融合的张量个数
"""
class MEFM(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()
        self.num_features = num_features
        self.multi_scale = MultiScaleFeatureExtractor(in_channels, in_channels)
        self.merge = Merge(in_channels, num_features)

    def forward(self, *features):
        assert len(features) == self.num_features
        processed = [self.multi_scale(feat) for feat in features]
        return self.merge(*processed)

if __name__ == '__main__':
    # 随机生成输入数据，假设输入的形状为 (batch_size=32, channels=512, height=7, width=7)
    feat1 = torch.randn(32, 512, 7, 7).cuda()  # 将随机生成的数据放置在GPU上
    feat2 = torch.randn(32, 512, 7, 7).cuda() 
    feat3 = torch.randn(32, 512, 7, 7).cuda() 
    feat4 = torch.randn(32, 512, 7, 7).cuda() 
    
    # 两特征融合
    mefm2 = MEFM(in_channel=512, num_features=2).cuda()
    output = mefm2(feat1, feat2)
    print(output.shape)

    # 三特征融合 
    mefm3 = MEFM(in_channel=512, num_features=3).cuda()
    output = mefm3(feat1, feat2, feat3)
    print(output.shape)

    feature_list = [feat1, feat2, feat3, feat4]
    
    # N特征融合
    mefmN = MEFM(in_channel=512, num_features=len(feature_list)).cuda()
    output = mefmN(*feature_list)
    # 输出经过模型处理后的结果张量的形状，用于确认模型的输出维度是否正确
    print(output.shape)  # 输出形状应为 (batch_size=2, channels=512, height=7, width=7)