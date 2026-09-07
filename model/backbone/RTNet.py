import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn
import torch.nn.functional as F
import pywt
import pywt.data
from functools import partial
from einops.layers.torch import Rearrange

import torchvision.models as models
from torchvision import transforms

from dataUtil.LungNoduleDataLoader import get_loaders
from attn.TMFM import TMFM

class MultiTaskRTNet(nn.Module):
    def __init__(self, attribute_classes, pretrained=False):
        super().__init__()
        # 共享骨干网络
        base_model = models.resnet34()
        self.attention = TMFM(in_channels=256, out_channels=256)
        self.feature_extractor = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            self.attention,
            base_model.layer4,
            base_model.avgpool
        )
        self.feature_dim = base_model.fc.in_features
        
        # 多任务输出头
        self.heads = nn.ModuleDict()
        for attr, num_classes in attribute_classes.items():
            self.heads[attr] = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return {attr: head(features) for attr, head in self.heads.items()}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'data_path': './dataset',
        'batch_size': 32,
        'checkpoint_path': './result/experiments/exp_001/checkpoints/best_model.pth',
        'result_dir': './result/experiments/exp_001/test_results'
    }
    # 初始化数据加载器
    _, test_loader, attribute_classes = get_loaders(
        root_dir=config['data_path'],
        batch_size=config['batch_size'],
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    model = MultiTaskRTNet(attribute_classes).to(device)
    # print(model)
    dummy_img = torch.randn(2, 3, 224, 224).cuda()
    output = model(dummy_img)  # 测试模型
    # print(f"Output shape: {output.shape}")
    print(output)