import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from timm.models import hrnet
from torchvision import transforms

from dataUtil.LungNoduleDataLoader import get_loaders

class MultiTaskHRNet(nn.Module):
    def __init__(self, attribute_classes, model, pretrained=False):
        super().__init__()
        # 共享骨干网络
        self.model = model

        self.feature_dim = 2048
                
        # 多任务输出头
        self.heads = nn.ModuleDict()
        for attr, num_classes in attribute_classes.items():
            self.heads[attr] = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        self.relu = nn.ReLU(inplace=False)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.model.forward_features(x)
        features = self.relu(features)
        features = self.avg(features)
        features = torch.flatten(features, 1)
        return {attr: head(features) for attr, head in self.heads.items()}
    
def MultiTaskHRNet_w18(attribute_classes):
    return MultiTaskHRNet(attribute_classes, hrnet.hrnet_w18(pretrained=False, num_classes=0))

def MultiTaskHRNet_w18_v2(attribute_classes):
    return MultiTaskHRNet(attribute_classes, hrnet.hrnet_w18_small_v2(pretrained=False, num_classes=0))

def MultiTaskHRNet_w32(attribute_classes):
    return MultiTaskHRNet(attribute_classes, hrnet.hrnet_w32(pretrained=False, num_classes=0))

def MultiTaskHRNet_w48(attribute_classes):
    return MultiTaskHRNet(attribute_classes, hrnet.hrnet_w48(pretrained=False, num_classes=0))

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
    model = MultiTaskHRNet_w48(attribute_classes).to(device)
    # print(model)
    dummy_img = torch.randn(32, 3, 224, 224).cuda()
    output = model(dummy_img)  # 测试模型
    print(output)