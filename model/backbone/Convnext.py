import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from timm.models import convnext
from torchvision import transforms

from dataUtil.LungNoduleDataLoader import get_loaders

class MultiTaskConvNext(nn.Module):
    def __init__(self, attribute_classes, model, feature_dim, pretrained=False):
        super().__init__()
        # 共享骨干网络
        self.model = model
        self.feature_dim = feature_dim
                
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
    
def MultiTaskConvNext_S(attribute_classes):
    return MultiTaskConvNext(attribute_classes, convnext.convnext_small(pretrained=False, num_classes=0), 768)

def MultiTaskConvNext_T(attribute_classes):
    return MultiTaskConvNext(attribute_classes, convnext.convnext_tiny(pretrained=False, num_classes=0), 768)

def MultiTaskConvNext_B(attribute_classes):
    return MultiTaskConvNext(attribute_classes, convnext.convnext_base(pretrained=False, num_classes=0), 1024)

def MultiTaskConvNext_L(attribute_classes):
    return MultiTaskConvNext(attribute_classes, convnext.convnext_large(pretrained=False, num_classes=0), 1536)

def MultiTaskConvNext_xL(attribute_classes):
    return MultiTaskConvNext(attribute_classes, convnext.convnext_xlarge_384_in22ft1k(pretrained=False, num_classes=0), 2048)


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
    model = MultiTaskConvNext_B(attribute_classes).to(device)
    # print(model)
    dummy_img = torch.randn(32, 3, 224, 224).cuda()
    output = model(dummy_img)  # 测试模型
    print(output)