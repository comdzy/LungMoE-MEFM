import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from dataUtil.LungNoduleDataLoader import get_loaders

class MultiTaskViT(nn.Module):
    def __init__(self, attribute_classes, model = models.vit_l_32(), pretrained=False):
        super().__init__()
        
        # 移除原始分类头并获取特征维度
        self.feature_dim = model.hidden_dim
        model.heads = nn.Identity()
 
        self.feature_extractor = model
        
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
    model = MultiTaskViT(attribute_classes).to(device)
    print(model)