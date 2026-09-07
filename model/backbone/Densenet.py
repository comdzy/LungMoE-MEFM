import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from dataUtil.LungNoduleDataLoader import get_loaders

class MultiTaskDenseNet(nn.Module):
    def __init__(self, attribute_classes, model, pretrained=False):
        super().__init__()
        # 共享骨干网络
        base_model = model
        self.feature_extractor = nn.Sequential(
            base_model.features,
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.feature_dim = base_model.classifier.in_features
                
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
    
def MultiTaskDenseNet121(attribute_classes):
    return MultiTaskDenseNet(attribute_classes, models.densenet121())

def MultiTaskDenseNet161(attribute_classes):
    return MultiTaskDenseNet(attribute_classes, models.densenet161())

def MultiTaskDenseNet169(attribute_classes):
    return MultiTaskDenseNet(attribute_classes, models.densenet169())

def MultiTaskDenseNet201(attribute_classes):
    return MultiTaskDenseNet(attribute_classes, models.densenet201())

def MultiTaskDenseNet264(attribute_classes):
    return MultiTaskDenseNet(attribute_classes, models.DenseNet(block_config=(6, 12, 48, 32)))

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
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    # print(model)
    dummy_img = torch.randn(32, 3, 224, 224).cuda()
    output = model(dummy_img)  # 测试模型
    for value in output:
        print(len(value))