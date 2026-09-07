import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataUtil.LungNoduleDataset import LungNoduleDataset

from collections import Counter

def get_loaders(root_dir, transform, batch_size=32):

    # 先创建临时训练数据集用于获取属性统计
    train_dataset = LungNoduleDataset(root_dir, mode='train')
    attribute_mapping = train_dataset.attribute_mapping
    attribute_classes = train_dataset.attribute_classes

   # 创建正式数据集
    train_dataset = LungNoduleDataset(
        root_dir, mode='train',
        transform=transform,
        attribute_mapping=attribute_mapping,
        attribute_classes=attribute_classes
    )

    test_dataset = LungNoduleDataset(
        root_dir, mode='test',
        transform=transform,
        attribute_mapping=attribute_mapping,
        attribute_classes=attribute_classes
    )

    # 自定义数据打包函数
    def collate_fn(batch):
        images = []
        labels = {attr: [] for attr in batch[0][1].keys()}
        
        for img, lbl_dict in batch:
            images.append(img)
            for attr, val in lbl_dict.items():
                labels[attr].append(val)
                
        return torch.stack(images), {k: torch.stack(v) for k, v in labels.items()}

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, test_loader, attribute_classes

def get_class(dataloader):
    # 使用 Counter 统计类别数量
    class_count = Counter()

    for _, labels in dataloader:
        class_count.update(labels.tolist())  # 将标签转换为列表后直接更新计数器

    print("Dataloader has:")
    for cls, count in class_count.items():
        print(f"Class {cls}: {count} samples")

    return class_count

if __name__ == "__main__":
    
    # 数据增强和转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 适应网络输入大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),    # 随机垂直翻转
        transforms.ToTensor(),
    ])
    
    train_dataloader, test_dataloader, attribute_classes= get_loaders("./dataset", transform)
    
    # 查看数据统计
    print("每个属性的类别数量：")
    for attr, num in attribute_classes.items():
        print(f"{attr}: {num} classes")
    
    # # 获取一个批次数据
    # images, labels = next(iter(train_dataloader))
    # print("图像尺寸:", images.shape)
    # print("标签示例:", {k: v.shape for k, v in labels.items()})