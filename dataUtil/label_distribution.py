import os
import json
import numpy as np
from collections import defaultdict
from LungNoduleDataset import LungNoduleDataset
import matplotlib.pyplot as plt

def analyze_label_distribution(root_dir):
    # 初始化统计字典
    distribution = defaultdict(lambda: defaultdict(int))
    
    # 遍历训练集和测试集
    for mode in ['train', 'test']:
        dataset = LungNoduleDataset(root_dir, mode=mode)
        for _, roi in dataset.samples:
            for attr in dataset.attributes:
                val = roi.get(attr, None)
                if val is not None:
                    distribution[attr][val] += 1

    # 保存统计结果
    with open('./result/data/label_distribution.json', 'w') as f:
        json.dump(distribution, f, indent=2)

    # 可视化
    plt.figure(figsize=(15, 15))
    for idx, (attr, counts) in enumerate(distribution.items()):
        plt.subplot(3, 3, idx+1)
        values = sorted(counts.keys())
        counts = [counts[v] for v in values]
        plt.bar(values, counts)
        plt.title(attr)
        plt.xlabel('Label Value')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('./result/data/label_distribution.png', dpi=1000)
    plt.close()

def collect_correlation_data(root_dir):
    # 数据结构：{属性: {该属性值: {Malignancy值: 计数}}}
    correlation_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for mode in ['train', 'test']:
        dataset = LungNoduleDataset(root_dir, mode=mode)
        for _, roi in dataset.samples:
            malignancy = roi.get('Malignancy', None)
            if malignancy is None:
                continue
                
            for attr in dataset.attributes:
                if attr == 'Malignancy':
                    continue
                val = roi.get(attr, None)
                if val is not None:
                    correlation_data[attr][val][malignancy] += 1

    # 保存数据
    with open('./result/data/correlation_data.json', 'w') as f:
        json.dump(correlation_data, f, indent=2)

if __name__ == "__main__":
    analyze_label_distribution("./dataset")
    collect_correlation_data("./dataset")