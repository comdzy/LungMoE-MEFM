import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve, 
                            average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns

from model.backbone.ResNet import MultiTaskResNet34
from model.backbone.ResNet import MultiTaskResNet50
from model.backbone.ResNet import MultiTaskResNet101
from model.backbone.ResNet import MultiTaskResNet152

from model.backbone.Densenet import MultiTaskDenseNet121
from model.backbone.Densenet import MultiTaskDenseNet161
from model.backbone.Densenet import MultiTaskDenseNet169
from model.backbone.Densenet import MultiTaskDenseNet201
from model.backbone.Densenet import MultiTaskDenseNet264

from model.backbone.ViT import MultiTaskViT
from model.backbone.RTNet import MultiTaskRTNet

from dataUtil.LungNoduleDataLoader import get_loaders

class Tester:
    def __init__(self, config, model, test_loader, attribute_classes, device):
        self.config = config
        self.result_dir = config['result_dir']
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.test_loader = test_loader
        self.attribute_classes = attribute_classes
        self.device = device
        
        # 初始化模型
        self.model = model
        self._load_checkpoint(config['checkpoint_path'])
        
        # 初始化结果存储
        self.all_preds = {attr: [] for attr in self.attribute_classes}
        self.all_labels = {attr: [] for attr in self.attribute_classes}
        self.all_probs = {attr: [] for attr in self.attribute_classes}

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {path}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                
                for attr in self.attribute_classes:
                    probs = torch.softmax(outputs[attr], dim=1)
                    _, preds = torch.max(probs, 1)
                    self.all_preds[attr].extend(preds.cpu().numpy())
                    self.all_labels[attr].extend(labels[attr].cpu().numpy())
                    self.all_probs[attr].extend(probs.cpu().numpy())
        
        # 保存完整结果
        self._save_full_results()
        
        # 生成评估报告
        results = {}
        for attr in self.attribute_classes:
            results[attr] = self._generate_attr_report(attr)
            
        return results

    def _generate_attr_report(self, attr):
        # 分类报告
        report = classification_report(
            self.all_labels[attr], self.all_preds[attr],
            target_names=[str(i) for i in range(self.attribute_classes[attr])],
            digits=4,
            output_dict=True,
            zero_division=1
        )
        
        # 保存文本报告
        with open(os.path.join(self.result_dir, f'{attr}_report.txt'), 'w') as f:
            f.write(json.dumps(report, indent=2))
        
        # 混淆矩阵
        self._plot_confusion_matrix(attr)
        
        # ROC曲线
        self._plot_roc_curve(attr)
        
        # PR曲线
        self._plot_pr_curve(attr)
        
        return {
            'accuracy': report['accuracy'],
            'roc_auc': self._calculate_roc_auc(attr),
            'average_precision': self._calculate_average_precision(attr)
        }

    def _plot_confusion_matrix(self, attr):
        cm = confusion_matrix(self.all_labels[attr], self.all_preds[attr])
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{attr} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.result_dir, f'{attr}_confusion_matrix.png'), dpi=1000)
        plt.close()

    def _plot_roc_curve(self, attr):
        probs = np.array(self.all_probs[attr])
        n_classes = self.attribute_classes[attr]
        
        plt.figure()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((np.array(self.all_labels[attr]) == i).astype(int), probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{attr} ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.result_dir, f'{attr}_roc_curve.png'), dpi=1000)
        plt.close()

    def _plot_pr_curve(self, attr):
        probs = np.array(self.all_probs[attr])
        n_classes = self.attribute_classes[attr]
        
        plt.figure()
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                (np.array(self.all_labels[attr]) == i).astype(int), probs[:, i])
            ap = average_precision_score(
                (np.array(self.all_labels[attr]) == i).astype(int), probs[:, i])
            plt.plot(recall, precision, label=f'Class {i} (AP = {ap:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{attr} Precision-Recall Curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.result_dir, f'{attr}_pr_curve.png'), dpi=1000)
        plt.close()

    def _calculate_roc_auc(self, attr):
        probs = np.array(self.all_probs[attr])
        return {i: auc(*roc_curve((np.array(self.all_labels[attr]) == i).astype(int), 
                       probs[:, i])[:2]) 
                for i in range(self.attribute_classes[attr])}

    def _calculate_average_precision(self, attr):
        probs = np.array(self.all_probs[attr])
        return {i: average_precision_score(
                    (np.array(self.all_labels[attr]) == i).astype(int), probs[:, i])
                for i in range(self.attribute_classes[attr])}
    
    def _save_full_results(self):
        results = {
            'labels': {
                attr: [int(label) for label in self.all_labels[attr]]
                for attr in self.attribute_classes
            },
            'preds': {
                attr: [int(pred) for pred in self.all_preds[attr]]
                for attr in self.attribute_classes
            },
            'probs': {
                attr: [list(map(float, prob)) for prob in self.all_probs[attr]]
                for attr in self.attribute_classes
            }
        }
        
        with open(os.path.join(self.result_dir, 'full_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

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
    
    tester = Tester(config, model, test_loader, attribute_classes, device)
    test_results = tester.evaluate()
        
    print("\n=== Final Test Results ===")
    for attr, res in test_results.items():
        print(f"{attr}:")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  Avg ROC AUC: {np.mean(list(res['roc_auc'].values())):.4f}")
        print(f"  Avg Precision: {np.mean(list(res['average_precision'].values())):.4f}")