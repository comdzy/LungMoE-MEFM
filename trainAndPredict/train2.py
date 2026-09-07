import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import random
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
import gc

import torch
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score)
import matplotlib.pyplot as plt

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
from model.backbone.DTNet import MultiTaskDTNet

from model.backbone.HRNet import MultiTaskHRNet_w18
from model.backbone.HRNet import MultiTaskHRNet_w18_v2
from model.backbone.HRNet import MultiTaskHRNet_w32
from model.backbone.HRNet import MultiTaskHRNet_w48

from model.backbone.Convnext import MultiTaskConvNext_S
from model.backbone.Convnext import MultiTaskConvNext_T
from model.backbone.Convnext import MultiTaskConvNext_B
from model.backbone.Convnext import MultiTaskConvNext_L
from model.backbone.Convnext import MultiTaskConvNext_xL

from model.mutiLungSC.LungMoE import LungMoE
from model.mutiLungSC.LungMutiSCMoE import LungMutiSCMoE

from dataUtil.LungNoduleDataLoader import get_loaders
from dataUtil.plot_learning_curves import plot_learning_curves
from trainAndPredict.test import Tester
from heatmap import draw_heatmap

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, attribute_classes, device):
        self.config = config
        
        # 初始化数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.attribute_classes = attribute_classes
        self.device = device
        
        # 初始化目录
        self.result_dir = self._create_result_dir()
        
        # 初始化模型
        self.model = model
        
        # 添加混合精度训练组件
        self.scaler = torch.amp.GradScaler(enabled=config['mixed_precision'])
        self.autocast = torch.amp.autocast(device_type=str(self.device), enabled=config['mixed_precision'])
        
        # 设置测试配置
        self.test_config = {
            'data_path': './dataset',
            'batch_size': 32,
            'checkpoint_path': f'{self.result_dir}/checkpoints/best_model.pth',
            'result_dir': f'{self.result_dir}/test_results'
        }
        
        # 优化配置
        self.optimizer = AdamW(self.model.parameters(), 
                             lr=config['lr'], 
                             weight_decay=config.get('weight_decay', 1e-4))
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                         mode='max', 
                                         patience=config.get('patience', 2), 
                                         factor=config.get('factor', 0.5))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lossConfig = {
            'Subtlety': 0.308522621716693, 
            'InternalStructure': 0.02480339584292037, 
            'Calcification': 0.6977934992625768, 
            'Sphericity': 0.09332310437887115, 
            'Margin': 0.1994661728501534, 
            'Lobulation': 0.25856170318998695, 
            'Spiculation': 0.31189084928182237, 
            'Texture': 0.1056386534769756, 
            'Malignancy': 1
            }
        
        # 训练记录
        self.best_metrics = {attr: 0.0 for attr in self.attribute_classes}
        self.history = []
        self.start_time = time.time()

    def _create_result_dir(self):
        base_dir = self.config.get('result_dir', 'results')
        os.makedirs(base_dir, exist_ok=True)
        experiment_id = len(os.listdir(base_dir)) + 1
        result_dir = os.path.join(base_dir, f'exp_{experiment_id:03d}')
        os.makedirs(result_dir, exist_ok=True)
        
        dirs = ['checkpoints', 'metrics', 'plots', 'heatmaps', 'test_results']
        for d in dirs:
            os.makedirs(os.path.join(result_dir, d), exist_ok=True)
        
        return result_dir

    def _seed_everything(self):
        seed = self.config.get('seed')
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        self._seed_everything()
        print(f"🚀 Starting training with config:\n{json.dumps(self.config, indent=2)}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练阶段
            train_metrics = self._train_epoch()
            
            # 验证阶段
            val_metrics = self._validate()
            
            # 保存记录
            self._save_metrics(epoch, train_metrics, val_metrics)
            
            # 保存最佳模型
            self._update_best_models(val_metrics)
            
            # 更新学习率
            self.scheduler.step(val_metrics['val_Malignancy']['accuracy'])
        
        # 最终处理
        self._finalize_training()
        print(f"\n✅ Training comMoEte. Results saved to: {self.result_dir}")

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        preds = {attr: [] for attr in self.attribute_classes}
        labels = {attr: [] for attr in self.attribute_classes}

        with tqdm(self.train_loader, desc="Training") as pbar:
            for images, batch_labels in pbar:
                images = images.to(self.device)
                batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

                # 混合精度前向传播
                with self.autocast:
                    outputs = self.model(images)
                    
                    # 计算多任务损失
                    losses = []
                    for attr in self.attribute_classes:
                        loss = self.lossConfig[attr] * self.criterion(outputs[attr], batch_labels[attr])
                        losses.append(loss)
                    total_loss = sum(losses)

                # 反向传播与梯度缩放
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()  # 缩放梯度
                
                # 梯度裁剪（在缩放后执行）
                self.scaler.unscale_(self.optimizer)  # 必须反缩放后才能裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 参数更新
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 记录指标（保持FP32精度）
                epoch_loss += total_loss.item()  # 这里会自动转换回FP32
                with torch.no_grad():  # 减少显存占用
                    for attr in self.attribute_classes:
                        _, attr_preds = torch.max(outputs[attr], 1)
                        preds[attr].extend(attr_preds.detach().cpu().numpy())
                        labels[attr].extend(batch_labels[attr].cpu().numpy())
                
                pbar.set_postfix({'loss': total_loss.item()})

        # 计算指标
        train_metrics = {
            'loss': epoch_loss/len(self.train_loader),
            **self._calculate_metrics(preds, labels, prefix='train')
        }
        return train_metrics

    def _validate(self):
        self.model.eval()
        preds = {attr: [] for attr in self.attribute_classes}
        labels = {attr: [] for attr in self.attribute_classes}

        with torch.no_grad(), tqdm(self.val_loader, desc="Validating") as pbar:
            for images, batch_labels in pbar:
                images = images.to(self.device)
                outputs = self.model(images)
                
                for attr in self.attribute_classes:
                    _, attr_preds = torch.max(outputs[attr], 1)
                    preds[attr].extend(attr_preds.cpu().numpy())
                    labels[attr].extend(batch_labels[attr].cpu().numpy())

        return self._calculate_metrics(preds, labels, prefix='val')

    def _calculate_metrics(self, preds, labels, prefix):
        metrics = {}
        for attr in self.attribute_classes:
            metrics[f'{prefix}_{attr}'] = {
                'accuracy': accuracy_score(labels[attr], preds[attr]),
                'precision': precision_score(labels[attr], preds[attr], average='macro', zero_division=1),
                'recall': recall_score(labels[attr], preds[attr], average='macro'),
                'f1': f1_score(labels[attr], preds[attr], average='macro')
            }
        return metrics

    def _save_metrics(self, epoch, train_metrics, val_metrics):
        record = {
            'epoch': epoch+1,
            'loss': train_metrics['loss'],
            **self._flatten_metrics(val_metrics)
        }
        
        # 保存到CSV
        csv_path = os.path.join(self.result_dir, 'metrics/training_metrics.csv')
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        
        # 保存到历史记录
        self.history.append(record)

    def _flatten_metrics(self, metrics):
        flattened = {}
        for key in metrics:
            if isinstance(metrics[key], dict):
                for subkey, value in metrics[key].items():
                    flattened[f"{key}_{subkey}"] = value
            else:
                flattened[key] = metrics[key]
        return flattened

    def _update_best_models(self, val_metrics):
        current_acc = val_metrics[f'val_Malignancy']['accuracy']
        if current_acc > self.best_metrics['Malignancy']:
            self.best_metrics['Malignancy'] = current_acc
            torch.save(
                self.model.state_dict(),
                os.path.join(self.result_dir, f'checkpoints/best_model.pth')
            )
            print(f"🌟 New best model saved with accuracy: {current_acc:.4f}")

    def _finalize_training(self):
        
        # 保存最终模型
        torch.save(
            self.model.state_dict(),
            os.path.join(self.result_dir, 'checkpoints/final_model.pth')
        )
        
        # # 生成热力图
        # self._generate_heatmaps()
        
        # 绘制训练图
        plot_learning_curves(self.result_dir)
        
        # 测试
        tester = Tester(self.test_config, self.model, self.val_loader, self.attribute_classes, self.device)
        test_results = tester.evaluate()
            
        print("\n=== Final Test Results ===")
        for attr, res in test_results.items():
            print(f"{attr}:")
            print(f"  Accuracy: {res['accuracy']:.4f}")
            print(f"  Avg ROC AUC: {np.mean(list(res['roc_auc'].values())):.4f}")
            print(f"  Avg Precision: {np.mean(list(res['average_precision'].values())):.4f}")
        
        # 输出训练时间
        training_time = time.time() - self.start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = training_time % 60
        print(f"\n⏱️ Total training time: {hours}h {minutes}m {seconds:.2f}s")

    def _generate_heatmaps(self):

        samMoE_ids = ['1017004304']  # 示例ID
        for img_id in samMoE_ids:
            try:
                draw_heatmap(
                    result_id=self.result_dir,
                    image_id=img_id,
                    model=self.model
                )
                print(f"🔥 Heatmap generated for {img_id}")
            except Exception as e:
                print(f"❌ Failed to generate heatmap for {img_id}: {str(e)}")

if __name__ == "__main__":

    gc.collect()

    torch.cuda.empty_cache()

    print(f"当前可用显存：{torch.cuda.mem_get_info()[0]/1024**2:.2f} MB")
    
    config = {
        'lr': 1e-4,
        'epochs': 500,
        'seed': random.randint(1, 10000000),
        'result_dir': './result/experiments',
        'mixed_precision': True,  # 控制是否启用
        'grad_accum_steps': 2,    # 梯度累积步数
        'use_amp': True           # 备用别名（可选）
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, attribute_classes= get_loaders(root_dir='./dataset', batch_size=32,
            # 数据增强和转换
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomVerticalFlip(),    # 随机垂直翻转
                transforms.RandomRotation(15),      # 随机旋转15度
                transforms.ColorJitter(brightness=0.2),  # 随机改变亮度
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 随机仿射变换
                transforms.ToTensor(),              # 转换为张量
                transforms.Normalize(mean=[0.5], std=[0.5]),  # 标准化，针对黑白图像
            ]))
    
    model_classes = [
        # MultiTaskResNet34,
        # MultiTaskResNet50,
        # MultiTaskResNet101,
        # MultiTaskResNet152,
        # MultiTaskDenseNet121,
        # MultiTaskDenseNet161,
        # MultiTaskDenseNet169,
        # MultiTaskDenseNet201,
        # # MultiTaskDenseNet264,
        # MultiTaskViT,
        # MultiTaskRTNet,
        # MultiTaskDTNet
        # MultiTaskHRNet_w18,
        # MultiTaskHRNet_w18_v2,
        # MultiTaskHRNet_w32,
        # MultiTaskHRNet_w48,
        # MultiTaskConvNext_S,
        # MultiTaskConvNext_T,
        # MultiTaskConvNext_B,
        # MultiTaskConvNext_L,
        MultiTaskConvNext_xL

    ]
    

    for ModelClass in model_classes:
        # 动态创建模型实例
        model = ModelClass(attribute_classes).to(device)
        
        # 初始化训练器
        trainer = Trainer(config, model, train_loader, val_loader, attribute_classes, device)
        
        # 执行训练
        trainer.train()
        
        # === 显存释放关键步骤 ===
        # 1. 删除模型和训练器引用
        del model
        del trainer
        
        # 2. 强制Python垃圾回收
        gc.collect()
        
        # 3. 清空PyTorch的CUDA缓存
        torch.cuda.empty_cache()
        
        # 4. 验证显存释放
        print(f"当前可用显存：{torch.cuda.mem_get_info()[0]/1024**2:.2f} MB")
    