# -*- coding: utf-8 -*-
"""
消融实验脚本：4 expert + 单小波 (db1 only)
- 复制 train_eval.py 的全部结构, 仅做必要的消融改动
- 其他任何不相关的代码不做任何修改
- 训练集: 7957 (mode='train' 全部)
- 验证集: 884 (mode='test', 即原 test 集作为 val)
- 关掉早停: patience=500 (跑满 250 epoch)
- epoch=250, T_max=32
"""
import os, sys, json, time, csv, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, average_precision_score)
from tqdm import tqdm

# ===== 路径 (与 train_eval.py 一致) — 必须先设置 sys.path 再 import =====
CODE_ROOT = r"/root/autodl-tmp/code"
DATA_DIR  = r"/root/autodl-tmp/data/dataset"
WORK_DIR  = r"/root/autodl-tmp/data/_eval_db1"   # 输出到独立目录, 跟主实验分开
sys.path.insert(0, CODE_ROOT)

# ===== Monkey-patch: MEFM 只用 db1 (替换 MultiScaleFeatureExtractor) =====
# 注意: attn 实际在 /root/autodl-tmp/code/model/attn/ 不是 /code/attn/
# 所以要用 model.attn.MEFM 全路径
import model.attn.MEFM as _mefm_module
from model.attn.MEFM import DepthwiseSeparableConvWithWTConv2d

class MultiScaleFeatureExtractorDB1(nn.Module):
    """消融: 只用 db1 小波 (原版用 db1 + bior1.1 两个)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConvWithWTConv2d(in_channels, out_channels, wt_type='db1')
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        return out1   # 消融: 去掉 bior1.1 分支, 不再 out1+out2

# 替换原版 MultiScaleFeatureExtractor
_mefm_module.MultiScaleFeatureExtractor = MultiScaleFeatureExtractorDB1
print("⚠️ [消融] MEFM.MultiScaleFeatureExtractor 已替换为 db1-only 版本", flush=True)

from dataUtil.LungNoduleDataset import LungNoduleDataset
from model.mutiLungSC.LungMutiSCMoE import LungMutiSCMoE   # 导入的 LungMutiSCMoE 用的就是上面 patch 过的 MEFM

# ===== 配置 (消融改动) =====
CONFIG = {
    'lr': 0.0001,
    'epochs': 250,                  # 消融: 固定 250 epoch, 不早停
    'patience': 500,                # 消融: 关早停 (500 > 250, 永远不触发)
    'seed': random.randint(1, 10000000),
    'data_path': DATA_DIR,
    'batch_size': 32,
    'mixed_precision': True,
    'result_dir': WORK_DIR,
    'cuda_device': 0,
}

# ===== 任务损失权重 (一字不差复制原 trainMuScMoE.Trainer.lossConfig) =====
LOSS_CONFIG = {
    'Subtlety': 0.308522621716693,
    'InternalStructure': 0.02480339584292037,
    'Calcification': 0.6977934992625768,
    'Sphericity': 0.09332310437887115,
    'Margin': 0.1994661728501534,
    'Lobulation': 0.25856170318998695,
    'Spiculation': 0.31189084928182237,
    'Texture': 0.1056386534769756,
    'Malignancy': 1,
}

TRAIN_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
EVAL_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ===== 数据加载 (消融: 7957 train, 884 val, 无 test) =====
def build_loaders():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("=== [消融] 数据加载 (7957 全部 train 当 train, 884 当 val, 无 test) ===", flush=True)

    base = LungNoduleDataset(CONFIG['data_path'], mode='train')
    attribute_classes = base.attribute_classes
    print(f"  attribute_classes: {attribute_classes}", flush=True)
    print(f"  train dir samples: {len(base.samples)}", flush=True)

    train_ds = LungNoduleDataset(CONFIG['data_path'], mode='train', transform=TRAIN_TFM,
                                 attribute_mapping=base.attribute_mapping,
                                 attribute_classes=base.attribute_classes)
    val_ds   = LungNoduleDataset(CONFIG['data_path'], mode='test',  transform=EVAL_TFM,
                                 attribute_mapping=base.attribute_mapping,
                                 attribute_classes=base.attribute_classes)

    def collate(batch):
        imgs = [b[0] for b in batch]
        lbls = {k: [b[1][k] for b in batch] for k in batch[0][1].keys()}
        return torch.stack(imgs), {k: torch.stack(v) for k, v in lbls.items()}

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=0, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=0, collate_fn=collate, pin_memory=True)
    print(f"  train batches: {len(train_loader)}    val samples: {len(val_ds)}", flush=True)
    return train_loader, val_loader, attribute_classes


# ===== Trainer (一字不差复制原 trainMuScMoE.Trainer, 仅替换 _calculate_metrics + train) =====
class Trainer:
    def __init__(self, config, model, train_loader, val_loader, attribute_classes, device):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.attribute_classes = attribute_classes
        self.device = device
        self.result_dir = self._create_result_dir()
        self.model = model
        self.scaler = GradScaler(enabled=config['mixed_precision'])
        self.autocast = torch.amp.autocast(device_type="cuda", enabled=config['mixed_precision'])
        self.test_config = {
            'data_path': './dataset',
            'batch_size': 32,
            'checkpoint_path': f'{self.result_dir}/checkpoints/best_model.pth',
            'result_dir': f'{self.result_dir}/test_results'
        }
        # ① 一字不差: 无 weight_decay
        self.optimizer = AdamW(self.model.parameters(), lr=config['lr'])
        # ② 消融: T_max=32 (vs 原版 64)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=32, eta_min=0, last_epoch=-1, verbose=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lossConfig = LOSS_CONFIG
        self.best_metrics = {attr: 0.0 for attr in self.attribute_classes}
        self.history = []
        self.start_time = time.time()
        # 早停 / 双权重状态
        self.no_improve = 0
        self.best_w_acc = -1.0
        self.best_epoch = -1
        self.stopped_epoch = None

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

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        preds = {attr: [] for attr in self.attribute_classes}
        labels = {attr: [] for attr in self.attribute_classes}

        with tqdm(self.train_loader, desc="Training") as pbar:
            for images, batch_labels in pbar:
                images = images.to(self.device)
                batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

                with self.autocast:
                    outputs = self.model(images)
                    losses = []
                    for attr in self.attribute_classes:
                        loss = self.lossConfig[attr] * self.criterion(outputs[attr], batch_labels[attr])
                        losses.append(loss)
                    total_loss = sum(losses)

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += total_loss.item()
                with torch.no_grad():
                    for attr in self.attribute_classes:
                        _, attr_preds = torch.max(outputs[attr], 1)
                        preds[attr].extend(attr_preds.detach().cpu().numpy())
                        labels[attr].extend(batch_labels[attr].cpu().numpy())
                pbar.set_postfix({'loss': total_loss.item()})

        train_metrics = {
            'loss': epoch_loss / len(self.train_loader),
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
            w_acc = balanced_accuracy_score(labels[attr], preds[attr])
            metrics[f'{prefix}_{attr}'] = {
                'accuracy': accuracy_score(labels[attr], preds[attr]),
                'precision': precision_score(labels[attr], preds[attr], average='macro', zero_division=1),
                'recall': recall_score(labels[attr], preds[attr], average='macro'),
                'f1': f1_score(labels[attr], preds[attr], average='macro'),
                'w_acc': float(w_acc),
            }
        return metrics

    def _save_metrics(self, epoch, train_metrics, val_metrics):
        record = {
            'epoch': epoch + 1,
            'loss': train_metrics['loss'],
            **self._flatten_metrics(val_metrics)
        }
        csv_path = os.path.join(self.result_dir, 'metrics/training_metrics.csv')
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
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
                self.model.module.state_dict(),
                os.path.join(self.result_dir, f'checkpoints/best_model.pth')
            )
            print(f"🌟 New best model saved with accuracy: {current_acc:.4f}", flush=True)

    def train(self):
        self._seed_everything()
        print(f"🚀 [消融] Starting training with config:\n{json.dumps(self.config, indent=2)}", flush=True)

        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}", flush=True)
            train_metrics = self._train_epoch()
            val_metrics = self._validate()
            self._save_metrics(epoch, train_metrics, val_metrics)
            self._update_best_models(val_metrics)
            self.scheduler.step()

            # 每 epoch 保存 final_model.pth
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.result_dir, 'checkpoints/final_model.pth')
            )

            # 早停判据 (本消融 patience=500 >> 250, 实际不触发)
            cur_w_acc = val_metrics['val_Malignancy']['w_acc']
            if cur_w_acc > self.best_w_acc:
                self.best_w_acc = cur_w_acc
                self.best_epoch = epoch + 1
                self.no_improve = 0
                print(f"  *** new best val W-Acc {cur_w_acc:.4f} at epoch {epoch+1} ***", flush=True)
            else:
                self.no_improve += 1
                if self.no_improve >= self.config['patience']:
                    self.stopped_epoch = epoch + 1
                    print(f"\n⛔ Early stop at epoch {epoch+1}, no improve for {self.no_improve} epochs", flush=True)
                    break

        # 训练结束 (250 epoch 全部跑完)
        torch.save(
            self.model.module.state_dict(),
            os.path.join(self.result_dir, 'checkpoints/final_model.pth')
        )
        with open(os.path.join(self.result_dir, 'stop_summary.json'), 'w') as f:
            json.dump({
                'best_epoch': self.best_epoch,
                'best_val_Malignancy_w_acc': self.best_w_acc,
                'stopped_epoch': self.stopped_epoch,
                'total_epochs_trained': self.stopped_epoch or self.config['epochs'],
                'best_metrics': self.best_metrics,
            }, f, indent=2)
        print(f"\n✅ [消融] Training done. best_epoch={self.best_epoch}  best_W-Acc={self.best_w_acc:.4f}  "
              f"stopped_epoch={self.stopped_epoch}", flush=True)


# ===== 消融评估: Malignancy 5 等级的 per-grade W-Acc/AUC/AP =====
# 输出格式跟原论文表 16/17/18 对齐
def ablation_compare(trainer, attribute_classes):
    print("\n=== [消融] Malignancy per-grade 评估 (best vs final on 884 val) ===", flush=True)
    device = trainer.device
    ckpt_dir = os.path.join(trainer.result_dir, 'checkpoints')
    num_classes = attribute_classes['Malignancy']   # = 5

    def evaluate_per_grade(ckpt_path, tag):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        trainer.model.module.load_state_dict(ckpt)
        trainer.model.eval()
        all_preds, all_probs, all_lbls = [], [], []
        with torch.no_grad():
            for imgs, lbls in trainer.val_loader:
                imgs = imgs.to(device)
                outputs = trainer.model(imgs)
                probs = torch.softmax(outputs['Malignancy'], dim=1)
                _, pred = torch.max(probs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_lbls.extend(lbls['Malignancy'].numpy())
        y_true = np.array(all_lbls)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        # 每类 (5 等级) 的 recall (per-grade W-Acc), AUC, AP
        per_grade = {}
        for c in range(num_classes):
            yt = (y_true == c).astype(int)
            yp = (y_true == c) & (y_pred == c)
            recall_c = float(yp.sum() / yt.sum()) if yt.sum() > 0 else 0.0
            if 0 < yt.sum() < len(yt):
                try: auc_c = float(roc_auc_score(yt, y_prob[:, c]))
                except: auc_c = 0.0
                try: ap_c = float(average_precision_score(yt, y_prob[:, c]))
                except: ap_c = 0.0
            else:
                auc_c = 0.0
                ap_c = 0.0
            per_grade[f'grade_{c+1}'] = {'recall': recall_c, 'auc': auc_c, 'ap': ap_c}
        # W-Acc (balanced accuracy)
        per_grade['w_acc'] = float(balanced_accuracy_score(y_true, y_pred))
        per_grade['n_samples'] = len(y_true)
        # support (每类样本数)
        per_grade['support'] = [int((y_true == c).sum()) for c in range(num_classes)]
        print(f"  [{tag}] evaluated on {len(y_true)} samples", flush=True)
        return per_grade

    print("--- 加载 best_model.pth ---", flush=True)
    best = evaluate_per_grade(os.path.join(ckpt_dir, 'best_model.pth'), 'best')
    print("--- 加载 final_model.pth ---", flush=True)
    final = evaluate_per_grade(os.path.join(ckpt_dir, 'final_model.pth'), 'final')

    # 打印 3 张表 (跟原论文表 16/17/18 格式一致)
    print(f"\n[消融表 1: Malignancy per-grade W-Acc (best vs final)]", flush=True)
    print(f"{'Method':<22} | {'G1':<7} | {'G2':<7} | {'G3':<7} | {'G4':<7} | {'G5':<7} | {'W-Acc':<7}")
    print("-" * 90)
    for tag, m in [('best', best), ('final', final)]:
        cells = [m[f'grade_{g+1}']['recall'] for g in range(num_classes)]
        print(f"  LungMoE-MEFM_S ({tag}) | {cells[0]:.4f} | {cells[1]:.4f} | {cells[2]:.4f} | {cells[3]:.4f} | {cells[4]:.4f} | {m['w_acc']:.4f}")
    print(f"\n[消融表 2: Malignancy per-grade AUC]", flush=True)
    print(f"{'Method':<22} | {'G1':<7} | {'G2':<7} | {'G3':<7} | {'G4':<7} | {'G5':<7}")
    print("-" * 75)
    for tag, m in [('best', best), ('final', final)]:
        cells = [m[f'grade_{g+1}']['auc'] for g in range(num_classes)]
        print(f"  LungMoE-MEFM_S ({tag}) | {cells[0]:.4f} | {cells[1]:.4f} | {cells[2]:.4f} | {cells[3]:.4f} | {cells[4]:.4f}")
    print(f"\n[消融表 3: Malignancy per-grade AP]", flush=True)
    print(f"{'Method':<22} | {'G1':<7} | {'G2':<7} | {'G3':<7} | {'G4':<7} | {'G5':<7}")
    print("-" * 75)
    for tag, m in [('best', best), ('final', final)]:
        cells = [m[f'grade_{g+1}']['ap'] for g in range(num_classes)]
        print(f"  LungMoE-MEFM_S ({tag}) | {cells[0]:.4f} | {cells[1]:.4f} | {cells[2]:.4f} | {cells[3]:.4f} | {cells[4]:.4f}")
    print(f"\n[support (样本分布)]: {best['support']}  (total: {best['n_samples']})", flush=True)

    # 保存 JSON
    out = {
        'model': 'LungMoE-MEFM_S (单小波 db1)',
        'config': {
            'shared_experts': 4, 'task_experts': 2,
            'wavelet': 'db1 only (no bior1.1)',
            'epochs': trainer.config['epochs'],
            't_max': 32, 'patience': trainer.config['patience'],
            'train_samples': 7957, 'val_samples': 884,
        },
        'best': best,
        'final': final,
    }
    out_path = os.path.join(trainer.result_dir, 'ablation_compare.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {out_path}")


# ===== 主入口 =====
if __name__ == "__main__":
    import torch.nn as nn
    t0 = time.time()
    train_loader, val_loader, attribute_classes = build_loaders()
    device_ids = [0, 1]
    main_device = torch.device(f"cuda:{device_ids[0]}")
    _ = torch.zeros(1).to(main_device)
    torch.cuda.synchronize()
    model = LungMutiSCMoE(attribute_classes, num_shared_experts=4, num_task_experts=2)
    model = nn.DataParallel(model.to(main_device), device_ids=device_ids)
    trainer = Trainer(CONFIG, model, train_loader, val_loader, attribute_classes, main_device)
    trainer.train()
    # 训练结束后, 消融评估 (per-grade 指标, best vs final)
    ablation_compare(trainer, attribute_classes)
    print(f"\n总训练耗时: {(time.time()-t0)/60:.2f} min", flush=True)
