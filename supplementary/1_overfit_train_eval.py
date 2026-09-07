# -*- coding: utf-8 -*-
"""
LIDC_first 训练评估（7161/796/884 + 早停 + 双权重 + W-Acc/AUC/AP）

设计原则：
- 原 trainMuScMoE.Trainer 的 __init__ / _train_epoch / _validate / _create_result_dir /
  _seed_everything 全部一字不差复制（含 device_type="cuda:3"、T_max=8、无 weight_decay）
- 仅替换 _calculate_metrics（追加 W-Acc, 不删原 4 个指标）
- 仅替换 train()（加早停 patience=30 + 每 epoch 保存 final_model.pth）
- 仅替换数据加载（7161/796 切分，原 trainMuScMoE.py 没有 val 数据集）
- 加 test_compare()：在 884 固定测试集上分别评估 best/final 两套权重
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

# AutoDL 容器默认部署路径：/root/autodl-tmp/ (数据盘, 50G)
# 系统盘只剩 1.7G, 必须放在 autodl-tmp 下
CODE_ROOT = r"/root/autodl-tmp/code"
DATA_DIR  = r"/root/autodl-tmp/data/dataset"
WORK_DIR  = r"/root/autodl-tmp/data/_eval"
sys.path.insert(0, CODE_ROOT)
from dataUtil.LungNoduleDataset import LungNoduleDataset
from model.mutiLungSC.LungMutiSCMoE import LungMutiSCMoE

# ===== 配置（只改 4 个：数据路径/work_dir/max_epochs/patience，其余与原 trainMuScMoE 一致） =====
CONFIG = {
    'lr': 0.0001,
    'epochs': 500,
    'patience': 30,                    # 新增：早停
    'seed': random.randint(1, 10000000),  # 一字不差复制自原 trainMuScMoE.py
    'data_path': DATA_DIR,
    'batch_size': 32,
    'mixed_precision': True,
    'result_dir': WORK_DIR,
    'cuda_device': 0,                  # 数据加载时 collate 后 batch.to(self.device) 用
}

# ===== 任务损失权重（一字不差复制自原 trainMuScMoE.Trainer.lossConfig） =====
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


# ===== 7161/796 文件级切分（与 datashffule.py 同策略：按文件名） =====
def build_loaders():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("=== 数据加载 (按文件名 9:1 切 7161/796, test 884 不动) ===", flush=True)

    base = LungNoduleDataset(CONFIG['data_path'], mode='train')
    attribute_classes = base.attribute_classes
    print(f"  attribute_classes: {attribute_classes}", flush=True)
    print(f"  train dir samples: {len(base.samples)}", flush=True)

    file_to_indices = defaultdict(list)
    file_to_mal = {}
    for i, (img_path, roi) in enumerate(base.samples):
        file_to_indices[img_path].append(i)
        if img_path not in file_to_mal:
            v = roi.get('Malignancy', -1)
            file_to_mal[img_path] = int(v) if isinstance(v, (int, np.integer)) else -1
    files = sorted(file_to_indices.keys())
    print(f"  total train files: {len(files)}", flush=True)
    # 跟原 datashffule.py 一字不差: random.shuffle 不设 seed, 不 stratify
    # 原 datashffule.py: random.shuffle(total_txt); train = random.sample(..., num_train)
    # 我用 random.shuffle + 取前 796 为 val, 余下为 train (等价)
    files_v = list(files)  # 用全部 7957 文件, 跟原 datashffule.py 一样不 drop 任何
    random.shuffle(files_v)
    val_files = sorted(files_v[:796])
    train_files = sorted(files_v[796:])
    train_idx = [i for f in train_files for i in file_to_indices[f]]
    val_idx = [i for f in val_files for i in file_to_indices[f]]
    print(f"  train files={len(train_files)} (ROI={len(train_idx)})  val files={len(val_files)} (ROI={len(val_idx)})", flush=True)
    assert len(train_files) == 7161
    assert len(val_files) == 796

    train_ds = LungNoduleDataset(CONFIG['data_path'], mode='train', transform=TRAIN_TFM,
                                 attribute_mapping=base.attribute_mapping,
                                 attribute_classes=base.attribute_classes)
    val_ds   = LungNoduleDataset(CONFIG['data_path'], mode='train', transform=EVAL_TFM,
                                 attribute_mapping=base.attribute_mapping,
                                 attribute_classes=base.attribute_classes)
    test_ds  = LungNoduleDataset(CONFIG['data_path'], mode='test', transform=EVAL_TFM,
                                 attribute_mapping=base.attribute_mapping,
                                 attribute_classes=base.attribute_classes)

    def collate(batch):
        imgs = [b[0] for b in batch]
        lbls = {k: [b[1][k] for b in batch] for k in batch[0][1].keys()}
        return torch.stack(imgs), {k: torch.stack(v) for k, v in lbls.items()}

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=0, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=0, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=0, collate_fn=collate, pin_memory=True)
    return train_loader, val_loader, test_loader, attribute_classes


# ===== Trainer（一字不差复制原 trainMuScMoE.Trainer, 仅替换 _calculate_metrics + train） =====
class Trainer:
    def __init__(self, config, model, train_loader, val_loader, attribute_classes, device):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.attribute_classes = attribute_classes
        self.device = device
        self.result_dir = self._create_result_dir()
        self.model = model
        # 云服务器 2×4090: device_type="cuda" (PyTorch 标准形式, 兼容 DataParallel)
        self.scaler = GradScaler(enabled=config['mixed_precision'])
        self.autocast = torch.amp.autocast(device_type="cuda", enabled=config['mixed_precision'])
        self.test_config = {
            'data_path': './dataset',
            'batch_size': 32,
            'checkpoint_path': f'{self.result_dir}/checkpoints/best_model.pth',
            'result_dir': f'{self.result_dir}/test_results'
        }
        # ② 一字不差: 无 weight_decay
        self.optimizer = AdamW(self.model.parameters(), lr=config['lr'])
        # ③ 一字不差: T_max=8
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=64, eta_min=0, last_epoch=-1, verbose=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lossConfig = LOSS_CONFIG  # 与原一字不差
        self.best_metrics = {attr: 0.0 for attr in self.attribute_classes}
        self.history = []
        self.start_time = time.time()
        # 早停 / 双权重状态（仅 train() 用, 不影响 _train_epoch/_validate 的逐字复制）
        self.no_improve = 0
        self.best_w_acc = -1.0
        self.best_epoch = -1
        self.stopped_epoch = None

    def _create_result_dir(self):
        # 一字不差复制
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
        # 一字不差复制
        seed = self.config.get('seed')
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _train_epoch(self):
        # 一字不差复制原 _train_epoch
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
        # 一字不差复制原 _validate (注意: 原代码验证时没用 autocast, 不动)
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
        # 原 4 个指标保留 (一字不差), 追加 W-Acc
        metrics = {}
        for attr in self.attribute_classes:
            w_acc = balanced_accuracy_score(labels[attr], preds[attr])
            metrics[f'{prefix}_{attr}'] = {
                'accuracy': accuracy_score(labels[attr], preds[attr]),
                'precision': precision_score(labels[attr], preds[attr], average='macro', zero_division=1),
                'recall': recall_score(labels[attr], preds[attr], average='macro'),
                'f1': f1_score(labels[attr], preds[attr], average='macro'),
                'w_acc': float(w_acc),  # 新增: W-Acc = balanced accuracy (每类 recall 等权)
            }
        return metrics

    def _save_metrics(self, epoch, train_metrics, val_metrics):
        # 一字不差复制
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
        # 一字不差复制
        flattened = {}
        for key in metrics:
            if isinstance(metrics[key], dict):
                for subkey, value in metrics[key].items():
                    flattened[f"{key}_{subkey}"] = value
            else:
                flattened[key] = metrics[key]
        return flattened

    def _update_best_models(self, val_metrics):
        # 一字不差复制 (按 Malignancy accuracy 选 best, 用 self.model.module.state_dict() 因 model 被 DataParallel 包装)
        current_acc = val_metrics[f'val_Malignancy']['accuracy']
        if current_acc > self.best_metrics['Malignancy']:
            self.best_metrics['Malignancy'] = current_acc
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.result_dir, f'checkpoints/best_model.pth')
            )
            print(f"🌟 New best model saved with accuracy: {current_acc:.4f}", flush=True)

    def _finalize_training(self):
        # 一字不差复制
        torch.save(
            self.model.module.state_dict(),
            os.path.join(self.result_dir, 'checkpoints/final_model.pth')
        )
        # 简化: 不画图不调 Tester (test_compare 我们单独跑)
        # 训练时间输出
        training_time = time.time() - self.start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = training_time % 60
        print(f"\n⏱️ Total training time: {hours}h {minutes}m {seconds:.2f}s", flush=True)

    def train(self):
        # 替换原 train(): 保留原 epoch 循环结构, 加早停 + 双权重 final_model 每 epoch 保存
        self._seed_everything()
        print(f"🚀 Starting training with config:\n{json.dumps(self.config, indent=2)}", flush=True)

        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}", flush=True)
            train_metrics = self._train_epoch()
            val_metrics = self._validate()
            self._save_metrics(epoch, train_metrics, val_metrics)
            self._update_best_models(val_metrics)
            self.scheduler.step()

            # 新增: 每 epoch 保存 final_model.pth (训练终止时刻的权重)
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.result_dir, 'checkpoints/final_model.pth')
            )

            # 新增: 早停逻辑 (按 val Malignancy w_acc)
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

        # 不调 _finalize_training (原版会调 plot_learning_curves + Tester, 我们都不需要)
        # 自己 final 一次: 保存双权重 + summary
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
        print(f"\n✅ Training done. best_epoch={self.best_epoch}  best_W-Acc={self.best_w_acc:.4f}  "
              f"stopped_epoch={self.stopped_epoch}", flush=True)


# ===== 测试对比 (best vs final 在 884 固定测试集上) =====
def test_compare(trainer, test_loader, attribute_classes):
    print("\n=== 测试对比 (test 884, final vs best) ===", flush=True)
    device = trainer.device
    ckpt_dir = os.path.join(trainer.result_dir, 'checkpoints')

    def evaluate(ckpt_path, tag):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        # 一字不差: 原 trainMutiGPU.py 用 self.model.module.load_state_dict() (因 checkpoint 存的是 module.state_dict())
        trainer.model.module.load_state_dict(ckpt)
        trainer.model.eval()
        all_preds = {a: [] for a in attribute_classes}
        all_probs = {a: [] for a in attribute_classes}
        all_lbls  = {a: [] for a in attribute_classes}
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(device)
                outputs = trainer.model(imgs)
                for a in attribute_classes:
                    probs = torch.softmax(outputs[a], dim=1)
                    _, pred = torch.max(probs, 1)
                    all_preds[a].extend(pred.cpu().numpy())
                    all_probs[a].extend(probs.cpu().numpy())
                    all_lbls[a].extend(lbls[a].numpy())
        per = {}
        for a in attribute_classes:
            yt = np.array(all_lbls[a]); yp = np.array(all_preds[a]); ypr = np.array(all_probs[a])
            w_acc = float(balanced_accuracy_score(yt, yp))
            auc_pc, ap_pc = [], []
            for c in range(ypr.shape[1]):
                yb = (yt == c).astype(int)
                if 0 < yb.sum() < len(yb):
                    try: auc_pc.append(roc_auc_score(yb, ypr[:, c]))
                    except: pass
                    try: ap_pc.append(average_precision_score(yb, ypr[:, c]))
                    except: pass
            per[a] = {'w_acc': w_acc,
                      'auc_macro': float(np.mean(auc_pc)) if auc_pc else 0.0,
                      'ap_macro': float(np.mean(ap_pc)) if ap_pc else 0.0}
        print(f"  [{tag}] evaluated on {len(test_loader.dataset)} samples", flush=True)
        return per

    print("--- 加载 best_model.pth ---", flush=True)
    best_m = evaluate(os.path.join(ckpt_dir, 'best_model.pth'), 'best')
    print("--- 加载 final_model.pth ---", flush=True)
    final_m = evaluate(os.path.join(ckpt_dir, 'final_model.pth'), 'final')

    rows = []
    for a in attribute_classes:
        b, f = best_m[a], final_m[a]
        rows.append({
            'task': a,
            'w_acc_best': b['w_acc'], 'w_acc_final': f['w_acc'], 'w_acc_drop': b['w_acc'] - f['w_acc'],
            'auc_best': b['auc_macro'], 'auc_final': f['auc_macro'], 'auc_drop': b['auc_macro'] - f['auc_macro'],
            'ap_best': b['ap_macro'], 'ap_final': f['ap_macro'], 'ap_drop': b['ap_macro'] - f['ap_macro'],
        })

    print(f"\n{'Task':20s} | {'W-Acc best':>10s} | {'W-Acc final':>11s} | {'Δ':>8s} | "
          f"{'AUC best':>9s} | {'AUC final':>10s} | {'Δ':>8s} | "
          f"{'AP best':>8s} | {'AP final':>9s} | {'Δ':>7s}")
    print("-" * 130)
    for r in rows:
        print(f"{r['task']:20s} | {r['w_acc_best']:10.4f} | {r['w_acc_final']:11.4f} | "
              f"{r['w_acc_drop']:8.4f} | {r['auc_best']:9.4f} | {r['auc_final']:10.4f} | "
              f"{r['auc_drop']:8.4f} | {r['ap_best']:8.4f} | {r['ap_final']:9.4f} | {r['ap_drop']:7.4f}")

    mm = {
        'w_acc_best': float(np.mean([r['w_acc_best'] for r in rows])),
        'w_acc_final': float(np.mean([r['w_acc_final'] for r in rows])),
        'auc_best': float(np.mean([r['auc_best'] for r in rows])),
        'auc_final': float(np.mean([r['auc_final'] for r in rows])),
        'ap_best': float(np.mean([r['ap_best'] for r in rows])),
        'ap_final': float(np.mean([r['ap_final'] for r in rows])),
    }
    print("-" * 130)
    print(f"{'MACRO MEAN':20s} | {mm['w_acc_best']:10.4f} | {mm['w_acc_final']:11.4f} | "
          f"{mm['w_acc_best']-mm['w_acc_final']:8.4f} | {mm['auc_best']:9.4f} | {mm['auc_final']:10.4f} | "
          f"{mm['auc_best']-mm['auc_final']:8.4f} | {mm['ap_best']:8.4f} | {mm['ap_final']:9.4f} | "
          f"{mm['ap_best']-mm['ap_final']:7.4f}")

    mal = next(r for r in rows if r['task'] == 'Malignancy')
    print(f"\n主任务 Malignancy:")
    print(f"  W-Acc: best={mal['w_acc_best']:.4f}  final={mal['w_acc_final']:.4f}  Δ={mal['w_acc_drop']:+.4f}")
    print(f"  AUC:   best={mal['auc_best']:.4f}    final={mal['auc_final']:.4f}    Δ={mal['auc_drop']:+.4f}")
    print(f"  AP:    best={mal['ap_best']:.4f}    final={mal['ap_final']:.4f}    Δ={mal['ap_drop']:+.4f}")

    overfit = sum(1 for r in rows if r['w_acc_drop'] > 0.01)
    print(f"\n过拟合评估 (W-Acc 下降 > 0.01 算明显):")
    print(f"  9 个任务中 {overfit}/9 个 best → final W-Acc 明显下降")
    if overfit >= 5:
        print("  → 持续训练产生明显过拟合, best_model 更可靠")
    elif overfit >= 1:
        print("  → 部分任务有轻度过拟合, 建议使用早停权重")
    else:
        print("  → 未观察到明显过拟合, final 与 best 接近")

    report = {
        'rows': rows, 'macro_mean': mm, 'malignancy': mal,
        'overfit_task_count': overfit, 'overfit_threshold': 0.01,
        'stop_summary': {
            'best_epoch': trainer.best_epoch,
            'best_w_acc': trainer.best_w_acc,
            'stopped_epoch': trainer.stopped_epoch,
        }
    }
    out = os.path.join(WORK_DIR, 'test_compare.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {out}")


if __name__ == "__main__":
    # 一字不差复制原 trainMuScMoE.py main 入口风格: device_ids + main_device + DataParallel 包装
    # 云服务器配置: 2 张 4090, device_ids = [0, 1]
    import torch.nn as nn
    t0 = time.time()
    train_loader, val_loader, test_loader, attribute_classes = build_loaders()
    device_ids = [0, 1]   # 2 张 4090
    main_device = torch.device(f"cuda:{device_ids[0]}")
    _ = torch.zeros(1).to(main_device)
    torch.cuda.synchronize()
    model = LungMutiSCMoE(attribute_classes, num_shared_experts=4, num_task_experts=2)
    # 一字不差: DataParallel 包装
    model = nn.DataParallel(model.to(main_device), device_ids=device_ids)
    trainer = Trainer(CONFIG, model, train_loader, val_loader, attribute_classes, main_device)
    trainer.train()
    print(f"\n总训练耗时: {(time.time()-t0)/60:.2f} min", flush=True)
    test_compare(trainer, test_loader, attribute_classes)
    print(f"\n全部完成: {(time.time()-t0)/60:.2f} min", flush=True)
