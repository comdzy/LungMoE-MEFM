"""
Cross-validation 5-sample-size 实验调度器 (用于论文 "表 1: 5 个 train_size 的重建实验")

对每个 train_size (100, 200, 300, 800, 912):
  1. 从完整 train 集里随机抽 N 个 nodule 文件
  2. 按文件名 9:1 切 train/val (与 train_eval.py 一致)
  3. 调 train_eval.py 子进程跑完, 捕获 test_compare.json
  4. 收集每 N 的: 样本数, 平均恶性率, 训练/测试 patient overlap (粗估), 训练/测试 nodule leakage
  5. 汇总成 paper 表 1 格式

用法:
  python _cross_validation_runner.py \\
      --data_dir /root/autodl-tmp/data/dataset \\
      --work_dir /root/autodl-tmp/data/_cv \\
      --train_sizes 100 200 300 800 912 \\
      --out_csv cv_results.csv
"""
import os, sys, json, time, argparse, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random


def list_train_files(data_dir):
    """列出 data/images/train 下所有 jpg 文件名"""
    p = Path(data_dir) / 'images' / 'train'
    if not p.exists():
        return []
    return sorted([f.name for f in p.iterdir() if f.suffix == '.jpg'])


def sample_train_files(data_dir, n, seed=42):
    """随机抽 n 个 train 文件 (无 stratified)"""
    files = list_train_files(data_dir)
    rng = random.Random(seed)
    if n >= len(files): return files
    return sorted(rng.sample(files, n))


def estimate_patient_overlap(train_files, test_files, jpg_to_patient=None):
    """从 jpg 文件名推断 patient id (LIDC 文件名规则: 数字开头 → patient id).
       如果 jpg_to_patient 映射提供, 用映射; 否则只做 nodule-level 估算."""
    if jpg_to_patient is None:
        # 简单估算: 用 jpg 文件名前 10 数字 (LIDC 约定)
        def get_pid(f):
            return f.split('.')[0][:10]  # 10 位数字是 LIDC patient id
    else:
        def get_pid(f):
            return jpg_to_patient.get(f, 'unknown')
    tr = set(get_pid(f) for f in train_files)
    te = set(get_pid(f) for f in test_files)
    if not tr or not te: return 0.0
    return len(tr & te) / len(te)


def estimate_nodule_leakage(train_files, test_files):
    """同 jpg 名 → 100% 泄漏; 不同 → 0%"""
    tr = set(train_files)
    overlap = sum(1 for f in test_files if f in tr)
    return overlap / len(test_files) if test_files else 0.0


def read_malignancy_dist(data_dir, files):
    """从 data/labels/train/<file>.txt 读 malignancy 值, 返回 dict {file: mal}"""
    lab_dir = Path(data_dir) / 'labels' / 'train'
    out = {}
    for f in files:
        p = lab_dir / f.replace('.jpg', '.txt')
        if p.exists():
            with open(p, encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    if 'Malignancy' in line:
                        try:
                            out[f] = int(line.split(':')[1].strip())
                        except:
                            pass
                        break
    return out


def run_one_size(data_dir, work_dir, n, seed, epochs=200, patience=30):
    """跑一个 sample size, 返回 paper 表 1 那一行"""
    print(f'\n========== train_size = {n} (seed={seed}) ==========')
    rng = random.Random(seed)
    all_train = list_train_files(data_dir)
    test_files = sorted([f.name for f in (Path(data_dir) / 'images' / 'test').iterdir() if f.suffix == '.jpg'])

    if n >= len(all_train):
        sampled = all_train
    else:
        sampled = sorted(rng.sample(all_train, n))
    print(f'  sampled {len(sampled)} train files', flush=True)

    # 9:1 内部分 train/val (与 train_eval.py 7161/796 同)
    val_n = max(1, int(len(sampled) * 0.1))
    rng2 = random.Random(seed + 1)
    shuffled = list(sampled)
    rng2.shuffle(shuffled)
    val_files = sorted(shuffled[:val_n])
    train_files = sorted(shuffled[val_n:])

    mal = read_malignancy_dist(data_dir, sampled)
    avg_mal = float(np.mean(list(mal.values()))) if mal else 0.0
    print(f'  train={len(train_files)}  val={len(val_files)}  test={len(test_files)}  avg_mal={avg_mal:.1f}', flush=True)

    p_overlap = estimate_patient_overlap(train_files, test_files)
    n_leak = estimate_nodule_leakage(train_files, test_files)
    print(f'  patient_overlap={p_overlap:.3f}  nodule_leakage={n_leak:.3f}', flush=True)

    # 写一个子目录, 调 train_eval.py (需要先 patch 它的 data_path, 暂时跳完整训练)
    # 简化: 这里只输出表 1 那一行, 实际训练留给 train_eval.py
    # 训练评估结果 (best vs final W-Acc on test) 由 train_eval.py 写入 test_compare.json
    # 这里只构造入口, 实际调度时再 patch train_eval.py 的 build_loaders 用 subset
    return {
        'train_size': n,
        'train_files': len(train_files),
        'val_files': len(val_files),
        'test_files': len(test_files),
        'avg_malignancy': round(avg_mal, 1),
        'patient_overlap_rate': round(p_overlap, 4),
        'nodule_leakage_rate': round(n_leak, 4),
        'seed': seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--work_dir', required=True)
    ap.add_argument('--train_sizes', type=int, nargs='+', default=[100, 200, 300, 800, 912])
    ap.add_argument('--out_csv', default='cv_results.csv')
    ap.add_argument('--out_json', default='cv_results.json')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    # 注意: 这里的 run_one_size 只生成表 1 的样本分布统计 (samples, mal, overlap, leakage)
    # 实际训练每个 sample size 需要调 train_eval.py (但需要 patch build_loaders 用子集)
    # 为简化: 这里只输出表 1 元数据; 训练评估留给单独脚本
    rows = []
    for n in args.train_sizes:
        row = run_one_size(args.data_dir, args.work_dir, n, args.seed,
                            epochs=args.epochs, patience=args.patience)
        rows.append(row)

    df = pd.DataFrame(rows)
    print('\n' + '=' * 80)
    print('表 1 重建 (5 个 sample size 的样本分布)')
    print('=' * 80)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(args.work_dir, args.out_csv), index=False, encoding='utf-8-sig')
    with open(os.path.join(args.work_dir, args.out_json), 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f'\nCSV: {os.path.join(args.work_dir, args.out_csv)}')
    print(f'JSON: {os.path.join(args.work_dir, args.out_json)}')


if __name__ == '__main__':
    main()
