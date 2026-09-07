# LungMoE-MEFM

**A Multi-Task Analysis of Multiple Signs for Pulmonary Nodule Malignancy Risk Assessment**

Code companion to the manuscript under review at *Medical Image Analysis* (submission `MEDIA-D-25-02844R1`).

---

## Overview

LungMoE-MEFM is a multi-task multi-expert framework for joint prediction of nine LIDC nodule characteristics (Malignancy + 8 morphological signs), driven by a wavelet-convolutional front end and a Mixture-of-Experts (MoE) gating network. The headline contribution is the **MEFM** (Multi-Expert Fusion Module), which combines shared and task-specific experts with a dual-wavelet (`db1` + `bior1.1`) branch.

This repository contains the full implementation, training pipeline, and the five supplementary experiments introduced in the R2 revision.

---

## Repository Layout

```
code/
├── dataProcess/        # LIDC XML → jpg+txt 预处理流水线
│   ├── xml_txt.py                # 从 LIDC XML 切出候选结节
│   ├── jpg_txt.py                # 配对 jpg + 标签 txt
│   ├── remove_malignancy_zero.py # 清洗未标注样本
│   ├── compared.py
│   └── del_name.py
│
├── dataUtil/           # 数据集 / DataLoader / 训练工具
│   ├── LungNoduleDataset.py      # 核心 Dataset（含 jpg 预读缓存）
│   ├── LungNoduleDataLoader.py
│   ├── datashffule.py            # 训练 / 验证 / 测试划分
│   ├── label_distribution.py     # 9 任务标签分布
│   ├── correlation_analysis.py
│   ├── loss_weight.py            # 多任务加权策略
│   └── plot_learning_curves.py
│
├── model/
│   ├── attn/                     # MEFM / TMFM / WTConv2d
│   │   ├── MEFM.py               #  ★ 多专家融合模块（核心）
│   │   ├── TMFM.py
│   │   └── WTConv2d.py           #  小波卷积
│   ├── backbone/                 # 7 个候选 backbone
│   │   ├── ResNet.py
│   │   ├── Densenet.py
│   │   ├── Convnext.py
│   │   ├── HRNet.py
│   │   ├── DTNet.py
│   │   ├── RTNet.py
│   │   └── ViT.py
│   └── mutiLungSC/
│       ├── LungMutiSCMoE.py      #  ★ 多任务多专家主模型
│       └── LungMoE.py
│
├── trainAndPredict/    # 主训练 / 评估 / 可视化
│   ├── trainMuScMoE.py           #  ★ 主训练入口
│   ├── train.py / train2.py      #  历史训练脚本
│   ├── test.py / testMutiGPU.py
│   ├── gate_analysis.py
│   ├── heatmap.py
│   ├── MuScMoEHeatmap.py
│   └── shaptest.py
│
└── supplementary/      # R2 审稿补充实验（新增）
    ├── 1_overfit_train_eval.py
    ├── 2_single_wavelet_train_eval_db1.py
    ├── 3_5sample_cv_runner.py
    ├── 4_reader_dist_diag.py
    └── 5_fleiss_kappa.py
```

---

## Environment

- Python 3.9+
- PyTorch ≥ 2.0 (CUDA 11.8 / 12.1)
- `pywavelets<1.8` (锁版本以保证 `WTConv2d` 的小波分解行为一致)
- `pydicom`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`

Quick install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm matplotlib pydicom 'pywavelets<1.8'
```

A pinned `requirements.txt` is provided in the GitHub-upload snapshot.

---

## Data

The LIDC-IDRI cohort is **not** redistributed in this repository due to licence terms. To reproduce:

1. Obtain LIDC-IDRI from the TCIA portal (or your institutional mirror).
2. Run `dataProcess/xml_txt.py` → `dataProcess/jpg_txt.py` → `dataProcess/remove_malignancy_zero.py` to produce a `data/<train|val|test>/<jpg|txt>/` tree.
3. Adjust the path constants in `dataUtil/LungNoduleDataset.py` if your layout differs.

For a quick sanity check, the supplementary scripts accept any folder of paired `*.jpg` / `*.txt` files with the 9-value label format.

---

## Training

```bash
# from the repo root (the `code/` directory)
python trainAndPredict/trainMuScMoE.py
```

Key config knobs (top of `trainMuScMoE.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `num_shared_experts` | 4 | MoE shared experts |
| `num_task_experts`  | 9 | MoE task-specific experts |
| `wavelet_variants`  | `['db1', 'bior1.1']` | Dual-wavelet front end |
| `batch_size`        | 32 | per-GPU |
| `epochs`            | 64 | with cosine LR |
| `lr`                | 1e-4 | AdamW |

---

## R2 Supplementary Experiments

All five scripts live under `supplementary/` and target specific reviewer concerns:

| # | File | Reviewer Question Addressed | Inputs Required |
| --- | --- | --- | --- |
| 1 | `1_overfit_train_eval.py` | "Convergence of training loss alone does not establish absence of overfitting" (Reviewer #4) — runs best vs final checkpoint on a fixed test split and reports ΔW-Acc / ΔAUC / ΔAP per task. | Trained checkpoints from the full 4-shared-9-task dual-wavelet model. |
| 2 | `2_single_wavelet_train_eval_db1.py` | "Why dual-wavelet?" — ablates to a single `db1` wavelet, all else identical, to isolate the contribution of the dual branch. | None beyond data. |
| 3 | `3_5sample_cv_runner.py` | "Include standard deviations or p-values based on multiple random splits" (Reviewer #2) — five stratified sub-samples (100 / 200 / 300 / 800 / 912) with repeated random splits. | None beyond data. |
| 4 | `4_reader_dist_diag.py` | "Reader distribution is unclear" — counts how many of the four LIDC radiologists annotated each nodule (4 / 3 / 2 / 1). | LIDC-IDRI raw XML tree. |
| 5 | `5_fleiss_kappa.py` | "Inter-rater agreement not reported" — Fleiss κ across the four radiologists on the nine characteristics. | LIDC-IDRI raw XML tree. |

Run any one of them with:

```bash
python supplementary/<file>.py
```

Each script is self-contained; absolute paths to data/weight directories live at the top of each file for easy editing.

---

## Reproducibility Notes

- Splits are deterministic and derived from `dataUtil/datashffule.py` with a fixed seed.
- The cloud runs that produced the reported numbers are reproducible from the same seed on the same PyTorch / CUDA versions; numerical drift of < 0.5% per metric is normal across hardware.
- The full 4-expert-9-task training run takes ~14 h on a single RTX 4090 (24 GB). The two supplementary ablations each take 4–6 h.

---

## Citation

If you use this code, please cite the corresponding *Medical Image Analysis* paper (full citation will be added upon publication).

---

## License

Research-only use. LIDC-IDRI is governed by the TCIA Data Use Agreement; end users are responsible for compliance.
