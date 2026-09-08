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
└── supplementary/      # 补充实验（新增）
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



## Citation

If you use this code, please cite the corresponding *Medical Image Analysis* paper (full citation will be added upon publication).

---

## License

Research-only use. LIDC-IDRI is governed by the TCIA Data Use Agreement; end users are responsible for compliance.
