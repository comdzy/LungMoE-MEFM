import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

# 添加必要的路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入模型和数据加载器
from model.mutiLungSC.LungMoE import LungMoE
from model.mutiLungSC.LungMutiSCMoE import LungMutiSCMoE

from dataUtil.LungNoduleDataLoader import get_loaders


# 临床意义解释映射
MEDICAL_INSIGHTS = {
    "Malignancy": "结节恶性程度",
    "Subtlety": "病灶明显度与恶性程度正相关，明显病灶更可能为恶性",
    "InternalStructure": "内部结构特征(如脂肪密度)与良性相关",
    "Calcification": "钙化模式是良恶性鉴别的重要指标",
    "Sphericity": "球形度越高良性概率越大，不规则形状提示恶性可能",
    "Margin": "边缘毛刺与恶性高度相关，光滑边缘多为良性",
    "Lobulation": "分叶征是恶性结节的重要特征",
    "Spiculation": "毛刺征是肺癌的典型表现",
    "Texture": "实性成分与恶性概率相关，混合磨玻璃结节风险较高"
}

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def save_analysis_data(config, all_gates, correlations, main_gates, test_loader):
    """保存分析数据到CSV文件以便后续调试"""
    data_dir = Path(config['result_dir']) / 'analysis_data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"保存分析数据到: {data_dir}")
    
    # 1. 保存门控权重 (每个任务单独CSV)
    gates_dir = data_dir / 'gate_weights'
    gates_dir.mkdir(exist_ok=True)
    
    for task, weights in all_gates.items():
        # 转换为DataFrame
        df = pd.DataFrame(weights.numpy())
        df.columns = [f'Expert_{i}' for i in range(weights.shape[1])]
        
        # 保存为CSV
        filename = f'gate_weights_{task.replace(" ", "_")}.csv'
        df.to_csv(gates_dir / filename, index=False)
    
    # 2. 保存相关性结果
    corr_data = []
    for task, metrics in correlations.items():
        row = {'Task': task}
        row.update(metrics)
        corr_data.append(row)
    
    corr_df = pd.DataFrame(corr_data)
    corr_df.to_csv(data_dir / 'correlations.csv', index=False)
    
    # 3. 保存主任务门控权重
    main_gates_df = pd.DataFrame(main_gates.numpy())
    main_gates_df.columns = [f'Expert_{i}' for i in range(main_gates.shape[1])]
    main_gates_df.to_csv(data_dir / 'main_gates.csv', index=False)
    
    # 4. 保存数据集信息
    dataset_info = {
        'num_samples': [len(test_loader.dataset)],
        'main_task': [config['main_task']],
        'analysis_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    # 添加次任务信息
    for i, task in enumerate(config['secondary_tasks']):
        dataset_info[f'secondary_task_{i+1}'] = [task]
    
    pd.DataFrame(dataset_info).to_csv(data_dir / 'dataset_info.csv', index=False)
    
    print("分析数据保存完成!")

def load_analysis_data(config):
    """从CSV文件加载保存的分析数据"""
    data_dir = Path(config['result_dir']) / 'analysis_data'
    
    if not data_dir.exists():
        raise FileNotFoundError(f"分析数据目录不存在: {data_dir}")
    
    print(f"从 {data_dir} 加载分析数据...")
    
    # 1. 加载门控权重
    gates_dir = data_dir / 'gate_weights'
    all_gates = {}
    
    for csv_file in gates_dir.glob('gate_weights_*.csv'):
        # 从文件名提取任务名
        task_name = csv_file.stem.replace('gate_weights_', '').replace('_', ' ')
        
        # 读取CSV
        df = pd.read_csv(csv_file)
        weights = torch.tensor(df.values, dtype=torch.float32)
        all_gates[task_name] = weights
    
    # 2. 加载相关性结果
    corr_df = pd.read_csv(data_dir / 'correlations.csv')
    correlations = {}
    
    for _, row in corr_df.iterrows():
        task = row['Task']
        metrics = {
            'pearson': row['pearson'],
            'agreement': row['agreement'],
            'weight_similarity': row['weight_similarity']
        }
        correlations[task] = metrics
    
    # 3. 加载主任务门控权重
    main_gates_df = pd.read_csv(data_dir / 'main_gates.csv')
    main_gates = torch.tensor(main_gates_df.values, dtype=torch.float32)
    
    # 4. 加载数据集信息
    dataset_info_df = pd.read_csv(data_dir / 'dataset_info.csv')
    dataset_info = {
        'num_samples': dataset_info_df['num_samples'].iloc[0],
        'main_task': dataset_info_df['main_task'].iloc[0],
        'secondary_tasks': []
    }
    
    # 提取次任务
    for col in dataset_info_df.columns:
        if col.startswith('secondary_task_') and not pd.isna(dataset_info_df[col].iloc[0]):
            dataset_info['secondary_tasks'].append(dataset_info_df[col].iloc[0])
    
    print("分析数据加载完成!")
    return all_gates, correlations, main_gates, dataset_info

def load_model(config, model_config, attribute_classes):
    """加载训练好的模型"""

    model = model_config["class"](
        attribute_classes, 
        num_shared_experts = model_config["shared"],
        num_task_experts = model_config["task"]
    ).to(config['device'])
    
    # 安全加载模型
    print(f"Loading checkpoint from: {config['checkpoint']}")
    
    # 加载权重
    checkpoint = torch.load(config['checkpoint'], map_location=config['device'], weights_only=True)

    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, attribute_classes

def collect_gate_weights(model, dataloader, config, model_config):
    """收集所有样本的门控权重"""
    model.eval()
    device = config['device']
    
    # 初始化存储结构
    all_tasks = [config['main_task']] + config['secondary_tasks']
    all_gates = {task: [] for task in all_tasks}
    
    # 计算总批次数
    total_batches = len(dataloader)
    
    # 创建进度条
    progress_bar = tqdm(
        total=total_batches,
        desc="Collecting gate weights",
        bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'
    )
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            _, gate_dict = model(inputs, return_gates=True)
            
            for task, weights in gate_dict.items():
                # 提取共享专家权重 (前shared个)
                shared_weights = weights[:, :model_config['shared']]
                all_gates[task].append(shared_weights.cpu())
            
            # 更新进度条
            progress_bar.set_postfix({
                "batch": f"{batch_idx+1}/{total_batches}",
                "mem": f"{torch.cuda.mem_get_info(device)[0]/(1024**2):.1f}MB"
            })
            progress_bar.update(1)
    
    progress_bar.close()
    
    # 合并批次数据
    for task in all_gates:
        all_gates[task] = torch.cat(all_gates[task], dim=0)
    
    return all_gates

def analyze_correlation(all_gates, config):
    """分析次任务与主任务的门控权重相关性"""
    main_task = config['main_task']
    main_gates = all_gates[main_task]
    
    # 初始化进度条
    tasks = config['secondary_tasks']
    progress_bar = tqdm(
        tasks,
        desc="Analyzing correlations",
        bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'
    )
    
    # 计算相关系数
    correlations = {}
    for task in progress_bar:
        task_gates = all_gates[task]
        
        # 计算Pearson相关系数
        cov = torch.mean(
            (main_gates - main_gates.mean(dim=0)) *
            (task_gates - task_gates.mean(dim=0))
        )
        std_product = torch.std(main_gates) * torch.std(task_gates)
        pearson_corr = (cov / std_product).item()
        
        # 计算专家选择一致性
        agreement = (main_gates.argmax(dim=1) == task_gates.argmax(dim=1)).float().mean().item()
        
        # 计算共享专家权重分布相似度
        weight_similarity = torch.cosine_similarity(
            main_gates.mean(dim=0),
            task_gates.mean(dim=0),
            dim=0
        ).item()
        
        correlations[task] = {
            'pearson': pearson_corr,
            'agreement': agreement,
            'weight_similarity': weight_similarity
        }
        
        # 更新进度条描述
        progress_bar.set_postfix({
            "task": task[:10] + ".." if len(task) > 10 else task,
            "corr": f"{pearson_corr:.3f}"
        })
    
    progress_bar.close()
    return correlations, main_gates

def visualize_results(correlations, main_gates, config):
    """可视化相关性结果"""
    # 创建结果目录
    os.makedirs(config['result_dir'], exist_ok=True)
    save_path = config['result_dir']
    
    # 创建进度条
    progress_bar = tqdm(
        total=4,
        desc="Visualizing results",
        bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'
    )
    
    # 1. 相关系数热力图 - 添加标签旋转
    plt.figure(figsize=(15, 10))
    tasks = config['secondary_tasks']
    corr_matrix = np.zeros((len(tasks), len(tasks)))
    
    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                corr_matrix[i, j] = correlations[task1]['pearson']
            else:
                corr_matrix[i, j] = correlations[task2]['pearson']
    
    plt.subplot(2, 2, 1)
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap="coolwarm", 
                     xticklabels=tasks, yticklabels=tasks,
                     annot_kws={"size": 10})
    
    # 旋转标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title('Task Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    progress_bar.update(1)
    
    # 2. 主任务相关性条形图 - 添加数值标签
    plt.subplot(2, 2, 2)
    pearson_values = [correlations[t]['pearson'] for t in tasks]
    bars = plt.bar(tasks, pearson_values, color='#1f77b4')
    plt.axhline(0, color='k', linestyle='--')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=9)
    
    plt.title(f'Correlation with Main Task ({config["main_task"]})', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=12)
    plt.ylim(-1.1, 1.1)  # 扩展范围以容纳标签
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    progress_bar.update(1)
    
    # 3. 主任务门控权重分布 - 添加数值标签
    plt.subplot(2, 2, 3)
    expert_weights = main_gates.mean(dim=0).numpy()
    bars = plt.bar(range(len(expert_weights)), expert_weights, color='#9467bd')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=9)
    
    plt.title(f'{config["main_task"]} Gate Weight Distribution', fontsize=14)
    plt.xlabel('Shared Expert Index', fontsize=12)
    plt.ylabel('Average Weight', fontsize=12)
    plt.xticks(range(len(expert_weights)), fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    progress_bar.update(1)
    
    # 4. 指标对比图 - 添加数值标签
    plt.subplot(2, 2, 4)
    x = np.arange(len(tasks))
    width = 0.25
    
    # 三组指标
    corr_bars = plt.bar(x - width, [correlations[t]['pearson'] for t in tasks], 
                        width, label='Correlation', color='#2ca02c')
    agree_bars = plt.bar(x, [correlations[t]['agreement'] for t in tasks], 
                         width, label='Agreement', color='#ff7f0e')
    sim_bars = plt.bar(x + width, [correlations[t]['weight_similarity'] for t in tasks], 
                       width, label='Weight Similarity', color='#d62728')
    
    # 添加数值标签
    for bars in [corr_bars, agree_bars, sim_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=8)
    
    plt.axhline(0, color='k', linewidth=0.8)
    plt.title('Correlation Metrics Comparison', fontsize=14)
    plt.xticks(x, tasks, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10,loc='center left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(os.path.join(save_path, 'gate_analysis.png'), dpi=1000, bbox_inches='tight')
    progress_bar.update(1)
    
    # 单独保存主任务专家权重分布 - 添加数值标签
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(expert_weights)), expert_weights, color='#9467bd')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=10)
    
    plt.title(f'{config["main_task"]} Expert Contribution', fontsize=16)
    plt.xlabel('Expert Index', fontsize=14)
    plt.ylabel('Average Gate Weight', fontsize=14)
    plt.xticks(range(len(expert_weights)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'expert_contribution.png'), dpi=1000)
    
    plt.close('all')
    progress_bar.close()

def save_text_report(correlations, config, test_len):
    """保存文本分析报告"""
    report_path = os.path.join(config['result_dir'], 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Lung Nodule MoE Gate Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Main Task: {config['main_task']}\n")
        f.write(f"Secondary Tasks: {', '.join(config['secondary_tasks'])}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {config['device']}\n")
        f.write(f"Dataset Size: {test_len} samples\n")
        f.write("\n" + "="*60 + "\n\n")
        
        # 任务相关性分析
        f.write("Task Correlation Analysis:\n")
        f.write("-"*60 + "\n")
        for task in config['secondary_tasks']:
            metrics = correlations[task]
            insight = MEDICAL_INSIGHTS.get(task, "No clinical insight available")
            
            f.write(f"Task: {task}\n")
            f.write(f"  Pearson Correlation: {metrics['pearson']:.4f}\n")
            f.write(f"  Expert Agreement: {metrics['agreement']:.4f}\n")
            f.write(f"  Weight Similarity: {metrics['weight_similarity']:.4f}\n")
            f.write(f"  Clinical Insight: {insight}\n")
            f.write("-"*60 + "\n")
        
        # 临床意义总结
        f.write("\nClinical Significance Summary:\n")
        f.write("-"*60 + "\n")
        for task, insight in MEDICAL_INSIGHTS.items():
            if task != config['main_task']:
                f.write(f"{task}: {insight}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Analysis complete. Results saved to:\n")
        f.write(f"  - {os.path.join(config['result_dir'], 'gate_analysis.png')}\n")
        f.write(f"  - {os.path.join(config['result_dir'], 'expert_contribution.png')}\n")
        f.write(f"  - {report_path}\n")
        f.write("="*60 + "\n")

def run_analysis(config, model_config, skip_to_visualization=False):
    """运行完整的分析流程"""
    print("="*60)
    print(f"Starting Gate Weight Analysis for Lung Nodule Classification")
    print(f"Main Task: {config['main_task']}")
    print(f"Secondary Tasks: {', '.join(config['secondary_tasks'])}")
    print(f"Device: {config['device']}")
    print("="*60)
    
    # 0. 全局计时
    start_time = datetime.now()
    
    if skip_to_visualization:
        # 直接加载之前保存的数据
        print("\n[1/5] 加载缓存的分析数据...")
        all_gates, correlations, main_gates, dataset_info = load_analysis_data(config)
        
        # 更新配置中的任务信息（以防保存后配置有变化）
        config['main_task'] = dataset_info['main_task']
        config['secondary_tasks'] = dataset_info['secondary_tasks']
        
        _, test_loader, attribute_classes = get_loaders(
            root_dir=config['data_path'],
            batch_size=config['batch_size'],
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        )
        
        test_len = len(test_loader.dataset)
        print(f"Loaded test dataset with {test_len} samples")

        print("[5/5] Visualizing and saving results...")
        visualize_results(correlations, main_gates, config)
        save_text_report(correlations, config, test_len)
        
        # 计算总耗时
        time_elapsed = datetime.now() - start_time
        print("\n" + "="*60)
        print("Analysis complete!")
        print(f"Total time: {time_elapsed}")
        print(f"Results saved to: {config['result_dir']}")
        print("="*60)
    else:
    
        # 1. 加载数据
        print("\n[1/5] Loading dataset...")
        _, test_loader, attribute_classes = get_loaders(
            root_dir=config['data_path'],
            batch_size=config['batch_size'],
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        )
        
        test_len = len(test_loader.dataset)
        print(f"Loaded test dataset with {test_len} samples")
        
        # 2. 加载模型
        print("[2/5] Loading model...")
        model, attribute_classes = load_model(config, model_config, attribute_classes)
        print(f"Model loaded with {model_config['class']} shared experts {model_config['shared']} and task experts {model_config['task']}")
        
        # 3. 收集门控权重
        print("[3/5] Collecting gate weights...")
        all_gates = collect_gate_weights(model, test_loader, config, model_config)
        print(f"Collected gate weights for {len(test_loader.dataset)} samples")
        
        # 4. 分析相关性
        print("[4/5] Analyzing correlations...")
        correlations, main_gates = analyze_correlation(all_gates, config)
        
        # 保存分析数据以便后续调试
        save_analysis_data(config, all_gates, correlations, main_gates, test_loader)
        
        # 5. 可视化和保存结果
        print("[5/5] Visualizing and saving results...")
        visualize_results(correlations, main_gates, config)
        save_text_report(correlations, config, test_len)
        
        # 计算总耗时
        time_elapsed = datetime.now() - start_time
        print("\n" + "="*60)
        print("Analysis complete!")
        print(f"Total time: {time_elapsed}")
        print(f"Results saved to: {config['result_dir']}")
        print("="*60)

if __name__ == "__main__":
    # ================= 配置字典 =================
    CONFIG = {
        'data_path': './dataset',  # 数据集路径
        'batch_size': 32,          # 批量大小
        'main_task': 'Malignancy',  # 主任务名称
        'secondary_tasks': ['Subtlety', 'InternalStructure', 'Calcification', 
                        'Sphericity', 'Margin', 'Lobulation', 'Spiculation', 'Texture'],  # 次任务列表

        'device': 'cuda:0',
        
        'checkpoint': './result/MutiExperimentsM3/exp_001/checkpoints/best_model.pth',  # 模型检查点
        'result_dir': './result/MutiExperimentsM3/exp_001/analysis_results',  # 结果保存目录
    }
    
        # 初始化模型
    MODEL_CONFIGS = [
        # LungMutiSCMoE 配置
        {"class": LungMutiSCMoE, "shared": 4, "task": 2},
        # {"class": LungMutiSCMoE, "shared": 4, "task": 1},
        # {"class": LungMutiSCMoE, "shared": 2, "task": 2},
        # # LungMoE 配置
        # {"class": LungMoE, "shared": 4, "task": 2},
        # {"class": LungMoE, "shared": 4, "task": 1},
        # {"class": LungMoE, "shared": 2, "task": 2}
        
    ]

    for model_config in MODEL_CONFIGS:

        # 直接使用CONFIG字典运行分析
        run_analysis(CONFIG, model_config, True)