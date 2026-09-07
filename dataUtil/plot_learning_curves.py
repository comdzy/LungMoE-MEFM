import os
import csv
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(result_dir):
    
    plots_dir = os.path.join(result_dir, 'plots')
    df = pd.read_csv(os.path.join(result_dir, 'metrics/training_metrics.csv'))  # 请替换为实际文件路径

    plt.rcParams.update({
        'font.size': 15})

    # ================= 保存Loss曲线 =================
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['loss'], label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize = 10)
    plt.grid(True)
    plt.savefig(f'{plots_dir}/Training_Loss_Curve.png', dpi=1000)  # 保存为高清图片
    plt.close()

    # ================= 保存四个指标曲线 =================
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for metric in metrics:
        plt.figure(figsize=(12, 10))
        
        # 筛选当前指标的所有列
        cols = [col for col in df.columns 
            if col.startswith('val_') and col.split('_')[-1].lower() == metric]
            
        # 绘制所有类别曲线
        for col in cols:
            category = col.split('_')[1]
            plt.plot(df['epoch'], df[col], label=category, alpha=0.8, linewidth=1.5)
        
        plt.title(f'Validation {metric.capitalize()} per Category')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend(fontsize = 10)
        plt.grid(True)
        plt.savefig(f"{plots_dir}/Validation_{metric.capitalize()}.png", dpi=1000)
        plt.close()  # 关闭当前图表


if __name__ == "__main__":
    result_dir = "result/experiments/exp_005"
    plot_learning_curves(result_dir)