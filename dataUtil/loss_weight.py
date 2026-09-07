import json
import numpy as np
from sklearn.metrics import mutual_info_score

# 加载数据
with open("result/data/correlation_data.json") as f:
    data = json.load(f)

# 计算每个任务与Malignancy的互信息
task_weights = {}
malignancy_key = "Malignancy"  # 主任务名称

for task, task_data in data.items():
    if task == malignancy_key:
        continue
    
    # 构建联合计数矩阵（添加拉普拉斯平滑）
    joint_matrix = []
    max_malign_grade = 5  # 根据数据确定Malignancy最大等级
    
    # 按任务等级排序（1,2,3,4,5）
    sorted_task_grades = sorted(task_data.keys(), key=lambda x: int(x))
    
    for task_grade in sorted(task_data.keys(), key=int):
        row = []
        for m_grade in range(1, max_malign_grade+1):
            # 添加平滑处理 (加1避免零值)
            count = task_data[task_grade].get(str(m_grade), 0) + 1
            row.append(count)
        joint_matrix.append(row)
    
    # 转换为NumPy数组
    contingency = np.array(joint_matrix)
    
    # 计算互信息（直接传入联合计数矩阵）
    try:
        mi = mutual_info_score(None, None, contingency=contingency)
    except ValueError as e:
        print(f"任务 {task} 计算失败: {e}")
        mi = 0
    
    # 计算归一化互信息（NMI）
    total = contingency.sum()
    pi = contingency.sum(axis=1) / total
    pj = contingency.sum(axis=0) / total
    
    h_i = -np.sum(pi * np.log(pi + 1e-10))  # 避免log(0)
    h_j = -np.sum(pj * np.log(pj + 1e-10))
    
    nmi = mi / np.sqrt(h_i * h_j) if (h_i * h_j) > 1e-10 else 0
    task_weights[task] = nmi

# 归一化权重
total_nmi = sum(task_weights.values())
loss_weights = {task: (nmi/total_nmi)*2.0 for task, nmi in task_weights.items()}

print("多任务损失权重字典：")
print(loss_weights)