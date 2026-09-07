import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def analyze_correlation():
    with open('./result/data/correlation_data.json') as f:
        data = json.load(f)
    
    results = []
    
    for attr, attr_values in data.items():
        values = sorted(map(int, attr_values.keys()))
        malignancies = sorted({int(m) for v in attr_values.values() for m in v.keys()})
        
        matrix = np.zeros((len(values), len(malignancies)))
        for vi, val in enumerate(values):
            for mi, mal in enumerate(malignancies):
                matrix[vi, mi] = attr_values[str(val)].get(str(mal), 0)
        
        chi2, p, dof, expected = chi2_contingency(matrix)
        n = np.sum(matrix)
        cramers_v = np.sqrt(chi2 / (n * (min(matrix.shape) - 1)))
        
        results.append({
            'Attribute': attr,
            'Chi-Squared': chi2,
            'P-Value': p,
            "Cramer's V": cramers_v,
            'Matrix': matrix.tolist()
        })
    
    with open('./result/data/correlation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame(results)[['Attribute', "Cramer's V", 'P-Value']]
    df = df.sort_values("Cramer's V", ascending=False)
    
    plt.subplot(2, 1, 1)
    sns.barplot(x="Cramer's V", y='Attribute', data=df,
               hue='Attribute', palette='viridis', legend=False)
    plt.title("Cramer's V Correlation with Malignancy")
    
    plt.subplot(2, 1, 2)
    pvalues = df.set_index('Attribute')['P-Value']
    sns.heatmap(pvalues.to_frame(), annot=True, cmap='coolwarm', 
                cbar_kws={'label': 'P-Value'}, fmt=".2e")
    plt.title("P-Values of Chi-Squared Tests")
    
    plt.tight_layout()
    plt.savefig('./result/data/correlation_analysis.png', dpi=1000)
    plt.show()

def plot_detailed_distribution(attr_name):
    with open('./result/data/correlation_data.json') as f:
        data = json.load(f)
    
    attr_data = data[attr_name]
    
    # 增强数据处理
    df = pd.DataFrame(attr_data).T.fillna(0)
    
    # 确保列名和索引为数值类型
    try:
        df.columns = df.columns.astype(int)
        df.index = df.index.astype(int)
    except:
        df.columns = pd.to_numeric(df.columns, errors='coerce')
        df.index = pd.to_numeric(df.index, errors='coerce')
    
    # 过滤无效值
    df = df.loc[:, ~df.columns.isna()]
    df = df.loc[~df.index.isna(), :]
    
    # 排序处理
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.reindex(sorted(df.index), axis=0)
    
    # 类型转换
    df = df.astype(float).astype(int)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu",
                cbar_kws={'label': 'Count'})
    plt.title(f"{attr_name} vs Malignancy Distribution")
    plt.xlabel('Malignancy')
    plt.ylabel(attr_name)
    plt.savefig(f'./result/data/{attr_name}_distribution.png', dpi=1000)
    plt.show()

if __name__ == "__main__":
    analyze_correlation()
    scList = ['Subtlety','InternalStructure','Calcification',
             'Sphericity','Margin','Lobulation','Spiculation','Texture']
    for sc in scList:
        try:
            plot_detailed_distribution(sc)
        except Exception as e:
            print(f"处理{sc}时出错: {str(e)}")
            continue