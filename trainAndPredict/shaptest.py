    # 加载训练好的模型
    checkpoint = torch.load(config['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 使用SHAP分析任务相关性
    shap_results, shap_values, sample_inputs = analyze_with_shap(
        model, 
        test_loader, 
        main_task="Malignancy",  # 替换为您的主任务名称
        device=main_device
    )
    
    print("\nSHAP分析 - 次任务对主任务的相关性贡献:")
    for i, (task, score) in enumerate(shap_results, 1):
        print(f"{i}. {task}: {score:.6f}")
    
    # 可视化结果
    # 1. 条形图显示任务贡献
    tasks = [t for t, _ in shap_results]
    scores = [s for _, s in shap_results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(tasks, scores, color='purple')
    plt.xlabel('次任务')
    plt.ylabel('平均|SHAP值|')
    plt.title('次任务对主任务的相关性贡献')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('shap_task_correlation.png')
    
    # 2. 单个样本的SHAP解释
    sample_idx = 0
    shap.image_plot(
        [shap_values[sample_idx]], 
        sample_inputs[sample_idx:sample_idx+1],
        show=False
    )
    plt.title(f"样本 {sample_idx} 的SHAP解释")
    plt.savefig('shap_sample_explanation.png')
    
    # 3. 全局特征重要性
    shap.summary_plot(
        shap_values, 
        sample_inputs, 
        feature_names=[f"特征_{i}" for i in range(sample_inputs.shape[1])],
        show=False,
        plot_type="bar"
    )
    plt.title("全局特征重要性")
    plt.savefig('shap_global_importance.png')