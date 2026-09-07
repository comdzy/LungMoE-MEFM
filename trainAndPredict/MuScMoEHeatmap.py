import os
import sys
import re
from torchvision import transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 添加必要的路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入模型和数据加载器
from model.backbone.ResNet import MultiTaskResNet34, MultiTaskResNet50, MultiTaskResNet101, MultiTaskResNet152
from model.backbone.Densenet import MultiTaskDenseNet121, MultiTaskDenseNet161, MultiTaskDenseNet169, MultiTaskDenseNet201
from model.backbone.ViT import MultiTaskViT
from model.backbone.RTNet import MultiTaskRTNet
from model.backbone.DTNet import MultiTaskDTNet
from model.mutiLungSC.LungMoE import LungMoE
from model.mutiLungSC.LungMutiSCMoE import LungMutiSCMoE
from dataUtil.LungNoduleDataLoader import get_loaders

class TaskSpecificWrapper(torch.nn.Module):
    """针对LungMutiSCMoE的包装类，用于特定任务的热力图生成"""
    def __init__(self, model, task_name):
        super().__init__()
        self.model = model
        self.task_name = task_name
        
    def forward(self, x):
        # 运行模型并返回特定任务的融合特征
        # 移除了 torch.no_grad() 以保留梯度信息
        # 获取共享特征
        shared_features = [expert(x) for expert in self.model.shared_experts]
        
        # shared_features_merge = self.model.mefm(*shared_features)
        
        shared_features_merge = torch.mean(torch.stack(shared_features), dim=0)  # 共享特征平均
        
        # 获取任务特定特征
        task_features = [expert(x) for expert in self.model.task_expert_pools[self.task_name]]
        
        # 投影共享特征
        # shared_to_task = [
        #     self.model.conv(feat)
        #     for feat in shared_features
        # ]
        
        # 合并所有特征
        all_features = shared_features + task_features
        
        # 获取门控权重
        # gate_weights = self.model.gates[self.task_name](self.model.tmfm(shared_features_merge))
        
        gate_weights = self.model.gates[self.task_name](shared_features_merge)
        
        # 融合特征
        combined = sum(
            weight.view(-1, 1, 1, 1) * feat 
            for weight, feat in zip(gate_weights.unbind(1), all_features))
        
        # 通过分类头得到分类得分
        x = self.model.heads[self.task_name](combined)
        return x

def get_target_layer(model, task_name):
    """获取任务特定的目标层（最后一个卷积层）"""
    # 获取该任务的最后一个任务专家
    task_expert = model.task_expert_pools[task_name][-1]
    
    # 在任务专家网络中查找最后一个卷积层
    for module in reversed(list(task_expert.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError(f"No Conv2d layer found for task: {task_name}")

def read_labels(label_path):
    """
    解析新的标签文件格式：
    ROI 1:
     Center: (x, y) Width: ... Height: ...
    """
    rois = []
    with open(label_path, 'r') as f:
        current_roi = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 检测ROI行
            if line.startswith("ROI"):
                if current_roi:
                    rois.append(current_roi)
                current_roi = {"label": int(re.search(r"ROI (\d+):", line).group(1))}
            else:
                # 解析属性
                parts = re.findall(r"(\w+):\s*([\d.]+|\([\d., ]+\))", line)
                for key, value in parts:
                    if key == "Center":
                        # 提取坐标 (x, y)
                        coords = re.findall(r"[\d.]+", value)
                        current_roi["x_center"] = float(coords[0])
                        current_roi["y_center"] = float(coords[1])
                    else:
                        current_roi[key.lower()] = float(value)
        if current_roi:
            rois.append(current_roi)
    return rois

def crop_and_resize(image, roi_info, o_image_path):
    """
    根据ROI信息裁剪图像
    """
    # 获取ROI参数
    x_center = roi_info["x_center"]
    y_center = roi_info["y_center"]
    width = roi_info["width"] + 10  # 根据需求扩展10像素
    height = roi_info["height"] + 10

    # 计算裁剪坐标
    top_left = (
        int(x_center - width/2),
        int(y_center - height/2)
    )
    bottom_right = (
        int(x_center + width/2),
        int(y_center + height/2)
    )

    # 边界检查
    h, w = image.shape[:2]
    top_left = (
        max(0, top_left[0]),
        max(0, top_left[1])
    )
    bottom_right = (
        min(w, bottom_right[0]),
        min(h, bottom_right[1])
    )

    # 执行裁剪
    cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # 调整大小并保存
    resized = cv2.resize(cropped, (224, 224))
    cv2.imwrite(o_image_path, resized)
    
    return resized

def main(image_id, result_path, model):
    pth_path = os.path.join(result_path, 'checkpoints/best_model.pth')
    images_bg = os.path.join('dataset', 'images', 'test', f'{image_id}.jpg')
    labels_path = os.path.join('dataset', 'labels', 'test', f'{image_id}.txt')
    
    output_dir = os.path.join(result_path, 'heatmaps', image_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(images_bg)
    if image is None:
        raise FileNotFoundError(f"Image not found: {images_bg}")

    # 解析标签
    rois = read_labels(labels_path)
    if not rois:
        raise ValueError("No ROIs found in label file")
    
    # 只处理第一个ROI
    roi_info = rois[0]
    
    # 裁剪和调整大小
    cropped_image = crop_and_resize(image, roi_info, 
                                  os.path.join(output_dir, f'cropped_{image_id}.jpg'))
    
    # 转换图像张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(cropped_image).unsqueeze(0).to(device)
    
    # 加载模型
    checkpoint = torch.load(pth_path, map_location=device, weights_only=True)  # 使用 weights_only=True 解决安全警告
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {pth_path}")
    model.eval()
    
    # 获取所有任务
    tasks = list(model.attribute_classes.keys())
    
    # 前向传播获取预测结果
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # 为每个任务生成热力图
    for task_name in tasks:
        # 获取预测类别
        pred_class = torch.argmax(outputs[task_name]).item()
        
        # 创建任务特定的包装模型
        wrapper = TaskSpecificWrapper(model, task_name).to(device)
        wrapper.eval()
        
        # 获取目标层
        target_layer = get_target_layer(model, task_name)
        
        # 配置CAM
        cam = GradCAMPlusPlus(
            model=wrapper,
            target_layers=[target_layer]
        )
        
        # 确保输入张量需要梯度
        input_tensor_requires_grad = input_tensor.clone().requires_grad_(True)
        
        # 生成热力图（仅预测类别）
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor_requires_grad, targets=targets)[0]
        
        # 可视化处理
        visualization = show_cam_on_image(
            np.float32(cropped_image)/255.0, 
            grayscale_cam,
            use_rgb=False
        )
        
        # 保存结果
        output_path = os.path.join(output_dir, f'heatmap_{task_name}_pred{pred_class}.jpg')
        cv2.imwrite(output_path, visualization)
        print(f"Generated {task_name} heatmap for class {pred_class}")

    print("All heatmaps generated.")

def draw_heatmap(result_id, image_id, model):
    # 定义结果文件夹和CSV文件路径
    result_dir = result_id  # 结果文件夹路径
    main(image_id, result_dir, model)
    print("heatmap saved.")

# 全局设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 确保设备设置正确
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    result_id = './result/MutiExperimentsM3/exp_002'
    
    config = {
        'data_path': './dataset',
        'batch_size': 32,
        'checkpoint_path': f'{result_id}/checkpoints/best_model.pth',
        'result_dir': f'{result_id}/test_results'
    }
    
    # 初始化数据加载器
    _, test_loader, attribute_classes = get_loaders(
        root_dir=config['data_path'],
        batch_size=config['batch_size'],
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    
    # 初始化模型
    model = LungMutiSCMoE(
        attribute_classes, 
        num_shared_experts=4,
        num_task_experts=1
    ).to(device)
    
    # model = LungMoE(
    #     attribute_classes, 
    #     num_shared_experts=4,
    #     num_task_experts=2
    # ).to(device)
    
    result_list = ['2776254256']
    for image_id in result_list:
        draw_heatmap(result_id, image_id, model)