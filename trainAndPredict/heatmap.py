import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re
import torch
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.backbone.ResNet import MultiTaskResNet34
from model.backbone.ResNet import MultiTaskResNet50
from model.backbone.ResNet import MultiTaskResNet101
from model.backbone.ResNet import MultiTaskResNet152

from model.backbone.Densenet import MultiTaskDenseNet121
from model.backbone.Densenet import MultiTaskDenseNet161
from model.backbone.Densenet import MultiTaskDenseNet169
from model.backbone.Densenet import MultiTaskDenseNet201

from model.backbone.ViT import MultiTaskViT
from model.backbone.RTNet import MultiTaskRTNet
from model.backbone.DTNet import MultiTaskDTNet

from model.backbone.Densenet import MultiTaskDenseNet264

from model.backbone.Convnext import MultiTaskConvNext_T
from model.backbone.Convnext import MultiTaskConvNext_S
from model.backbone.Convnext import MultiTaskConvNext_B
from model.backbone.Convnext import MultiTaskConvNext_L
from model.backbone.Convnext import MultiTaskConvNext_xL

from model.backbone.HRNet import MultiTaskHRNet_w18
from model.backbone.HRNet import MultiTaskHRNet_w18_v2
from model.backbone.HRNet import MultiTaskHRNet_w32
from model.backbone.HRNet import MultiTaskHRNet_w48

from dataUtil.LungNoduleDataLoader import get_loaders

class MultiTaskWrapper(torch.nn.Module):
    """多任务模型包装类，用于指定特定任务"""
    def __init__(self, model, task_name):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.head = model.heads[task_name]
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return self.head(features)

def vit_reshape_transform(tensor):
    """ViT专用特征重组函数"""
    # 去掉cls token：[batch, num_tokens, dim]
    tokens = tensor[:, 1:, :]
    
    # 计算空间维度
    batch_size, num_tokens, dim = tokens.shape
    height = width = int(num_tokens ** 0.5)  # 假设是正方形排列
    
    # 重塑为空间格式并调整维度顺序
    result = tokens.reshape(batch_size, height, width, dim)
    result = result.transpose(2, 3).transpose(1, 2)  # [batch, dim, height, width]
    return result
    
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
    pth_path = os.path.join(result_path, 'checkpoints/best_model.pth')  # 修改模型路径
    images_bg = os.path.join('dataset', 'images', 'train', f'{image_id}.jpg')
    labels_path = os.path.join('dataset', 'labels', 'train', f'{image_id}.txt')
    
    # 创建输出目录
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
    checkpoint = torch.load(pth_path, map_location=device)
    # def _load_checkpoint(self, path):
    #     checkpoint = torch.load(path, map_location=self.device, weights_only=True)
    #     self.model.module.load_state_dict(checkpoint)
    #     print(f"Loaded checkpoint from {path}")
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {pth_path}")
    model.eval()
    
    # 定义各任务参数
    # resnet34
    # tasks = {
    #     'Malignancy': model.feature_extractor[7][-1].conv2,
    #     'Subtlety': model.feature_extractor[7][-1].conv2,
    #     'InternalStructure': model.feature_extractor[7][-1].conv2,
    #     'Calcification': model.feature_extractor[7][-1].conv2,
    #     'Sphericity': model.feature_extractor[7][-1].conv2,
    #     'Margin': model.feature_extractor[7][-1].conv2,
    #     'Lobulation': model.feature_extractor[7][-1].conv2,
    #     'Spiculation': model.feature_extractor[7][-1].conv2,
    #     'Texture': model.feature_extractor[7][-1].conv2
    # }
        # RTNet
    # tasks = {
    #     'Malignancy': model.feature_extractor[8][-1].conv2,
    #     'Subtlety': model.feature_extractor[8][-1].conv2,
    #     'InternalStructure': model.feature_extractor[8][-1].conv2,
    #     'Calcification': model.feature_extractor[8][-1].conv2,
    #     'Sphericity': model.feature_extractor[8][-1].conv2,
    #     'Margin': model.feature_extractor[8][-1].conv2,
    #     'Lobulation': model.feature_extractor[8][-1].conv2,
    #     'Spiculation': model.feature_extractor[8][-1].conv2,
    #     'Texture': model.feature_extractor[8][-1].conv2
    # }
    # resnet50/101/152
    # tasks = {
    #     'Malignancy': model.feature_extractor[7][-1].conv3,
    #     'Subtlety': model.feature_extractor[7][-1].conv3,
    #     'InternalStructure': model.feature_extractor[7][-1].conv3,
    #     'Calcification': model.feature_extractor[7][-1].conv3,
    #     'Sphericity': model.feature_extractor[7][-1].conv3,
    #     'Margin': model.feature_extractor[7][-1].conv3,
    #     'Lobulation': model.feature_extractor[7][-1].conv3,
    #     'Spiculation': model.feature_extractor[7][-1].conv3,
    #     'Texture': model.feature_extractor[7][-1].conv3
    # }
    # densenet121
    # tasks = {
    #     'Malignancy': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'Subtlety': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'InternalStructure': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'Calcification': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'Sphericity': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'Margin': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'Lobulation': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'Spiculation': model.feature_extractor[0].denseblock4.denselayer16.conv2,
    #     'Texture': model.feature_extractor[0].denseblock4.denselayer16.conv2
    # }
    # DTNet
    # tasks = {
    #     'Malignancy': model.feature_extractor[2][0].denselayer16.conv2,
    #     'Subtlety': model.feature_extractor[2][0].denselayer16.conv2,
    #     'InternalStructure': model.feature_extractor[2][0].denselayer16.conv2,
    #     'Calcification': model.feature_extractor[2][0].denselayer16.conv2,
    #     'Sphericity': model.feature_extractor[2][0].denselayer16.conv2,
    #     'Margin': model.feature_extractor[2][0].denselayer16.conv2,
    #     'Lobulation': model.feature_extractor[2][0].denselayer16.conv2,
    #     'Spiculation': model.feature_extractor[2][0].denselayer16.conv2,
    #     'Texture': model.feature_extractor[2][0].denselayer16.conv2
    # }
    # densenet161
    # tasks = {
    #     'Malignancy': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'Subtlety': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'InternalStructure': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'Calcification': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'Sphericity': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'Margin': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'Lobulation': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'Spiculation': model.feature_extractor[0].denseblock4.denselayer24.conv2,
    #     'Texture': model.feature_extractor[0].denseblock4.denselayer24.conv2
    # }
    # densenet169/201
    # tasks = {
    #     'Malignancy': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'Subtlety': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'InternalStructure': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'Calcification': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'Sphericity': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'Margin': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'Lobulation': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'Spiculation': model.feature_extractor[0].denseblock4.denselayer32.conv2,
    #     'Texture': model.feature_extractor[0].denseblock4.denselayer32.conv2
    # }
    # ViT
    # tasks = {
    #     # 高级语义任务使用深层特征
    #     'Malignancy': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'Subtlety': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'InternalStructure': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'Calcification': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'Sphericity': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'Margin': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'Lobulation': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'Spiculation': model.feature_extractor.encoder.layers[-1].ln_1,
    #     'Texture': model.feature_extractor.encoder.layers[-1].ln_1,
    # }
    
    # densenet264
    tasks = {
        'Malignancy': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'Subtlety': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'InternalStructure': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'Calcification': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'Sphericity': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'Margin': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'Lobulation': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'Spiculation': model.feature_extractor[0].denseblock4.denselayer32.conv2,
        'Texture': model.feature_extractor[0].denseblock4.denselayer32.conv2
    }
    
    # convnext
    
    # HRNet
    

    # 前向传播获取预测结果
    with torch.no_grad():
        features = model.feature_extractor(input_tensor)
        flattened = torch.flatten(features, 1)
        outputs = {task: model.heads[task](flattened) for task in tasks}

    # 为每个任务生成单张热力图
    for task_name, target_layer in tasks.items():
    
        # 获取预测类别
        pred_class = torch.argmax(outputs[task_name]).item()
        
        # 配置CAM（自动处理ViT的特殊需求）
        cam = GradCAMPlusPlus(
            model=MultiTaskWrapper(model, task_name).to(device),
            target_layers=[target_layer],
            # reshape_transform=vit_reshape_transform  # ViT自动启用变换
        )

        # 生成热力图（仅预测类别）
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor, targets=targets)[0]

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有 GPU 可用

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    result_id = './result/MutiExperimentsM3/exp_001'
    
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
        
    # model = MultiTaskResNet34(attribute_classes).to(device)
    # model = MultiTaskResNet50(attribute_classes).to(device)
    # model = MultiTaskResNet101(attribute_classes).to(device)
    # model = MultiTaskResNet152(attribute_classes).to(device)
    # model = MultiTaskDenseNet121(attribute_classes).to(device)
    # model = MultiTaskDenseNet161(attribute_classes).to(device)
    # model = MultiTaskDenseNet169(attribute_classes).to(device)
    # model = MultiTaskDenseNet201(attribute_classes).to(device)
    # model = MultiTaskViT(attribute_classes).to(device)
    # model = MultiTaskRTNet(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    model = MultiTaskDenseNet264(attribute_classes).to(device)
    
    
    result_list = ['1017004304']
    for image_id in result_list:
        draw_heatmap(result_id, image_id, model)
