import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import re

class LungNoduleDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, attribute_mapping=None, attribute_classes=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images', mode)
        self.label_dir = os.path.join(root_dir, 'labels', mode)
        self.samples = []
        self.attributes = ['Malignancy', 'Subtlety', 'InternalStructure', 'Calcification',
                          'Sphericity', 'Margin', 'Lobulation', 'Spiculation', 'Texture']
        
        self.attribute_mapping = attribute_mapping or {}
        self.attribute_classes = attribute_classes or {}

        if not self.attribute_mapping:
            self._collect_samples(validate=True)
            self._build_attribute_mappings()
        else:
            self._collect_samples(validate=False)

    def _collect_samples(self, validate=True):
        for img_file in os.listdir(self.image_dir):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(self.image_dir, img_file)
                txt_path = os.path.join(self.label_dir, img_file.replace('.jpg', '.txt'))
                if not os.path.exists(txt_path):
                    continue
                rois = self._parse_txt(txt_path, validate)
                for roi in rois:
                    self.samples.append((img_path, roi))

    def _parse_txt(self, txt_path, validate):
        rois = []
        with open(txt_path, 'r') as f:
            content = f.read().splitlines()
        
        current_roi = {}
        roi_pattern = re.compile(r'ROI\s+\d+:')
        attr_pattern = re.compile(r'(\w+):\s*(.*?)(?=\s+\w+:|$)', re.DOTALL)
        
        for line in content:
            line = line.strip()
            if not line:
                continue
            if roi_pattern.match(line):
                if current_roi:
                    if self._validate_roi(current_roi) or not validate:
                        rois.append(current_roi)
                current_roi = {}
            else:
                matches = attr_pattern.findall(line)
                for key, value in matches:
                    key = key.strip()
                    value = value.strip()
                    if key == 'Center':
                        coords = re.findall(r'\d+\.?\d*', value)
                        if len(coords) >= 2:
                            current_roi['Center'] = (float(coords[0]), float(coords[1]))
                    elif key in ['Width', 'Height']:
                        current_roi[key] = int(value)
                    elif key in self.attributes:
                        current_roi[key] = int(value)
        if current_roi:
            if self._validate_roi(current_roi) or not validate:
                rois.append(current_roi)
        return rois

    def _validate_roi(self, roi):
        try:
            # 计算有效区域
            cx, cy = roi['Center']
            w = roi['Width'] + 10
            h = roi['Height'] + 10
            s = max(w, h)
            
            # 转换为整数坐标
            left = cx - s/2
            upper = cy - s/2
            right = cx + s/2
            lower = cy + s/2
            
            # 检查有效性
            return (right > left) and (lower > upper)
        except:
            return False

    def _build_attribute_mappings(self):
        attr_values = {attr: set() for attr in self.attributes}
        for _, roi in self.samples:
            for attr in self.attributes:
                if attr in roi:
                    attr_values[attr].add(roi[attr])
        
        for attr in self.attributes:
            sorted_values = sorted(attr_values[attr])
            self.attribute_mapping[attr] = {v: i for i, v in enumerate(sorted_values)}
            self.attribute_classes[attr] = len(sorted_values)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, roi = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 计算正方形ROI区域
        cx, cy = roi['Center']
        w = roi['Width'] + 10
        h = roi['Height'] + 10
        s = max(w, h)
        
        # 计算边界坐标
        left = cx - s/2
        upper = cy - s/2
        right = cx + s/2
        lower = cy + s/2
        
        # 转换为整数坐标
        left, upper, right, lower = map(lambda x: int(round(x)), [left, upper, right, lower])
        
        # 计算填充边界
        pad_left = max(0, -left)
        pad_upper = max(0, -upper)
        pad_right = max(0, right - image.width)
        pad_bottom = max(0, lower - image.height)
        
        # 实际裁剪区域
        crop_coords = (
            max(0, left),
            max(0, upper),
            min(image.width, right),
            min(image.height, lower)
        )
        
        # 裁剪并填充
        roi_img = image.crop(crop_coords)
        if any([pad_left, pad_upper, pad_right, pad_bottom]):
            roi_img = ImageOps.expand(roi_img, 
                                    (pad_left, pad_upper, pad_right, pad_bottom),
                                    fill=0)
        
        # 应用变换
        if self.transform:
            roi_img = self.transform(roi_img)
        
        # 处理标签
        labels = {}
        for attr in self.attributes:
            val = roi.get(attr, 0)
            labels[attr] = torch.tensor(self.attribute_mapping[attr][val], dtype=torch.long)
        
        return roi_img, labels