import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from timm.models import convnext
from timm.models import hrnet
from torchvision import transforms

from torch.nn.parallel import DistributedDataParallel as DDP

from dataUtil.LungNoduleDataLoader import get_loaders

import torch
import torch.nn as nn
from timm.models import convnext, hrnet


import gc

class LungMoE(nn.Module):
    def __init__(self, 
                 attribute_classes,
                 num_shared_experts=4,
                 num_task_experts=1,
                 base_dim=512):  # 统一所有专家输出到512通道
        super().__init__()
        self.attribute_classes = attribute_classes
        self.tasks = list(attribute_classes.keys())
        self.base_dim = base_dim
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts

        # ================= 网络架构配置 =================
        self.backbone_config = {
            # 共享专家网络
            "hrnet_w18": {
                "builder": hrnet.hrnet_w18,
                "out_channels": 2048,
                "adapter": nn.Conv2d(2048, self.base_dim, 1)
            },
            "convnext_pico": {
                "builder": lambda **kwargs: convnext.convnext_pico(**kwargs),
                "out_channels": 512,
                "adapter": nn.Conv2d(512, self.base_dim, 1)
            },
            # 任务专家网络
            "convnext_femto": {
                "builder": convnext.convnext_femto,
                "out_channels": 384,
                "adapter": nn.Conv2d(384, self.base_dim, 1)
            },
            "convnext_atto": {
                "builder": lambda **kwargs: convnext.convnext_atto(**kwargs),
                "out_channels": 320,
                "adapter": nn.Conv2d(320, self.base_dim, 1)
            }
        }

        # ================= 共享专家 =================
        self.shared_experts = nn.ModuleList([
            self._build_shared_expert(i)
            for i in range(self.num_shared_experts)
        ])

        # ================= 任务专家池 =================
        self.task_expert_pools = nn.ModuleDict({
            task: nn.ModuleList([
                self._build_task_expert("convnext_femto" if task == "Malignancy" else "convnext_atto")
                for _ in range(self.num_task_experts)
            ]) for task in self.tasks
        })

        # ================= 动态门控 =================
        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.base_dim, 128),
                nn.GELU(),
                nn.Linear(128, self.num_shared_experts + self.num_task_experts),
                nn.Softmax(dim=1)
            ) for task in self.tasks
        })

        # ================= 分类头 =================
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.base_dim, num_classes)
            ) for task, num_classes in self.attribute_classes.items()
        })

    def _build_shared_expert(self, expert_id):
        """构建交替的共享专家"""
        backbone_type = "hrnet_w18" if expert_id % 2 == 0 else "convnext_pico"
        cfg = self.backbone_config[backbone_type]
        
        # HRNet特殊处理
        if backbone_type == "hrnet_w18":
            class HRNetExpertWrapper(nn.Module):
                def __init__(self, hrnet_model):
                    super().__init__()
                    self.model = hrnet_model
                
                def forward(self, x):
                    features = self.model.forward_features(x)
                    return features  # 取最后一层 [B, 2048, 7, 7]
            
            backbone = HRNetExpertWrapper(cfg["builder"](
                pretrained=False,
                num_classes=0
            ))
        else:
            class ConvNeXtExpertWrapper(nn.Module):
                def __init__(self, convnext_model):
                    super().__init__()
                    self.model = convnext_model
                
                def forward(self, x):
                    features = self.model.forward_features(x)
                    return features  # 取最后一层 [B, X, 7, 7]
            
            backbone = ConvNeXtExpertWrapper(cfg["builder"](
                pretrained=False,
                num_classes=0
            ))
        
        return nn.Sequential(
            backbone,
            cfg["adapter"],  # 维度转换到base_dim
            nn.BatchNorm2d(self.base_dim),
            nn.GELU()
        )

    def _build_task_expert(self, backbone_type):
        """构建任务专家"""
        cfg = self.backbone_config[backbone_type]
        
        class ConvNeXtTaskWrapper(nn.Module):
            def __init__(self, convnext_model):
                super().__init__()
                self.model = convnext_model
                
            def forward(self, x):
                features = self.model.forward_features(x)
                return features  # 取最后一层 [B, X, 7, 7]
            
        backbone = ConvNeXtTaskWrapper(cfg["builder"](
                pretrained=False,
                num_classes=0
            ))
        
        return nn.Sequential(
            backbone,
            cfg["adapter"],  # 维度转换到base_dim
            nn.BatchNorm2d(self.base_dim),
            nn.GELU()
        )

    def forward(self, x, return_gates=False):
        # 共享特征提取 [B, 512, 7, 7]
        shared_features = [expert(x) for expert in self.shared_experts]
        shared_avg = torch.mean(torch.stack(shared_features), dim=0)  # 共享特征平均
        
        outputs = {}
        gate_dict = {}
        
        for task in self.tasks:
            # 任务特征提取 [B, 512, 7, 7]
            task_features = [expert(x) for expert in self.task_expert_pools[task]]
            
            # 合并所有特征
            all_features = shared_features + task_features
            
            # 门控权重计算
            gate_weights = self.gates[task](shared_avg)  # [B, total_experts]
            gate_dict[task] = gate_weights  # 存储门控权重
            
            # 空间特征融合
            combined = sum(
                weight.view(-1, 1, 1, 1) * feat 
                for weight, feat in zip(gate_weights.unbind(1), all_features)
            )
            
            # 分类预测
            outputs[task] = self.heads[task](combined)
            
        if return_gates:  # 新增返回选项
            return outputs, gate_dict
        return outputs

if __name__ == "__main__":
    # 初始化主设备
    device_ids = [3]
    main_device = torch.device(f"cuda:{device_ids[0]}")
    
    config = {
        'data_path': './dataset',
        'batch_size': 32,
        'checkpoint_path': './result/experiments/exp_001/checkpoints/best_model.pth',
        'result_dir': './result/experiments/exp_001/test_results'
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
    
    # 模型初始化并移至主设备
    model = LungMoE(attribute_classes, num_shared_experts=8,
                 num_task_experts=2).to(main_device)
    
    # DataParallel 封装
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    
    # 输入数据在主设备
    dummy = torch.randn(1, 3, 224, 224).to(main_device)
    
    # # 维度验证（通过 model.module 访问原始模型）
    # print("共享专家输出：")
    # for i, expert in enumerate(model.module.shared_experts):
    #     out = expert(dummy)
    #     print(f"专家{i+1}: {out.shape} (设备: {out.device})")
    
    # print("\n任务专家输出：")
    # for task in model.module.tasks:  # 通过 model.module 访问原始模型
    #     # 获取第一个任务专家
    #     expert = model.module.task_expert_pools[task][0]
        
    #     # 前向传播
    #     out = expert(dummy)
    #     print(f"{task}专家：", out.shape)
    
    # print("\n分类头输出：")
    # outputs = model(dummy)
    # for task, out in outputs.items():
    #     print(f"{task}: {out.shape} (设备: {out.device})")
        
    gc.collect()

    torch.cuda.empty_cache()

    for gpu_id in device_ids:
        print(f"GPU{gpu_id} 当前可用显存：{torch.cuda.mem_get_info(gpu_id)[0]/1024**2:.4f} MB")
        print(f"GPU{gpu_id} 总显存：{torch.cuda.get_device_properties(gpu_id).total_memory/1024**2:.4f} MB")
        
    # 前向传播测试
    print("\n前向传播输出：")
    outputs = model(dummy)
    for task, out in outputs.items():
        print(f"{task}输出：{out.shape} (设备: {out.device})")  # 应与attribute_classes对应
        
    print(model)
