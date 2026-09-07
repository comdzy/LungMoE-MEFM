import timm
import torch
from timm.models import hrnet
from timm.models import convnext

model1 = hrnet.hrnet_w18(
                in_chans=3,
                num_classes=0  # 禁用默认分类头
                ).cuda()
model2 = convnext.convnext_nano(
                in_chans=3,
                num_classes=0  # 禁用默认分类头
                ).cuda()
dummy = torch.randn(1,3,224,224).cuda()
output = model1.forward_features(dummy)
print(output.shape)
output = model2.forward_features(dummy)
print(output.shape)
