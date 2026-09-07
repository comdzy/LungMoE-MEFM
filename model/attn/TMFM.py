import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from attn.WTConv2d import DepthwiseSeparableConvWithWTConv2d

"""
全局中值池化函数：对输入的特征图在空间维度上取每个通道的中值
输入:
    x - 维度为(batch_size, channels, height, width)的四维Tensor
输出:
    经过全局中值池化后的Tensor，维度为(batch_size, channels, 1, 1)
"""
def global_median_pooling(x):  # 对输入特征图进行全局中值池化操作。

    # 将 x 重新塑形为 (batch_size, channels, height*width)，并在最后一个维度上求中值
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    # 将结果 reshape 回 (batch_size, channels, 1, 1) 的形状，方便后续计算
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled


"""
通道注意力模块：生成每个通道的权重，用于调整通道的重要性
初始化参数：
    input_channels - 输入张量的通道数
    internal_neurons - 用于减少计算量的瓶颈神经元数量
"""
class ChannelAttention(nn.Module):  

    def __init__(self, input_channels, internal_neurons):
        # 调用父类的构造函数
        super(ChannelAttention, self).__init__()
        
        # 创建一个自适应平均池化层，将输入张量的空间维度缩小为 (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 创建一个自适应最大池化层，也将输入张量缩小为 (1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 第一个1x1卷积层用于降维，输出通道数是internal_neurons
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, 1, bias=True)
        # 第二个1x1卷积层用于恢复维度，将通道数变回input_channels
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, 1, bias=True)
        
        # ReLU激活层
        self.relu = nn.ReLU(inplace=False)
        self.input_channels = input_channels  # 保存输入的通道数

    """
    计算权重并将其应用于输入特征
    参数:
        poolchange_x - 池化后的张量，形状为 (batch_size, channels, 1, 1)
        x - 原始输入张量，形状为 (batch_size, channels, height, width)
    返回:
        返回加权后的输入张量
    """
    def expand(self, poolchange_x, x):
        y = self.fc1(poolchange_x)  # 用第一个1x1卷积层对池化结果降维
        y = self.relu(y)  # 应用ReLU激活
        y = self.fc2(y)  # 用第二个1x1卷积层恢复通道维度
        return x * y.expand_as(x)  # 对原始输入x按通道进行加权调整

    """
    通道注意力模块的前向传播过程
    参数:
        x - 输入张量，形状为 (batch_size, channels, height, width)
    返回:
        输出张量，加权后各通道的特征
    """
    def forward(self, x):
        # 获取输入张量x的尺寸信息
        B, C, _, _ = x.size()
        avg_pool = self.avg_pool(x)  # 平均池化，生成一个 (batch_size, channels, 1, 1) 张量
        max_pool = self.max_pool(x)  # 最大池化，生成同样形状的张量
        median_pool = global_median_pooling(x)  # 中值池化，生成同样形状的张量

        # 对三个池化结果分别计算通道注意力，并加权输入
        avg_out = self.expand(avg_pool, x)
        max_out = self.expand(max_pool, x)
        median_out = self.expand(median_pool, x)

        # 将平均池化、最大池化和中值池化的结果相加，得到最终的输出
        out = avg_out + max_out + median_out
        
        return out

"""
空间注意力模块：用于在空间维度上为输入特征图分配权重，突出重要区域
初始化参数：
    in_channels - 输入张量的通道数
"""
class SpatialAttention(nn.Module):
    
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        
        # 使用1x1卷积用于产生空间激励
        self.Conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        
        # Sigmoid激活函数，用于对输出进行归一化，确保注意力值在[0, 1]之间
        self.sigmoid = nn.Sigmoid()

    """
    空间注意力模块的前向传播过程
    参数:
        x - 输入张量，形状为(batch_size, channels, height, width)
    返回:
        加权后的输出张量，强调空间上重要的区域
    """
    def forward(self, x):
        # 获取输入张量x的尺寸信息
        B, _, H, W = x.size()
        # 进行不同的池化操作
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 平均池化，形状为(B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化，形状为(B, 1, H, W)
        median_pool, _ = torch.median(x, dim=1, keepdim=True)  # 中值池化，形状为 (B, 1, H, W)

        # 将三种池化结果拼接在一起 (B, 3, H, W)
        pool_combined = torch.cat([avg_pool, max_pool, median_pool], dim=1)  # 拼接通道维度
        
        # 使用1x1卷积产生空间激励
        out = self.Conv(pool_combined)  # 输出 (B, 1, H, W)
        
        # 应用Sigmoid激活函数
        out = self.sigmoid(out)  # 输出空间注意力图 (B, 1, H, W)
        
        # 将注意力图扩展到与输入张量相同的形状，并与输入进行逐元素相乘
        out = x * out.expand_as(x)  # 将注意力权重应用到输入张量上

        return out  # 返回加权后的输入特征

"""
像素注意力模块：用于生成像素级别的注意力图，从而在空间维度上为每个像素分配权重。
初始化参数：
    in_channels - 输入张量的通道数
"""
class PixelAttention(nn.Module):

    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()

        # 定义一个卷积层，该卷积层的输入通道数为2 * in_channels，输出通道数为in_channels。
        # 卷积核大小为7，使用reflect填充模式，并且使用了组卷积（groups=in_channels）来实现每个通道的独立卷积操作
        self.fc = nn.Conv2d(2 * in_channels, in_channels, 7, padding=3, 
                            padding_mode='reflect', groups=in_channels, bias=True)

        # 使用Sigmoid激活函数，将卷积结果映射到[0, 1]之间
        self.sigmoid = nn.Sigmoid()

    """
    像素注意力模块的前向传播过程
    参数:
        x - 输入张量，形状为(batch_size, channels, height, width)
        pattn - 生成的第一张注意力图，形状为(batch_size, channels, height, width)
    返回:
        out - 加权后的注意力图，形状与x相同
    """
    def forward(self, x, pattn):
        
        # 获取输入张量x的尺寸信息
        B, C, H, W = x.size()

        # 对输入x和pattn进行扩展，添加一个维度，使其形状变为(batch_size, channels, 1, height, width)
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn = pattn.unsqueeze(dim=2)  # B, C, 1, H, W

        # 在第三个维度（即通道维度）上将x和pattn1拼接在一起，形成一个新的张量
        # 拼接后的形状为(batch_size, channels, 2, height, width)，即在通道维度上拼接
        x2 = torch.cat([x, pattn], dim=2)  # B, C, 2, H, W

        # 使用Rearrange操作重排维度，将张量从(batch_size, channels, 2, height, width)重排为
        # (batch_size, channels * 2, height, width)，相当于将通道维度展开为两个通道的拼接
        # Rearrange 是一个用于改变张量维度的操作，具体作用是将张量的维度按照指定顺序排列
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)

        # 对重排后的张量进行卷积操作，得到注意力图out，卷积操作使用了groups=in_channels
        # 这意味着每个通道的卷积操作是独立的，卷积结果将是每个像素的注意力权重
        out = self.fc(x2)

        # 对卷积结果应用Sigmoid函数，将输出值压缩到[0, 1]之间，得到最终的注意力图
        out = self.sigmoid(out)

        # 返回计算后的注意力图out
        return out

    
"""
特征融合模块：用于将空间注意力、通道注意力和像素注意力提取的特征进行融合。
初始化参数：
    in_channels - 输入张量的通道数
"""
class Merge(nn.Module):    
    
    def __init__(self, in_channels):
        # 调用父类构造函数
        super().__init__()

        # 创建通道注意力子模块，in_channels表示输入通道数，in_channels // 4为通道注意力的瓶颈神经元数
        self.channel = ChannelAttention(in_channels, in_channels // 4)

        # 创建空间注意力子模块，in_channels表示输入通道数
        self.space = SpatialAttention(in_channels)

        # 创建像素注意力子模块，in_channels表示输入通道数
        self.pixel = PixelAttention(in_channels)

        # 使用Sigmoid激活函数，将结果值压缩到[0, 1]之间
        self.sigmoid = nn.Sigmoid()

        # 定义一个1x1卷积层，用于最终的特征图生成
        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=True)

    """
    特征合并模块的前向传播过程
    参数:
        x - 输入张量，形状为(batch_size, channels, height, width)
    返回:
        out - 经过注意力加权合并后的输出张量，形状与x和y相同
    """
    def forward(self, x):

        # 使用通道注意力模块对初始特征图进行加权调整
        cattn = self.channel(x)  # B, C, H, W

        # 使用空间注意力模块对初始特征图进行加权调整
        sattn = self.space(x)  # B, C, H, W

        # 将通道注意力和空间注意力的结果相加，得到初步的注意力图
        pattn1 = sattn + cattn  # B, C, H, W

        # 使用像素注意力模块对特征图进行加权，得到最终的注意力图
        pattn2 = self.sigmoid(self.pixel(x, pattn1))  # B, C, H, W

        # 将注意力图应用到输入x上，得到加权后的特征图
        out = x + pattn2 * pattn1  # B, C, H, W

        # 使用1x1卷积层对加权后的特征图进行卷积操作，进一步融合特征
        out = self.conv(out)  # B, C, H, W

        # 返回最终的合并结果
        return out

"""
多尺度多感受野多注意力融合模块 （Multi-Scale Multi-Receptive-Field Multi-Attention Fusion Module）TMFM：
该模块通过多个分支使用不同的空洞卷积、小波卷积和其他操作提取多尺度特征，
然后将其融合，通过通道、空间和像素注意力进一步增强特征。

初始化参数：
    in_channels - 输入张量的通道数
    out_channels - 输出张量的通道数
    rate - 空洞卷积的基准空洞率，控制感受野的扩展
    bn_mom - 批归一化的动量，用于控制均值和方差的更新速度
"""
class TMFM(nn.Module):  
    
    def __init__(self, in_channels, out_channels, rate=1, bn_mom=0.1):
        """
        初始化TMFM模块，包含多个分支进行不同规模的空洞卷积操作，并整合通道和空间特征。
        
        参数：
            in_channels - 输入特征图的通道数
            out_channels - 输出特征图的通道数
            rate - 空洞卷积的基准空洞率
            bn_mom - 批归一化动量
        """
        super(TMFM, self).__init__()

        # 第一分支：1x1卷积，保持通道维度不变
        # 该分支不使用空洞卷积，仅用于提取局部特征
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        # 第二分支：3x3卷积，空洞率为3
        # 该分支通过空洞卷积扩展感受野，能够提取较大范围的特征
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        # 第三分支：3x3卷积，空洞率为5
        # 进一步扩大感受野，有助于捕捉更大范围的上下文信息
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=5 * rate, dilation=5 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        # 第四分支：3x3卷积，空洞率为7
        # 该分支使用更大的空洞率来最大化感受野，捕捉更多的上下文信息
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=7 * rate, dilation=7 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        # 第五分支：小波卷积 (Depthwise Separable Conv with WTConv2d)
        # 使用小波卷积进一步扩展感受野，并减少计算量
        self.branch5 = nn.Sequential(
            DepthwiseSeparableConvWithWTConv2d(in_channels, out_channels, wt_type='db1'),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        # 第六分支：全局特征提取，使用全局平均池化后通过1x1卷积处理
        # 该分支提取全局特征，用于捕捉全局上下文信息
        self.branch6_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.branch6_bn = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.branch6_relu = nn.ReLU(inplace=False)

        # 合并所有分支的输出，并通过1x1卷积降维到out_channels通道数
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 6, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        # 合并空间、通道和像素特征的模块，用于增强特征表达
        self.merge = Merge(in_channels=out_channels * 6)


    """
    前向传播过程，将输入x通过多个分支进行处理并合并特征。

    参数：
        x - 输入张量，形状为(batch_size, channels, height, width)
    
    返回：
        out - 经过多个分支处理和合并的输出张量，形状为(batch_size, out_channels, height, width)
    """
    def forward(self, x):

        # 获取输入张量的形状
        B, C, H, W = x.size()

        # 应用每个分支的卷积操作，提取不同尺度的特征
        conv1x1 = self.branch1(x)  # 1x1卷积分支
        conv3x3_1 = self.branch2(x)  # 3x3卷积，空洞率为3
        conv3x3_2 = self.branch3(x)  # 3x3卷积，空洞率为5
        conv3x3_3 = self.branch4(x)  # 3x3卷积，空洞率为7
        wtconv = self.branch5(x)  # 小波卷积分支

        # 全局特征提取：通过全局平均池化得到全局特征
        global_feature = torch.mean(x, 2, True)  # 全局平均池化，去掉height维度
        global_feature = torch.mean(global_feature, 3, True)  # 去掉width维度
        global_feature = self.branch6_conv(global_feature)  # 通过1x1卷积处理全局特征
        global_feature = self.branch6_bn(global_feature)  # 批归一化
        global_feature = self.branch6_relu(global_feature)  # ReLU激活

        # 将全局特征的尺寸调整为与输入特征相同 (H, W)
        global_feature = F.interpolate(global_feature, (H, W), mode='bilinear', align_corners=True)

        # 将所有分支的输出特征进行拼接
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, wtconv, global_feature], dim=1)

        # 使用合并模块进一步增强特征，通过通道、空间和像素级别的注意力增强特征
        merge = self.merge(feature_cat)
        merge_feature_cat = merge * feature_cat

        # 使用1x1卷积对最终的特征进行降维，生成最终的输出
        out = self.conv_cat(merge_feature_cat)

        return out

if __name__ == '__main__':
    # 随机生成输入数据，假设输入的形状为 (batch_size=2, channels=512, height=7, width=7)
    input = torch.randn(2, 512, 7, 7).cuda()  # 将随机生成的数据放置在GPU上
    # 实例化TMFM模块，指定输入和输出的通道数，并将模型放置在GPU上
    model = TMFM(in_channels=512, out_channels=512).cuda()
    # 将输入数据通过模型，得到输出
    output = model(input)
    # 输出经过模型处理后的结果张量的形状，用于确认模型的输出维度是否正确
    print(output.shape)  # 输出形状应为 (batch_size=2, channels=512, height=7, width=7)

    # 使用torchsummary的summary函数来打印模型的结构信息
    # 传入模型和输入张量的形状 (512, 64, 64) 作为summary函数的参数
    # 这将输出每层的参数数量、输出形状等详细信息，方便查看模型结构
    # summary(model, (512, 64, 64))
