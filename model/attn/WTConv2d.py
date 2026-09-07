import pywt
import pywt.data
import torch
from torch import nn
import torch.nn.functional as F

from functools import partial

def create_wavelet_filter(wave, in_size, out_size, device, dtype=torch.float32):
    w = pywt.Wavelet(wave)
    
    # 直接在目标设备上创建张量
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype, device=device)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype, device=device)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0).to(device)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype, device=device).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype, device=device).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0).to(device)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    # 强制滤波器与输入同一设备
    filters = filters.to(x.device)
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    # 强制滤波器与输入同一设备
    filters = filters.to(x.device)
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type, kernel_size=5, 
                 stride=1, bias=True, wt_levels=1, device=None):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels
        
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        
        # 确定设备（假设后续会调用 model.to(device)）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 生成滤波器时传入设备
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, self.device, torch.float)

        # 直接注册为 Buffer（不需要梯度，自动同步设备）
        self.register_buffer("wt_filter", wt_filter)
        self.register_buffer("iwt_filter", iwt_filter)
        
        # 使用 lambda 绑定动态设备（确保每次调用时更新）
        self.wt_function = lambda x: wavelet_transform(x, self.wt_filter)
        self.iwt_function = lambda x: inverse_wavelet_transform(x, self.iwt_filter)
        
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

# 示例：在 DepthwiseSeparableConvWithWTConv2d 中初始化 WTConv2d
class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type, kernel_size=3, device=None):
        super().__init__()
        
        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(
            in_channels, in_channels, wt_type, 
            kernel_size=kernel_size, 
            device=device  # 传递设备信息
        )
        
        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, device=device)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

if __name__ == "__main__":
    # 单 GPU 测试
    model = WTConv2d(3, 3, "haar").cuda(0)
    x = torch.randn(1, 3, 32, 32).cuda(0)
    print(model.wt_filter.device)  # 输出: cuda:0
    out = model(x)  # 正常执行

    # 多 GPU 测试
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    x_multi = torch.randn(64, 3, 32, 32).cuda(0)
    out_multi = model(x_multi)  # 无设备错误