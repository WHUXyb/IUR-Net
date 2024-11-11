import sys
sys.path.append('..')  # 将父级目录添加到导入搜索路径中 改  board and google
from models.HRNet import conv3x3, BasicBlock
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
BN_MOMENTUM = 0.01


class CSAMBlock(nn.Module):
    def __init__(self, out_feat=48, stage_channels=[48, 96, 192, 384], attention='TripletAttention'):
    # def __init__(self, out_feat=48, stage_channels=[192, 192, 192, 192], attention='TripletAttention'):      
        super(CSAMBlock, self).__init__()
        self.check_layer = nn.Sequential(
            conv3x3(1, out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            conv3x3(out_feat, out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        # print(stage_channels)
        self.stages = nn.ModuleList([MSCSAM(ic, out_feat, attention=attention) for ic in stage_channels])

        self.dfl = nn.Sequential(
            conv3x3(out_feat * len(stage_channels), out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

    def forward(self, y_list, check_map):
        c = self.check_layer(check_map)
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        list_feature = []
        for i, f in enumerate(y_list):
            x = self.stages[i](c, f)
            list_feature.append(F.upsample(x, size=(x0_h, x0_w), mode='bilinear', align_corners=True))
        x = self.dfl(torch.cat(list_feature, dim=1))
        return x


class MSCSAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention='TripletAttention'):
        super().__init__()
        self.print = (in_channels, out_channels)
        self.feature_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.res_fuse_layer = nn.Sequential(
            conv3x3(out_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        # if attention == 'scam':
        #     self.attention = CSAM(inchannels=out_channels)
        # 确保在所有情况下都初始化 self.attention
        if attention == 'TripletAttention':
            self.attention = TripletAttention()
            print("TripletAttention")
        else:
            # 如果不是 'TripletAttention'，则可以选择初始化为 None 或其他适当的默认值
            self.attention = None  # 或其他默认的注意力机制
            print("None TripletAttention")
        

    def forward(self, c, f):
        _, _, m, n = f.shape
        f = self.feature_layer(f)
        x = self.res_fuse_layer(f - F.interpolate(c, size=(m, n), mode='bilinear'))  #差分***********重点
        # x = self.attention(x)
        # 在使用 self.attention 之前，检查它是否已经初始化
        if self.attention is not None:
            x = self.attention(x)
        return x


class CSAM(nn.Module):
    def __init__(self, inchannels, kernel_size=7):
        super().__init__()
        self.CA = ChannelAttention(in_planes=inchannels)
        self.SA = SpatialAttention(kernel_size=kernel_size)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels, inchannels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        ca = self.CA(x) * x
        sa = self.SA(x) * x
        out = x + ca + sa
        out = self.conv1x1(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_planes = max(64, in_planes//2)
        self.fc1   = nn.Conv2d(in_planes, mid_planes, 1, bias=False)
        self.relu1 = nn.functional.leaky_relu
        self.fc2   = nn.Conv2d(mid_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    # 假设stage_channels对应于您从HRNet中得到的多尺度特征的通道数
    stage_channels = [48, 96, 192, 384]  # 根据实际情况调整这些值
    csam_block = CSAMBlock(out_feat=48, stage_channels=stage_channels, attention='TripletAttention').cuda()
    
    # 模拟y_list（多尺度特征图）
    y_list = [torch.randn(2, c, 128 // (2**i), 128 // (2**i)).cuda() for i, c in enumerate(stage_channels)]

    check_map = torch.randn(2, 1, 512, 512).cuda()

    with torch.no_grad():
        csam_output = csam_block(y_list, check_map)
        print("CSAMBlock output size:", csam_output.size())
    # 还可以打印出y_list中每个元素的尺寸来检查输入尺寸是否正确
    for i, feature in enumerate(y_list):
        print(f"Input feature map {i} size:", feature.size())
        # print(y_list)