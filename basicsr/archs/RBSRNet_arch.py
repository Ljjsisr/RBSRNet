import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


#  Enhanced Hierarchical Feature Fusion Block（增强式分层特征融合块）
class EHFFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EHFFB, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_d2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3 // 2)

        self.conv1x1_1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)

        self.eca = eca_layer(channel=in_channels)

    def forward(self, x):
        d1 = F.relu(self.conv_d1(x))
        x1 = F.relu(self.conv_1(d1))
        x2 = F.relu(self.conv_2(x1 + x))
        c1 = torch.cat((x1, x2), dim=1)

        x11 = self.conv1x1_1(c1)
        d2 = F.relu(self.conv_d2(x11))
        c2 = torch.cat((d1, d2), dim=1)
        x12 = self.conv1x1_2(c2)

        x3 = F.relu(self.conv_3(x12))
        c3 = torch.cat((x1, x2, x3), dim=1)
        x13 = self.conv1x1_3(c3)

        x13 = self.eca(x13)
        out = x13 + x

        return out


#  Hierarchical Fusion Network（分层式融合网络）
class HFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFN, self).__init__()
        self.ehb1 = EHFFB(in_channels=in_channels, out_channels=out_channels)
        self.ehb2 = EHFFB(in_channels=in_channels, out_channels=out_channels)
        self.ehb3 = EHFFB(in_channels=in_channels, out_channels=out_channels)
        self.ehb4 = EHFFB(in_channels=in_channels, out_channels=out_channels)
        self.ehb5 = EHFFB(in_channels=in_channels, out_channels=out_channels)
        self.ehb6 = EHFFB(in_channels=in_channels, out_channels=out_channels)
        self.ehb7 = EHFFB(in_channels=in_channels, out_channels=out_channels)
        self.conv1_sac = nn.Conv2d(in_channels * 6, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.ehb1(x)
        x2 = self.ehb2(x1)
        x3 = self.ehb3(x2)
        x4 = self.ehb4(x3)
        x5 = self.ehb5(x4)
        x6 = self.ehb6(x5)
        y = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        y = self.conv1_sac(y)
        out = self.ehb7(y)

        return out


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        channel = 64
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


#  Lateral Inhibition Module（侧抑制模块）
class LIM(nn.Module):
    def __init__(self, in_channels):
        super(LIM, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 使用卷积进行侧抑制
        suppressed_output = self.conv(x)
        output = x * torch.sigmoid(suppressed_output)  # 模拟侧抑制操作

        return output


@ARCH_REGISTRY.register()
class RBSRNet(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_feat,
                 num_out_ch,
                 num_block,
                 upscale=2,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(RBSRNet, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # 浅层特征提取层
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=3 // 2)  # 3 -> 64

        # 侧抑制模块
        self.lim = LIM(num_feat)

        # 深层特征提取层
        self.body = make_layer(HFN, num_block, in_channels=num_feat, out_channels=num_feat)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=3 // 2)  # 64 -> 64

        # 重建层
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, padding=3 // 2)  # 64 -> 3

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # 1、浅层特征提取层
        x1 = self.conv_first(x)  # 3 -> 64

        # 侧抑制模块
        hc = self.lim(x1)

        # 3、深层特征提取层
        m1 = self.body(hc)
        y2 = self.conv_after_body(m1) + x1

        # 重建层
        y3 = self.upsample(y2)
        out = self.conv_last(y3)

        out = out / self.img_range + self.mean
        return out
