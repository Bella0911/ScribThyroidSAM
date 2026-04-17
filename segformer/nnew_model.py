from turtle import st

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from segformer.model import segformer_b2



# ================= 1. 简化后的核心模块 =================

class FusionModule(nn.Module):


    def __init__(self, in_channels=6, hidden_channels=32):
        super(FusionModule, self).__init__()

        # 1. 特征提取：简单的卷积层
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

        # 2. 权重生成：直接输出 3 个通道 (对应 Ori, Small, Large)
        self.conv_out = nn.Conv2d(hidden_channels, 3, kernel_size=1)

    def forward(self, x):
        # x: [Batch, 6, H, W] (拼接后的 logits)

        feat = self.conv_in(x)
        feat = self.relu(feat)

        # 生成融合权重 [B, 3, H, W]，并在通道维度归一化
        fusion_weights = F.softmax(self.conv_out(feat), dim=1)

        # 返回权重和 None (替代原本的 att_map，保持接口兼容)
        return fusion_weights, None


# ================= 2. 改进后的 Three Model =================

class three_model(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载 SegFormer B2
        self.model = segformer_b2(pretrained=True, num_classes=2)

        # 使用新的简化版融合模块
        # 输入是 3个尺度的 logit (2+2+2=6通道)
        self.fusion_module = FusionModule(in_channels=6, hidden_channels=32)

    def forward(self, x):
        img_size = x.size()

        # --- 1. 多尺度输入 ---
        img_small = F.interpolate(x, scale_factor=0.7, mode='bilinear', align_corners=True)
        img_large = F.interpolate(x, scale_factor=1.5, mode='bilinear', align_corners=True)

        # --- 2. 前向传播 (共享权重) ---
        logit_ori = self.model(x)

        logit_small = self.model(img_small)
        logit_small = F.interpolate(logit_small, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)

        logit_large = self.model(img_large)
        logit_large = F.interpolate(logit_large, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)

        # --- 3. 智能融合 ---
        # 拼接 Logits (通道数 2+2+2=6)
        concat_logits = torch.cat((logit_ori, logit_small, logit_large), 1)

        # 获取融合权重 (att_map 为 None)
        fusion_weights, _ = self.fusion_module(concat_logits)

        w_ori = fusion_weights[:, 0:1, :, :]
        w_small = fusion_weights[:, 1:2, :, :]
        w_large = fusion_weights[:, 2:3, :, :]

        # 加权求和，得到最终融合结果
        logit_fusion = w_ori * logit_ori + w_small * logit_small + w_large * logit_large

        # 返回 5 个值，以保持与训练循环 unpack 逻辑兼容
        # 最后一个返回值原本是 att_map，现在设为 None
        return logit_fusion
        # return logit_fusion, logit_ori, logit_small, logit_large, None


class single_model(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载标准的 SegFormer B2
        # pretrained=True 会加载 ImageNet 预训练权重，这是必须的
        self.model = segformer_b2(pretrained=True, num_classes=2)

    def forward(self, x):
        # 直接前向传播，不做任何缩放
        # 输出形状: [B, 2, H, W]
        logit = self.model(x)

        # 为了兼容你 train.py 的解包逻辑 (虽然有点丑，但改动最小)
        # 我们可以只返回一个 logit，然后在 train.py 里改接收方式
        # 或者在这里返回 5 个值 (后4个是 None)，但建议直接改 train.py 比较清爽
        return logit