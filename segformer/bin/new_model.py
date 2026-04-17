import torch
from torch import nn
import torch.nn.functional as F
from segformer.model import segformer_b2


# ================= 1. 从 SPANET 移植的核心模块 =================

class irnn_layer(nn.Module):
    """
    捕捉四个方向（上下左右）的上下文信息，替代普通的空洞卷积
    """

    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        # 创建四个副本用于不同方向的传递
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()

        # 使用切片操作模拟 RNN 的递归传递 (比循环快)
        # Left -> Right
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        # Right -> Left
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        # Up -> Down
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        # Down -> Up
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)

        return top_up, top_right, top_down, top_left


class DirectionalAttention(nn.Module):
    """
    生成四个方向的注意力权重
    """

    def __init__(self, in_channels):
        super(DirectionalAttention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=2, dilation=2)
        self.relu2 = nn.ReLU()
        # 输出4个通道，对应4个方向的权重
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmoid(out)  # 归一化到 0~1
        return out


class FusionModule(nn.Module):
    """
    融合模块：结合 SAM 的思想，生成多尺度融合权重
    """

    def __init__(self, in_channels=6, hidden_channels=32):
        super(FusionModule, self).__init__()

        # 1. 特征降维/预处理
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

        # 2. 方向特征提取 (IRNN)
        self.irnn = irnn_layer(hidden_channels)

        # 3. 方向注意力计算
        self.attention_layer = DirectionalAttention(hidden_channels)

        # 4. 最终融合权重生成 (输出3个通道，分别对应 Ori, Small, Large)
        # 输入维度是 hidden_channels * 4 (因为 IRNN 输出4个方向拼接)
        self.conv_out = nn.Conv2d(hidden_channels * 4, 3, kernel_size=1)

        # 5. 辅助输出：用于计算 L_att 的单通道注意力图
        self.conv_att_map = nn.Conv2d(hidden_channels * 4, 1, kernel_size=1)

    def forward(self, x):
        # x: [Batch, 6, H, W] (拼接后的 logits)

        feat = self.conv_in(x)
        feat = self.relu(feat)

        # 计算4个方向的注意力权重 [B, 4, H, W]
        dir_weights = self.attention_layer(feat)

        # 计算4个方向的特征
        top_up, top_right, top_down, top_left = self.irnn(feat)

        # 核心：Attention Residual Block 的变体
        # 用注意力权重加权对应方向的特征
        top_up = top_up * dir_weights[:, 0:1, :, :]
        top_right = top_right * dir_weights[:, 1:2, :, :]
        top_down = top_down * dir_weights[:, 2:3, :, :]
        top_left = top_left * dir_weights[:, 3:4, :, :]

        # 拼接所有方向的特征
        concat_feat = torch.cat([top_up, top_right, top_down, top_left], dim=1)

        # 生成融合权重 [B, 3, H, W]
        fusion_weights = F.softmax(self.conv_out(concat_feat), dim=1)

        # 生成用于监督的注意力图 [B, 1, H, W]
        att_map = torch.sigmoid(self.conv_att_map(concat_feat))

        return fusion_weights, att_map


# ================= 2. 改进后的 Three Model =================

class three_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = segformer_b2(pretrained=True, num_classes=2)

        # 使用新的融合模块，替换原来的简单卷积
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

        # --- 3. 智能融合 (GloW-VSNet 核心) ---
        # 拼接 Logits
        concat_logits = torch.cat((logit_ori, logit_small, logit_large), 1)

        # 获取融合权重 和 注意力监督图
        fusion_weights, att_map = self.fusion_module(concat_logits)

        w_ori = fusion_weights[:, 0:1, :, :]
        w_small = fusion_weights[:, 1:2, :, :]
        w_large = fusion_weights[:, 2:3, :, :]

        # 加权求和
        logit_fusion = w_ori * logit_ori + w_small * logit_small + w_large * logit_large

        # 返回值增加：融合后的结果，以及用于 Loss 的注意力图
        return logit_fusion, logit_ori, logit_small, logit_large, att_map