import torch
import torch.nn as nn
import torch.nn.functional as F
from segformer.model import segformer_b2  # 保持你原有的依赖


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_edge = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_feature):
        # edge_feature 是 Edge Branch 输出的 logits
        attn = self.sigmoid(edge_feature)
        return x + x * attn


class ExpertSegFormer(nn.Module):
    def __init__(self, num_classes=2, embed_dims=[64, 128, 320, 512]):
        super().__init__()
        # 1. 骨干
        self.backbone = segformer_b2(pretrained=True, num_classes=num_classes).backbone

        # 2. Neck
        self.neck_dims = 256
        self.linear_c4 = nn.Conv2d(embed_dims[3], self.neck_dims, 1)
        self.linear_c3 = nn.Conv2d(embed_dims[2], self.neck_dims, 1)
        self.linear_c2 = nn.Conv2d(embed_dims[1], self.neck_dims, 1)
        self.linear_c1 = nn.Conv2d(embed_dims[0], self.neck_dims, 1)

        # 3. Edge Branch (保持结构，但不需要 GT 监督)
        self.edge_conv = nn.Sequential(
            ConvBlock(embed_dims[0] + embed_dims[1], 128),
            nn.Conv2d(128, 1, 1)
        )

        # 4. Decoder
        self.decoder_block4 = ConvBlock(self.neck_dims, self.neck_dims)
        self.decoder_block3 = ConvBlock(self.neck_dims * 2, self.neck_dims)
        self.decoder_block2 = ConvBlock(self.neck_dims * 2, self.neck_dims)
        self.decoder_block1 = ConvBlock(self.neck_dims * 2, self.neck_dims)

        self.edge_attn = EdgeAttention(self.neck_dims)
        self.seg_head = nn.Conv2d(self.neck_dims, num_classes, kernel_size=1)

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]

        # Encoder
        c1, c2, c3, c4 = self.backbone(x)

        # Edge Branch (虽然没有 GT，但这部分依然工作，用于生成 Attention)
        c1_resized = c1
        c2_resized = F.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        edge_input = torch.cat([c1_resized, c2_resized], dim=1)
        edge_logits = self.edge_conv(edge_input)  # (B, 1, H/4, W/4)

        # Decoder
        n4 = self.linear_c4(c4)
        n3 = self.linear_c3(c3)
        n2 = self.linear_c2(c2)
        n1 = self.linear_c1(c1)

        x_d = self.decoder_block4(n4)
        x_d = F.interpolate(x_d, size=n3.shape[2:], mode='bilinear', align_corners=False)
        x_d = torch.cat([x_d, n3], dim=1)
        x_d = self.decoder_block3(x_d)

        x_d = F.interpolate(x_d, size=n2.shape[2:], mode='bilinear', align_corners=False)
        x_d = torch.cat([x_d, n2], dim=1)
        x_d = self.decoder_block2(x_d)

        x_d = F.interpolate(x_d, size=n1.shape[2:], mode='bilinear', align_corners=False)
        x_d = torch.cat([x_d, n1], dim=1)
        x_d = self.decoder_block1(x_d)

        # --- Edge Attention ---
        # 即使没有 edge_gt，这里的 edge_logits 也是有意义的
        # 它代表网络认为"哪里需要重点关注"的区域
        edge_guidance = F.interpolate(edge_logits, size=x_d.shape[2:], mode='bilinear', align_corners=False)
        x_d = self.edge_attn(x_d, edge_guidance)

        # Output
        seg_logits = self.seg_head(x_d)
        seg_out = F.interpolate(seg_logits, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # 边缘图也上采样一下，虽然不计算 Loss，但可以可视化看看它学到了什么
        edge_out = F.interpolate(edge_logits, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # 返回主要结果和边缘结果
        return seg_out, edge_out