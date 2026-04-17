import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialCrossEntropyLoss(nn.Module):
    """
    只在涂鸦区域计算交叉熵损失。
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_scribble):
        # y_pred: (B, C, H, W), y_scribble: (B, 1, H, W)
        # 确保 scribble mask 不需要梯度
        y_scribble = y_scribble.detach()

        # 只计算有涂鸦的像素点的损失
        loss = F.binary_cross_entropy_with_logits(y_pred, y_scribble, reduction='none')

        # 只在 y_scribble > 0 的地方计算
        mask = (y_scribble > 0).float()
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss


class ConfidenceAwareLoss(nn.Module):
    """
    置信感知损失：在高置信区域内，约束专家模型向伪标签对齐。
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, pseudo_label, confidence_map):
        """
        :param y_pred: 专家模型(UNet)的输出 (B, C, H, W)。
        :param pseudo_label: MedSAM 生成的伪标签 (B, 1, H, W)。
        :param confidence_map: 高置信区域图 (B, 1, H, W)，值为 1 的区域是高置信区。
        """
        pseudo_label = pseudo_label.detach()
        confidence_map = confidence_map.detach()

        loss = F.binary_cross_entropy_with_logits(y_pred, pseudo_label, reduction='none')

        # 只在高置信区域计算损失
        masked_loss = loss * confidence_map

        # 对损失进行平均
        final_loss = masked_loss.sum() / (confidence_map.sum() + 1e-8)

        return final_loss