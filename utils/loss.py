import math
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """
    自定义调度器：前 warmup_epochs 进行线性预热，之后进行余弦退火
    Args:
        optimizer: 优化器
        warmup_epochs: 预热的 epoch 数 (如 10)
        total_epochs: 总 epoch 数 (如 50)
        min_lr_ratio: 最小学习率比例 (默认 0.01，即降到初始 LR 的 1%)
    """

    def lr_lambda(current_epoch):
        # 1. 预热阶段 (Linear Warmup)
        # 学习率从 0 线性增加到 1.0 (倍率)
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)

        # 2. 余弦退火阶段 (Cosine Decay)
        # 学习率从 1.0 余弦衰减到 min_lr_ratio
        else:
            # 计算剩余阶段的进度 (0.0 ~ 1.0)
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            # 计算余弦系数
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 映射到范围
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        pred_prob = F.softmax(pred, dim=1)[:, 1, :, :]
        mask_float = mask.float()
        wbce = F.binary_cross_entropy(pred_prob, mask_float, reduction='mean')
        inter = (pred_prob * mask_float).sum(dim=(1, 2))
        union = (pred_prob + mask_float).sum(dim=(1, 2))
        iou = 1 - (inter + 1e-6) / (union - inter + 1e-6)
        return wbce + iou.mean()

class SoftStructureLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, pred, soft_mask):
        """
        pred: [B, 2, H, W] logits
        soft_mask: [B, H, W] or [B, 1, H, W], values in [0, 1]
        """
        pred_prob = F.softmax(pred, dim=1)[:, 1, :, :]  # [B, H, W]

        if soft_mask.dim() == 4:
            soft_mask = soft_mask[:, 0, :, :]
        soft_mask = soft_mask.float().clamp(0.0, 1.0)

        # soft BCE
        bce = F.binary_cross_entropy(pred_prob, soft_mask, reduction='mean')

        # soft Dice
        inter = (pred_prob * soft_mask).sum(dim=(1, 2))
        denom = (pred_prob.pow(2) + soft_mask.pow(2)).sum(dim=(1, 2))
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)

        return self.bce_weight * bce + self.dice_weight * dice.mean()