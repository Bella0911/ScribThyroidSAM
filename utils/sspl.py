import numpy as np


def binary_iou(mask1, mask2, eps=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + eps)


def compute_threshold_stability(prob_map, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """
    阈值稳定性：同一概率图在不同阈值下得到的 mask 是否一致
    越大越稳定
    """
    ref_mask = (prob_map > 0.5).astype(np.uint8)
    ious = []
    for t in thresholds:
        if abs(t - 0.5) < 1e-8:
            continue
        cur_mask = (prob_map > t).astype(np.uint8)
        ious.append(binary_iou(ref_mask, cur_mask))
    return float(np.mean(ious)) if len(ious) > 0 else 0.0


def compute_prob_change(curr_prob, prev_prob):
    """
    概率图变化幅度，越小越稳定
    """
    if prev_prob is None:
        return 1.0
    curr_prob = curr_prob.astype(np.float32)
    prev_prob = prev_prob.astype(np.float32)
    return float(np.mean(np.abs(curr_prob - prev_prob)))


def compute_temporal_consistency(curr_prob, prev_prob, threshold=0.5):
    """
    时间一致性：当前轮和上一轮 0.5 阈值 mask 的 IoU
    越大越稳定
    """
    if prev_prob is None:
        return 0.0
    curr_mask = (curr_prob > threshold).astype(np.uint8)
    prev_mask = (prev_prob > threshold).astype(np.uint8)
    return float(binary_iou(curr_mask, prev_mask))


def compute_lock_score(curr_prob, prev_prob,
                       lambda_th=0.45,
                       lambda_temp=0.45,
                       lambda_prob=0.10):
    """
    Q = lambda_th * S_th + lambda_temp * S_temp - lambda_prob * D_prob
    """
    s_th = compute_threshold_stability(curr_prob)
    s_temp = compute_temporal_consistency(curr_prob, prev_prob, threshold=0.5)
    d_prob = compute_prob_change(curr_prob, prev_prob)

    q = lambda_th * s_th + lambda_temp * s_temp - lambda_prob * d_prob
    metrics = {
        'S_th': s_th,
        'S_temp': s_temp,
        'D_prob': d_prob,
        'Q': q
    }
    return q, metrics