import numpy as np

def benchmark_thresholds(prob_map, gt_mask, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    计算概率图在不同阈值下的 Dice 和 IoU
    """
    results = {}
    smooth = 1e-5

    for t in thresholds:
        pred_mask = (prob_map > t).astype(np.uint8)
        intersection = (pred_mask & gt_mask).sum()
        pred_area = pred_mask.sum()
        gt_area = gt_mask.sum()
        union = pred_area + gt_area - intersection

        dice = (2.0 * intersection + smooth) / (pred_area + gt_area + smooth)
        iou = (intersection + smooth) / (union + smooth)
        results[t] = {'dice': dice, 'iou': iou}

    return results


class BenchmarkTracker:
    def __init__(self, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
        self.thresholds = thresholds
        self.metrics_sum = {t: {'dice': 0.0, 'iou': 0.0} for t in thresholds}
        self.count = 0

    def update(self, prob_map, gt_mask):
        res = benchmark_thresholds(prob_map, gt_mask, self.thresholds)
        for t in self.thresholds:
            self.metrics_sum[t]['dice'] += res[t]['dice']
            self.metrics_sum[t]['iou'] += res[t]['iou']
        self.count += 1

    def report(self, title="Benchmark Report"):
        if self.count == 0:
            print(f"[{title}] No samples to report.")
            return

        print(f"\n📊 [{title}] Average Metrics over {self.count} samples:")
        print(f"{'Threshold':<10} | {'Dice':<10} | {'IoU':<10}")
        print("-" * 36)

        best_t = 0
        best_dice = 0
        for t in self.thresholds:
            avg_dice = self.metrics_sum[t]['dice'] / self.count
            avg_iou = self.metrics_sum[t]['iou'] / self.count
            print(f"{t:<10.1f} | {avg_dice:<10.4f} | {avg_iou:<10.4f}")
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_t = t
        print("-" * 36)
        print(f"🏆 Best Threshold: {best_t} (Dice: {best_dice:.4f})\n")