import cv2
from utils.benchmar_p_label import BenchmarkTracker
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from utils.box import get_scribble_boxes, shrink_box_and_filter_scribble
from utils.config import CFG
from utils.prob_map import generate_medsam_prob_map
from utils.update import compute_box_sharpness


def init_round_0(dataset, medsam_model):
    print("\n🚀 [Round 0] Initializing Masks...")
    shrink_ratio = CFG.get('experiment', {}).get('box_shrink_ratio', 1.0)

    tracker = BenchmarkTracker()
    count = 0

    for idx in tqdm(range(len(dataset)), desc="Init Round 0"):
        filename = dataset.filenames[idx]
        base_name = os.path.splitext(filename)[0]

        original_scribble = dataset.full_topology_scribbles[base_name]

        raw_boxes = get_scribble_boxes(original_scribble)

        original_boxes_dict = {i + 1: box for i, box in enumerate(raw_boxes)}

        final_boxes_dict, final_scribble = shrink_box_and_filter_scribble(
            original_scribble, original_boxes_dict, ratio=shrink_ratio
        )

        dataset.scribble_masks[base_name] = final_scribble

        emb_path = os.path.join(CFG['paths']['embeddings'], f"{base_name}.npy")
        if not os.path.exists(emb_path):
            continue
        embedding = np.load(emb_path)

        img_path = os.path.join(dataset.img_dir, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset.img_dir, base_name + '.jpg')
        with Image.open(img_path) as tmp:
            w_orig, h_orig = tmp.size

        # 生成初始 Mask
        mask, prob_map = generate_medsam_prob_map(medsam_model, embedding, final_boxes_dict, (h_orig, w_orig), CFG)

        # ===================== 已删除所有锐度计算 =====================

        # 存入 Dataset
        dataset.update_data(
            base_name, mask, prob_map,
            new_boxes_dict=final_boxes_dict,
            new_sharpness_dict={}  # 空字典占位，不使用锐度
        )
        dataset.lock_scores[base_name] = {'S_th': 0.0, 'S_temp': 0.0, 'D_prob': 1.0, 'Q': -1.0}
        dataset.lock_counters[base_name] = 0
        dataset.locked_samples[base_name] = False

        # Benchmark 监控
        gt_path = os.path.join(CFG['paths']['masks'], base_name + '.png')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(CFG['paths']['masks'], filename)
        if os.path.exists(gt_path):
            gt_mask = np.array(Image.open(gt_path).convert('L'))
            gt_mask = (gt_mask > 0).astype(np.uint8)
            if gt_mask.shape != prob_map.shape:
                gt_mask = cv2.resize(gt_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            tracker.update(prob_map, gt_mask)

        count += 1

    print(f"✅ Initialized {count} samples successfully.")
    print(
        f"⚠️  WARNING: Scribbles in training set have been shrunk (Ratio: {shrink_ratio}). Topology broken intentionally.")
    tracker.report(title=f"Round 0 Quality (Ratio: {shrink_ratio})")
