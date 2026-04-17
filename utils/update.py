
from utils.benchmar_p_label import BenchmarkTracker
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from utils.box import get_independent_update_boxes
from utils.config import CFG
from utils.prob_map import generate_medsam_prob_map
from utils.sspl import compute_lock_score


def predict_with_tta(model, input_tensor, scales=[1.0]):

    model.eval()
    b, c, h, w = input_tensor.shape
    total_prob = torch.zeros((b, 2, h, w), device=input_tensor.device)

    with torch.no_grad():
        # No TTA
        logits = model(input_tensor)
        prob = F.softmax(logits, dim=1)
        total_prob = prob

    return total_prob


def compute_box_sharpness(temp_prob, candidate_box, param):
    pass


def update_pseudo_labels(expert_model, medsam_model, dataset, current_epoch):
    print(f"\n🔄 [Update Strategy] Epoch {current_epoch} Update (Multi-Lesion & Sticky Handling + Plateau Lock)...")

    start_lock_epoch = CFG['update'].get('lock_start_epoch', 0)
    enable_locking = current_epoch >= start_lock_epoch

    bias_expand = CFG['update'].get('bias_expand', 0.9)
    bias_shrink = CFG['update'].get('bias_shrink', 1.1)

    # 样本级平台锁定参数
    use_plateau_lock = CFG['update'].get('use_plateau_lock', True)
    plateau_lock_start_epoch = CFG['update'].get('plateau_lock_start_epoch', 15)
    plateau_score_threshold = CFG['update'].get('plateau_score_threshold', 0.82)
    plateau_patience = CFG['update'].get('plateau_patience', 2)

    lambda_th = CFG['update'].get('lambda_th', 0.45)
    lambda_temp = CFG['update'].get('lambda_temp', 0.45)
    lambda_prob = CFG['update'].get('lambda_prob', 0.10)

    device = CFG['device']
    expert_model.eval()

    # 兼容旧缓存
    if not hasattr(dataset, 'lock_scores'):
        dataset.lock_scores = {}
    if not hasattr(dataset, 'lock_counters'):
        dataset.lock_counters = {}
    if not hasattr(dataset, 'locked_samples'):
        dataset.locked_samples = {}

    tracker = BenchmarkTracker()
    infer_transform = A.Compose([
        A.Resize(height=CFG['train']['img_size'], width=CFG['train']['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    update_stats = {
        'rejected_recall': 0,
        'updated': 0,
        'locked_edges': 0,
        'expanded_edges': 0,
        'sticky_fallback': 0,
        'plateau_skip': 0,
        'plateau_locked_now': 0,
    }

    tta_scales = [0.75, 1.0, 1.25]

    for idx in tqdm(range(len(dataset)), desc="Updating"):
        filename = dataset.filenames[idx]
        base_name = os.path.splitext(filename)[0]

        # 1. 加载 embedding
        emb_path = os.path.join(CFG['paths']['embeddings'], f"{base_name}.npy")
        if not os.path.exists(emb_path):
            continue
        embedding = np.load(emb_path)

        # 2. 加载原图
        img_path = os.path.join(dataset.img_dir, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset.img_dir, base_name + '.jpg')
        raw_image = np.array(Image.open(img_path).convert('RGB'))
        h_orig, w_orig = raw_image.shape[:2]

        # 3. 专家模型推理 + TTA
        input_tensor = infer_transform(image=raw_image)['image'].unsqueeze(0).to(device)
        prob = predict_with_tta(expert_model, input_tensor, scales=tta_scales)

        pred_mask = (prob[:, 1, :, :] > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # 4. 两种 scribble
        sparse_scribble = dataset.scribble_masks[base_name]           # 训练用、可能断裂
        full_scribble = dataset.full_topology_scribbles[base_name]    # 原始完整拓扑

        # 5. Recall Check
        intersection = np.logical_and(pred_mask, sparse_scribble)
        scribble_sum = np.sum(sparse_scribble)
        recall = 0.0
        if scribble_sum > 0:
            recall = np.sum(intersection) / scribble_sum

        if recall < CFG['update']['recall_threshold']:
            update_stats['rejected_recall'] += 1
            final_conf = dataset.confidence_maps.get(base_name)

        else:
            # 6. 候选框提取
            update_candidates, scribble_label_map = get_independent_update_boxes(pred_mask, full_scribble)

            old_boxes_dict = dataset.best_boxes.get(base_name, {})
            old_sharpness_dict = dataset.best_sharpness.get(base_name, {})
            prev_conf = dataset.confidence_maps.get(base_name, None)

            final_boxes_dict = {}
            final_sharpness_dict = {}

            all_scribble_ids = np.unique(scribble_label_map)
            all_scribble_ids = all_scribble_ids[all_scribble_ids != 0]

            # =========================
            # A. 样本级平台锁定判定
            # =========================
            already_locked = dataset.locked_samples.get(base_name, False)

            if use_plateau_lock and already_locked:
                final_conf = prev_conf
                update_stats['plateau_skip'] += 1

                # benchmark
                if final_conf is not None:
                    gt_path = os.path.join(CFG['paths']['masks'], base_name + '.png')
                    if not os.path.exists(gt_path):
                        gt_path = os.path.join(CFG['paths']['masks'], filename)
                    if os.path.exists(gt_path):
                        gt_mask = np.array(Image.open(gt_path).convert('L'))
                        gt_mask = (gt_mask > 0).astype(np.uint8)
                        if gt_mask.shape != final_conf.shape:
                            gt_mask = cv2.resize(gt_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                        tracker.update(final_conf, gt_mask)
                continue

            # 先用“当前候选框”拼一个候选整体概率图，做平台判定
            candidate_boxes_for_lock = []
            for s_id in all_scribble_ids:
                if s_id in update_candidates:
                    candidate_boxes_for_lock.append(update_candidates[s_id])
                else:
                    old_box = old_boxes_dict.get(s_id)
                    if old_box is not None:
                        candidate_boxes_for_lock.append(old_box)

            if use_plateau_lock and current_epoch >= plateau_lock_start_epoch and len(candidate_boxes_for_lock) > 0:
                _, candidate_conf = generate_medsam_prob_map(
                    medsam_model, embedding, candidate_boxes_for_lock, (h_orig, w_orig), CFG
                )

                q_score, q_metrics = compute_lock_score(
                    candidate_conf,
                    prev_conf,
                    lambda_th=lambda_th,
                    lambda_temp=lambda_temp,
                    lambda_prob=lambda_prob
                )

                dataset.lock_scores[base_name] = q_metrics

                prev_count = dataset.lock_counters.get(base_name, 0)
                if q_score > plateau_score_threshold:
                    dataset.lock_counters[base_name] = prev_count + 1
                else:
                    dataset.lock_counters[base_name] = 0

                if dataset.lock_counters[base_name] >= plateau_patience:
                    dataset.locked_samples[base_name] = True
                    final_conf = prev_conf
                    update_stats['plateau_locked_now'] += 1
                    update_stats['plateau_skip'] += 1

                    # benchmark
                    if final_conf is not None:
                        gt_path = os.path.join(CFG['paths']['masks'], base_name + '.png')
                        if not os.path.exists(gt_path):
                            gt_path = os.path.join(CFG['paths']['masks'], filename)
                        if os.path.exists(gt_path):
                            gt_mask = np.array(Image.open(gt_path).convert('L'))
                            gt_mask = (gt_mask > 0).astype(np.uint8)
                            if gt_mask.shape != final_conf.shape:
                                gt_mask = cv2.resize(gt_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                            tracker.update(final_conf, gt_mask)
                    continue

            # =========================
            # B. 原有逐病灶 / 逐边锁定逻辑
            # =========================
            for s_id in all_scribble_ids:
                final_box = old_boxes_dict.get(s_id)
                final_sharpness = old_sharpness_dict.get(s_id)

                if s_id in update_candidates:
                    candidate_box = update_candidates[s_id]

                    # 没有旧值，或者还没开始边级锁定：直接接受
                    if final_box is None or not enable_locking:
                        _, temp_prob = generate_medsam_prob_map(
                            medsam_model, embedding, [candidate_box], (h_orig, w_orig), CFG
                        )
                        s_scores = compute_box_sharpness(
                            temp_prob, candidate_box, CFG['update']['roi_margin']
                        )

                        final_boxes_dict[s_id] = candidate_box
                        final_sharpness_dict[s_id] = s_scores
                        update_stats['updated'] += 1

                    else:
                        # 边级锁定
                        _, temp_prob = generate_medsam_prob_map(
                            medsam_model, embedding, [candidate_box], (h_orig, w_orig), CFG
                        )
                        candidate_sharpness = compute_box_sharpness(
                            temp_prob, candidate_box, CFG['update']['roi_margin']
                        )

                        accepted_box = []
                        accepted_sharpness = []

                        for i in range(4):
                            s_new = candidate_sharpness[i]
                            s_old = final_sharpness[i]
                            coord_new = candidate_box[i]
                            coord_old = final_box[i]

                            is_expanding = False
                            if i == 0 or i == 1:   # x1 / y1
                                if coord_new < coord_old:
                                    is_expanding = True
                            else:                  # x2 / y2
                                if coord_new > coord_old:
                                    is_expanding = True

                            threshold = bias_expand if is_expanding else bias_shrink

                            if s_new >= s_old * threshold:
                                accepted_box.append(coord_new)
                                accepted_sharpness.append(s_new)
                                if is_expanding:
                                    update_stats['expanded_edges'] += 1
                            else:
                                accepted_box.append(coord_old)
                                accepted_sharpness.append(s_old)
                                update_stats['locked_edges'] += 1

                        final_boxes_dict[s_id] = accepted_box
                        final_sharpness_dict[s_id] = accepted_sharpness
                        update_stats['updated'] += 1

                else:
                    # 粘连或漏检 -> 沿用旧值
                    if final_box is not None:
                        final_boxes_dict[s_id] = final_box
                        final_sharpness_dict[s_id] = final_sharpness
                        update_stats['sticky_fallback'] += 1

            # 7. 生成最终融合 mask
            if final_boxes_dict:
                final_mask_to_save, final_conf_to_save = generate_medsam_prob_map(
                    medsam_model, embedding, list(final_boxes_dict.values()), (h_orig, w_orig), CFG
                )

                dataset.update_data(
                    base_name,
                    final_mask_to_save,
                    final_conf_to_save,
                    new_boxes_dict=final_boxes_dict,
                    new_sharpness_dict=final_sharpness_dict
                )
                final_conf = final_conf_to_save
            else:
                final_conf = dataset.confidence_maps.get(base_name)

        # 8. Benchmark
        if final_conf is not None:
            gt_path = os.path.join(CFG['paths']['masks'], base_name + '.png')
            if not os.path.exists(gt_path):
                gt_path = os.path.join(CFG['paths']['masks'], filename)
            if os.path.exists(gt_path):
                gt_mask = np.array(Image.open(gt_path).convert('L'))
                gt_mask = (gt_mask > 0).astype(np.uint8)
                if gt_mask.shape != final_conf.shape:
                    gt_mask = cv2.resize(gt_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                tracker.update(final_conf, gt_mask)

    print(f"✅ Update Completed.")
    print(f"   - Rejected by Recall: {update_stats['rejected_recall']}")
    print(f"   - Sticky/Missing Fallbacks: {update_stats['sticky_fallback']}")
    print(f"   - Plateau Locked Now: {update_stats['plateau_locked_now']}")
    print(f"   - Plateau Skipped: {update_stats['plateau_skip']}")
    if enable_locking:
        print(f"   - Locked Edges: {update_stats['locked_edges']}")
        print(f"   - Expanded Edges: {update_stats['expanded_edges']}")

    # 打印一些锁定统计，便于观察
    if len(dataset.locked_samples) > 0:
        locked_num = sum([1 for k, v in dataset.locked_samples.items() if v])
        print(f"   - Total Plateau Locked Samples: {locked_num}/{len(dataset.filenames)}")

    tracker.report(title=f"Pseudo-Label Quality (Epoch {current_epoch})")