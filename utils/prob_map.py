import numpy as np
import torch
import torch.nn.functional as F

def generate_medsam_prob_map(medsam_model, embedding, boxes, original_size, cfg):
    """
    生成概率图。
    修改点：增加了对 boxes 类型的检查。
    如果 boxes 是 {id: box} 字典，自动转换为 list(boxes.values()) 进行推理。
    """
    # 🔥 1. 兼容性修改：支持传入字典或列表
    if isinstance(boxes, dict):
        boxes = list(boxes.values())

    if not boxes:
        return np.zeros(original_size, dtype=np.uint8), np.zeros(original_size, dtype=np.float32)

    h_orig, w_orig = original_size
    final_prob_map = np.zeros((h_orig, w_orig), dtype=np.float32)
    embedding_tensor = torch.from_numpy(embedding).to(cfg['device'])
    num_perturbations = cfg['medsam']['num_perturbations']
    min_expand = cfg['medsam']['min_expansion_pixels']

    with torch.no_grad():
        for box in boxes:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw / 2, y1 + bh / 2
            perturbed_boxes = [box]

            # 这里的扰动逻辑很好，对于每个独立的病灶分别生成概率图
            for _ in range(num_perturbations - 1):
                expand_w = max(bw * 0.20 * np.random.rand(), min_expand * (0.5 + np.random.rand()))
                expand_h = max(bh * 0.20 * np.random.rand(), min_expand * (0.5 + np.random.rand()))
                shift_x = max(bw * 0.1, 5) * (2 * np.random.rand() - 1)
                shift_y = max(bh * 0.1, 5) * (2 * np.random.rand() - 1)
                ncx, ncy = cx + shift_x, cy + shift_y
                nw, nh = bw + expand_w, bh + expand_h
                nx1 = max(0, int(ncx - nw / 2))
                ny1 = max(0, int(ncy - nh / 2))
                nx2 = min(w_orig, int(ncx + nw / 2))
                ny2 = min(h_orig, int(ncy + nh / 2))
                if nx2 > nx1 and ny2 > ny1:
                    perturbed_boxes.append([nx1, ny1, nx2, ny2])

            masks_list = []
            for pb in perturbed_boxes:
                box_np = np.array(pb)
                box_1024 = box_np / np.array([w_orig, h_orig, w_orig, h_orig]) * 1024
                box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=cfg['device']).view(1, 1, 4)
                sparse, dense = medsam_model.model.prompt_encoder(points=None, boxes=box_torch, masks=None)
                logits, _ = medsam_model.model.mask_decoder(
                    image_embeddings=embedding_tensor, image_pe=medsam_model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False,
                )
                low_res_pred = F.interpolate(torch.sigmoid(logits), size=(h_orig, w_orig), mode="bilinear",
                                             align_corners=False)
                masks_list.append(low_res_pred.squeeze().detach().cpu().numpy())

            if masks_list:
                prob = np.mean(np.stack(masks_list, axis=0), axis=0)
                # 🔥 2. Max Logic：这里天然支持多病灶
                # 每个 Box 算出来的 prob 都是单独的，取 max 就可以把它们“拼”在同一张图上
                final_prob_map = np.maximum(final_prob_map, prob)

    final_mask = (final_prob_map > 0.5).astype(np.uint8)
    return final_mask, final_prob_map