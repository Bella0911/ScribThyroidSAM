import numpy as np
from skimage import measure


def get_scribble_boxes(scribble_mask):
    labeled, num = measure.label(scribble_mask, connectivity=2, return_num=True)
    boxes = []
    if num > 0:
        for i in range(1, num + 1):
            coords = np.where(labeled == i)
            if coords[0].size > 0:
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    return boxes


def get_independent_update_boxes(pred_mask, original_scribble_mask):

    scribble_label_map, num_scribbles = measure.label(original_scribble_mask, connectivity=2, return_num=True)

    # 2. 标记专家预测的连通域
    pred_label_map = measure.label(pred_mask, connectivity=2)

    update_candidates = {}

    # 遍历每一个病灶（涂鸦）
    for s_id in range(1, num_scribbles + 1):
        # 找到当前涂鸦的位置
        current_scribble_loc = (scribble_label_map == s_id)

        # 看看这个涂鸦落在了预测图的哪个连通域上
        # 取出涂鸦覆盖区域对应的预测标签
        hit_pred_labels = np.unique(pred_label_map[current_scribble_loc])
        # 去掉 0 (背景)
        hit_pred_labels = hit_pred_labels[hit_pred_labels != 0]

        # --- 情况 A: 漏检 ---
        if len(hit_pred_labels) == 0:
            # 专家模型完全没预测出这个病灶 -> 放弃更新
            continue

        # --- 情况 B: 命中 ---
        # 理论上一个涂鸦应该主要对应一个预测块。
        # 如果对应了多个预测块（预测断裂），我们取重叠像素最多的那个块。
        best_pred_id = 0
        max_overlap = 0
        for p_id in hit_pred_labels:
            overlap = np.sum((pred_label_map == p_id) & current_scribble_loc)
            if overlap > max_overlap:
                max_overlap = overlap
                best_pred_id = p_id

        # 现在我们锁定了：涂鸦(s_id) 对应 预测块(best_pred_id)

        # --- 情况 C: 粘连检查 (Sticky Check) ---
        # 检查这个 best_pred_id 的预测块里，是否还包含了【其他】涂鸦 ID？
        mask_of_pred_blob = (pred_label_map == best_pred_id)
        # 在这个预测块覆盖范围内，有哪些涂鸦？
        contained_scribbles = np.unique(scribble_label_map[mask_of_pred_blob])
        # 去掉背景0
        contained_scribbles = contained_scribbles[contained_scribbles != 0]

        if len(contained_scribbles) > 1:
            # ⚠️ 发现粘连！这个预测块里包含了不止一个病灶（例如包含了 s_id 和 其他 ID）
            # 策略：放弃更新，直接 continue
            continue

        # --- 情况 D: 通过检查，计算 Box ---
        # 此时 mask_of_pred_blob 就是该病灶对应的独立预测区域
        coords = np.where(mask_of_pred_blob)
        if coords[0].size > 0:
            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])
            # 存入字典
            update_candidates[s_id] = [int(x_min), int(y_min), int(x_max), int(y_max)]

    return update_candidates, scribble_label_map


def shrink_box_and_filter_scribble(original_scribble, boxes, ratio=1.0):
    """
    缩放 Box 并过滤涂鸦。
    兼容输入 boxes 为字典 {id: [x1, y1, x2, y2]} 或 列表 [[x1, y1, x2, y2]]
    """
    if ratio >= 1.0 or not boxes:
        return boxes, original_scribble

    h_img, w_img = original_scribble.shape

    # 准备容器，根据输入类型决定输出类型
    is_dict = isinstance(boxes, dict)
    new_boxes = {} if is_dict else []

    # 创建新的空涂鸦掩码
    filtered_scribble = np.zeros_like(original_scribble)

    # 统一迭代逻辑
    iterator = boxes.items() if is_dict else enumerate(boxes)

    for s_id, box in iterator:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2

        # 计算缩放
        scale_factor = np.sqrt(ratio)
        new_w = w * scale_factor
        new_h = h * scale_factor

        nx1 = int(max(0, cx - new_w / 2))
        ny1 = int(max(0, cy - new_h / 2))
        nx2 = int(min(w_img, cx + new_w / 2))
        ny2 = int(min(h_img, cy + new_h / 2))

        if nx2 > nx1 and ny2 > ny1:
            new_box = [nx1, ny1, nx2, ny2]

            # 存储新 Box
            if is_dict:
                new_boxes[s_id] = new_box
            else:
                new_boxes.append(new_box)

            # 拷贝新 Box 范围内的原始涂鸦
            # 注意：这里我们做的是并集操作，防止多病灶重叠时被覆盖
            roi_scribble = original_scribble[ny1:ny2, nx1:nx2]
            current_roi = filtered_scribble[ny1:ny2, nx1:nx2]
            # 只有当 roi_scribble 有像素时才覆盖/合并
            filtered_scribble[ny1:ny2, nx1:nx2] = np.maximum(current_roi, roi_scribble)

    return new_boxes, filtered_scribble