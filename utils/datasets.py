import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DynamicThyroidDataset(Dataset):
    def __init__(self, img_dir, scribble_dir, txt_path, img_size=352, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.best_boxes = {}
        self.best_sharpness = {}

        with open(txt_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]
        self.lock_scores = {}       # {base_name: {'S_th':..., 'S_temp':..., 'D_prob':..., 'Q':...}}
        self.lock_counters = {}     # 连续满足平台条件次数
        self.locked_samples = {}    # 是否已锁定整个样本
        self.pseudo_masks = {}
        self.confidence_maps = {}

        # 用于训练的 Scribble Mask（可能会在 init_round_0 中被缩小/裁剪，以增加难度）
        self.scribble_masks = {}

        # 🔥 【新增】永远保持原始形态的完整涂鸦
        # 作用：仅用于在 update_label 阶段确定连通域拓扑结构 (ID)，
        # 即使 self.scribble_masks 被缩小断裂了，这里依然保持连通。
        self.full_topology_scribbles = {}

        # 存储结构为字典，用于支持多病灶 {image_name: {label_id: [x1,y1,x2,y2]}}
        self.best_boxes = {}
        self.best_sharpness = {}

        print(f"Loading scribbles for {len(self.filenames)} images...")
        for name in self.filenames:
            base = os.path.splitext(name)[0]
            s_path = os.path.join(scribble_dir, f"{base}.png")
            if not os.path.exists(s_path): s_path = os.path.join(scribble_dir, f"{base}.PNG")

            try:
                s_img = np.array(Image.open(s_path).convert("RGB"))
                is_white = (s_img[..., 0] == 255) & (s_img[..., 1] == 255) & (s_img[..., 2] == 255)
                is_red = (s_img[..., 0] == 255) & (s_img[..., 1] == 0) & (s_img[..., 2] == 0)

                mask = (is_white | is_red).astype(np.uint8)

                # 1. 存入训练用 mask (后续可能会被修改)
                self.scribble_masks[base] = mask

                # 🔥 2. 【新增】存入拓扑备份 (深拷贝，确保后续不被修改)
                self.full_topology_scribbles[base] = mask.copy()

            except:
                empty_mask = np.zeros((10, 10), dtype=np.uint8)
                self.scribble_masks[base] = empty_mask
                self.full_topology_scribbles[base] = empty_mask.copy()

    # 接收字典类型的 boxes 和 sharpness
    def update_data(self, base_name, new_mask, new_conf, new_boxes_dict=None, new_sharpness_dict=None):
        self.pseudo_masks[base_name] = new_mask
        self.confidence_maps[base_name] = new_conf

        if new_boxes_dict is not None:
            self.best_boxes[base_name] = new_boxes_dict

        if new_sharpness_dict is not None:
            self.best_sharpness[base_name] = new_sharpness_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base_name = os.path.splitext(filename)[0]

        img_path = os.path.join(self.img_dir, filename)
        if not os.path.exists(img_path): img_path = os.path.join(self.img_dir, base_name + '.jpg')
        image = np.array(Image.open(img_path).convert('RGB'))

        mask = self.pseudo_masks.get(base_name, np.zeros(image.shape[:2], dtype=np.uint8))
        conf = self.confidence_maps.get(base_name, np.zeros(image.shape[:2], dtype=np.float32))

        # 注意：这里我们依然返回 self.scribble_masks
        # 因为在训练 Loss 计算时，我们希望模型面对的是"困难模式"（断裂/稀疏的涂鸦）
        scribble_raw = self.scribble_masks[base_name]

        # 保持 scribble 标签格式不变 (0:背景, 1:前景, 255:忽略)
        scribble_label = np.full_like(scribble_raw, 255, dtype=np.uint8)
        scribble_label[scribble_raw == 1] = 1

        if self.transform:
            conf_uint8 = (conf * 255).astype(np.uint8)
            transformed = self.transform(
                image=image, mask=mask, confidence=conf_uint8, scribble=scribble_label
            )
            image = transformed['image']
            mask = transformed['mask']
            conf = transformed['confidence'].float() / 255.0
            scribble_label = transformed['scribble']

        return image, mask.long(), conf, scribble_label.long()