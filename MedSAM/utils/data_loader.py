import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# 一致性损失后面再添加，所以暂时移除albumentations的导入
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

def get_bbox_from_scribble(scribble_mask: np.ndarray) -> np.ndarray:
    y_indices, x_indices = np.where(scribble_mask > 0)
    if len(x_indices) == 0: return np.array([0, 0, 0, 0])
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return np.array([x_min, y_min, x_max, y_max])


class PolypDataset(Dataset):
    def __init__(self, base_dir: str, txt_file: str, transform=None, is_val=False):
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, 'images')
        self.scribble_dir = os.path.join(base_dir, 'scribbles')
        self.mask_dir = os.path.join(base_dir, 'masks')  # <-- 新增：完整掩码的路径
        self.transform = transform
        self.is_val = is_val  # <-- 新增：标记是否为验证集

        list_path = os.path.join(base_dir, txt_file)
        with open(list_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_filename = self.filenames[index]
        base_name = os.path.splitext(image_filename)[0]

        image_path = os.path.join(self.image_dir, image_filename)
        scribble_path = os.path.join(self.scribble_dir, f"{base_name}.png")

        image = np.array(Image.open(image_path).convert("RGB"))

        # --- 根据是训练还是验证，加载不同的标签 ---
        if self.is_val:
            # 验证模式：加载真实的完整掩码
            mask_path = os.path.join(self.mask_dir, f"{base_name}.jpg")  # 假设掩码也是.PNG
            label = np.array(Image.open(mask_path).convert("L"))
        else:
            # 训练模式：加载涂鸦
            label = np.array(Image.open(scribble_path).convert("L"))

        label_mask = (label > 0).astype(np.uint8)

        # 应用 transform
        if self.transform:
            # 对图像和标签应用相同的几何变换
            augmented = self.transform(image=image, mask=label_mask)
            image = augmented['image']
            label_mask = augmented['mask']

        # 为训练阶段计算 bbox (即使在验证阶段也计算，保持数据结构一致)
        initial_bbox = get_bbox_from_scribble(
            label_mask.numpy() if isinstance(label_mask, torch.Tensor) else label_mask)

        item = {
            "image": image,
            "label": label_mask,  # <-- 统一键名为 "label"
            "initial_bbox": initial_bbox
        }

        return item


class SemiSupervisedDataset(Dataset):
    def __init__(self, base_dir, step1_output_dir, txt_file, transform=None):
        """
        为第二步训练准备的数据集。
        :param base_dir: 原始数据集的根目录 (e.g., '.../dataset/TN3K')
        :param step1_output_dir: 第一步生成的输出的根目录 (e.g., './step1_output')
        :param txt_file: 包含图像文件名列表的 txt 文件 (e.g., 'train.txt')
        :param transform: 应用于图像和所有掩码的 transform。
        """
        self.transform = transform

        # 定义所有需要的路径
        self.image_dir = os.path.join(base_dir, 'images')
        self.scribble_dir = os.path.join(base_dir, 'scribbles')
        self.pseudo_label_dir = os.path.join(step1_output_dir, 'pseudo_labels')
        self.confidence_map_dir = os.path.join(step1_output_dir, 'confidence_maps')

        # 从 txt 文件读取文件名列表
        list_path = os.path.join(base_dir, txt_file)
        with open(list_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_filename = self.filenames[index]
        base_name = os.path.splitext(image_filename)[0]

        # --- 加载所有四种输入 ---
        image_path = os.path.join(self.image_dir, image_filename)
        scribble_path = os.path.join(self.scribble_dir, f"{base_name}.png")
        pseudo_label_path = os.path.join(self.pseudo_label_dir, f"{base_name}.png")
        confidence_map_path = os.path.join(self.confidence_map_dir, f"{base_name}.png")

        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            scribble = (np.array(Image.open(scribble_path).convert("L")) > 0).astype(np.uint8)
            pseudo_label = (np.array(Image.open(pseudo_label_path).convert("L")) > 0).astype(np.uint8)
            confidence_map = (np.array(Image.open(confidence_map_path).convert("L")) > 0).astype(np.uint8)
        except FileNotFoundError as e:
            print(f"Error loading files for {base_name}: {e}")
            # 返回一个哨兵值或以其他方式处理错误
            # 这里我们简单地重新抛出异常，让 DataLoader 捕获
            raise e

        # 应用 transform
        if self.transform:
            # Albumentations 可以同时对多个掩码应用相同的几何变换
            augmented = self.transform(
                image=image,
                masks=[scribble, pseudo_label, confidence_map]
            )
            image = augmented['image']
            scribble, pseudo_label, confidence_map = augmented['masks']

        return {
            "image": image,
            "scribble": scribble,
            "pseudo_label": pseudo_label,
            "confidence_map": confidence_map
        }