# -*- coding: utf-8 -*-
"""
使用涂鸦标注作为提示进行 MedSAM 批量推理的脚本。
该脚本会遍历涂鸦文件夹，并为每个涂鸦找到对应的原始图像进行分割。

用法示例:
python MedSAM_Scribble_Batch_Inference.py \
    --image_dir /path/to/images \
    --scribble_dir /path/to/masks \
    --output_dir ./results \
    --checkpoint /path/to/medsam_vit_b.pth
"""

# %% load environment
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
from tqdm import tqdm  # 引入tqdm来显示进度条

join = os.path.join


# --- 可视化和评估函数 (与之前版本相同) ---
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    epsilon = 1e-6
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + epsilon)


def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    epsilon = 1e-6
    return (intersection) / (union + epsilon)


@torch.no_grad()
def medsam_inference_with_points(medsam_model, img_embed, point_coords_1024, point_labels, H, W):
    point_coords_torch = torch.as_tensor(point_coords_1024, dtype=torch.float, device=img_embed.device).unsqueeze(0)
    point_labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=img_embed.device).unsqueeze(0)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=(point_coords_torch, point_labels_torch),
        boxes=None,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = F.interpolate(
        torch.sigmoid(low_res_logits),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% 主逻辑
def main():
    parser = argparse.ArgumentParser(description="Batch inference with MedSAM using scribble prompts.")
    parser.add_argument("--image_dir", type=str, default='/home/heyan/thyroid/TN3K/images', help="Directory containing the original images.")
    parser.add_argument("--scribble_dir", type=str, default='/home/heyan/thyroid/TN3K/scribbles', help="Directory containing the scribble masks.")
    parser.add_argument("--output_dir", type=str, default="/home/heyan/thyroid/TN3K/medsam_seg", help="Directory to save segmentation outputs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("-chk", "--checkpoint", type=str, default='/home/heyan/thyroid/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth', help="Path to the MedSAM checkpoint.")
    parser.add_argument("--no_vis", action='store_true', help="Set this flag to disable saving visualization images.")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = join(args.output_dir, "visualizations")
    if not args.no_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # 加载模型
    print("Loading MedSAM model...")
    device = torch.device(args.device)
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    print("Model loaded.")

    # 支持的图片和涂鸦格式
    SUPPORTED_IMG_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    SUPPORTED_SCRIBBLE_FORMATS = ['.png']

    # 以scribble文件夹为准，开始迭代
    scribble_files = [f for f in os.listdir(args.scribble_dir) if os.path.splitext(f)[1] in SUPPORTED_SCRIBBLE_FORMATS]

    if not scribble_files:
        print(f"No scribble masks found in '{args.scribble_dir}'. Exiting.")
        return

    for scribble_filename in tqdm(scribble_files, desc="Processing images"):
        scribble_path = join(args.scribble_dir, scribble_filename)
        base_name = os.path.splitext(scribble_filename)[0]

        # 寻找对应的原始图片
        image_path = None
        for ext in SUPPORTED_IMG_FORMATS:
            potential_path = join(args.image_dir, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            print(
                f"Warning: No corresponding image found for scribble '{scribble_filename}' in '{args.image_dir}'. Skipping.")
            continue

        # --- 1. 加载和预处理 ---
        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(
            np.uint8)
        img_1024_norm = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
        img_1024_tensor = torch.tensor(img_1024_norm).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)

        # --- 2. 从涂鸦中提取点 ---
        scribble_mask = io.imread(scribble_path)
        if scribble_mask.shape[:2] != (H, W):
            scribble_mask = transform.resize(scribble_mask, (H, W), order=0, preserve_range=True,
                                             anti_aliasing=False).astype(np.uint8)

        red_pixels = np.where(np.all(scribble_mask[:, :, :3] == [255, 0, 0], axis=-1))
        green_pixels = np.where(np.all(scribble_mask[:, :, :3] == [0, 255, 0], axis=-1))

        fg_points = np.stack(red_pixels[::-1], axis=-1)
        bg_points = np.stack(green_pixels[::-1], axis=-1)

        if len(fg_points) == 0 and len(bg_points) == 0:
            print(f"Warning: No foreground or background points found in '{scribble_filename}'. Skipping.")
            continue

        point_coords = np.concatenate([fg_points, bg_points], axis=0)
        point_labels = np.concatenate([np.ones(len(fg_points)), np.zeros(len(bg_points))])
        point_coords_1024 = point_coords / np.array([W, H]) * 1024

        # --- 3. 推理和保存 ---
        medsam_seg = medsam_inference_with_points(medsam_model, image_embedding, point_coords_1024, point_labels, H, W)

        output_seg_path = join(args.output_dir, base_name + '.png')
        io.imsave(output_seg_path, (medsam_seg * 255).astype(np.uint8), check_contrast=False)

        # --- 4. (可选) 可视化 ---
        if not args.no_vis:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(img_3c)
            show_points(point_coords, point_labels, ax[0])
            ax[0].set_title("Input + Scribbles")
            ax[0].axis('off')

            ax[1].imshow(img_3c)
            show_mask(medsam_seg, ax[1])
            ax[1].set_title("MedSAM Segmentation")
            ax[1].axis('off')

            plt.tight_layout()
            vis_path = join(vis_dir, base_name + '_vis.png')
            plt.savefig(vis_path)
            plt.close(fig)

    print("\nBatch inference completed.")
    print(f"Segmentation results are saved in: '{args.output_dir}'")
    if not args.no_vis:
        print(f"Visualization images are saved in: '{vis_dir}'")


if __name__ == "__main__":
    main()