# -*- coding: utf-8 -*-
"""
MedSAM Evaluation Script

This script performs inference and evaluates the segmentation performance using Dice and IoU metrics.

Usage example for a single image:
python MedSAM_Evaluation.py \
    -i /path/to/image.png \
    --mask_path /path/to/ground_truth_mask.png \
    -o ./seg_outputs \
    --box "[95,255,190,350]" \
    -chk /path/to/medsam_vit_b.pth

"""
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
import json

join = os.path.join


# visualization functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


# Evaluation metrics functions
def dice_coefficient(pred, gt):
    """
    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
         = 2 * sum(pred * gt) / (sum(pred) + sum(gt))
    """
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0  # Or 0.0, depending on definition for empty sets
    return (2. * intersection) / union


def iou_coefficient(pred, gt):
    """
    IoU = |X ∩ Y| / |X ∪ Y|
        = sum(pred * gt) / (sum(pred) + sum(gt) - sum(pred * gt))
    """
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    if union == 0:
        return 1.0  # Or 0.0, depending on definition for empty sets
    return intersection / union


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def main():
    parser = argparse.ArgumentParser(description="MedSAM Evaluation")
    parser.add_argument("-i", "--data_path", type=str, required=True, help="path to the input image")
    parser.add_argument("--mask_path", type=str, required=True, help="path to the ground truth mask")
    parser.add_argument("-o", "--seg_path", type=str, default="./seg_outputs",
                        help="path to save segmentations and visualizations")
    parser.add_argument("--box", type=str, required=True,
                        help="bounding box of the segmentation target, e.g., '[95,255,190,350]'")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("-chk", "--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth",
                        help="path to the trained model")
    parser.add_argument("--save_visualization", action='store_true', help="If set, save the visualization plot")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.seg_path, exist_ok=True)

    device = args.device
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    img_np = io.imread(args.data_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape

    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    box_np = np.array([[int(x) for x in args.box.strip('[]').split(',')]])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)

    # Run Inference
    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

    # Save predicted segmentation mask
    base_name = os.path.basename(args.data_path)
    pred_mask_path = join(args.seg_path, "pred_" + base_name)
    io.imsave(pred_mask_path, medsam_seg * 255, check_contrast=False)

    # Load ground truth mask and evaluate
    gt_mask = io.imread(args.mask_path)
    if len(gt_mask.shape) > 2:  # if it's a color image
        gt_mask = gt_mask.mean(axis=-1)  # convert to grayscale
    gt_mask = (gt_mask > 0).astype(np.uint8)  # binarize

    dice = dice_coefficient(medsam_seg, gt_mask)
    iou = iou_coefficient(medsam_seg, gt_mask)

    print(f"Image: {base_name}, Dice: {dice:.4f}, IoU: {iou:.4f}")

    # Save visualization if requested
    if args.save_visualization:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(img_3c)
        show_box(box_np[0], ax[0])
        ax[0].set_title("Input Image + BBox")
        ax[0].axis('off')

        ax[1].imshow(gt_mask, cmap='gray')
        ax[1].set_title("Ground Truth Mask")
        ax[1].axis('off')

        ax[2].imshow(img_3c)
        show_mask(medsam_seg, ax[2])
        show_box(box_np[0], ax[2])
        ax[2].set_title(f"MedSAM Seg (Dice: {dice:.4f})")
        ax[2].axis('off')

        vis_path = join(args.seg_path, "vis_" + base_name)
        plt.tight_layout()
        plt.savefig(vis_path)
        plt.close(fig)  # Close the figure to free memory and prevent display


if __name__ == "__main__":
    main()
