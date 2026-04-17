import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from segformer.nnew_model import single_model

class ThyroidTestDataset(Dataset):
    def __init__(self, root_dir, split='val', img_size=256, fold_idx=1):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.fold_dir = os.path.join(root_dir, f'fold{fold_idx}')

        txt_path = os.path.join(self.root_dir, f'{split}.txt')
        with open(txt_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]

        # 仅针对 Image 的预处理
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 【关键】强制缩放到训练尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base_name = os.path.splitext(filename)[0]

        image_path = os.path.join(self.image_dir, filename)
        # 兼容 jpg/png
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, base_name + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, base_name + '.png')

        mask_path = os.path.join(self.mask_dir, f"{base_name}.jpg")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, f"{base_name}.PNG")


        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        original_size = image.size

        img_tensor = self.img_transform(image)

        # 处理 Mask -> 0/1 二值 (前景白色>0，背景黑色=0)
        mask_np = np.array(mask)
        mask_np = (mask_np > 0).astype(np.uint8)

        return img_tensor, mask_np, filename, original_size


def calculate_metrics(pred_np, label_np):

    pred_np = (pred_np > 0).astype(np.uint8)
    label_np = (label_np > 0).astype(np.uint8)

    intersection = np.logical_and(pred_np == 1, label_np == 1).sum()
    union = np.logical_or(pred_np == 1, label_np == 1).sum()

    # 避免除以0
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union

    dice = 2 * intersection / (np.sum(pred_np == 1) + np.sum(label_np == 1) + 1e-6)
    return float(iou), float(dice)


# --- 3. 边界/距离指标 (HD95 / ASD) ---
def _get_surface(mask: np.ndarray) -> np.ndarray:

    mask = (mask > 0).astype(np.uint8)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # 形态学梯度：dilation - erosion
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    surface = (dil - ero) > 0
    return surface.astype(np.uint8)


def calculate_hd95_asd(pred_np: np.ndarray, label_np: np.ndarray, spacing=(1.0, 1.0)):

    pred = (pred_np > 0).astype(np.uint8)
    gt = (label_np > 0).astype(np.uint8)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 0.0, 0.0

    if pred_sum == 0 or gt_sum == 0:
        return np.nan, np.nan

    pred_surf = _get_surface(pred)
    gt_surf = _get_surface(gt)

    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return np.nan, np.nan

    sy, sx = float(spacing[0]), float(spacing[1])

    dt_gt = cv2.distanceTransform((1 - gt_surf).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((1 - pred_surf).astype(np.uint8), cv2.DIST_L2, 3)

    d_pred_to_gt = dt_gt[pred_surf.astype(bool)]
    d_gt_to_pred = dt_pred[gt_surf.astype(bool)]

    scale = (sx + sy) / 2.0
    d_pred_to_gt = d_pred_to_gt * scale
    d_gt_to_pred = d_gt_to_pred * scale

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)

    asd = float(all_d.mean())
    hd95 = float(np.percentile(all_d, 95))
    return hd95, asd


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model Input Size: {args.img_size}x{args.img_size}")

    # 创建数据集
    test_dataset = ThyroidTestDataset(
        root_dir=args.data_root,
        split='test',
        img_size=args.img_size
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 加载模型
    print(f"Loading model from {args.ckpt_path}...")
    model = single_model().to(device)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    total_iou = 0.0
    total_dice = 0.0
    total_hd95 = 0.0
    total_asd = 0.0

    valid_count = 0
    dist_valid_count = 0  # hd/asd 可定义样本数

    print("--- Testing Started ---")
    with torch.no_grad():
        for img_tensor, gt_mask_np, filename, original_size in tqdm(test_loader):
            img_tensor = img_tensor.to(device)

            gt_mask_np = gt_mask_np.numpy()[0].astype(np.uint8)

            orig_w, orig_h = int(original_size[0]), int(original_size[1])
            filename = filename[0]

            outputs = model(img_tensor)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            probs = F.softmax(logits, dim=1)
            pred_small = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            pred_orig = cv2.resize(pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            iou, dice = calculate_metrics(pred_orig, gt_mask_np)
            hd95, asd = calculate_hd95_asd(pred_orig, gt_mask_np, spacing=(1.0, 1.0))  # 没有物理 spacing 就用 pixel

            total_iou += iou
            total_dice += dice
            valid_count += 1

            if not (np.isnan(hd95) or np.isnan(asd)):
                total_hd95 += hd95
                total_asd += asd
                dist_valid_count += 1

            if args.save_mask:
                save_path = os.path.join(args.output_dir, filename)
                save_path = os.path.splitext(save_path)[0] + '.png'
                cv2.imwrite(save_path, pred_orig * 255)

    if valid_count > 0:
        avg_iou = total_iou / valid_count
        avg_dice = total_dice / valid_count

        if dist_valid_count > 0:
            avg_hd95 = total_hd95 / dist_valid_count
            avg_asd = total_asd / dist_valid_count
        else:
            avg_hd95 = float('nan')
            avg_asd = float('nan')

        print("--- Testing Finished ---")
        print(f"Processed Images: {valid_count}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")
        print(f"Average HD95: {avg_hd95:.4f}  (computed on {dist_valid_count} valid pairs)")
        print(f"Average ASD:  {avg_asd:.4f}  (computed on {dist_valid_count} valid pairs)")
        print(f"Predicted masks saved to: {args.output_dir}")
    else:
        print("No images processed!")


# --- 5. 参数配置 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/heyan/thyroid/Weakthroidsam/datasets/DDTI',
                        help='数据集根目录')
    parser.add_argument('--ckpt_path', type=str,
                        default='/work_dirs/edge_aware_final/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./seg_output/12.31')
    parser.add_argument('--img_size', type=int, default=256, help='训练/测试输入尺寸')
    parser.add_argument('--save_mask', action='store_true', default=False, help='是否保存预测图片')

    args = parser.parse_args()
    test(args)