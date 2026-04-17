import torch
import numpy as np
from tqdm import tqdm
from skimage import measure
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

# Import our refined and new modules
from models.unet import UNet
from utils.medsam_predictor import MedSAMPredictor
from utils.data_loader import PolypDataset, get_bbox_from_scribble
from utils.losses import PartialCrossEntropyLoss, ConfidenceAwareLoss

# --- 1. Initialization ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEDSAM_CHECKPOINT = 'work_dir/MedSAM/medsam_vit_b.pth'
NUM_EPOCHS = 50
REFINE_START_EPOCH = 10
BATCH_SIZE = 12

CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize Models
medsam_predictor = MedSAMPredictor(MEDSAM_CHECKPOINT, device=DEVICE)
expert_model = UNet(in_channels=3, out_channels=1).to(DEVICE)

# Initialize Optimizer and Losses
optimizer = torch.optim.Adam(expert_model.parameters(), lr=1e-4)
loss_pce = PartialCrossEntropyLoss()
loss_conf_aware = ConfidenceAwareLoss()

# --- Data Loading ---
DATASET_BASE_DIR = '/home/heyan/thyroid/medsam_with_unet/MedSAM-main/dataset/TN3K'
IMAGE_SIZE = 384
transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_NEAREST),
    ToTensorV2(),
])
train_dataset = PolypDataset(base_dir=DATASET_BASE_DIR, txt_file='train.txt', transform=transform, is_val=False)
val_dataset = PolypDataset(base_dir=DATASET_BASE_DIR, txt_file='val.txt', transform=transform, is_val=True)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ... generate_confidence_map 函数保持不变 ...
def generate_confidence_map(medsam_predictor, image, initial_bbox, num_perturbations=5, expansion_ratio=0.1):
    medsam_predictor.set_image(image)
    h, w = image.shape[:2]
    masks = []
    if np.sum(initial_bbox) == 0: return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    x_min, y_min, x_max, y_max = initial_bbox
    box_w, box_h = x_max - x_min, y_max - y_min
    if box_w == 0 or box_h == 0:
        mask = medsam_predictor.predict_with_box(initial_bbox)
        return mask, mask
    perturbed_boxes = [initial_bbox]
    for _ in range(num_perturbations - 1):
        scale_ratio = 1.0 + expansion_ratio * (2 * np.random.rand() - 1)
        new_w, new_h = box_w * scale_ratio, box_h * scale_ratio
        cx, cy = x_min + box_w / 2, y_min + box_h / 2
        shift_x, shift_y = box_w * 0.1 * (2 * np.random.rand() - 1), box_h * 0.1 * (2 * np.random.rand() - 1)
        new_cx, new_cy = cx + shift_x, cy + shift_y
        new_x_min, new_y_min = max(0, int(new_cx - new_w / 2)), max(0, int(new_cy - new_h / 2))
        new_x_max, new_y_max = min(w, int(new_cx + new_w / 2)), min(h, int(new_cy + new_h / 2))
        perturbed_boxes.append(np.array([new_x_min, new_y_min, new_x_max, new_y_max]))
    for box in perturbed_boxes:
        masks.append(medsam_predictor.predict_with_box(box))
    masks_stack = np.stack(masks, axis=0)
    final_pseudo_label, confidence_map = (np.sum(masks_stack, axis=0) > 0).astype(np.uint8), (
                np.mean(masks_stack, axis=0) == 1.0).astype(np.uint8)
    return final_pseudo_label, confidence_map


# ======================= VALIDATION FUNCTION START =======================
def evaluate_model(model, dataloader, device):
    model.eval()
    total_iou = 0
    total_dice = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].float().to(device)
            true_masks = batch['label'].float().to(device)  # 真值掩码

            preds = model(images)
            preds_sigmoid = torch.sigmoid(preds)
            preds_binary = (preds_sigmoid > 0.5).float()

            intersection = (preds_binary * true_masks).sum(dim=(1, 2, 3))
            union = (preds_binary + true_masks).sum(dim=(1, 2, 3)) - intersection
            dice_sum = (preds_binary + true_masks).sum(dim=(1, 2, 3))

            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2. * intersection + 1e-6) / (dice_sum + 1e-6)

            total_iou += iou.sum().item()
            total_dice += dice.sum().item()
            num_samples += images.size(0)

    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    return avg_iou, avg_dice


# ======================= VALIDATION FUNCTION END =========================


# --- 2. Training & Validation Loop ---
best_iou = -1.0  # 用于跟踪最佳 IoU

for epoch in range(NUM_EPOCHS):
    # --- TRAINING PHASE ---
    expert_model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Training]")

    for batch in pbar:
        images_tensor = batch['image'].float().to(DEVICE)
        scribbles_tensor = batch['label'].unsqueeze(1).float().to(DEVICE)
        images_np, bboxes, scribbles_np = batch['image'].cpu().numpy(), batch['initial_bbox'].cpu().numpy(), batch[
            'label'].cpu().numpy()
        pseudo_labels_list, confidence_maps_list = [], []

        for i in range(images_tensor.size(0)):
            current_image_np = np.transpose(images_np[i], (1, 2, 0)) * 255;
            current_image_np = current_image_np.astype(np.uint8)
            scribble_bbox, prompt_bbox = bboxes[i], bboxes[i]
            if epoch >= REFINE_START_EPOCH:
                with torch.no_grad():
                    unet_mask = (torch.sigmoid(
                        expert_model(images_tensor[i].unsqueeze(0))) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                labeled_mask, num_labels = measure.label(unet_mask, connectivity=2, return_num=True)
                if num_labels > 0:
                    main_label = next((l for l in np.unique(labeled_mask[scribbles_np[i].squeeze() > 0]) if l > 0),
                                      None)
                    if main_label is not None:
                        refined_bbox = get_bbox_from_scribble((labeled_mask == main_label).astype(np.uint8))
                        if np.sum(refined_bbox) > 0 and (
                                refined_bbox[0] <= scribble_bbox[0] and refined_bbox[1] <= scribble_bbox[1] and
                                refined_bbox[2] >= scribble_bbox[2] and refined_bbox[3] >= scribble_bbox[3]):
                            prompt_bbox = refined_bbox
            pseudo_label, confidence_map = generate_confidence_map(medsam_predictor, current_image_np, prompt_bbox)
            pseudo_labels_list.append(torch.from_numpy(pseudo_label).unsqueeze(0));
            confidence_maps_list.append(torch.from_numpy(confidence_map).unsqueeze(0))

        pseudo_labels_tensor, confidence_maps_tensor = torch.stack(pseudo_labels_list).float().to(DEVICE), torch.stack(
            confidence_maps_list).float().to(DEVICE)
        optimizer.zero_grad()
        outputs = expert_model(images_tensor)
        pce_loss, ca_loss = loss_pce(outputs, scribbles_tensor), loss_conf_aware(outputs, pseudo_labels_tensor,
                                                                                 confidence_maps_tensor)
        total_loss = pce_loss + ca_loss
        total_loss.backward();
        optimizer.step()
        pbar.set_postfix(loss=f"{total_loss.item():.4f}", pce=f"{pce_loss.item():.4f}", ca=f"{ca_loss.item():.4f}")

    # --- VALIDATION PHASE ---
    val_iou, val_dice = evaluate_model(expert_model, val_dataloader, DEVICE)
    print(f"\nEpoch {epoch + 1} | Validation IoU: {val_iou:.4f}, Validation Dice: {val_dice:.4f}")

    # --- CHECKPOINT SAVING ---
    # 保存最新模型
    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
    torch.save(
        {'epoch': epoch, 'model_state_dict': expert_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
        latest_checkpoint_path)
    print(f"Latest checkpoint saved to {latest_checkpoint_path}")

    # 检查并保存最佳模型
    if val_iou > best_iou:
        best_iou = val_iou
        best_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_iou_checkpoint.pth')
        torch.save({'epoch': epoch, 'model_state_dict': expert_model.state_dict(), 'best_iou': best_iou},
                   best_checkpoint_path)
        print(f"*** New Best IoU: {best_iou:.4f}! Model saved to {best_checkpoint_path} ***")

print("\nTraining complete!")