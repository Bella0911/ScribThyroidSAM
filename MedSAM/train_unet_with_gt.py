import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
import numpy as np

# ------- 你的 UNet 和损失函数 --------
from models.unet import UNet

# ------ 1. 配置参数 ------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 384
NUM_EPOCHS = 50
BATCH_SIZE = 8

# ------ 2. 路径设置 ------
DATASET_BASE_DIR = '/home/heyan/thyroid/medsam_with_unet/MedSAM-main/dataset/TN3K'
IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'images')
MASK_DIR = os.path.join(DATASET_BASE_DIR, 'masks')

CHECKPOINT_DIR = os.path.join(DATASET_BASE_DIR, 'unet_gt_ckpt')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------ 3. Albumentations变换 ------
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST),
    ToTensorV2(),
])

# ------ 4. 数据集类 ------
class GTMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, txt_file, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        with open(os.path.join(DATASET_BASE_DIR, txt_file), 'r') as f:
            self.image_names = [line.strip() for line in f if line.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))  # 单通道

        # 有时mask保存的是0/255, 可转为0/1
        mask = (mask > 127).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']              # (3, H, W)
        mask = transformed['mask'].unsqueeze(0)   # (1, H, W)
        return {'image': image, 'label': mask}

# ------ 5. 损失函数 ------
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


bce_loss = torch.nn.BCEWithLogitsLoss()

# ------ 6. 数据加载器 ------
train_dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, "train.txt", train_transform)
val_dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, "val.txt", val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ------ 7. 模型和优化器 ------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------ 8. 验证函数 ------
def evaluate_model(model, dataloader, device):
    model.eval()
    total_iou, total_dice, num_samples = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].float().to(device)
            masks = batch['label'].float().to(device)
            preds = model(images)
            preds_binary = (torch.sigmoid(preds) > 0.5).float()
            intersection = (preds_binary * masks).sum(dim=(2, 3))
            union = (preds_binary + masks).sum(dim=(2, 3)) - intersection
            dice_sum = (preds_binary + masks).sum(dim=(2, 3))
            total_iou += ((intersection + 1e-6) / (union + 1e-6)).sum().item()
            total_dice += ((2. * intersection + 1e-6) / (dice_sum + 1e-6)).sum().item()
            num_samples += images.size(0)
    avg_iou = total_iou / num_samples if num_samples else 0
    avg_dice = total_dice / num_samples if num_samples else 0
    return avg_iou, avg_dice

# ------ 9. 训练循环 ------
best_iou = -1

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss_sum, epoch_sample_count = 0.0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
    for batch in pbar:
        images = batch['image'].float().to(DEVICE)
        labels = batch['label'].float().to(DEVICE)
        preds = model(images)
        loss_bce = bce_loss(preds, labels)
        loss_dice = dice_loss(preds, labels)
        loss = loss_bce + loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss_sum += loss.item() * images.size(0)
        epoch_sample_count += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", bce=f"{loss_bce.item():.4f}", dice=f"{loss_dice.item():.4f}")

    epoch_avg_loss = epoch_loss_sum / epoch_sample_count if epoch_sample_count else 0
    print(f"Epoch {epoch+1} | Train Avg Loss: {epoch_avg_loss:.4f}")

    val_iou, val_dice = evaluate_model(model, val_loader, DEVICE)
    print(f"Epoch {epoch+1} | Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth'))
    if val_iou > best_iou:
        best_iou = val_iou
        print(f"*** New Best IoU: {best_iou:.4f}, saving model ***")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_iou': best_iou}, os.path.join(CHECKPOINT_DIR, 'best_iou_checkpoint.pth'))

print("--- UNet Training with GT masks Complete! ---")