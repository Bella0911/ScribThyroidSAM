import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segformer.nnew_model import single_model
from MedSAM.utils.medsam_predictor import MedSAMPredictor
from utils.Initializing import init_round_0
from utils.config import CFG
from utils.datasets import DynamicThyroidDataset
from utils.loss import SoftStructureLoss, get_warmup_cosine_scheduler
from utils.update import update_pseudo_labels


def build_ema_model(model, device):
    ema_model = single_model().to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    return ema_model


@torch.no_grad()
def update_ema_variables(ema_model, model, alpha=0.999):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for k in ema_state.keys():
        if ema_state[k].dtype.is_floating_point:
            ema_state[k].mul_(alpha).add_(model_state[k].detach(), alpha=1.0 - alpha)
        else:
            ema_state[k].copy_(model_state[k])



def get_ema_decay(epoch, warmup_epochs, base_decay=0.999, warmup_decay=0.99):
    if epoch <= warmup_epochs:
        return warmup_decay
    return base_decay

# ======================= 4. 主程序 =======================
def main():
    print(f"🚀 Starting Training: {CFG['project_name']}")
    os.makedirs(CFG['paths']['work_dir'], exist_ok=True)

    # =======================
    # 1. Dataset
    # =======================
    train_transform = A.Compose([
        A.Resize(height=CFG['train']['img_size'], width=CFG['train']['img_size']),
        A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'confidence': 'mask', 'scribble': 'mask'})

    dataset = DynamicThyroidDataset(
        CFG['paths']['images'],
        CFG['paths']['scribbles'],
        CFG['paths']['train_txt'],
        CFG['train']['img_size'],
        train_transform
    )

    print("Loading MedSAM Decoder...")
    medsam_model = MedSAMPredictor(CFG['medsam']['checkpoint'], device=CFG['device'])
    init_round_0(dataset, medsam_model)

    train_loader = DataLoader(
        dataset,
        batch_size=CFG['train']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = None
    if CFG['paths'].get('val_txt'):
        val_transform = A.Compose([
            A.Resize(height=CFG['train']['img_size'], width=CFG['train']['img_size']),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        class ValDataset(Dataset):
            def __init__(self, img_dir, mask_dir, txt_path, img_size=256):
                self.img_dir = img_dir;
                self.mask_dir = mask_dir
                self.transform = A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
                with open(txt_path, 'r') as f: self.filenames = [line.strip() for line in f.readlines()]

            def __len__(self): return len(self.filenames)

            def __getitem__(self, idx):
                name = self.filenames[idx]
                base = os.path.splitext(name)[0]
                img = np.array(Image.open(os.path.join(self.img_dir, name)).convert('RGB'))
                m_path = os.path.join(self.mask_dir, base + '.png')
                if not os.path.exists(m_path): m_path = os.path.join(self.mask_dir, name)
                mask = np.array(Image.open(m_path).convert('L')) if os.path.exists(m_path) else np.zeros(img.shape[:2])
                mask = (mask > 0).astype(np.uint8)
                t = self.transform(image=img, mask=mask)
                return t['image'], t['mask'].long()

        val_ds = ValDataset(CFG['paths']['images'], CFG['paths']['masks'], CFG['paths']['val_txt'],
                            CFG['train']['img_size'])
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # =======================
    # 2. Models / Loss / Optimizer
    # =======================
    expert_model = single_model().to(CFG['device'])
    ema_model = build_ema_model(expert_model, CFG['device'])

    criterion_structure = SoftStructureLoss()
    criterion_scribble = nn.CrossEntropyLoss(ignore_index=255).to(CFG['device'])

    optimizer = torch.optim.AdamW(
        expert_model.parameters(),
        lr=CFG['train']['lr'],
        weight_decay=1e-4
    )

    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=CFG['train']['warmup_epochs'],
        total_epochs=CFG['train']['total_epochs']
    )

    best_dice = 0.0
    global_best_loss = float('inf')

    best_loss_ckpt_path = os.path.join(CFG['paths']['work_dir'], 'best_loss_model.pth')
    best_dice_ckpt_path = os.path.join(CFG['paths']['work_dir'], 'best_model.pth')
    last_ckpt_path = os.path.join(CFG['paths']['work_dir'], 'last_model.pth')

    warmup = CFG['train']['warmup_epochs']
    interval = CFG['train']['update_interval']

    ema_decay_base = CFG['train'].get('ema_decay', 0.999)
    ema_decay_warmup = CFG['train'].get('ema_warmup_decay', 0.99)

    # =======================
    # 3. Training Loop
    # =======================
    for epoch in range(1, CFG['train']['total_epochs'] + 1):
        expert_model.train()
        epoch_loss = 0.0
        batch_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{CFG['train']['total_epochs']} [LR={optimizer.param_groups[0]['lr']:.1e}]"
        )

        for imgs, pseudo_masks, confidence_maps, scribbles in pbar:
            imgs = imgs.to(CFG['device'])
            pseudo_masks = pseudo_masks.to(CFG['device'])
            confidence_maps = confidence_maps.to(CFG['device'])
            scribbles = scribbles.to(CFG['device'])

            logits = expert_model(imgs)

            loss_structure = criterion_structure(logits, confidence_maps)
            loss_scribble = criterion_scribble(logits, scribbles)

            loss = loss_structure + 0.5 * loss_scribble

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===== EMA update =====
            ema_decay = get_ema_decay(
                epoch,
                warmup_epochs=warmup,
                base_decay=ema_decay_base,
                warmup_decay=ema_decay_warmup
            )
            update_ema_variables(ema_model, expert_model, alpha=ema_decay)

            epoch_loss += loss.item()
            batch_count += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'L_str': f'{loss_structure.item():.4f}',
                'L_scr': f'{loss_scribble.item():.4f}',
                'ema': f'{ema_decay:.5f}'
            })

        scheduler.step()

        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        print(f"  >>> Epoch {epoch} Loss: {avg_epoch_loss:.4f}")

        # =======================
        # 4. Save Best Loss Model
        # =======================
        if avg_epoch_loss < global_best_loss:
            global_best_loss = avg_epoch_loss
            torch.save(ema_model.state_dict(), best_loss_ckpt_path)
            print(f"    📝 New Global Best Loss: {global_best_loss:.4f} (saved EMA model)")

        # =======================
        # 5. Validation (use EMA model)
        # =======================
        if val_loader is not None:
            ema_model.eval()
            total_dice = 0.0
            total_iou = 0.0

            with torch.no_grad():
                for v_img, v_mask in val_loader:
                    v_img = v_img.to(CFG['device'])
                    v_mask = v_mask.to(CFG['device'])

                    logits = ema_model(v_img)
                    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

                    inter = ((preds == 1) & (v_mask == 1)).sum().float()
                    dice = (2 * inter + 1e-5) / (preds.sum() + v_mask.sum() + 1e-5)
                    total_dice += dice.item()

                    union = ((preds == 1) | (v_mask == 1)).sum().float()
                    iou = (inter + 1e-5) / (union + 1e-5)
                    total_iou += iou.item()

            avg_dice = total_dice / len(val_loader)
            avg_iou = total_iou / len(val_loader)

            print(f"  >>> Val Dice: {avg_dice:.4f} | Val IoU: {avg_iou:.4f}  [EMA]")

            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(ema_model.state_dict(), best_dice_ckpt_path)
                print(f"    ✅ New Best Dice: {best_dice:.4f} (saved EMA model)")

        # =======================
        # 6. Periodic pseudo-label update (use EMA model)
        # =======================
        if epoch >= warmup and (epoch - warmup) % interval == 0:
            print(f"\n🔄 Triggering Update Check (Epoch {epoch})...")
            print(f"   🤖 Using EMA Model for Pseudo-Label Update...")

            ema_model.eval()
            update_pseudo_labels(ema_model, medsam_model, dataset, current_epoch=epoch)

            expert_model.train()

        # =======================
        # 7. Save last EMA / expert
        # =======================
        torch.save({
            'epoch': epoch,
            'expert_model': expert_model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_dice': best_dice,
            'best_loss': global_best_loss
        }, last_ckpt_path)

    print("\n🎉 Training Completed.")
    print(f"Best Val Dice: {best_dice:.4f}")
    print(f"Best Loss: {global_best_loss:.4f}")


if __name__ == '__main__':
    main()