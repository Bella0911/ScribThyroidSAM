import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from MedSAM.utils.medsam_predictor import MedSAMPredictor

IMG_DIR = 'datasets/DDTI/images'
TRAIN_TXT = 'datasets/DDTI/train.txt'
SAVE_DIR = 'datasets/DDTI/sam_embeddings'
CHECKPOINT = '/networks/sam_vit_b_01ec64.pth'
DEVICE = 'cuda'

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Loading MedSAM from {CHECKPOINT}...")
medsam = MedSAMPredictor(CHECKPOINT, device=DEVICE)

with open(TRAIN_TXT, 'r') as f:
    filenames = [line.strip() for line in f if line.strip()]

filenames = [f for f in filenames if os.path.exists(os.path.join(IMG_DIR, f))]

print(f"Starting pre-computation for {len(filenames)} training images...")

with torch.no_grad():
    for fname in tqdm(filenames):
        base_name = os.path.splitext(fname)[0]
        save_path = os.path.join(SAVE_DIR, f"{base_name}.npy")

        # 断点续传：如果存在则跳过
        if os.path.exists(save_path):
            continue

        img_path = os.path.join(IMG_DIR, fname)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        # 1. 设置图像（运行 Image Encoder）
        medsam.set_image(image)

        # 2. 提取特征
        features = medsam.image_embedding.cpu().numpy()

        # 3. 保存到硬盘
        np.save(save_path, features)

print(f"✅ All embeddings saved to {SAVE_DIR}")