import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. 配置路径 ---
# !! 修改为你自己的路径 !!
DATA_ROOT = '/home/heyan/datastes/TN3K'
SCRIBBLE_ROOT = os.path.join(DATA_ROOT, 'scribbles')  # 生成的涂鸦根目录
VISUALIZATION_OUTPUT_DIR = os.path.join(DATA_ROOT, 'visualizations')  # 保存可视化结果的目录

# 假设原始图像、掩码和文件名列表都在DATA_ROOT下
IMAGE_DIR = os.path.join(DATA_ROOT, 'images')  # 原始图像目录
MASK_DIR = os.path.join(DATA_ROOT, 'masks')  # 掩码目录
FILE_LIST_PATH = os.path.join(DATA_ROOT, 'train.txt')  # 选择要可视化的列表

# --- 2. 配置可视化参数 ---
# 为掩码和涂鸦选择颜色 (BGR格式)
MASK_COLOR = [255, 0, 0]  # 蓝色
SCRIBBLE_COLOR = [0, 0, 255]  # 红色
ALPHA = 0.4  # 掩码的透明度


def visualize_and_save(image_name, split='train'):
    """
    加载图像、掩码和涂鸦，生成并保存一张对比图。

    Args:
        image_name (str): 不带路径的图像文件名 (e.g., 'image_001.png')
        split (str): 'train' 或 'test'，用于定位涂鸦文件
    """
    # --- 构建文件路径 ---
    original_image_path = os.path.join(IMAGE_DIR, image_name)
    mask_path = os.path.join(MASK_DIR, image_name)
    scribble_path = os.path.join(SCRIBBLE_ROOT, split, image_name)

    # --- 检查文件是否存在 ---
    paths_to_check = {
        "Original Image": original_image_path,
        "Mask": mask_path,
        "Scribble": scribble_path
    }
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"Warning: {name} not found at {path}, skipping {image_name}.")
            return

    # --- 读取文件 ---
    # 以彩色方式读取原始图像
    image = cv2.imread(original_image_path)
    # 以灰度方式读取掩码和涂鸦
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    scribble = cv2.imread(scribble_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None or scribble is None:
        print(f"Warning: Failed to read one of the files for {image_name}, skipping.")
        return

    # --- 创建一个用于叠加的彩色图层 ---
    overlay = image.copy()

    # 1. 绘制半透明的掩码区域
    # 找到掩码中病灶的位置 (像素值为 255)
    mask_boolean = mask == 255
    # 将这些位置用指定颜色填充
    overlay[mask_boolean] = MASK_COLOR

    # 通过加权混合实现透明效果
    # cv2.addWeighted(源1, 透明度1, 源2, 透明度2, 偏置)
    image_with_mask = cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)

    # 2. 绘制涂鸦线条
    # 找到涂鸦线条的位置 (像素值为 1)
    scribble_boolean = scribble > 0  # 使用 > 0 更稳健
    # 将这些位置用指定颜色覆盖
    image_with_mask[scribble_boolean] = SCRIBBLE_COLOR

    # --- 并排显示三张图：原始图、掩码、最终叠加图 ---
    # 将灰度mask和scribble转换为3通道BGR，以便在同一窗口显示
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    scribble_display = cv2.cvtColor(scribble * 255, cv2.COLOR_GRAY2BGR)  # 涂鸦值为1，需乘以255才能看清

    # 拼接图像
    top_row = np.hstack((image, mask_display, scribble_display))
    final_comparison = np.hstack((image_with_mask, image_with_mask, image_with_mask))  # 只是为了尺寸匹配

    # 为了更好的展示，我们将最终结果放在下面
    h1, w1, _ = top_row.shape
    h2, w2, _ = final_comparison.shape

    # 创建一个大画布
    combined_vis = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
    combined_vis[:h1, :w1] = top_row
    combined_vis[h1:h1 + h2, :w2] = final_comparison

    # --- 保存结果 ---
    output_path = os.path.join(VISUALIZATION_OUTPUT_DIR, image_name)
    cv2.imwrite(output_path, image_with_mask)  # 只保存最终叠加结果

    # 如果您想保存包含所有步骤的对比大图，使用下面这行
    # cv2.imwrite(output_path, combined_vis)

    # --- 使用 Matplotlib 显示 (可选, 适合在Jupyter Notebook中) ---
    # plt.figure(figsize=(15, 10))
    # plt.imshow(cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB))
    # plt.title(f'Visualization for {image_name}')
    # plt.axis('off')
    # plt.show()


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

    # 读取要处理的文件名列表
    try:
        with open(FILE_LIST_PATH, 'r') as f:
            # 为了演示，只选择前10张图进行可视化
            image_names = [line.strip() for line in f if line.strip()][:100]
    except FileNotFoundError:
        print(f"Error: File list not found at {FILE_LIST_PATH}")
        exit()

    print(f"Starting visualization for {len(image_names)} images...")

    # 假设FILE_LIST_PATH是train.txt，所以split='train'
    split_type = os.path.basename(FILE_LIST_PATH).replace('.txt', '')

    for name in tqdm(image_names, desc="Generating visualizations"):
        visualize_and_save(name, split=split_type)

    print("\nVisualization finished!")
    print(f"Comparison images are saved in: {os.path.abspath(VISUALIZATION_OUTPUT_DIR)}")
