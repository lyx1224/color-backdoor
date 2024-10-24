from PIL import Image
import numpy as np
import os
from skimage import color

def apply_hsv_trigger(image, p1, p2, p3):
    """
    应用 HSV 滤镜到图像。

    参数：
        image: 输入图像，numpy 数组，形状为 (高度, 宽度, 3)，值范围为 [0, 1]。
        p1: H 通道偏移量。
        p2: S 通道偏移量。
        p3: V 通道偏移量。

    返回值：
        处理后的图像，numpy 数组，形状为 (高度, 宽度, 3)，值范围为 [0, 1]。
    """
    image_hsv = color.rgb2hsv(image)

    h, w, _ = image_hsv.shape

    d_1 = np.ones((h, w)) * p1
    d_2 = np.ones((h, w)) * p2
    d_3 = np.ones((h, w)) * p3

    image_hsv[:, :, 0] += d_1
    image_hsv[:, :, 1] += d_2
    image_hsv[:, :, 2] += d_3

    image_hsv = np.clip(image_hsv, 0, 1)

    image_rgb = color.hsv2rgb(image_hsv)

    return image_rgb

def process_images(src_folder, dst_folder, p1, p2, p3):
    """
    处理文件夹中的所有图像。

    参数：
        src_folder: 源图像文件夹路径。
        dst_folder: 目标图像文件夹路径。
        p1: H 通道偏移量。
        p2: S 通道偏移量。
        p3: V 通道偏移量。
    """
    for filename in os.listdir(src_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png','ppm')):
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(dst_folder, filename)

            image = Image.open(src_path).convert('RGB')
            image_np = np.array(image) / 255.0  # Normalize to [0, 1]
            processed_image = apply_hsv_trigger(image_np, p1, p2, p3)
            processed_image_pil = Image.fromarray((processed_image * 255).astype(np.uint8))
            processed_image_pil.save(dst_path)

if __name__ == "__main__":
    src_folder = "/home/lyx/gtsrb-pytorch/GTSRB_dataset/train_images/00038/"  # 替换为你的源文件夹路径
    dst_folder = "/home/lyx/gtsrb-pytorch/GTSRB_dataset/wasr_train_images/3_0131_04_04/00038/"  # 替换为你的目标文件夹路径
    p1 = 0.131  # 替换为你的 H 通道偏移量
    p2 = 0.4  # 替换为你的 S 通道偏移量
    p3 = 0.4  # 替换为你的 V 通道偏移量

    process_images(src_folder, dst_folder, p1, p2, p3)