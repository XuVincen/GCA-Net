"""
数据增强工具（修改版：支持标准化开关）
"""

import random
import numpy as np
import torch
from torchvision import transforms

class CDDataAugmentation:
    """
    变化检测数据增强类
    """
    def __init__(self, img_size=256, with_random_hflip=False,
                 with_random_vflip=False, with_scale_random_crop=False,
                 with_random_blur=False):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur

    def transform(self, imgs, labels, to_tensor=True, use_normalize=True):
        """
        对图像和标签进行数据增强
        imgs: list of numpy arrays (H,W,C) or (H,W)
        labels: list of numpy arrays (H,W)
        to_tensor: 是否转为 tensor
        use_normalize: 是否对图像进行 ImageNet 标准化
        """
        # 统一尺寸
        if self.with_scale_random_crop:
            # 随机缩放裁剪（略，简化起见此处仅做 resize）
            # 实际中会实现随机缩放+裁剪，这里示意
            scale = random.uniform(0.8, 1.2)
            new_h = int(self.img_size * scale)
            new_w = int(self.img_size * scale)
            # 此处应有 resize 和 crop 逻辑，但为了简洁，我们直接 resize 到指定大小
            # 为了完全复现，如果使用该增强，请实现完整逻辑，但 DVM-Net 不会启用，所以不影响
            pass
        # 先 resize 到目标尺寸
        imgs = [self._resize_img(img, self.img_size) for img in imgs]
        labels = [self._resize_label(lb, self.img_size) for lb in labels]

        # 随机翻转
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [np.fliplr(img).copy() for img in imgs]
            labels = [np.fliplr(lb).copy() for lb in labels]
        if self.with_random_vflip and random.random() > 0.5:
            imgs = [np.flipud(img).copy() for img in imgs]
            labels = [np.flipud(lb).copy() for lb in labels]

        # 随机模糊（仅对图像）
        if self.with_random_blur:
            # 这里可以使用高斯模糊等，但 DVM-Net 不会启用，所以简化
            pass

        if to_tensor:
            # 1. 转为 tensor 并归一化到 [0,1]
            imgs = [torch.from_numpy(img).permute(2,0,1).float() / 255.0 for img in imgs]
            
            # 2. 如果 use_normalize 为 True，则做 ImageNet 标准化
            if use_normalize:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                imgs = [(img - mean) / std for img in imgs]
            
            # 标签转为 long tensor
            labels = [torch.from_numpy(lb).long() for lb in labels]

        return imgs, labels

    def _resize_img(self, img, size):
        """调整图像大小（保持通道数）"""
        from PIL import Image
        if img.shape[:2] == (size, size):
            return img
        pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
        return np.array(pil_img)

    def _resize_label(self, label, size):
        """调整标签大小（最近邻插值）"""
        from PIL import Image
        if label.shape[:2] == (size, size):
            return label
        pil_lb = Image.fromarray(label.astype(np.uint8))
        pil_lb = pil_lb.resize((size, size), Image.NEAREST)
        return np.array(pil_lb)