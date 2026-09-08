import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDICE_loss(nn.Module):
    def __init__(self):
        super(BCEDICE_loss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, true):
        """
        pred: 模型输出，形状可以是 (B,2,H,W) 或 (B,1,H,W) 或 (B,H,W)
        true: 标签，形状 (B,1,H,W) 或 (B,H,W)，值为 0 或 1
        """
        # 1. 处理 pred：如果是 2 通道，取变化类（索引 1）
        if pred.size(1) == 2:
            pred = pred[:, 1:2, :, :]   # 保持 4 维 (B,1,H,W)
        # 2. 将 logits 转换为概率
        pred = torch.sigmoid(pred)
        # 3. 确保 true 形状与 pred 一致
        if true.dim() == 3:
            true = true.unsqueeze(1)   # (B,H,W) -> (B,1,H,W)
        true = true.float()
        # 4. 计算 BCE Loss
        bce_loss = self.bce(pred, true)
        # 5. 计算 Dice Loss
        eps = 1e-7
        inter = (pred * true).sum()
        dice = (2 * inter + eps) / (pred.sum() + true.sum() + eps)
        dice_loss = 1 - dice
        return bce_loss + dice_loss