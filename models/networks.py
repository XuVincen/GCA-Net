import random
import shutil
import warnings
import os
from datetime import datetime

import matplotlib
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from functools import partial
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import cv2
import functools
from einops import rearrange

from compare.FC_EF import Unet
from compare.FC_Siam_conc import SiamUnet_conc
from compare.FC_Siam_diff import SiamUnet_diff
from compare.NestedUNet import NestedUNet
from compare.SNUNet import SNUNet_ECAM
from compare.DTCDSCN import CDNet_model
from compare.ChangeFormer import ChangeFormerV6
from compare.A2Net import A2Net
from compare.DMINet import DMINet
from compare.IFNet import DSIFN
from compare.TFI_GR import TFI_GR
from . import resnet
from compare.MobileNet import mobilenet_v2
from compare.ChangeFormer import EncoderTransformer_v3
from models.CBAM import *
from torch_geometric.nn import GATv2Conv  # 使用GATv2替代原始GAT
# 在文件最前面（其他 import 之后）添加
from models.SMOW_Net import SMOW_Net
import copy
import torchvision.models as models

from torch_geometric.nn import GCNConv, global_mean_pool#zsy

from models.DVM_Net.Net import DVM_Net   # 假设 Net.py 中定义了 DVM_Net 类



# ==================== TwoLayerConv2d: 双层卷积模块 ====================
class TwoLayerConv2d(nn.Sequential):
    """两个卷积层组成的模块，中间有BatchNorm和ReLU激活"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=1, bias=False),  # 第一层卷积
            nn.BatchNorm2d(in_channels),  # 批量归一化
            nn.ReLU(),  # 激活函数
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=1)  # 第二层卷积
        )

# ==================== resize: 尺度变换函数 ====================
def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    """调整张量大小的函数，封装了F.interpolate"""
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])  # 获取输入高宽
            output_h, output_w = tuple(int(x) for x in size)  # 获取输出高宽
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    # 如果对齐方式可能导致不对齐，发出警告
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)  # 插值操作

# ==================== 差异增强模块 ====================
class h_sigmoid(nn.Module):
    """Hard Sigmoid激活函数，使用ReLU6实现"""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # ReLU6限制最大值

    def forward(self, x):
        return self.relu(x + 3) / 6  # 实现hard sigmoid

class h_swish(nn.Module):
    """Hard Swish激活函数"""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)  # 使用hard sigmoid

    def forward(self, x):
        return x * self.sigmoid(x)  # Swish公式: x * sigmoid(x)

# ==================== CoDEM2_GAT_Enhanced: 增强版图注意力模块 ====================
class CoDEM2_GAT_Enhanced(nn.Module):
    """增强版图注意力CoDEM2模块，添加全局上下文补偿提升召回率"""
    def __init__(self, channel_dim):
        super(CoDEM2_GAT_Enhanced, self).__init__()
        self.channel_dim = channel_dim

        self.shared_conv = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, 3, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True)
        )#可学习差分用的共享卷积

        self.conv_cat = nn.Conv2d(2*channel_dim, channel_dim, 1)#拼接后卷积

        self.conv_mix = nn.Conv2d(3*channel_dim, channel_dim, 1)  # 混合差分（Concat + |X1-X2| 三者拼接）

        
        # 原始CoDEM2结构
        self.Conv3 = nn.Conv2d(2*channel_dim, 2*channel_dim, kernel_size=3, stride=1, padding=1)
        self.Conv1 = nn.Conv2d(2*channel_dim, channel_dim, kernel_size=1, stride=1, padding=0)
        self.BN1 = nn.BatchNorm2d(2*channel_dim)
        self.BN2 = nn.BatchNorm2d(channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        
        # 图注意力机制
        self.graph_att = EnhancedGATv2(channel_dim)
        
        # 轻量化全局上下文补偿模块 (提升召回率)
        self.global_compensation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，获取全局信息
            nn.Conv2d(channel_dim, channel_dim//8, 1),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim//8, channel_dim, 1),  # 升维
            nn.Sigmoid()  # 生成0-1的权重
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        
        # 差异特征：计算两幅图像的绝对差值
        f_d = torch.abs(x1 - x2)#原差异差分

        # x1_proj = self.shared_conv(x1)
        # x2_proj = self.shared_conv(x2)
        # f_d = torch.abs(x1_proj - x2_proj)#这三行是可学习差分

        # f_cat = torch.cat([x1, x2], dim=1)
        # f_d = self.conv_cat(f_cat)#这两行是拼接后卷积

        # f_raw = torch.abs(x1 - x2)
        # f_cat = torch.cat([x1, x2, f_raw], dim=1)   # [B, 3C, H, W]
        # f_d = self.conv_mix(f_cat)#这三行是混合差分（Concat + |X1-X2| 三者拼接）
        
        # 连接特征：将两幅图像沿通道维度拼接
        f_c = torch.cat((x1, x2), dim=1)
        # 对连接特征进行卷积处理
        z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
        
        # 图注意力：应用GATv2生成注意力图
        att_map = self.graph_att(f_d)
        # 用sigmoid激活注意力图并与差异特征相乘
        z_d = f_d * att_map.sigmoid()
        
        # 全局补偿：增强弱变化信号
        global_comp = self.global_compensation(f_d)
        # 增强变化信号：原始特征 + 全局补偿
        z_d_enhanced = z_d * (1 + global_comp)
        
        # 融合输出：差异特征 + 连接特征
        out = z_d_enhanced + z_c
        return out

# ==================== EnhancedGATv2: 强化版图注意力 ====================
class EnhancedGATv2(nn.Module):
    """强化版GATv2：动态注意力+保留高维特征+多头注意力+残差连接+位置感知"""
    def __init__(self, in_channels, global_k=8, heads=4, dropout=0.3):
        super(EnhancedGATv2, self).__init__()
        self.in_channels = in_channels
        self.global_k = global_k  # 全局语义连接的top-k值
        self.heads = heads  # 多头注意力头数
        
        # 轻量降维（保留更多特征）
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),  # 1x1卷积降维
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # GATv2动态注意力层（关键改进）
        self.gatv2 = GATv2Conv(
            in_channels=in_channels // 2,  # 输入通道减半
            out_channels=in_channels // (2 * heads),  # 每个头的输出维度
            heads=heads,  # 多头注意力
            concat=True,  # 多头输出拼接
            dropout=dropout,  # 随机失活防止过拟合
            add_self_loops=True,  # 添加自连接
            edge_dim=None  # 不使用边特征
        )
        
        # 位置编码（增强空间感知）
        self.pos_enc = nn.Sequential(
            nn.Conv2d(2, in_channels // 2, kernel_size=1),  # 将2D坐标编码为特征
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 输出融合+残差
        self.output_proj = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels, 1),  # 恢复通道数
            nn.BatchNorm2d(in_channels),
            nn.Dropout(dropout)  # 输出前dropout
        )
        self.residual = nn.Conv2d(in_channels, in_channels, 1)  # 残差连接，保持维度

    def build_hybrid_graph(self, x):
        """构建局部-全局混合图：空间邻域+语义相似性"""
        batch_size, channels, height, width = x.size()
        device = x.device
        N = height * width  # 总节点数 = 像素数
        
        # 1. 局部空间连接（4邻域）
        local_edges = []
        for i in range(height):
            for j in range(width):
                node_idx = i * width + j  # 当前节点索引
                # 上邻接
                if i > 0:
                    local_edges.append((node_idx, (i-1)*width + j))
                # 下邻接
                if i < height-1:
                    local_edges.append((node_idx, (i+1)*width + j))
                # 左邻接
                if j > 0:
                    local_edges.append((node_idx, i*width + (j-1)))
                # 右邻接
                if j < width-1:
                    local_edges.append((node_idx, i*width + (j+1)))
        
        # 2. 全局语义连接（特征相似性top-k）
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, N, C]
        x_norm = F.normalize(x_flat, dim=-1)  # L2归一化
        
        # 计算特征相似度（仅用第一个样本构建图）
        sim_matrix = torch.matmul(x_norm[0], x_norm[0].t())  # [N, N] 余弦相似度
        sim_matrix.fill_diagonal_(-1e9)  # 排除自身，用极小值填充对角线
        _, topk_indices = torch.topk(sim_matrix, k=self.global_k, dim=1)  # [N, global_k]
        
        global_edges = []
        for i in range(N):
            for j in topk_indices[i]:
                global_edges.append((i, j.item()))  # 添加相似度最高的k个连接
        
        # 合并局部+全局边（GATv2需要双向边）
        all_edges = local_edges + global_edges
        
        # 添加反向边（确保对称性）
        reverse_edges = [(j, i) for i, j in all_edges]
        all_edges += reverse_edges
        
        # 添加自环（GATv2需要显式添加）
        self_loops = [(i, i) for i in range(N)]
        all_edges += self_loops
        
        # 去重
        unique_edges = list(set(all_edges))
        
        # 转换为tensor格式：2 x num_edges
        edge_index = torch.tensor(unique_edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        device = x.device
        
        # 生成坐标特征（用于位置感知）
        x_coord = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width).repeat(batch_size, 1, height, 1)
        y_coord = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1).repeat(batch_size, 1, 1, width)
        # 将坐标信息编码为特征
        coord_feat = self.pos_enc(torch.cat([x_coord, y_coord], dim=1))  # [B, C//2, H, W]
        
        # 特征降维+位置特征融合
        x_reduced = self.channel_reduction(x) + coord_feat  # [B, C//2, H, W]
        
        # 构建混合图（缓存以加速）
        if not hasattr(self, 'cached_edge_index') or self.cached_edge_index.size(1) != height * width:
            self.cached_edge_index = self.build_hybrid_graph(x_reduced)
        
        # 批处理节点特征
        N = height * width
        x_nodes = x_reduced.view(batch_size, -1, N).permute(0, 2, 1)  # [B, N, C//2]
        x_nodes = x_nodes.reshape(-1, x_reduced.size(1))  # [B*N, C//2]
        
        # 扩展边索引以匹配批处理
        edge_index = self.cached_edge_index
        if batch_size > 1:
            # 为每个样本复制图结构
            edge_index = self.repeat_edge_index_for_batch(edge_index, N, batch_size)
        
        # GATv2动态注意力（关键改进）
        att_nodes = self.gatv2(x_nodes, edge_index)  # [B*N, heads * out_channels]
        
        # 恢复特征图形状 [B, C//2, H, W]
        att_map = att_nodes.view(batch_size, N, -1).permute(0, 2, 1).view(batch_size, -1, height, width)
        
        # 输出投影+残差连接
        out = self.output_proj(att_map) + self.residual(x)
        return out

    def repeat_edge_index_for_batch(self, edge_index, num_nodes, batch_size):
        """将图结构复制到批处理中的每个样本"""
        offsets = torch.arange(0, batch_size * num_nodes, num_nodes, device=edge_index.device)
        offset_matrix = torch.stack([offsets, offsets]).unsqueeze(-1)
        repeated_edges = edge_index.unsqueeze(1) + offset_matrix
        return repeated_edges.view(2, -1).contiguous()

# ==================== CoordAtt: 坐标注意力机制 ====================
class CoordAtt(nn.Module):
    """坐标注意力机制，分别处理水平和垂直方向"""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向池化

        mip = max(8, inp // reduction)  # 中间层通道数，至少为8

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()  # 使用hard swish激活

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 水平方向卷积
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 垂直方向卷积

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1]

        y = torch.cat([x_h, x_w], dim=2)  # 拼接水平和垂直特征
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分割回水平和垂直特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # 水平注意力
        a_w = self.conv_w(x_w).sigmoid()  # 垂直注意力
        # 扩展到原始尺寸
        a_h = a_h.expand(-1, -1, h, w)
        a_w = a_w.expand(-1, -1, h, w)

        return a_w, a_h  # 返回两个方向的注意力图

# ==================== CoDEM2: 坐标嵌入的差异增强模块 ====================
class CoDEM2(nn.Module):
    '''增加坐标嵌入的版本（修复通道不匹配错误）'''
    def __init__(self, channel_dim):
        super(CoDEM2, self).__init__()
        self.channel_dim = channel_dim
        
        # 关键修正：Conv3的输入通道数 = 2*(原始通道数 + 坐标通道数)
        self.Conv3 = nn.Conv2d(
            in_channels=2 * (self.channel_dim + 2),  # 输入：2*(C+2)
            out_channels=2 * self.channel_dim,  # 输出：2C
            kernel_size=3, stride=1, padding=1
        )
        self.Conv1 = nn.Conv2d(
            in_channels=2 * self.channel_dim,  # 输入：2C
            out_channels=self.channel_dim,  # 输出：C
            kernel_size=1, stride=1, padding=0
        )
        self.BN1 = nn.BatchNorm2d(2 * self.channel_dim)
        self.BN2 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)  # 坐标注意力

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        
        # 生成坐标嵌入（2通道）
        x_coords = torch.linspace(-1, 1, W, device=x1.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        y_coords = torch.linspace(-1, 1, H, device=x1.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
        coord_emb = torch.cat([x_coords, y_coords], dim=1)  # [B, 2, H, W]
        
        # x1和x2各增加2个坐标通道
        x1_with_coord = torch.cat([x1, coord_emb], dim=1)  # [B, C+2, H, W]
        x2_with_coord = torch.cat([x2, coord_emb], dim=1)  # [B, C+2, H, W]
        
        # 拼接后通道数：(C+2) + (C+2) = 2C + 4
        f_c = torch.cat((x1_with_coord, x2_with_coord), dim=1)  # [B, 2C+4, H, W]
        
        # 对连接特征进行处理
        z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
        
        # 计算差异特征
        f_d = torch.abs(x1_with_coord - x2_with_coord)  # [B, C+2, H, W]
        
        # 仅对原始通道应用注意力（前C个通道）
        d_aw, d_ah = self.coAtt_1(f_d[:, :C, :, :])
        z_d = f_d[:, :C, :, :] * d_aw * d_ah
        
        # 融合输出
        out = z_d + z_c
        return out

# ==================== ConvLayer: 基本卷积层 ====================
class ConvLayer(nn.Module):
    """简单的卷积层封装"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out

# ==================== ResidualBlock: 残差块 ====================
class ResidualBlock(torch.nn.Module):
    """残差连接的基本块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x  # 保留输入
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1  # 缩放输出
        out = torch.add(out, residual)  # 残差连接
        return out

# ==================== ChannelAttention: 通道注意力 ====================
class ChannelAttention(nn.Module):
    """通道注意力机制（SE模块）"""
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()  # 生成0-1的权重

    def forward(self, x):
        # 平均池化路径
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        
        # 最大池化路径
        max_out_out = self.max_pool(x)
        max_out = self.fc2(self.relu1(self.fc1(max_out_out)))
        
        # 合并两条路径
        out = avg_out + max_out
        return self.sigmod(out)  # sigmoid激活得到通道权重

# ==================== ACFF2_TopK: 自适应特征融合模块（Top-K版） ====================
class ACFF2_TopK(nn.Module):
    '''基于Top-K图池化的自适应特征融合模块'''
    def __init__(self, channel_L, channel_H, pool_ratio=0.5, reduction_ratio=16):
        super(ACFF2_TopK, self).__init__()
        
        # 高层特征处理路径：1x1卷积调整通道数 + 上采样
        self.conv1 = nn.Conv2d(channel_H, channel_L, kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍上采样
        
        # 特征融合路径
        self.conv = nn.Conv2d(2 * channel_L, channel_L, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(channel_L)
        self.relu = nn.ReLU(inplace=True)
        
        # Top-K图池化参数
        self.pool_ratio = pool_ratio  # 池化比例
        self.reduction_ratio = reduction_ratio  # 降维比例
        
        # 用于计算节点重要性分数的层
        self.score_layer = nn.Conv2d(channel_L, 1, kernel_size=1)
        
        # 通道注意力MLP
        self.mlp = nn.Sequential(
            nn.Linear(channel_L, channel_L // reduction_ratio),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel_L // reduction_ratio, channel_L),  # 升维
            nn.Sigmoid()  # 生成权重
        )
        
        # 批归一化层，稳定训练
        self.bn = nn.BatchNorm1d(channel_L)

    def forward(self, f_low, f_high):
        """
        前向传播
        Args:
            f_low: 低层特征 [B, C_l, H, W] (高分辨率，细节丰富)
            f_high: 高层特征 [B, C_h, H/2, W/2] (低分辨率，语义丰富)
        Returns:
            out: 融合后的特征 [B, C_l, H, W]
        """
        # 处理高层特征：上采样 + 调整通道数
        f_high = self.relu(self.BN(self.conv1(self.up(f_high))))
        
        # 特征融合（加法方式）
        f_cat = f_high + f_low
        
        # 计算自适应权重 - 使用Top-K图池化
        batch_size, C, H, W = f_cat.size()
        num_nodes = H * W
        
        # 1. 将特征图转换为节点特征矩阵
        x_nodes = f_cat.view(batch_size, C, -1).permute(0, 2, 1)  # [B, N, C]
        
        # 2. 计算每个节点的重要性分数
        score = self.score_layer(f_cat).view(batch_size, -1)  # [B, N]
        
        # 3. Top-K 图池化：选择最重要的节点
        k = int(self.pool_ratio * num_nodes)  # 要保留的节点数
        topk_score, topk_indices = torch.topk(score, k=k, dim=1)  # [B, k]
        
        # 4. 使用softmax归一化分数，使其和为1
        topk_score = torch.softmax(topk_score, dim=1)  # [B, k]
        
        # 5. 根据索引选取最重要的节点特征
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        x_topk = torch.gather(x_nodes, dim=1, index=topk_indices_expanded)  # [B, k, C]
        
        # 6. 加权聚合：使用分数对选出的节点特征进行加权平均
        x_pooled = torch.sum(x_topk * topk_score.unsqueeze(-1), dim=1)  # [B, C]
        
        # 7. 批归一化稳定训练
        x_pooled = self.bn(x_pooled)
        
        # 8. 通过MLP生成最终的通道权重
        adaptive_w = self.mlp(x_pooled)  # [B, C]
        adaptive_w = adaptive_w.view(batch_size, C, 1, 1)  # [B, C, 1, 1]
        
        # 加权融合：低层特征权重 + 高层特征权重
        out = f_low * adaptive_w + f_high * (1 - adaptive_w)
        
        return out

# ==================== ACFF2: 自适应特征融合模块（简化版） ====================
class ACFF2(nn.Module):
    '''最新版本的ACFF 4.21,将cat改成+，去掉卷积'''
    def __init__(self, channel_L, channel_H):
        super(ACFF2, self).__init__()
        self.conv1 = nn.Conv2d(channel_H, channel_L, kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(2*channel_L, channel_L, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(channel_L)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(in_channels=channel_L, ratio=16)  # 通道注意力

    def forward(self, f_low, f_high):
        # 处理高层特征
        f_high = self.relu(self.BN(self.conv1(self.up(f_high))))
        # 特征融合
        f_cat = f_high + f_low  # 加法融合（原版是拼接）
        # 计算自适应权重
        adaptive_w = self.ca(f_cat)  # 通道注意力权重
        # 加权融合
        out = f_low * adaptive_w + f_high * (1 - adaptive_w)
        return out

# ==================== CatUP: 特征拼接上采样模块 ====================
class CatUP(nn.Module):
    """拼接特征后上采样的模块"""
    def __init__(self, channel_L, channel_H):
        super(CatUP, self).__init__()
        self.conv1 = nn.Conv2d(channel_H, channel_L, kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 拼接后的卷积处理
        self.Conv = nn.Sequential(
            nn.Conv2d(channel_L+channel_H, channel_L, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_L),
            nn.ReLU(),
            nn.Conv2d(channel_L, channel_L, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_L),
        )
        self.sigmod = nn.Sigmoid()
        self.ca = ChannelAttention(in_channels=channel_L, ratio=16)  # 通道注意力

    def forward(self, f_low, f_high):
        f_high = self.up(f_high)  # 上采样高层特征
        f_cat = torch.cat((f_low, f_high), dim=1)  # 拼接低层和高层特征
        out = self.Conv(f_cat)  # 卷积处理
        att = self.ca(out)  # 通道注意力
        output = att * out  # 注意力加权
        return output

# ==================== SupervisedAttentionModule: 监督注意力模块 ====================
class SupervisedAttentionModule(nn.Module):
    """带有CBAM注意力的监督模块"""
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        self.cbam = CBAM(channel=self.mid_d)  # CBAM注意力（通道+空间）
        # 后续卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        context = self.cbam(x)  # 应用CBAM注意力
        x_out = self.conv2(context)  # 卷积处理
        return x_out

# ==================== 辅助函数 ====================
def get_scheduler(optimizer, args):
    """返回学习率调度器"""
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)  # 线性衰减
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)  # 阶梯衰减
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler

class Identity(nn.Module):
    """恒等映射层"""
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """返回归一化层"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()  # 无归一化
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 卷积层和线性层的权重初始化
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)  # 正态分布
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)  # Xavier初始化
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # Kaiming初始化
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)  # 正交初始化
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)  # 偏置初始化为0
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm层初始化
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # 递归应用初始化函数

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """初始化网络并移动到GPU"""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])  # 移动到第一个GPU
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # 多GPU并行
    init_weights(net, init_type, init_gain=init_gain)
    return net

# ==================== 模型定义函数 ====================
def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """根据参数选择生成器模型"""
    if args.net_G == 'FC_EF':
        net = Unet(input_nbr=3, label_nbr=2)
    elif args.net_G == 'FC_Siam_conc':
        net = SiamUnet_conc(input_nbr=3, label_nbr=2)
    elif args.net_G == 'FC_Siam_diff':
        net = SiamUnet_diff(input_nbr=3, label_nbr=2)
    elif args.net_G == 'UNet++':
        net = NestedUNet(num_classes=2, input_channels=6, deep_supervision=True)
    elif args.net_G == 'SNUNet':
        net = SNUNet_ECAM(in_ch=3, out_ch=2)
    elif args.net_G == 'DTCDSCN':
        net = CDNet_model(in_channels=3)
    elif args.net_G == 'ChangeFormer':
        net = ChangeFormerV6(embed_dim=args.embed_dim)
    elif args.net_G == 'A2Net':
        net = A2Net(input_nc=3, output_nc=2)
    elif args.net_G == 'DMINet':
        net = DMINet(pretrained=True)
    elif args.net_G == 'IFNet':
        net = DSIFN()
    elif args.net_G == 'TFI-GR':
        net = TFI_GR(input_nc=3, output_nc=2)
    # 新网络
    elif args.net_G == 'SEIFNet':
        net = SEIFNet(args, input_nc=3, output_nc=2)  # 最终版的网络
    elif args.net_G == 'SMOWNet':
        # 加载预训练的 resnet18
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net = SMOW_Net(copy.deepcopy(resnet18))
    elif args.net_G == 'DVM_Net':
        net = DVM_Net(input_channel=3, input_size=args.img_size)   # input_size 可以从 args 读取，默认256
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)

# ==================== Backbone: 骨干网络基类 ====================
class Backbone(torch.nn.Module):
    """骨干网络基类"""
    def __init__(self, args, input_nc, output_nc,
                 resnet_stages_num=5,
                 output_sigmoid=False, if_upsample_2x=True):
        super(Backbone, self).__init__()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='nearest')  # 2倍上采样
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 4倍上采样
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)  # 分类器
        self.if_upsample_2x = if_upsample_2x  # 是否2倍上采样标志
        self.output_sigmoid = output_sigmoid  # 是否使用sigmoid输出
        self.sigmoid = nn.Sigmoid()  # sigmoid激活

    def forward_single0(self, x):
        """BIT网络的前向传播"""
        x = self.backbone(x)
        x0, x1, x2, x3 = x  # 解包特征
        if self.if_upsample_2x:
            x = self.upsamplex2(x2)  # 2倍上采样
        else:
            x = x2
        x = self.conv_pred(x)  # 分类卷积
        return x

    def forward_single(self, x):
        """SEIFNet的前向传播"""
        f = self.backbone(x)  # 提取特征
        return f

    def forward_down(self, x):
        """下采样"""
        f = self.downsample(x)
        return f

# ==================== DM: 差异模块 ====================
class DM(nn.Module):
    """差异模块 (DM) - 基于连接的方法"""
    def __init__(self, channel_dim):
        super(DM, self).__init__()
        self.channel_dim = channel_dim
        # 3×3卷积操作来学习最优距离度量
        self.conv = nn.Sequential(
            nn.Conv2d(2 * channel_dim, channel_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)  # 拼接特征
        out = self.conv(x_cat)  # 卷积处理
        return out

# ==================== SEIFNet: 主网络架构 ====================
class SEIFNet(Backbone):
    """4.4 最新版本，改进了Diff（DEM2_sobel)和ACFF2"""
    def __init__(self, args, input_nc, output_nc,
                 decoder_softmax=False, embed_dim=64,
                 Building_Bool=False):
        super(SEIFNet, self).__init__(args, input_nc, output_nc)

        self.stage_dims = [64, 128, 256, 512]  # 各阶段通道数
        self.output_nc = output_nc
        self.backbone = mobilenet_v2(pretrained=True)  # MobileNetV2骨干网络
        
        # 差异增强模块（不同层次使用不同版本）
        self.diff1 = CoDEM2(self.stage_dims[0])  # 低层：基础版
        self.diff2 = CoDEM2(self.stage_dims[1])  # 中低层：基础版
        self.diff3 = CoDEM2_GAT_Enhanced(self.stage_dims[2])  # 中层：增强版（带GAT）
        self.diff4 = CoDEM2_GAT_Enhanced(self.stage_dims[3])  # 高层：增强版（带GAT）
        
        # 自适应特征融合模块（不同层次使用不同配置）
        self.ACFF3 = ACFF2_TopK(
            channel_L=self.stage_dims[2], 
            channel_H=self.stage_dims[3],
            pool_ratio=0.4  # 保留40%的节点
        )
        self.ACFF2 = ACFF2_TopK(
            channel_L=self.stage_dims[1], 
            channel_H=self.stage_dims[2],
            pool_ratio=0.5  # 保留70%的节点
        )
        self.ACFF1 = ACFF2(channel_L=self.stage_dims[0], channel_H=self.stage_dims[1])  # 低层：基础版
        
        # 监督注意力模块（RM模块）
        self.sam_p4 = SupervisedAttentionModule(self.stage_dims[3])  # 高层
        self.sam_p3 = SupervisedAttentionModule(self.stage_dims[2])  # 中层
        self.sam_p2 = SupervisedAttentionModule(self.stage_dims[1])  # 中低层
        self.sam_p1 = SupervisedAttentionModule(self.stage_dims[0])  # 低层
        
        # 上采样层
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        # 通道调整卷积
        self.conv4 = nn.Conv2d(512, 64, kernel_size=1)  # 512通道→64通道
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)  # 256通道→64通道
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)  # 128通道→64通道
        
        # 最终分类卷积
        self.conv_final1 = nn.Conv2d(64, output_nc, kernel_size=1)  # 输出通道为类别数

    def forward(self, x1, x2):
        """前向传播：处理两个时相的图像"""
        # 提取特征（使用MobileNetV2）
        f1 = self.backbone(x1)  # 时相1的特征
        f2 = self.backbone(x2)  # 时相2的特征
        
        # 解包特征（各层次特征）
        x1_0, x1_1, x1_2, x1_3 = f1
        x2_0, x2_1, x2_2, x2_3 = f2
        
        # 差异增强（各层次分别处理）
        d1 = self.diff1(x1_0, x2_0)  # 低层差异
        d2 = self.diff2(x1_1, x2_1)  # 中低层差异
        d3 = self.diff3(x1_2, x2_2)  # 中层差异（GAT增强）
        d4 = self.diff4(x1_3, x2_3)  # 高层差异（GAT增强）
        
        # 自顶向下的特征融合（类似FPN）
        p4 = self.sam_p4(d4)  # 高层特征+注意力
        
        ACFF_43 = self.ACFF3(d3, p4)  # 融合中层和高层
        p3 = self.sam_p3(ACFF_43)  # 中层特征+注意力
        
        ACFF_32 = self.ACFF2(d2, p3)  # 融合中低层和中层
        p2 = self.sam_p2(ACFF_32)  # 中低层特征+注意力
        
        ACFF_21 = self.ACFF1(d1, p2)  # 融合低层和中低层
        p1 = self.sam_p1(ACFF_21)  # 低层特征+注意力
        
        # 多尺度特征融合（上采样+通道对齐）
        p4_up = self.upsample8(p4)  # 8倍上采样
        p4_up = self.conv4(p4_up)  # 通道对齐
        
        p3_up = self.upsample4(p3)  # 4倍上采样
        p3_up = self.conv3(p3_up)  # 通道对齐
        
        p2_up = self.upsample2(p2)  # 2倍上采样
        p2_up = self.conv2(p2_up)  # 通道对齐
        
        # 特征金字塔融合（所有层次相加）
        p = p1 + p2_up + p3_up + p4_up
        
        # 最终上采样
        p_up = self.upsample4(p)  # 4倍上采样到原图尺寸
        
        # 分类输出
        output = self.conv_final1(p_up)
        
        return output

# ==================== 其他特征提取模块 ====================
def Local_Attention(in_channel, r):
    """局部注意力模块"""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel/r), kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(int(in_channel/r)),
        nn.ReLU(),
        nn.Conv2d(in_channels=int(in_channel / r), out_channels=in_channel, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(in_channel)
    )

class Global_Attention(nn.Module):
    """全局注意力模块"""
    def __init__(self, in_channel=64, r=2):
        super(Global_Attention, self).__init__()
        self.PWConv_1 = nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel/r), kernel_size=1, stride=1, padding=0)
        self.PWConv_2 = nn.Conv2d(in_channels=int(in_channel /r), out_channels=in_channel, kernel_size=1, stride=1, padding=0)
        self.ReLU = nn.ReLU()
        self.BN_1 = nn.BatchNorm2d(int(in_channel/r))
        self.BN_2 = nn.BatchNorm2d(in_channel)

    def forward(self, input):
        x = nn.functional.adaptive_avg_pool2d(input, (1,1))  # 全局平均池化
        x = self.PWConv_1(x)  # 点卷积降维
        x = self.BN_1(x)
        x = self.ReLU(x)
        x = self.PWConv_2(x)  # 点卷积升维
        x = self.BN_2(x)
        x = self.ReLU(x)
        y = x.expand_as(input)  # 扩展到原始尺寸
        return y

class FDEM(nn.Module):
    """ST-DEM对比方法-2022 EGDE-Net"""
    def __init__(self, channel_dim):
        super(FDEM, self).__init__()
        reduction = 2
        self.channel_dim = channel_dim
        self.c_r = int(channel_dim/reduction)  # 降维后的通道数
        self.Local = Local_Attention(in_channel=self.channel_dim, r=reduction)  # 局部注意力
        self.Global = Global_Attention(in_channel=self.channel_dim, r=reduction)  # 全局注意力
        self.sigmod = nn.Sigmoid()

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)  # 差异特征
        f_l = self.Local(diff)  # 局部特征
        f_g = self.Global(diff)  # 全局特征
        w = self.sigmod(f_l + f_g)  # 权重融合
        output = w * diff + (1 - w) * diff  # 加权输出
        return output

class DifferenceModule(nn.Module):
    '''ST-DEM对比方法-2022 ChangeFormer'''
    def __init__(self, channel_dim):
        super(DifferenceModule, self).__init__()
        self.channel_dim = channel_dim
        self.conv_diff = nn.Sequential(
            nn.Conv2d(2*self.channel_dim, self.channel_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.channel_dim),
            nn.Conv2d(self.channel_dim, self.channel_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        f_c = torch.cat((x1, x2), dim=1)  # 拼接特征
        out = self.conv_diff(f_c)  # 卷积处理
        return out

class DEFM(nn.Module):
    '''ST-DEM对比方法-2022 DEFM'''
    def __init__(self, in_channels, conv_cfg=None, norm_cfg=dict(type='IN'), act_cfg=dict(type='GELU')):
        super(DEFM, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels * 2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        f1 = self.conv1(x1)
        f2 = self.conv1(x2)
        fuse = torch.cat([f1, f2], dim=1)  # 特征拼接
        tpo = self.conv3(fuse)  # 生成光流
        f2_ = self.warp(x2, tpo)  # 图像扭曲
        output = f2_ + x1  # 残差连接
        return output

    @staticmethod
    def warp(x, flow):
        """图像扭曲函数（基于光流）"""
        n, c, h, w = x.size()
        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)  # 归一化因子
        # 创建网格
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm  # 添加光流偏移
        output = F.grid_sample(x, grid, align_corners=True)  # 双线性采样
        return output

def make_prediction(in_channels, out_channels):
    """简单的预测头"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )















































# # 消融实验注意力消融SE注意力---------------------------------------------------------------
# import torch
# import torch.nn as nn

# # SE注意力模块
# class SEAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEAttention, self).__init__()
#         assert isinstance(reduction, int) and reduction > 0, "reduction必须是正整数"
#         assert channel >= reduction, f"channel({channel})必须大于等于reduction({reduction})"
        
#         self.reduction = reduction
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         mid_channels = max(1, channel // reduction)
        
#         self.fc = nn.Sequential(
#             nn.Linear(channel, mid_channels, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid_channels, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# # 使用SE注意力的CoDEM2模块
# class CoDEM2_SE(nn.Module):
#     def __init__(self, channel_dim, reduction=16):
#         super(CoDEM2_SE, self).__init__()
#         self.channel_dim = channel_dim
#         assert isinstance(reduction, int) and reduction > 0, "reduction必须是正整数"
        
#         self.Conv3 = nn.Conv2d(2 * (self.channel_dim + 2), 2 * self.channel_dim, kernel_size=3, stride=1, padding=1)
#         self.Conv1 = nn.Conv2d(2 * self.channel_dim, self.channel_dim, kernel_size=1, stride=1, padding=0)
#         self.BN1 = nn.BatchNorm2d(2 * self.channel_dim)
#         self.BN2 = nn.BatchNorm2d(self.channel_dim)
#         self.ReLU = nn.ReLU(inplace=True)
#         self.attention = SEAttention(channel_dim, reduction)

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
        
#         # 生成坐标嵌入
#         x_coords = torch.linspace(-1, 1, W, device=x1.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
#         y_coords = torch.linspace(-1, 1, H, device=x1.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
#         coord_emb = torch.cat([x_coords, y_coords], dim=1)
        
#         # 添加坐标信息
#         x1_with_coord = torch.cat([x1, coord_emb], dim=1)
#         x2_with_coord = torch.cat([x2, coord_emb], dim=1)
        
#         # 连接特征处理
#         f_c = torch.cat((x1_with_coord, x2_with_coord), dim=1)
#         z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
        
#         # 差异特征处理
#         f_d = torch.abs(x1_with_coord - x2_with_coord)
#         z_d = self.attention(f_d[:, :C, :, :])
        
#         out = z_d + z_c
#         return out

# # 使用本地ResNet的SEIFNet_SE类
# class SEIFNet_SE(nn.Module):
#     """
#     使用SE注意力的SEIFNet版本 - 使用本地ResNet实现
#     """

#     def __init__(self, args, reduction=16):
#         super(SEIFNet_SE, self).__init__()
        
#         # 从args中获取输入输出通道数
#         input_nc = getattr(args, 'input_nc', 3)  # 默认值为3
#         output_nc = getattr(args, 'n_class', 2)  # 使用n_class而不是output_nc，默认值为2
        
#         # 验证reduction参数
#         assert isinstance(reduction, int) and reduction > 0, "reduction必须是正整数"

#         self.stage_dims = [64, 128, 256, 512]
#         self.output_nc = output_nc
        
#         # 使用本地的ResNet18作为骨干网络
#         from models.resnet import resnet18
#         self.backbone = resnet18(pretrained=True)
        
#         # 修改ResNet的第一层以匹配输入通道数
#         if input_nc != 3:
#             print(f"调整ResNet第一层从3通道到{input_nc}通道")
#             # 保存原始权重
#             original_conv1_weight = self.backbone.conv1.weight.data
            
#             # 创建新的卷积层
#             new_conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
#             # 初始化新权重
#             if input_nc > 3:
#                 # 如果输入通道更多，重复使用原始权重
#                 new_conv1_weight = original_conv1_weight.repeat(1, input_nc // 3, 1, 1)
#                 if input_nc % 3 != 0:
#                     # 处理余数
#                     extra_channels = input_nc % 3
#                     extra_weight = original_conv1_weight[:, :extra_channels, :, :]
#                     new_conv1_weight = torch.cat([new_conv1_weight, extra_weight], dim=1)
#             else:
#                 # 如果输入通道更少，取前input_nc个通道
#                 new_conv1_weight = original_conv1_weight[:, :input_nc, :, :]
            
#             new_conv1.weight.data = new_conv1_weight
#             self.backbone.conv1 = new_conv1
        
#         # 使用SE注意力的DEM模块，传递reduction参数
#         self.diff1 = CoDEM2_SE(self.stage_dims[0], reduction)
#         self.diff2 = CoDEM2_SE(self.stage_dims[1], reduction)
#         self.diff3 = CoDEM2_SE(self.stage_dims[2], reduction)
#         self.diff4 = CoDEM2_SE(self.stage_dims[3], reduction)

#         # 上采样和卷积层
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
#         self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')

#         self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
#         self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
#         self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

#         # 确保输出2个通道，与标签匹配
#         self.conv_final1 = nn.Conv2d(64, output_nc, kernel_size=1)

#     def forward(self, x1, x2):
#         # 提取骨干网络的特征
#         f1 = self.backbone(x1)  # 返回 (x1_0, x1_1, x1_2, x1_3)
#         f2 = self.backbone(x2)  # 返回 (x2_0, x2_1, x2_2, x2_3)

#         x1_0, x1_1, x1_2, x1_3 = f1
#         x2_0, x2_1, x2_2, x2_3 = f2

#         # 差分特征提取
#         d1 = self.diff1(x1_0, x2_0)
#         d2 = self.diff2(x1_1, x2_1)
#         d3 = self.diff3(x1_2, x2_2)
#         d4 = self.diff4(x1_3, x2_3)

#         # 多尺度特征融合
#         p4_up = self.upsample8(d4)
#         p4_up = self.conv4(p4_up)

#         p3_up = self.upsample4(d3)
#         p3_up = self.conv3(p3_up)

#         p2_up = self.upsample2(d2)
#         p2_up = self.conv2(p2_up)

#         p = d1 + p2_up + p3_up + p4_up

#         # 最终上采样和输出
#         p_up = self.upsample4(p)
#         output = self.conv_final1(p_up)

#         return output
# #消融实验全部换成CBAM注意力--------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # CBAM注意力模块
# class CBAMAttention(nn.Module):
#     """
#     Convolutional Block Attention Module
#     论文: https://arxiv.org/abs/1807.06521
    
#     参数:
#         channel: 输入特征图的通道数
#         reduction: 通道注意力中的维度缩减比例，默认16
#         kernel_size: 空间注意力中的卷积核大小，默认7
#     """
#     def __init__(self, channel, reduction=16, kernel_size=7):
#         super(CBAMAttention, self).__init__()
        
#         # 通道注意力
#         self.channel_attention = ChannelAttention(channel, reduction)
        
#         # 空间注意力
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         # 先应用通道注意力
#         x = self.channel_attention(x)
        
#         # 再应用空间注意力
#         x = self.spatial_attention(x)
        
#         return x

# # 通道注意力子模块
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return x * self.sigmoid(out)

# # 空间注意力子模块
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size % 2 == 1, "Kernel size must be odd"
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 输入x的形状: [B, C, H, W]
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
#         x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
#         attention_map = self.conv(x_cat)  # [B, 1, H, W]
#         attention_map = self.sigmoid(attention_map)
        
#         # 应用注意力权重到所有通道
#         return x * attention_map  # 广播机制: [B, C, H, W] * [B, 1, H, W]
# # 使用CBAM注意力的CoDEM2模块
# class CoDEM2_CBAM(nn.Module):
#     '''
#     使用CBAM注意力替换的CoDEM2版本
#     '''
#     def __init__(self, channel_dim, reduction=16, kernel_size=7):
#         super(CoDEM2_CBAM, self).__init__()
#         self.channel_dim = channel_dim
        
#         # 验证参数
#         assert isinstance(reduction, int) and reduction > 0, "reduction必须是正整数"
#         assert isinstance(kernel_size, int) and kernel_size > 0, "kernel_size必须是正整数"
        
#         self.Conv3 = nn.Conv2d(
#             in_channels=2 * (self.channel_dim + 2),
#             out_channels=2 * self.channel_dim,
#             kernel_size=3, stride=1, padding=1
#         )
#         self.Conv1 = nn.Conv2d(
#             in_channels=2 * self.channel_dim,
#             out_channels=self.channel_dim,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.BN1 = nn.BatchNorm2d(2 * self.channel_dim)
#         self.BN2 = nn.BatchNorm2d(self.channel_dim)
#         self.ReLU = nn.ReLU(inplace=True)
        
#         # 使用CBAM注意力
#         self.attention = CBAMAttention(channel_dim, reduction, kernel_size)

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
        
#         # 生成坐标嵌入
#         x_coords = torch.linspace(-1, 1, W, device=x1.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
#         y_coords = torch.linspace(-1, 1, H, device=x1.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
#         coord_emb = torch.cat([x_coords, y_coords], dim=1)
        
#         # 添加坐标信息
#         x1_with_coord = torch.cat([x1, coord_emb], dim=1)
#         x2_with_coord = torch.cat([x2, coord_emb], dim=1)
        
#         # 连接特征处理
#         f_c = torch.cat((x1_with_coord, x2_with_coord), dim=1)
#         z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
        
#         # 差异特征处理
#         f_d = torch.abs(x1_with_coord - x2_with_coord)
#         z_d = self.attention(f_d[:, :C, :, :])  # 应用CBAM注意力
        
#         out = z_d + z_c
#         return out

# # 使用CBAM注意力的ACFF模块
# class ACFF2_CBAM(nn.Module):
#     '''
#     使用CBAM注意力的ACFF2模块
#     '''
#     def __init__(self, channel_L, channel_H, reduction=16, kernel_size=7):
#         super(ACFF2_CBAM, self).__init__()
#         self.conv1 = nn.Conv2d(channel_H, channel_L, kernel_size=1)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
#         # 使用CBAM注意力
#         self.attention = CBAMAttention(channel_L, reduction, kernel_size)

#     def forward(self, f_low, f_high):
#         f_high_up = self.up(self.conv1(f_high))
#         f_cat = f_high_up + f_low
#         adaptive_w = self.attention(f_cat)
#         return f_low * adaptive_w + f_high_up * (1 - adaptive_w)

# # 使用CBAM注意力的SEIFNet主类

# class SEIFNet_CBAM(nn.Module):
#     """
#     使用CBAM注意力的SEIFNet版本 - 用于消融实验
#     """

#     def __init__(self, args, reduction=16, kernel_size=7):
#         super(SEIFNet_CBAM, self).__init__()
        
#         # 从args中获取输入输出通道数
#         input_nc = getattr(args, 'input_nc', 3)  # 默认值为3
#         output_nc = getattr(args, 'n_class', 2)  # 使用n_class而不是output_nc，默认值为2
        
#         # 验证参数
#         assert isinstance(reduction, int) and reduction > 0, "reduction必须是正整数"
#         assert isinstance(kernel_size, int) and kernel_size > 0, "kernel_size必须是正整数"

#         self.stage_dims = [64, 128, 256, 512]
#         self.output_nc = output_nc
        
#         # 使用本地的ResNet18作为骨干网络
#         from models.resnet import resnet18
#         self.backbone = resnet18(pretrained=True)
        
#         # 修改ResNet的第一层以匹配输入通道数
#         if input_nc != 3:
#             print(f"调整ResNet第一层从3通道到{input_nc}通道")
#             # 保存原始权重
#             original_conv1_weight = self.backbone.conv1.weight.data
            
#             # 创建新的卷积层
#             new_conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
#             # 初始化新权重
#             if input_nc > 3:
#                 # 如果输入通道更多，重复使用原始权重
#                 new_conv1_weight = original_conv1_weight.repeat(1, input_nc // 3, 1, 1)
#                 if input_nc % 3 != 0:
#                     # 处理余数
#                     extra_channels = input_nc % 3
#                     extra_weight = original_conv1_weight[:, :extra_channels, :, :]
#                     new_conv1_weight = torch.cat([new_conv1_weight, extra_weight], dim=1)
#             else:
#                 # 如果输入通道更少，取前input_nc个通道
#                 new_conv1_weight = original_conv1_weight[:, :input_nc, :, :]
            
#             new_conv1.weight.data = new_conv1_weight
#             self.backbone.conv1 = new_conv1
        
#         # 使用CBAM注意力的DEM模块
#         self.diff1 = CoDEM2_CBAM(self.stage_dims[0], reduction, kernel_size)
#         self.diff2 = CoDEM2_CBAM(self.stage_dims[1], reduction, kernel_size)
#         self.diff3 = CoDEM2_CBAM(self.stage_dims[2], reduction, kernel_size)
#         self.diff4 = CoDEM2_CBAM(self.stage_dims[3], reduction, kernel_size)

#         # 使用CBAM注意力的ACFF模块
#         self.ACFF3 = ACFF2_CBAM(channel_L=self.stage_dims[2], channel_H=self.stage_dims[3], 
#                                reduction=reduction, kernel_size=kernel_size)
#         self.ACFF2 = ACFF2_CBAM(channel_L=self.stage_dims[1], channel_H=self.stage_dims[2], 
#                                reduction=reduction, kernel_size=kernel_size)
#         self.ACFF1 = ACFF2_CBAM(channel_L=self.stage_dims[0], channel_H=self.stage_dims[1], 
#                                reduction=reduction, kernel_size=kernel_size)
        
#         # 监督注意力模块（保持原有实现）
#         self.sam_p4 = SupervisedAttentionModule(self.stage_dims[3])
#         self.sam_p3 = SupervisedAttentionModule(self.stage_dims[2])
#         self.sam_p2 = SupervisedAttentionModule(self.stage_dims[1])
#         self.sam_p1 = SupervisedAttentionModule(self.stage_dims[0])

#         # 上采样和卷积层
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
#         self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')

#         self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
#         self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
#         self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

#         # 确保输出2个通道，与标签匹配
#         self.conv_final1 = nn.Conv2d(64, output_nc, kernel_size=1)

#     def forward(self, x1, x2):
#         # 提取骨干网络的特征
#         f1 = self.backbone(x1)  # 返回 (x1_0, x1_1, x1_2, x1_3)
#         f2 = self.backbone(x2)  # 返回 (x2_0, x2_1, x2_2, x2_3)

#         x1_0, x1_1, x1_2, x1_3 = f1
#         x2_0, x2_1, x2_2, x2_3 = f2

#         # 差分特征提取
#         d1 = self.diff1(x1_0, x2_0)
#         d2 = self.diff2(x1_1, x2_1)
#         d3 = self.diff3(x1_2, x2_2)
#         d4 = self.diff4(x1_3, x2_3)

#         # 监督注意力
#         p4 = self.sam_p4(d4)

#         # 特征融合
#         ACFF_43 = self.ACFF3(d3, p4)
#         p3 = self.sam_p3(ACFF_43)

#         ACFF_32 = self.ACFF2(d2, p3)
#         p2 = self.sam_p2(ACFF_32)

#         ACFF_21 = self.ACFF1(d1, p2)
#         p1 = self.sam_p1(ACFF_21)

#         # 多尺度特征融合
#         p4_up = self.upsample8(p4)
#         p4_up = self.conv4(p4_up)

#         p3_up = self.upsample4(p3)
#         p3_up = self.conv3(p3_up)

#         p2_up = self.upsample2(p2)
#         p2_up = self.conv2(p2_up)

#         p = p1 + p2_up + p3_up + p4_up

#         # 最终上采样和输出
#         p_up = self.upsample4(p)
#         output = self.conv_final1(p_up)

#         return output

# # 监督注意力模块（保持原有实现）
# class SupervisedAttentionModule(nn.Module):
#     def __init__(self, mid_d):
#         super(SupervisedAttentionModule, self).__init__()
#         self.mid_d = mid_d

#         self.cbam = CBAM(channel = self.mid_d)

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(self.mid_d),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         context = self.cbam(x)
#         x_out = self.conv2(context)
#         return x_out

# # CBAM模块（保持原有实现）
# class CBAM(nn.Module):
#     def __init__(self, channel, reduction=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(channel, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x
    

# # 融合模块消融-------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# import numpy as np

# # ------------------------------------------------------------------
# class CoDEM2_GAT_Enhanced(nn.Module):
#     """
#     增强版图注意力CoDEM2模块
#     添加全局上下文补偿模块提升召回率
#     """
#     def __init__(self, channel_dim):
#         super(CoDEM2_GAT_Enhanced, self).__init__()
#         self.channel_dim = channel_dim
        
#         # 原始CoDEM2结构
#         self.Conv3 = nn.Conv2d(2*channel_dim, 2*channel_dim, kernel_size=3, stride=1, padding=1)
#         self.Conv1 = nn.Conv2d(2*channel_dim, channel_dim, kernel_size=1, stride=1, padding=0)
#         self.BN1 = nn.BatchNorm2d(2*channel_dim)
#         self.BN2 = nn.BatchNorm2d(channel_dim)
#         self.ReLU = nn.ReLU(inplace=True)
        
#         # 图注意力机制
#         self.graph_att = EnhancedGATv2(channel_dim)
        
#         # 轻量化全局上下文补偿模块 (提升召回率)
#         self.global_compensation = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # 全局平均池化
#             nn.Conv2d(channel_dim, channel_dim//8, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel_dim//8, channel_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
        
#         # 差异特征
#         f_d = torch.abs(x1 - x2)
        
#         # 连接特征
#         f_c = torch.cat((x1, x2), dim=1)
#         z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
        
#         # 图注意力
#         att_map = self.graph_att(f_d)
#         z_d = f_d * att_map.sigmoid()
        
#         # 全局补偿 (增强弱变化信号)
#         global_comp = self.global_compensation(f_d)
#         z_d_enhanced = z_d * (1 + global_comp)  # 增强变化信号
        
#         # 融合输出
#         out = z_d_enhanced + z_c
#         return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv  # 使用GATv2替代原始GAT

# class EnhancedGATv2(nn.Module):
#     """强化版GATv2：动态注意力+保留高维特征+多头注意力+残差连接+位置感知"""
#     def __init__(self, in_channels, global_k=8, heads=4, dropout=0.3):
#         super(EnhancedGATv2, self).__init__()
#         self.in_channels = in_channels
#         self.global_k = global_k  # 全局语义连接的top-k
#         self.heads = heads
        
#         # 轻量降维（保留更多特征）
#         self.channel_reduction = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, 1),
#             nn.BatchNorm2d(in_channels // 2),
#             nn.ReLU(inplace=True)
#         )
        
#         # GATv2动态注意力层（关键改进）
#         self.gatv2 = GATv2Conv(
#             in_channels=in_channels // 2,
#             out_channels=in_channels // (2 * heads),  # 每个头的输出维度
#             heads=heads,
#             concat=True,  # 多头输出拼接
#             dropout=dropout,
#             add_self_loops=True,  # GATv2需要显式添加自环
#             edge_dim=None  # 不使用边特征
#         )
        
#         # 位置编码（增强空间感知）
#         self.pos_enc = nn.Sequential(
#             nn.Conv2d(2, in_channels // 2, kernel_size=1),
#             nn.BatchNorm2d(in_channels // 2),
#             nn.ReLU(inplace=True)
#         )
        
#         # 输出融合+残差
#         self.output_proj = nn.Sequential(
#             nn.Conv2d(in_channels // 2, in_channels, 1),
#             nn.BatchNorm2d(in_channels),
#             nn.Dropout(dropout)
#         )
#         self.residual = nn.Conv2d(in_channels, in_channels, 1)  # 残差连接

#     def build_hybrid_graph(self, x):
#         """构建局部-全局混合图：空间邻域+语义相似性"""
#         batch_size, channels, height, width = x.size()
#         device = x.device
#         N = height * width  # 总节点数
        
#         # 1. 局部空间连接（4邻域）
#         local_edges = []
#         for i in range(height):
#             for j in range(width):
#                 node_idx = i * width + j
#                 # 上
#                 if i > 0:
#                     local_edges.append((node_idx, (i-1)*width + j))
#                 # 下
#                 if i < height-1:
#                     local_edges.append((node_idx, (i+1)*width + j))
#                 # 左
#                 if j > 0:
#                     local_edges.append((node_idx, i*width + (j-1)))
#                 # 右
#                 if j < width-1:
#                     local_edges.append((node_idx, i*width + (j+1)))
        
#         # 2. 全局语义连接（特征相似性top-k）
#         x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, N, C]
#         x_norm = F.normalize(x_flat, dim=-1)
        
#         # 计算特征相似度（仅用第一个样本构建图）
#         sim_matrix = torch.matmul(x_norm[0], x_norm[0].t())  # [N, N]
#         sim_matrix.fill_diagonal_(-1e9)  # 排除自身
#         _,topk_indices = torch.topk(sim_matrix, k=self.global_k, dim=1)  # [N, global_k]
        
#         global_edges = []
#         for i in range(N):
#             for j in topk_indices[i]:
#                 global_edges.append((i, j.item()))
        
#         # 合并局部+全局边（GATv2需要双向边）
#         all_edges = local_edges + global_edges
        
#         # 添加反向边（确保对称性）
#         reverse_edges = [(j, i) for i, j in all_edges]
#         all_edges += reverse_edges
        
#         # 添加自环（GATv2需要显式添加）
#         self_loops = [(i, i) for i in range(N)]
#         all_edges += self_loops
        
#         # 去重
#         unique_edges = list(set(all_edges))
        
#         # 转换为tensor格式
#         edge_index = torch.tensor(unique_edges, dtype=torch.long, device=device).t().contiguous()
#         return edge_index

#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         device = x.device
        
#         # 生成坐标特征（用于位置感知）
#         x_coord = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width).repeat(batch_size, 1, height, 1)
#         y_coord = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1).repeat(batch_size, 1, 1, width)
#         coord_feat = self.pos_enc(torch.cat([x_coord, y_coord], dim=1))  # [B, C//2, H, W]
        
#         # 特征降维+位置特征融合
#         x_reduced = self.channel_reduction(x) + coord_feat  # [B, C//2, H, W]
        
#         # 构建混合图（缓存以加速）
#         if not hasattr(self, 'cached_edge_index') or self.cached_edge_index.size(1) != height * width:
#             self.cached_edge_index = self.build_hybrid_graph(x_reduced)
        
#         # 批处理节点特征
#         N = height * width
#         x_nodes = x_reduced.view(batch_size, -1, N).permute(0, 2, 1)  # [B, N, C//2]
#         x_nodes = x_nodes.reshape(-1, x_reduced.size(1))  # [B*N, C//2]
        
#         # 扩展边索引以匹配批处理
#         edge_index = self.cached_edge_index
#         if batch_size > 1:
#             # 为每个样本复制图结构
#             edge_index = self.repeat_edge_index_for_batch(edge_index, N, batch_size)
        
#         # GATv2动态注意力（关键改进）
#         att_nodes = self.gatv2(x_nodes, edge_index)  # [B*N, heads * out_channels]
        
#         # 恢复特征图形状 [B, C//2, H, W]
#         att_map = att_nodes.view(batch_size, N, -1).permute(0, 2, 1).view(batch_size, -1, height, width)
        
#         # 输出投影+残差连接
#         out = self.output_proj(att_map) + self.residual(x)
#         return out

#     def repeat_edge_index_for_batch(self, edge_index, num_nodes, batch_size):
#         """将图结构复制到批处理中的每个样本"""
#         offsets = torch.arange(0, batch_size * num_nodes, num_nodes, device=edge_index.device)
#         offset_matrix = torch.stack([offsets, offsets]).unsqueeze(-1)
#         repeated_edges = edge_index.unsqueeze(1) + offset_matrix
#         return repeated_edges.view(2, -1).contiguous()
# #---------------------------------------------------------------------------------------


# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))

#         mip = max(8, inp // reduction)

#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()

#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):

#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)

#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)

#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)

#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         a_h = a_h.expand(-1,-1,h,w)
#         a_w = a_w.expand(-1, -1, h, w)

#         # out = identity * a_w * a_h

#         return a_w , a_h
# class CoDEM2(nn.Module):
#     '''
#     增加坐标嵌入的版本（修复通道不匹配错误）
#     '''
#     def __init__(self, channel_dim):
#         super(CoDEM2, self).__init__()
#         self.channel_dim = channel_dim
        
#         # 关键修正：Conv3的输入通道数 = 2*(原始通道数 + 坐标通道数)
#         # 因为x1和x2各带2个坐标通道，拼接后总通道数为2*(channel_dim + 2)
#         self.Conv3 = nn.Conv2d(
#             in_channels=2 * (self.channel_dim + 2),  # 修正：从2*channel_dim + 2 → 2*(channel_dim + 2)
#             out_channels=2 * self.channel_dim,
#             kernel_size=3, stride=1, padding=1
#         )
#         self.Conv1 = nn.Conv2d(
#             in_channels=2 * self.channel_dim,
#             out_channels=self.channel_dim,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.BN1 = nn.BatchNorm2d(2 * self.channel_dim)
#         self.BN2 = nn.BatchNorm2d(self.channel_dim)
#         self.ReLU = nn.ReLU(inplace=True)
#         self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape  # C = channel_dim
        
#         # 生成坐标嵌入（2通道）
#         x_coords = torch.linspace(-1, 1, W, device=x1.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
#         y_coords = torch.linspace(-1, 1, H, device=x1.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
#         coord_emb = torch.cat([x_coords, y_coords], dim=1)  # [B, 2, H, W]
        
#         # x1和x2各增加2个坐标通道（变为 C+2 通道）
#         x1_with_coord = torch.cat([x1, coord_emb], dim=1)  # [B, C+2, H, W]
#         x2_with_coord = torch.cat([x2, coord_emb], dim=1)  # [B, C+2, H, W]
        
#         # 拼接后通道数：(C+2) + (C+2) = 2C + 4（与Conv3的in_channels匹配）
#         f_c = torch.cat((x1_with_coord, x2_with_coord), dim=1)  # [B, 2C+4, H, W]
        
#         # 后续处理（此时Conv3输入通道匹配）
#         z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
#         # 计算差异特征f_d（关键修复）
#         f_d = torch.abs(x1_with_coord - x2_with_coord)  # [B, C+2, H, W]
#         d_aw, d_ah = self.coAtt_1(f_d[:, :C, :, :])  # 仅对原始通道应用注意力
#         z_d = f_d[:, :C, :, :] * d_aw * d_ah
#         out = z_d + z_c
#         return out
# class SEIFNet_PlainFusion(Backbone):
#     """
#     将ACFF模块替换为普通融合模块的版本，用于相融实验
#     """

#     def __init__(self, args, input_nc, output_nc,
#                  decoder_softmax=False, embed_dim=64,
#                  Building_Bool=False):
#         super(SEIFNet_PlainFusion, self).__init__(args, input_nc, output_nc)

#         self.stage_dims = [64, 128, 256, 512]
#         self.output_nc = output_nc
#         self.backbone = resnet.resnet18(pretrained=True)

#         # 保持原有的差异增强模块不变
#         self.diff1 = CoDEM2(self.stage_dims[0])
#         self.diff2 = CoDEM2(self.stage_dims[1])
#         self.diff3 = CoDEM2_GAT_Enhanced(self.stage_dims[2])
#         self.diff4 = CoDEM2_GAT_Enhanced(self.stage_dims[3])

#         # 将ACFF模块替换为普通融合模块
#         # 使用简单的上采样+卷积+加法融合
#         self.fusion3 = PlainFusion(channel_L=self.stage_dims[2], channel_H=self.stage_dims[3])
#         self.fusion2 = PlainFusion(channel_L=self.stage_dims[1], channel_H=self.stage_dims[2])
#         self.fusion1 = PlainFusion(channel_L=self.stage_dims[0], channel_H=self.stage_dims[1])

#         # 保持SAM模块不变
#         self.sam_p4 = SupervisedAttentionModule(self.stage_dims[3])
#         self.sam_p3 = SupervisedAttentionModule(self.stage_dims[2])
#         self.sam_p2 = SupervisedAttentionModule(self.stage_dims[1])
#         self.sam_p1 = SupervisedAttentionModule(self.stage_dims[0])

#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
#         self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')

#         self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
#         self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
#         self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

#         self.conv_final1 = nn.Conv2d(64, output_nc, kernel_size=1)

#     def forward(self, x1, x2):
#         # 保持原有的特征提取流程不变
#         f1 = self.backbone(x1)
#         f2 = self.backbone(x2)

#         x1_0, x1_1, x1_2, x1_3 = f1
#         x2_0, x2_1, x2_2, x2_3 = f2

#         # 差异特征提取
#         d1 = self.diff1(x1_0, x2_0)
#         d2 = self.diff2(x1_1, x2_1)
#         d3 = self.diff3(x1_2, x2_2)
#         d4 = self.diff4(x1_3, x2_3)

#         p4 = self.sam_p4(d4)

#         # 使用普通融合模块替代ACFF
#         fusion_43 = self.fusion3(d3, p4)
#         p3 = self.sam_p3(fusion_43)

#         fusion_32 = self.fusion2(d2, p3)
#         p2 = self.sam_p2(fusion_32)

#         fusion_21 = self.fusion1(d1, p2)
#         p1 = self.sam_p1(fusion_21)

#         # 多尺度特征融合
#         p4_up = self.upsample8(p4)
#         p4_up = self.conv4(p4_up)

#         p3_up = self.upsample4(p3)
#         p3_up = self.conv3(p3_up)

#         p2_up = self.upsample2(p2)
#         p2_up = self.conv2(p2_up)

#         p = p1 + p2_up + p3_up + p4_up
#         p_up = self.upsample4(p)
#         output = self.conv_final1(p_up)

#         return output


# class PlainFusion(nn.Module):
#     """
#     普通融合模块：简单的上采样+卷积+加法融合
#     """

#     def __init__(self, channel_L, channel_H):
#         super(PlainFusion, self).__init__()
#         # 高层特征降维和上采样
#         self.conv_high = nn.Conv2d(channel_H, channel_L, kernel_size=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
#         # 可选：添加一个简单的卷积融合层
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(channel_L * 2, channel_L, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel_L),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, f_low, f_high):
#         # 高层特征处理
#         f_high_processed = self.upsample(self.conv_high(f_high))
        
#         # 简单加法融合
#         # out = f_low + f_high_processed
        
#         # 或者使用concat+conv融合
#         fused = torch.cat([f_low, f_high_processed], dim=1)
#         out = self.fusion_conv(fused)
        
#         return out
# ==================== 导入库和模块 ====================
