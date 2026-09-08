import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import Parameter
import torch.nn.functional as F


class SMOW_Net(nn.Module):
    def __init__(self, resnet18):
        super(SMOW_Net, self).__init__()
        self.resnet = ResNet3D(resnet18)

        self.OFW = OFW(32)

        self.Conv3d = BasicConv3d(64, 32)
        self.Conv3d1 = BasicConv3d(64, 32)
        self.Conv3d2 = BasicConv3d(128, 64)
        self.Conv3d3 = BasicConv3d(256, 128)
        self.Conv3d4 = BasicConv3d(512, 256)

        self.MaxPool = max_pooling_3d()

        self.C3DT1 = conv_trans_block_3d(256, 256)
        self.C3D1 = conv_block_2_3d(512, 128)
        self.C3DT2 = conv_trans_block_3d(128, 128)
        self.C3D2 = conv_block_2_3d(256, 64)
        self.C3DT3 = conv_trans_block_3d(64, 64)
        self.C3D3 = conv_block_2_3d(128, 64)
        self.C3DT4 = conv_trans_block_3d(64, 64)
        self.C3D4 = conv_block_2_3d(96, 32)
        self.C3DT5 = conv_trans_block_3d(32, 32)
        self.C3D5 = conv_block_2_3d(64, 32)

        self.Transformer_Encoder = Transformer_Encoder(in_chan=32)
        self.Transformer_Decoder = Transformer_Decoder(in_chan=128)
        self.decoder = Classifier(in_chan=128, n_class=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        x = torch.cat([x1, x2], 2)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)

        x0 = self.Conv3d(x0)
        x8 = self.OFW(x0)

        x8 = self.Transformer_Encoder(x8)
        
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x1 = self.Conv3d1(x1)
        x2 = self.Conv3d2(x2)
        x3 = self.Conv3d3(x3)
        x4 = self.Conv3d4(x4)

        b, c, t, h, w = x0.shape
        x0 = F.interpolate(x0, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x1.shape
        x1 = F.interpolate(x1, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x2.shape
        x2 = F.interpolate(x2, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x3.shape
        x3 = F.interpolate(x3, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x4.shape
        x4 = F.interpolate(x4, size=(4, h, w), mode='trilinear', align_corners=True)
        
        maxpool = self.MaxPool(x4)

        c3dt1 = self.C3DT1(maxpool)
        concat1 = torch.cat([c3dt1, x4], dim=1)
        c3d1 = self.C3D1(concat1)
        
        c3dt2 = self.C3DT2(c3d1)
        concat2 = torch.cat([c3dt2, x3], dim=1)
        c3d2 = self.C3D2(concat2)
        
        c3dt3 = self.C3DT3(c3d2)
        concat3 = torch.cat([c3dt3, x2], dim=1)
        c3d3 = self.C3D3(concat3)
        
        c3dt4 = self.C3DT4(c3d3)
        concat4 = torch.cat([c3dt4, x1], dim=1)
        c3d4 = self.C3D4(concat4)

        c3dt5 = self.C3DT5(c3d4)
        concat5 = torch.cat([c3dt5, x0], dim=1)
        c3d5 = self.C3D5(concat5)

        x16 = self.Transformer_Decoder(c3d5, x8)
        x16 = self.decoder(x16)
        x16 = self.sigmoid(x16)

        return x16


class conv_trans_block_3d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_trans_block_3d, self).__init__()
        self.conv3d_spatial = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), output_padding=(0, 1, 1))
        self.conv3d_time_1 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3d_time_2 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3d_time_3 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3d_time_4 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3d_time_5 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        torch.nn.init.constant_(self.conv3d_time_1.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_time_2.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_time_3.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_time_4.weight, 0.0)
        torch.nn.init.eye_(self.conv3d_time_5.weight[:, :, 0, 0, 0])
        self.batch = nn.BatchNorm3d(out_dim)
        self.leaky =  nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_spatial = self.conv3d_spatial(x)
        T1 = x_spatial[:, :, 0:1, :, :]
        T2 = x_spatial[:, :, 1:2, :, :]
        T3 = x_spatial[:, :, 2:3, :, :]
        T4 = x_spatial[:, :, 3:4, :, :]
        T1_F1 = self.conv3d_time_5(T1)
        T2_F1 = self.conv3d_time_5(T2)
        T3_F1 = self.conv3d_time_5(T3)
        T4_F1 = self.conv3d_time_5(T4)
        T1_F2 = self.conv3d_time_1(T1)
        T2_F2 = self.conv3d_time_2(T2)
        T3_F2 = self.conv3d_time_3(T3)
        T4_F2 = self.conv3d_time_4(T4)
        x = torch.cat([T1_F1 + T2_F2, T2_F1 + T3_F2,  T3_F1 + T4_F2,  T4_F1 + T1_F2], dim=2)
        x = self.batch(x)
        x = self.leaky(x)

        return x


class conv_block_2_3d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_block_2_3d, self).__init__()
        self.conv_block_2_3d = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_dim)
        )

    def forward(self, x):
        x = self.conv_block_2_3d(x)

        return x

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

class Transformer_Encoder(nn.Module):
    def __init__(self, in_chan=32, token_len=8, heads=8):
        super(Transformer_Encoder, self).__init__()
        self.token_len = token_len
        # This conv layer will now operate on individual time steps
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        # Positional embedding for each time step
        self.pos_embedding = nn.Parameter(torch.randn(4, token_len, in_chan))  # 4 for the time steps
        self.transformer = Transformer(dim=in_chan*4, depth=1, heads=heads, dim_head=in_chan*4, mlp_dim=in_chan*4, dropout=0)

    def forward(self, x):
        b, c, t, h, w = x.shape
        assert t == 4, "The time dimension (t) must be 4."
        
        # Process each time step independently
        tokens_all_time_steps = []
        for time_step in range(t):
            x_t = x[:, :, time_step, :, :]
            spatial_attention = self.conv_a(x_t)
            spatial_attention = spatial_attention.view(b, self.token_len, -1).contiguous()
            spatial_attention = torch.softmax(spatial_attention, dim=-1)
            x_t = x_t.view(b, c, -1).contiguous()
            tokens = torch.einsum('bln, bcn -> blc', spatial_attention, x_t)
            tokens += self.pos_embedding[time_step]
            tokens_all_time_steps.append(tokens)
        
        tokens_concat = torch.cat(tokens_all_time_steps, dim=2)  
        tokens_transformed = self.transformer(tokens_concat)

        return tokens_transformed


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x)

        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)


class Transformer_Decoder(nn.Module):
    def __init__(self, in_chan = 128, heads = 8):
        super(Transformer_Decoder, self).__init__()
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, t, h, w = x.shape
        x = x.view(b, c * t, h, w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)

        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2, **kwargs):

        return self.fn(x, x2, **kwargs) + x


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):

        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_chan, n_class, scale=2, pad=0):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, n_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        x = x_permuted.permute(0, 3, 2, 1)

        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_bn(x)

        return x


class Decompose_conv(nn.Module):
    def __init__(self, conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False):
        super(Decompose_conv, self).__init__()
        self.time_dim = time_dim
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        if time_dim == 1:
            self.conv3d = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_dim, padding=padding,
                                          dilation=dilation, stride=stride)

            weight_2d = conv2d.weight.data
            weight_3d = weight_2d.unsqueeze(2)

            self.conv3d.weight = Parameter(weight_3d)
            self.conv3d.bias = conv2d.bias
        else:
            self.conv3d_spatial = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels,
                                                  kernel_size=(1, kernel_dim[1], kernel_dim[2]),
                                                  padding=(0, padding[1], padding[2]),
                                                  dilation=(1, dilation[1], dilation[2]),
                                                  stride=(1, stride[1], stride[2])
                                                  )
            weight_2d = conv2d.weight.data
            self.conv3d_spatial.weight = Parameter(weight_2d.unsqueeze(2))
            self.conv3d_spatial.bias = conv2d.bias
            self.conv3d_time_1 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            self.conv3d_time_2 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            self.conv3d_time_3 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            torch.nn.init.constant_(self.conv3d_time_1.weight, 0.0)
            torch.nn.init.constant_(self.conv3d_time_3.weight, 0.0)
            torch.nn.init.eye_(self.conv3d_time_2.weight[:, :, 0, 0, 0])

    def forward(self, x):
        if self.time_dim == 1:
            return self.conv3d(x)
        else:
            x_spatial = self.conv3d_spatial(x)
            T1 = x_spatial[:, :, 0:1, :, :]
            T2 = x_spatial[:, :, 1:2, :, :]
            T1_F1 = self.conv3d_time_2(T1)
            T2_F1 = self.conv3d_time_2(T2)
            T1_F2 = self.conv3d_time_1(T1)
            T2_F2 = self.conv3d_time_3(T2)
            x = torch.cat([T1_F1 + T2_F2, T1_F2 + T2_F1], dim=2)

            return x

def Decompose_norm(batch2d):
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d

def Decompose_pool(pool2d, time_dim=1, time_padding=0, time_stride=None, time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(kernel_dim, padding=padding, dilation=dilation, stride=stride,
                                        ceil_mode=pool2d.ceil_mode)
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError('{} is not among known pooling classes'.format(type(pool2d)))
    return pool3d

def inflate_conv(conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False):
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_dim, padding=padding,
                             dilation=dilation, stride=stride)

    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d

class ResNet3D(torch.nn.Module):
    def __init__(self, resnet2d):
        super(ResNet3D, self).__init__()
        self.conv1 = Decompose_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = Decompose_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = Decompose_pool(resnet2d.maxpool, time_dim=1, time_padding=0, time_stride=1)
        
        self.layer1 = Decompose_layer(resnet2d.layer1)
        self.layer2 = Decompose_layer(resnet2d.layer2)
        self.layer3 = Decompose_layer(resnet2d.layer3)
        self.layer4 = Decompose_layer(resnet2d.layer4)

def Decompose_layer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        self.conv1 = Decompose_conv(bottleneck2d.conv1, time_dim=3, time_padding=1,
                                                time_stride=1, center=True)
        self.bn1 = Decompose_norm(bottleneck2d.bn1)

        self.conv2 = Decompose_conv(bottleneck2d.conv2, time_dim=3, time_padding=1,
                                                time_stride=1, center=True)
        self.bn2 = Decompose_norm(bottleneck2d.bn2)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = Decompose_downsample(bottleneck2d.downsample, time_stride=1)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

def Decompose_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        Decompose_norm(downsample2d[1]))
    return downsample3d

class OFW(nn.Module):
    def __init__(self, inplane):
        super(OFW, self).__init__()

        self.down = nn.Sequential(
            nn.Conv3d(inplane, inplane, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), groups=inplane),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplane, inplane, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), groups=inplane),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplane, inplane, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), groups=inplane),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv3d(inplane * 2, 2, kernel_size=(3,3,3), padding=(1,1,1), bias=False)

    def forward(self, x):
        size = x.size()[3:]
        seg_down = self.down(x)
        seg_down = F.interpolate(seg_down, size=(2, 128, 128), mode='trilinear', align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        return seg_flow_warp

    def flow_warp(self, input, flow, size):
        B, C, T, H, W = input.size()
        _, _, _, flow_H, flow_W = flow.size()
        out_h, out_w = size
    
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_grid = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(B, 1, 1, 1).type_as(input).to(input.device)
        output_list = []
        for t in range(T):
            current_frame_input = input[:, :, t, :, :]
            current_frame_flow = flow[:, :, t, :, :]

            norm = torch.tensor([[[[flow_W, flow_H]]]]).type_as(input).to(input.device)

            flow_field = current_frame_flow.permute(0, 2, 3, 1) / norm

            warped_frame = F.grid_sample(current_frame_input, (grid + flow_field).clamp(-1, 1), mode='bilinear', padding_mode='border', align_corners=True)
            output_list.append(warped_frame.unsqueeze(2))

        x1 = input[:, :, 0:1, :, :]
        x2 = input[:, :, 1:2, :, :]
        output = torch.cat([x1] + output_list + [x2], dim=2)
        
        return output