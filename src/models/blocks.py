import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups = 32, num_channels = in_channels)
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1,
        )

        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups = 32, num_channels = out_channels)
        self.dropout = nn.Dropout(p = 0.1)
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1,
        )

        self.skip_connection = nn.Identity()

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
            )
    
    def forward(self, x, time_embeddings):  # x: [B, C, H, W], time_embeddings: [B, time_dim]
        start_x = x

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        time_projection = self.time_mlp(F.silu(time_embeddings))
        x = x + time_projection[:, :, None, None]

        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x + self.skip_connection(start_x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads = 8):
        super().__init__()

        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups = 32, num_channels = channels)

        self.qkv = nn.Conv2d(
            in_channels = channels, 
            out_channels = channels * 3,
            kernel_size = 1,
        )

        self.final_proj = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 1,
        )
    
    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape

        start_x = x

        x = self.norm(x)
        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim = 1)

        head_dim = C // self.num_heads

        def reshape(tensor):
            return tensor.reshape(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        
        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        attn_weights = q @ k.transpose(-2, -1) * (head_dim ** -0.5)  # [B, self.num_heads, H*W, H*W]
        attn_weights = attn_weights.softmax(dim = -1)

        output = attn_weights @ v  # [B, self.num_heads, H*W, head_dim]

        output = output.transpose(2, 3).reshape(B, C, H, W)
        output = self.final_proj(output)

        return output + start_x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            padding = 1,
            stride = 2,
        )
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1,
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, need_attention = False):
        super().__init__()

        self.resblock1 = DiffusionResBlock(in_channels, out_channels, time_dim)

        self.attention = SelfAttentionBlock(out_channels) if need_attention else None

        self.resblock2 = DiffusionResBlock(out_channels, out_channels, time_dim)

        self.downsample = Downsample(out_channels)
    
    def forward(self, x, time_embeddings):
        x = self.resblock1(x, time_embeddings)

        if self.attention is not None:
            x = self.attention(x)
        
        x = self.resblock2(x, time_embeddings)

        skip_x = x
        x = self.downsample(x)

        return x, skip_x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim, need_attention = False):
        super().__init__()

        self.upsample = Upsample(in_channels)

        self.resblock1 = DiffusionResBlock(in_channels + skip_channels, out_channels, time_dim)

        self.attention = SelfAttentionBlock(out_channels) if need_attention else None

        self.resblock2 = DiffusionResBlock(out_channels, out_channels, time_dim)
    
    def forward(self, x, skip_x, time_embeddings):
        x = self.upsample(x)

        x = torch.cat([x, skip_x], dim = 1)

        x = self.resblock1(x, time_embeddings)

        if self.attention is not None:
            x = self.attention(x)

        x = self.resblock2(x, time_embeddings)

        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()

        self.resblock1 = DiffusionResBlock(channels, channels, time_dim)
        self.attention = SelfAttentionBlock(channels)
        self.resblock2 = DiffusionResBlock(channels, channels, time_dim)
    
    def forward(self, x, time_embeddings):
        x = self.resblock1(x, time_embeddings)
        x = self.attention(x)
        x = self.resblock2(x, time_embeddings)

        return x