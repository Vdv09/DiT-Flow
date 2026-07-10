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