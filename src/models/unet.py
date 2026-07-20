import torch.nn as nn
from .unet_blocks import DownBlock, MiddleBlock, UpBlock
from .time_embeddings import SinusoidalTimeEmbedding, MLPTimeEmbedding
from class_registry import class_registry


@class_registry.add_to_registry("unet")
class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        start_channels,
        num_blocks,
        embedding_dim,
        num_classes,
        num_blocks_with_attention,
    ):
        super().__init__()

        self.get_time_embeddings = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            MLPTimeEmbedding(embedding_dim, output_dim = embedding_dim)
        )
        
        self.class_embedding = nn.Embedding(num_classes + 1, embedding_dim)

        self.first_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = start_channels,
            kernel_size = 3,
            padding = 1
        )

        self.down_blocks = nn.ModuleList([
            DownBlock(
                in_channels = start_channels * 2 ** i,
                out_channels = start_channels * 2 ** (i + 1),
                time_dim = embedding_dim,
                need_attention = (i >= num_blocks - num_blocks_with_attention)
            )

            for i in range(num_blocks)
        ])

        self.middle_block = MiddleBlock(
            channels = start_channels * 2 ** num_blocks,
            time_dim = embedding_dim
        )

        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels = start_channels * 2 ** (num_blocks - i),
                skip_channels = start_channels * 2 ** (num_blocks - i),
                out_channels = start_channels * 2 ** (num_blocks - i - 1),
                time_dim = embedding_dim,
                need_attention = (i >= num_blocks - num_blocks_with_attention)
            )
            for i in range(num_blocks)
        ])

        self.last_conv = nn.Conv2d(
            in_channels = start_channels,
            out_channels = in_channels,
            kernel_size = 3,
            padding = 1
        )
    
    def forward(self, x, t, y = None):
        time_embeddings = self.get_time_embeddings(t)
        
        if y is not None:
            class_embedding = self.class_embedding(y)
            time_embeddings = time_embeddings + class_embedding

        x = self.first_conv(x)

        skip_connections = []

        for down_block in self.down_blocks:
            x, skip_x = down_block(x, time_embeddings)
            skip_connections.append(skip_x)

        x = self.middle_block(x, time_embeddings)

        for up_block, skip_x in zip(self.up_blocks, skip_connections[::-1]):
            x = up_block(x, skip_x, time_embeddings)

        x = self.last_conv(x)

        return x