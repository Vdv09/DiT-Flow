import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, period = 10_000):
        super().__init__()

        self.dim = dim
        self.period = period

        half_dim = self.dim // 2

        self.one_position = torch.exp(-math.log(self.period) * torch.arange(half_dim) / half_dim)

    def forward(self, timesteps):
        inner_args = timesteps[:, None] * self.one_position[None, :]

        sin_embeddings = torch.sin(inner_args)
        cos_embeddings = torch.cos(inner_args)

        all_data = torch.stack([sin_embeddings, cos_embeddings], dim = -1)

        return all_data.flatten(start_dim = -2)
    

class MLPTimeEmbedding(nn.Module):
    def __init__(self, dim, output_dim = None, coefficient = 4):
        super().__init__()

        if output_dim is None:
            output_dim = dim * coefficient

        self.net = nn.Sequential(
            nn.Linear(dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, embeddings):
        return self.net(embeddings)