import torch
import torch.nn as nn
import torch.nn.functional as F

from class_registry import class_registry


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups = 8, num_channels = in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.GroupNorm(num_groups = 8, num_channels = out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size = 3,
                padding = 1
            ),
        )
        
        self.skip = nn.Identity()
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = 1
            )
    
    def forward(self, x):
        return self.net(x) + self.skip(x)


class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels = 3, 
        base_channels = 64,
        channels_multiplier = [1, 2],
        latent_channels = 4
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.GroupNorm(num_groups = 8, num_channels = base_channels),
            nn.SiLU()
        ]

        current_channels = base_channels
        for multiplier in channels_multiplier:
            out_channels = base_channels * multiplier

            layers.extend([
                ResBlock(
                    current_channels,
                    out_channels
                ),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size = 3,
                    stride = 2,
                    padding = 1
                ),
                nn.GroupNorm(num_groups = 8, num_channels = out_channels),
                nn.SiLU()
            ])

            current_channels = out_channels
        
        layers.extend([
            ResBlock(
                current_channels,
                current_channels
            ),
            nn.GroupNorm(num_groups = 8, num_channels = current_channels),
            nn.SiLU()
        ])
            
        self.net = nn.Sequential(*layers)
        
        self.get_moments = nn.Conv2d(
            current_channels,
            latent_channels * 2,
            kernel_size = 3,
            padding = 1
        )
    
    def forward(self, x):
        x = self.net(x)
        
        mean_val, log_var = self.get_moments(x).chunk(2, dim = 1)
        
        return mean_val, log_var


class Decoder(nn.Module):
    def __init__(
        self, 
        final_channels = 3, 
        base_channels = 64,
        channels_multiplier = [1, 2],
        latent_channels = 4
    ):
        super().__init__()
        
        channels_multiplier = channels_multiplier[::-1]
        current_channels = base_channels * channels_multiplier[0]
        
        layers = [
            nn.Conv2d(
                latent_channels,
                current_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.SiLU()
        ]
        
        for multiplier in channels_multiplier:
            out_channels = base_channels * multiplier

            layers.extend([
                ResBlock(
                    current_channels,
                    out_channels,
                ),
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size = 3,
                    padding = 1
                ),
                nn.GroupNorm(num_groups = 8, num_channels = out_channels),
                nn.SiLU()
            ])

            current_channels = out_channels
        
        layers.append(
            nn.Conv2d(
                current_channels,
                final_channels,
                kernel_size = 3,
                padding = 1
            )
        )
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


@class_registry.add_to_registry("vae")
class VAE(nn.Module):
    def __init__(
        self, 
        in_channels = 3, 
        base_channels = 64, 
        channels_multiplier = [1, 2], 
        latent_channels = 4
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, base_channels, channels_multiplier, latent_channels)

        self.decoder = Decoder(in_channels, base_channels, channels_multiplier, latent_channels)

    def decode(self, x):
        return self.decoder(x)
    
    def sample(self, mean, log_var):
        std = torch.exp(log_var * 0.5)
        return mean + std * torch.randn_like(mean)
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        sample = self.sample(mean, log_var)
        return self.decode(sample), mean, log_var
    
    @torch.no_grad()
    def encode(self, x, return_sample = True):
        mean, log_var = self.encoder(x)
        
        if return_sample:
            return self.sample(mean, log_var)
        
        return mean

    def training_loss(self, x, kl_weight = 1e-6):
        x_reconstructed, mean, log_var = self.forward(x)
        
        mse_loss = F.mse_loss(x_reconstructed, x)
        
        kl_loss = torch.mean(mean ** 2 + torch.exp(log_var) - log_var - 1) / 2
        
        return mse_loss + kl_weight * kl_loss, {"Reconstruction_loss": mse_loss.item(), "KL_loss": kl_loss.item()}