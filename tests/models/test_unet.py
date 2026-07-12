from models.unet import UNet

import torch


def test_unet_without_attention():
    unet = UNet(
        in_channels = 3,
        start_channels = 64,
        num_blocks = 2,
        time_dim = 256,
        num_blocks_with_attention = 0
    )

    x = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 1000, (1,))

    assert unet(x, t).shape == (1, 3, 32, 32)


def test_unet_with_attention():
    unet = UNet(
        in_channels = 3,
        start_channels = 64,
        num_blocks = 2,
        time_dim = 256,
        num_blocks_with_attention = 1
    )

    x = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 1000, (1,))

    assert unet(x, t).shape == (1, 3, 32, 32)