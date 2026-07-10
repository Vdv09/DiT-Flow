from models.blocks import DiffusionResBlock, SelfAttentionBlock, Downsample, Upsample, DownBlock, UpBlock
import torch


def test_resblock_same_channels():
    block = DiffusionResBlock(64, 64, time_dim=256)
    x = torch.randn(2, 64, 32, 32)
    time_embeddings = torch.randn(2, 256)

    assert block(x, time_embeddings).shape == (2, 64, 32, 32)


def test_resblock_change_channels():
    block = DiffusionResBlock(64, 128, time_dim=256)
    x = torch.randn(2, 64, 16, 16)
    time_embeddings = torch.randn(2, 256)

    assert block(x, time_embeddings).shape == (2, 128, 16, 16)


def test_self_attention_block():
    block = SelfAttentionBlock(64)
    x = torch.randn(2, 64, 32, 32)

    assert block(x).shape == (2, 64, 32, 32)


def test_downsample():
    block = Downsample(64)
    x = torch.randn(2, 64, 32, 32)

    assert block(x).shape == (2, 64, 16, 16)


def test_upsample():
    block = Upsample(64)
    x = torch.randn(2, 64, 16, 16)

    assert block(x).shape == (2, 64, 32, 32)


def test_downblock():
    block = DownBlock(64, 128, time_dim=256)

    x = torch.randn(2, 64, 32, 32)
    time_embeddings = torch.randn(2, 256)

    assert block(x, time_embeddings)[0].shape == (2, 128, 16, 16)
    assert block(x, time_embeddings)[1].shape == (2, 128, 32, 32)


def test_upblock():
    block = UpBlock(128, 128, 64, time_dim=256)

    skip_x = torch.randn(2, 128, 32, 32)
    x = torch.randn(2, 128, 16, 16)
    time_embeddings = torch.randn(2, 256)

    assert block(x, skip_x, time_embeddings).shape == (2, 64, 32, 32)