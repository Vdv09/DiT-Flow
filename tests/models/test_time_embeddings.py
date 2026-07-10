from models.time_embeddings import SinusoidalTimeEmbedding, MLPTimeEmbedding

import torch


def test_sinusoidal_time_embedding():
    embedding = SinusoidalTimeEmbedding(dim=512)
    timesteps = torch.linspace(0, 1, 10, dtype=torch.long)
    embeddings = embedding(timesteps)

    assert embeddings.shape == (10, 512)


def test_mlp_time_embedding():
    mlp_embedding = MLPTimeEmbedding(dim=512, coefficient=4)

    sinusoidal_embedding = SinusoidalTimeEmbedding(dim=512)
    timesteps = torch.linspace(0, 1, 10, dtype=torch.long)
    sinusoidal_embeddings = sinusoidal_embedding(timesteps)
    mlp_embeddings = mlp_embedding(sinusoidal_embeddings)

    assert mlp_embeddings.shape == (10, 2048)