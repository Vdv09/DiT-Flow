from pathlib import Path
import sys

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

import torch
from class_registry import class_registry
import yaml
from tqdm import tqdm

import models.autoencoder
import data.cifar10 


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = load_config("configs/vae_cifar10.yaml")
    device = torch.device(config["experiment"]["device"])

    vae = class_registry.get_from_registry("vae", **config["model"]["params"])
    vae.to(device).eval()

    checkpoint_path = config["sampling"]["checkpoint"]
    ckpt = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(ckpt["model"])
    
    train_loader, _ = class_registry.get_from_registry(
        config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"]
    )
    
    stats = []
    
    for x, _ in tqdm(train_loader, desc="Computing VAE stats"):
        x = x.to(device)

        latent = vae.encode(x, return_sample = False)

        stats.append(latent.cpu())

    stats = torch.cat(stats, dim = 0)
    
    std = stats.std().item()
    
    print(f"Scale_factor = {1 / std}")
    

if __name__ == "__main__":
    main()