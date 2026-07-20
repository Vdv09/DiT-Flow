import models.autoencoder
import data.cifar10

from class_registry import class_registry

import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm

from utils.seed import set_seed
from utils.EMA import EMA


class Trainer:
    def __init__(
        self, 
        model,
        optimizer,
        sample_every,
        save_every,
        num_samples_to_check,
        output_dir,
        train_loader,
        device,
        kl_weight = 1e-6
    ):
        self.model = model
        self.optimizer = optimizer
        self.sample_every = sample_every
        self.save_every = save_every
        self.num_samples_to_check = num_samples_to_check
        self.output_dir = output_dir
        self.train_loader = train_loader
        self.device = device
        self.kl_weight = kl_weight

        os.makedirs(output_dir, exist_ok = True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok = True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok = True)

    def train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        
        total_loss = 0
        total_reconstruction_loss = 0
        total_kl_loss = 0
        total_samples = 0
        
        for x, _ in tqdm(self.train_loader, desc = f"Training Epoch {epoch + 1}/{num_epochs}"):
            x = x.to(self.device)
            batch_size = x.shape[0]
            
            self.optimizer.zero_grad()
            
            loss, loss_dict = self.model.training_loss(x, self.kl_weight)
            
            total_loss += loss.item() * batch_size
            total_reconstruction_loss += loss_dict["Reconstruction_loss"] * batch_size
            total_kl_loss += loss_dict["KL_loss"] * batch_size
            total_samples += batch_size
            
            loss.backward()
            self.optimizer.step()
            
        return {
            "Total_loss": total_loss / total_samples,
            "Reconstruction_loss": total_reconstruction_loss / total_samples,
            "KL_loss": total_kl_loss / total_samples
        }
    
    @torch.no_grad()
    def sample(self, epoch):
        self.model.eval()
        
        x, _ = next(iter(self.train_loader))
        x = x[:self.num_samples_to_check].to(self.device)
        
        reconstructed_x, _, _ = self.model(x)
        
        images = torch.cat([x, reconstructed_x], dim = 0)

        images = (images.clamp(-1, 1) + 1) / 2

        save_image(
            images,
            os.path.join(self.output_dir, "images", f"epoch_{epoch}.png"),
            nrow = self.num_samples_to_check,
        )
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint["epoch"]

    def save_model(self, epoch):
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok = True)

        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, os.path.join(self.output_dir, "models", f"epoch_{epoch}.pth"))

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_results = self.train_one_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}/{num_epochs}: {train_results}")

            if (epoch + 1) % self.sample_every == 0:
                self.sample(epoch)

            if (epoch + 1) % self.save_every == 0:
                self.save_model(epoch)


def main(config):
    set_seed(config["experiment"]["seed"])

    device = torch.device(config["experiment"]["device"])

    model = class_registry.get_from_registry(
        config["model"]["type"], **config["model"]["params"]
    )
    model.to(device)

    train_loader, _ = class_registry.get_from_registry(
        config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        output_dir=config["train"]["output_dir"],
        sample_every=config["train"]["sample_every"],
        save_every=config["train"]["save_every"],
        kl_weight=config["train"]["kl_weight"],
        num_samples_to_check=config["train"]["num_samples_to_check"]
    )
    trainer.train(config["train"]["epochs"])

    print("VAE training complete")