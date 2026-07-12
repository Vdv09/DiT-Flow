import models.unet
import diffusion.schedules
import diffusion.ddpm
import data.cifar10

from class_registry import class_registry

import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm


class Trainer:
    def __init__(self, model, diffusion, optimizer, sample_every, num_rows, num_columns, output_dir, train_loader, device):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.sample_every = sample_every
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.output_dir = output_dir
        self.train_loader = train_loader
        self.device = device

        os.makedirs(output_dir, exist_ok = False)

    def train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        total_loss = 0
        total_samples = 0

        for x, _ in tqdm(self.train_loader, desc = f"Training Epoch {epoch + 1}/{num_epochs}"):
            x = x.to(self.device)

            batch_size = x.shape[0]
            timesteps = torch.randint(0, self.diffusion.schedule.number_steps, (batch_size,), device = self.device, dtype = torch.long)

            self.optimizer.zero_grad()

            loss = self.diffusion.training_loss(self.model, x, timesteps)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            loss.backward()
            self.optimizer.step()
        
        return {
            "MSE_loss": total_loss / total_samples,
        }
    
    @torch.no_grad()
    def sample(self, epoch, num_rows, num_columns, output_dir):
        self.model.eval()

        images = self.diffusion.sample(
            self.model,
            image_shape = (num_rows * num_columns, 3, 32, 32),
            device = self.device
        )

        images = (images.clamp(-1, 1) + 1) / 2

        save_image(
            images,
            os.path.join(output_dir, f"epoch_{epoch}.png"),
            nrow = num_rows
        )

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_results = self.train_one_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}/{num_epochs}: {train_results}")

            if epoch % self.sample_every == 0:
                self.sample(epoch, self.num_rows, self.num_columns, self.output_dir)


def main(config):
    model = class_registry.get_from_registry(config["model"]["type"], **config["model"]["params"])
    device = torch.device(config["experiment"]["device"])
    model.to(device)

    schedule = class_registry.get_from_registry(config["diffusion"]["schedule"], number_steps = config["diffusion"]["number_steps"])
    schedule.to(device)

    diffusion = class_registry.get_from_registry(config["diffusion"]["method"], schedule = schedule)

    train_loader, _ = class_registry.get_from_registry(
        config["data"]["dataset"],
        batch_size = config["data"]["batch_size"],
        num_workers = config["data"]["num_workers"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr = config["train"]["lr"], weight_decay = config["train"]["weight_decay"])

    sample_every = config["train"]["sample_every"]
    num_rows = config["sampling"]["num_rows"]
    num_columns = config["sampling"]["num_columns"]
    output_dir = config["train"]["output_dir"]

    trainer = Trainer(model, diffusion, optimizer, sample_every, num_rows, num_columns, output_dir, train_loader, device)

    trainer.train(config["train"]["epochs"])
    print("Training complete")