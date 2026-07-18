import models.unet
import diffusion.schedules
import diffusion.ddpm
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
        diffusion, 
        optimizer, 
        sample_every, 
        save_every, 
        num_rows, 
        num_columns, 
        output_dir, 
        train_loader, 
        device, 
        ema_beta, 
        num_classes,
        uncond_prob = 0.1,
        guidance_scale = 10
    ):
        self.model = model
        self.ema_model = EMA(model, ema_beta)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.sample_every = sample_every
        self.save_every = save_every
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.output_dir = output_dir
        self.train_loader = train_loader
        self.device = device
        self.num_classes = num_classes
        self.uncond_prob = uncond_prob
        self.guidance_scale = guidance_scale

        os.makedirs(output_dir, exist_ok = True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok = True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok = True)

    def train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        total_loss = 0
        total_samples = 0

        for x, y in tqdm(self.train_loader, desc = f"Training Epoch {epoch + 1}/{num_epochs}"):
            x = x.to(self.device)
            y = y.to(self.device)

            batch_size = x.shape[0]
            timesteps = torch.randint(0, self.diffusion.schedule.number_steps, (batch_size,), device = self.device, dtype = torch.long)

            self.optimizer.zero_grad()
            
            drop_indices = torch.rand((batch_size,), device = self.device) < self.uncond_prob
            y[drop_indices] = self.num_classes

            loss = self.diffusion.training_loss(self.model, x, timesteps, y)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            loss.backward()
            self.optimizer.step()
            self.ema_model.update(self.model)
        
        return {
            "MSE_loss": total_loss / total_samples,
        }
    
    @torch.no_grad()
    def sample(self, epoch, num_columns, output_dir):
        self.model.eval()

        classes = torch.arange(self.num_classes, device = self.device).repeat_interleave(num_columns)

        images = self.diffusion.sample(
            self.ema_model.ema_model,
            image_shape = (self.num_classes * num_columns, 3, 32, 32),
            device = self.device,
            y = classes,
            guidance_scale = self.guidance_scale,
            null_class_label = self.num_classes
        )

        images = (images.clamp(-1, 1) + 1) / 2

        save_image(
            images,
            os.path.join(output_dir, "images", f"epoch_{epoch}.png"),
            nrow = num_columns,
        )
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint["epoch"]

    def save_model(self, epoch):
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok = True)

        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, os.path.join(self.output_dir, "models", f"epoch_{epoch}.pth"))

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_results = self.train_one_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}/{num_epochs}: {train_results}")

            if (epoch + 1) % self.sample_every == 0:
                self.sample(epoch, self.num_columns, self.output_dir)

            if (epoch + 1) % self.save_every == 0:
                self.save_model(epoch)


def main(config):
    set_seed(config["experiment"]["seed"])

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
    save_every = config["train"]["save_every"]
    num_rows = config["sampling"]["num_rows"]
    num_columns = config["sampling"]["num_columns"]
    output_dir = config["train"]["output_dir"]
    ema_beta = config["model"]["ema"]["beta"]
    num_classes = config["model"]["params"]["num_classes"]
    uncond_prob = config["train"]["uncond_prob"]
    guidance_scale = config["sampling"]["guidance_scale"]

    trainer = Trainer(model, diffusion, optimizer, sample_every, save_every, num_rows, num_columns, output_dir, train_loader, device, ema_beta, num_classes, uncond_prob, guidance_scale)

    trainer.train(config["train"]["epochs"])
    print("Training complete")