import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

import torch
import yaml
from torchvision.utils import save_image

import models.unet
import diffusion.ddpm
import diffusion.schedules
from class_registry import class_registry
from utils.seed import set_seed


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(config["experiment"]["device"])
    num_classes = config["model"]["params"]["num_classes"]

    model = class_registry.get_from_registry(config["model"]["type"], **config["model"]["params"])
    model.to(device).eval()
    
    checkpoint_path = config["sampling"]["checkpoint"]

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["ema_model"])

    schedule = class_registry.get_from_registry(
        config["diffusion"]["schedule"], number_steps=config["diffusion"]["number_steps"]
    )

    schedule.to(device)

    diffusion = class_registry.get_from_registry(config["diffusion"]["method"], schedule=schedule)

    classes = torch.arange(num_classes, device=device).repeat_interleave(config["sampling"]["per_class"])

    Path(config["sampling"]["output_dir"]).mkdir(exist_ok=True)

    for scale in config["sampling"]["scales"]:
        set_seed(config["experiment"]["seed"])

        images = diffusion.sample(
            model,
            image_shape=(classes.shape[0], 3, 32, 32),
            device=device,
            y=classes,
            guidance_scale=scale,
            null_class_label=num_classes
        )

        images = (images.clamp(-1, 1) + 1) / 2
        save_image(
            images, 
            f'{config["sampling"]["output_dir"]}/cfg_scale_{scale}.png', 
            nrow=config["sampling"]["per_class"]
        )

        print(f"Done cfg_scale={scale}")


if __name__ == "__main__":
    main()