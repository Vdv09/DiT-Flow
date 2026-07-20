import argparse
from pathlib import Path
import sys
import json

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

import torch
import yaml

import models.unet
import diffusion.ddpm
import diffusion.ddim
import diffusion.schedules
import data.cifar10
from class_registry import class_registry
from utils.seed import set_seed
from metrics.fid import FID


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_uint8(x):
    x = (x.clamp(-1, 1) + 1) / 2
    return (x * 255).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(config["experiment"]["device"])
    num_classes = config["model"]["params"]["num_classes"]
    eval_config = config["eval"]

    num_samples = eval_config["num_samples"]
    sample_batch = eval_config["sample_batch"]
    threshold = eval_config["threshold"]
    threshold_quantile = eval_config["threshold_quantile"]

    model = class_registry.get_from_registry(config["model"]["type"], **config["model"]["params"])
    model.to(device).eval()

    ckpt = torch.load(eval_config["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["ema_model"])

    schedule = class_registry.get_from_registry(
        config["diffusion"]["schedule"], number_steps=config["diffusion"]["number_steps"]
    )
    schedule.to(device)

    diffusion = class_registry.get_from_registry(config["diffusion"]["method"], schedule=schedule)

    train_loader, _ = class_registry.get_from_registry(
        config["data"]["dataset"],
        batch_size=sample_batch,
        num_workers=config["data"]["num_workers"],
    )

    real_images = []
    collected = 0

    for x, _ in train_loader:
        real_images.append(to_uint8(x).cpu())
        collected += x.shape[0]

        if collected >= num_samples:
            break

    real_images = torch.cat(real_images, dim = 0)[:num_samples]

    print("Real images have collected")

    out = Path(eval_config["output_dir"])
    out.mkdir(exist_ok = True)

    results = {}

    for w in eval_config["scales"]:
        set_seed(config["experiment"]["seed"])

        metric = FID(device)

        for i in range(0, real_images.shape[0], sample_batch):
            metric.update_real(real_images[i:i + sample_batch].to(device))

        made = 0

        while made < num_samples:
            b = min(sample_batch, num_samples - made)

            y = torch.arange(b, device = device) % num_classes

            sample_kwargs = dict(
                image_shape=(b, 3, 32, 32),
                device=device,
                y=y,
                guidance_scale=w,
                null_class_label=num_classes,
                threshold=threshold,
                threshold_quantile=threshold_quantile,
            )

            if config["diffusion"]["method"] == "ddim":
                sample_kwargs["number_steps"] = eval_config["sampling_steps"]

            fake = diffusion.sample(model, **sample_kwargs)

            metric.update_fake(to_uint8(fake))
            made += b

            print(f"[scale={w}] {made}/{num_samples}")

        results[str(w)] = metric.compute()
        print(f"scale={w}: {results[str(w)]}")

        with open(out / "metrics.json", "w") as f:
            json.dump(results, f, indent = 2)

    print("Metrics have saved")


if __name__ == "__main__":
    main()