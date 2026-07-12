import yaml
import argparse
from pathlib import Path
import sys
import importlib


SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type = str, required = True)

    args = parser.parse_args()

    config = load_config(args.config)

    module_path = config["train"]["module_path"]
    module = importlib.import_module(module_path)

    module.main(config)