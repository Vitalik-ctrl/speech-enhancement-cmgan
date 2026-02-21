import os
import yaml
import argparse
import logging
import torch
from pathlib import Path

from enhancement.dataset.loader import get_dataloader
from enhancement.dataset.audio import AudioManager
from enhancement.training.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train CMGAN Speech Enhancement Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    workspace_dir = Path(config.get("system", {}).get("workspace", "."))

    sph2pipe_path = workspace_dir / config.get("paths", {}).get("sph2pipe", "")

    audio_manager = AudioManager(sph2pipe_path)

    train_manifest = workspace_dir / config.get("paths", {}).get("train_manifest", "manifest/train.csv")
    eval_manifest = workspace_dir / config.get("paths", {}).get("eval_manifest", "manifest/train.csv")

    logger.info("Initializing DataLoaders...")
    train_loader = get_dataloader(config, train_manifest, audio_manager, is_train=True)
    eval_loader = get_dataloader(config, eval_manifest, audio_manager, is_train=False)

    logger.info("Initializing Trainer...")
    trainer = ModelTrainer(config, train_loader, eval_loader, device)
    
    logger.info("Starting training loop...")
    trainer.train()

if __name__ == "__main__":
    main()
