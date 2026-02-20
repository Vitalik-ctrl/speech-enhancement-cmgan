import torch
from torch.utils.data import DataLoader
from trainer import ModelTrainer
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def smoke_test():

    config = {
        "audio": {"n_fft": 400, "hop_length": 100},
        "model": {"num_channel": 64},
        "training": {"init_lr": 5e-4, "decay_epoch": 8, "loss_weights": [0.1, 0.9, 0.2, 0.05], "epochs": 1, "log_interval": 1},
        "paths": {"save_dir": "test_ckpts"}
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_noisy = torch.randn(2, 32000)
    dummy_clean = torch.randn(2, 32000)

    dummy_loader = [(dummy_noisy, dummy_clean)]

    logger.info(f"device: {device}")
    trainer = ModelTrainer(config, dummy_loader, dummy_loader, device)
    loss_g, loss_d = trainer.train_step(dummy_noisy.to(device), dummy_clean.to(device))
    logger.info(f"Step successful! loss_g: {loss_g}, loss_d: {loss_d}")


if __name__ == "__main__":
    smoke_test()