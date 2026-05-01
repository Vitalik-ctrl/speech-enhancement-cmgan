import os
import time
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from enhancement.models.cmgan.generator import TSCNet
from enhancement.models.cmgan.discriminator import Discriminator, batch_pesq
from enhancement.models.cmgan.utils import power_compress, power_uncompress

logger = logging.getLogger(__name__)

class ModelTrainer:

    def __init__(self, config: dict, train_loader: DataLoader, validation_loader: DataLoader, device: torch.device):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = validation_loader
        self.device = device

        self.n_fft = config["audio"]["n_fft"]
        self.hop = config["audio"]["hop_length"]

        self.loss_weights = config["training"]["loss_weights"]

        self.model = TSCNet(
            num_channel=config["model"]["num_channel"],
            num_features=self.n_fft // 2 + 1
        ).to(self.device)

        self.discriminator = Discriminator(
            ndf=16,
            in_channel=2
        ).to(self.device)

        init_lr = float(config["training"]["init_lr"])
        self.optimizer_generator = torch.optim.AdamW(self.model.parameters(), lr=init_lr)
        self.optimizer_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=1.6 * init_lr)

        decay_epoch = config["training"]["decay_epoch"]
        self.scheduler_generator = torch.optim.lr_scheduler.StepLR(self.optimizer_generator, step_size=decay_epoch, gamma=0.5)
        self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.optimizer_discriminator, step_size=decay_epoch, gamma=0.5)
        self.scaler = torch.cuda.amp.GradScaler()

        self.save_dir = config.get('paths', {}).get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

    def forward_generator_step(self, clean, noisy):
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        window = torch.hamming_window(self.n_fft).to(self.device)

        # STFT -> (Batch, Freq, Time)
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=window, onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=window, onesided=True, return_complex=True)

        # Prepare for power_compress (Needs Real/Imag at the end: (B, F, T, 2))
        noisy_spec = torch.view_as_real(noisy_spec)
        clean_spec = torch.view_as_real(clean_spec)

        # Power Compress -> Returns (B, 2, F, T)
        noisy_spec = power_compress(noisy_spec)
        clean_spec = power_compress(clean_spec)

        # Permute (B, 2, F, T) -> (B, 2, T, F)
        noisy_spec = noisy_spec.permute(0, 1, 3, 2)

        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)  # (B, 1, F, T)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)  # (B, 1, F, T)

        est_real, est_imag = self.model(noisy_spec)  # Returns (B, 1, T, F)

        # Calculate Magnitudes (B, 1, T, F)
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        # Important: Transpose clean_mag to match (B, 1, T, F)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).transpose(2, 3)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)  # (B, 2, T, F)

        # Convert back to complex for istft: (B, T, F)
        est_spec_complex = torch.complex(est_spec_uncompress[:, 0], est_spec_uncompress[:, 1])

        # Original power_uncompress returns (B, 1, T, F, 2)
        est_spec_uncompress = power_uncompress(est_real, est_imag)

        # Shape before slice: (B, 1, T, F, 2)
        est_spec_uncompress = est_spec_uncompress.squeeze(1)  # Result: (B, T, F, 2)

        est_spec_complex = torch.complex(
            est_spec_uncompress[..., 0],
            est_spec_uncompress[..., 1]
        )  # Result: (B, T, F)

        est_audio = torch.istft(est_spec_complex.transpose(1, 2), self.n_fft, self.hop, window=window, onesided=True)

        return {
            "est_real": est_real, "est_imag": est_imag, "est_mag": est_mag,
            "clean_real": clean_real.transpose(2, 3),
            "clean_imag": clean_imag.transpose(2, 3),
            "clean_mag": clean_mag,
            "est_audio": est_audio, "clean_audio": clean
        }

    def train_step(self, noisy, clean):
        batch_size = noisy.size(0)
        one_labels = torch.ones(batch_size).to(self.device)

        self.optimizer_generator.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            gen_outputs = self.forward_generator_step(clean, noisy)

            predict_fake_metric = self.discriminator(gen_outputs["clean_mag"], gen_outputs["est_mag"])
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

            loss_mag = F.mse_loss(gen_outputs["est_mag"], gen_outputs["clean_mag"])
            loss_ri = F.mse_loss(gen_outputs["est_real"], gen_outputs["clean_real"]) + \
                      F.mse_loss(gen_outputs["est_imag"], gen_outputs["clean_imag"])

            time_loss = torch.mean(torch.abs(gen_outputs["est_audio"] - gen_outputs["clean_audio"]))

            loss_generator = (
                    self.loss_weights[0] * loss_ri +
                    self.loss_weights[1] * loss_mag +
                    self.loss_weights[2] * time_loss +
                    self.loss_weights[3] * gen_loss_GAN
            )

        self.scaler.scale(loss_generator).backward()
        self.scaler.step(self.optimizer_generator)

        self.optimizer_discriminator.zero_grad()
        length = gen_outputs["est_audio"].size(-1)

        est_audio_list = list(gen_outputs["est_audio"].detach().float().cpu().numpy())
        clean_audio_list = list(gen_outputs["clean_audio"].float().cpu().numpy()[:, :length])

        pesq_score = batch_pesq(clean_audio_list, est_audio_list)

        if pesq_score is not None:
            pesq_score = torch.tensor(pesq_score, dtype=torch.float32).to(self.device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                predict_enhance_metric = self.discriminator(gen_outputs["clean_mag"], gen_outputs["est_mag"].detach())
                predict_max_metric = self.discriminator(gen_outputs["clean_mag"], gen_outputs["clean_mag"])

                loss_discriminator = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                         F.mse_loss(predict_enhance_metric.flatten(), pesq_score)

            self.scaler.scale(loss_discriminator).backward()
            self.scaler.step(self.optimizer_discriminator)
        else:
            loss_discriminator = torch.tensor(0.0)

        self.scaler.update()

        return loss_generator.item(), loss_discriminator.item()

    @torch.no_grad()
    def eval_step(self, noisy, clean):
        batch_size = noisy.size(0)
        one_labels = torch.ones(batch_size).to(self.device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            gen_outputs = self.forward_generator_step(clean, noisy)

            predict_fake_metric = self.discriminator(gen_outputs["clean_mag"], gen_outputs["est_mag"])
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

            loss_mag = F.mse_loss(gen_outputs["est_mag"], gen_outputs["clean_mag"])
            loss_ri = F.mse_loss(gen_outputs["est_real"], gen_outputs["clean_real"]) + \
                      F.mse_loss(gen_outputs["est_imag"], gen_outputs["clean_imag"])

            time_loss = torch.mean(torch.abs(gen_outputs["est_audio"] - gen_outputs["clean_audio"]))

            loss_generator = (
                    self.loss_weights[0] * loss_ri +
                    self.loss_weights[1] * loss_mag +
                    self.loss_weights[2] * time_loss +
                    self.loss_weights[3] * gen_loss_GAN
            )

        length = gen_outputs["est_audio"].size(-1)
        # Conversion back to float32 for PESQ calculation
        est_audio_list = list(gen_outputs["est_audio"].float().cpu().numpy())
        clean_audio_list = list(gen_outputs["clean_audio"].float().cpu().numpy()[:, :length])

        pesq_score = batch_pesq(clean_audio_list, est_audio_list)

        current_batch_pesq = 0.0

        if pesq_score is not None:
            pesq_tensor = torch.tensor(pesq_score, dtype=torch.float32, device=self.device).clone().detach()
            current_batch_pesq = pesq_tensor.mean().item()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                predict_enhance_metric = self.discriminator(gen_outputs["clean_mag"], gen_outputs["est_mag"])
                predict_max_metric = self.discriminator(gen_outputs["clean_mag"], gen_outputs["clean_mag"])

                loss_discriminator = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                     F.mse_loss(predict_enhance_metric.flatten(), pesq_tensor)
        else:
            loss_discriminator = torch.tensor(0.0)

        return loss_generator.item(), loss_discriminator.item(), current_batch_pesq


    def train(self):
        epochs = self.config.get('training', {}).get('epochs', 120)
        log_interval = self.config.get('training', {}).get('log_interval', 100)

        job_id = os.environ.get("PBS_JOBID", "").split(".")[0]

        if not job_id:
            job_id = time.strftime("run_%Y%m%d_%H%M%S")

        self.save_dir = os.path.join(self.save_dir, job_id)
        os.makedirs(self.save_dir, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        logger.info(f"batch_size: {self.config['training']['batch_size']}, "
                    f"init_lr: {self.config['training']['init_lr']}, "
                    f"decay_epoch: {self.config['training']['decay_epoch']}, "
                    f"loss_weights: {self.config['training']['loss_weights']}, "
                    f"num_workers: {self.config['training']['num_workers']}")

        for epoch in range(epochs):
            self.model.train()
            self.discriminator.train()

            logger.info(f"--- Epoch {epoch} Training ---")
            for step, (noisy, clean) in enumerate(self.train_loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                loss_generator, loss_discriminator = self.train_step(noisy, clean)

                if step % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch} | Step {step} | Train Loss G: {loss_generator:.4f} | Train Loss D: {loss_discriminator:.4f}")

            self.model.eval()
            self.discriminator.eval()
            eval_gen_loss_total = 0.0
            eval_disc_loss_total = 0.0
            eval_pesq_total = 0.0

            logger.info(f"--- Epoch {epoch} Evaluation ---")
            for eval_step_idx, (noisy, clean) in enumerate(self.eval_loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                val_loss_g, val_loss_d, val_pesq = self.eval_step(noisy, clean)

                eval_gen_loss_total += val_loss_g
                eval_disc_loss_total += val_loss_d
                eval_pesq_total += val_pesq

            avg_val_loss_g = eval_gen_loss_total / len(self.eval_loader)
            avg_val_loss_d = eval_disc_loss_total / len(self.eval_loader)
            avg_val_pesq = eval_pesq_total / len(self.eval_loader)

            logger.info(
                f"Epoch {epoch} Summary | Avg Eval Loss G: {avg_val_loss_g:.4f} | Avg Eval Loss D: {avg_val_loss_d:.4f} | Avg Eval PESQ: {avg_val_pesq:.4f}"
            )

            self.scheduler_generator.step()
            self.scheduler_discriminator.step()

            ckpt_path = os.path.join(self.save_dir, f"cmgan_epoch_{epoch}_valG_{avg_val_loss_g:.4f}.pth")
            torch.save(self.model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
