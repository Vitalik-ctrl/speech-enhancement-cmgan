import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .audio import AudioManager

logger = logging.getLogger(__name__)

class ModelDataset(Dataset):
    """Custom Dataset for loading audio files and their corresponding labels from a CSV file."""

    def __init__(self, manifest_path: str | Path, audio_manager: AudioManager, segment_seconds: float):
        self.manifest_path = Path(manifest_path)
        self.audio_manager = audio_manager
        self.segment_seconds = segment_seconds

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        logger.info(f"Loading manifest: {self.manifest_path}")
        self.dataframe = pd.read_csv(self.manifest_path)

        self.dataframe = self.dataframe[self.dataframe["additive_noise"] == True].reset_index(drop=True)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def extract_random_segment(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Extracts a random segment of target_length from the audio."""
        audio_length = len(audio)
        if audio_length > target_length:
            start_idx = random.randint(0, audio_length - target_length)
            return audio[start_idx: start_idx + target_length]
        elif audio_length < target_length:
            padding = target_length - audio_length
            return np.pad(audio, (0, padding), 'constant')
        return audio

    def mix_at_snr(self, clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Mixes clean and noise arrays at the exact SNR requested.
        Imported from legacy make_noisy.py logic."""
        if len(noise) < len(clean):
            noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
        noise = noise[:len(clean)]

        ps = np.mean(clean ** 2)
        pn = np.mean(noise ** 2)

        if ps == 0 or pn == 0:
            return clean + noise

        pn_target = ps / (10.0 ** (snr_db / 10.0))
        noise = noise * np.sqrt(pn_target / (pn + 1e-12))

        return clean + noise

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads the clean and noisy audio segments for the given index, applies necessary processing, and returns them as tensors."""
        row = self.dataframe.iloc[idx]

        clean_path = row["clean_path"]
        noise_path = row["additive_noise_path"]
        snr_db = float(row["snr_db"])
        target_sr = int(row["sr"])

        clean, sr_clean = self.audio_manager.load_audio(clean_path, target_sr=target_sr)
        noise, sr_noise = self.audio_manager.load_audio(noise_path, target_sr=target_sr)

        target_samples = int(self.segment_seconds * target_sr)
        clean_segment = self.extract_random_segment(clean, target_samples)

        noisy_segment = self.mix_at_snr(clean_segment, noise, snr_db)

        max_amp = np.max(np.abs(noisy_segment)) + 1e-9
        if max_amp > 1.0:
            noisy_segment /= max_amp
            clean_segment /= max_amp

        noisy_tensor = torch.from_numpy(noisy_segment.astype(np.float32))
        clean_tensor = torch.from_numpy(clean_segment.astype(np.float32))

        return noisy_tensor, clean_tensor


def get_dataloader(config: dict, manifest_path: Path, audio_manager: AudioManager, is_train: bool = True) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the ModelDataset.
    """
    dataset = ModelDataset(
        manifest_path=manifest_path,
        audio_manager=audio_manager,
        segment_seconds=config['audio']['segment_seconds']
    )

    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_mem = config['training']['pin_memory']

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_mem
    )
