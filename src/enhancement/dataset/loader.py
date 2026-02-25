import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from enum import Enum

from .audio import AudioManager

logger = logging.getLogger(__name__)

class MixScenario(Enum):
    ADDITIVE = "additive"
    CONVOLUTIONAL = "convolutional"
    BOTH = "both"

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

        # self.dataframe = self.dataframe[self.dataframe["additive_noise"] == True].reset_index(drop=True)

        self.noise_paths = self.dataframe["additive_noise_path"].dropna().tolist()

        if "convolutional_noise_path" in self.dataframe.columns:
            self.rir_paths = [p for p in self.dataframe["convolutional_noise_path"].fillna("").tolist() if p]
        else:
            self.rir_paths = []

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        clean_path = row["clean_path"]
        target_sr = int(row["sr"])

        snr_db = random.uniform(-5.0, 15.0)

        clean, _ = self.audio_manager.load_audio(clean_path, target_sr=target_sr)
        target_samples = int(self.segment_seconds * target_sr)
        clean_segment = self.extract_random_segment(clean, target_samples)
        audio_mix = clean_segment.copy()

        available_scenarios = [MixScenario.ADDITIVE]
        if hasattr(self, 'rir_paths') and self.rir_paths:
            logger.info(f"RIR paths available: {len(self.rir_paths)}")
            available_scenarios.extend([MixScenario.CONVOLUTIONAL, MixScenario.BOTH])

        mix_scenario = random.choice(available_scenarios)

        if mix_scenario in [MixScenario.CONVOLUTIONAL, MixScenario.BOTH]:
            logger.info(f"Applying convolutional noise using RIR")
            random_rir_path = random.choice(self.rir_paths)
            rir, _ = self.audio_manager.load_audio(random_rir_path, target_sr=target_sr)
            audio_mix = self.audio_manager.apply_rir(audio_mix, rir)

        if mix_scenario in [MixScenario.ADDITIVE, MixScenario.BOTH]:
            logger.info(f"Applying additive noise at {snr_db:.2f} dB SNR using noise")
            random_noise_path = random.choice(self.noise_paths)
            noise, _ = self.audio_manager.load_audio(random_noise_path, target_sr=target_sr)
            noise_segment = self.extract_random_segment(noise, target_samples)
            audio_mix = self.audio_manager.mix_at_snr(audio_mix, noise_segment, snr_db)

        max_amp = np.max(np.abs(audio_mix)) + 1e-9
        if max_amp > 1.0:
            audio_mix /= max_amp
            clean_segment /= max_amp

        noisy_tensor = torch.from_numpy(audio_mix.astype(np.float32))
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
