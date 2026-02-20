import logging
import random
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
import yaml

from audio import AudioManager

logger = logging.getLogger(__name__)

SCHEMA = [
    "clean_path", "additive_noise_path", "convolutional_noise_path",
    "noisy_path", "length_sec", "snr_db", "additive_noise",
    "convolutional_noise", "sr", "remarks"
]

class DatasetManager:
    """Manages dataset paths, gathers audio files, and generates a manifest for training."""

    def __init__(self, config_path: str | Path):
        """Initializes the DatasetManager with a configuration file.
        :param config_path: Path to the YAML configuration file containing dataset parameters."""
        self.config_path = Path(config_path)

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        sys_config = self.config["system"]
        path_config = self.config["paths"]

        self.base_root = Path(sys_config["root"])
        self.scratch_root = Path(sys_config["scratch"])
        self.workspace_root = Path(sys_config["workspace"])

        self.sph2pipe_path = self.workspace_root / path_config["sph2pipe"]
        self.audio_manager = AudioManager(sph2pipe_path=str(self.sph2pipe_path))

        self.clean = [self.base_root / p for p in path_config["clean_data"]]
        self.noise = [self.scratch_root / p for p in path_config["noise_data"]]
        self.manifest_dir = self.workspace_root / path_config["manifest_dir"]

        self.target_sr = self.config['audio']['sample_rate']
        self.snr_levels = self.config['audio']['target_snr']


    def get_files(self, source: List[Path], extensions: tuple = ('.wav', '.wv1')) -> List[Path]:
        """Recursively gathers all audio files from clean and noise directories."""
        files = []
        for item in source:
            if not item.exists():
                logger.warning(f"Path not found: {item}")
                continue

            if item.is_file():
                if item.suffix in extensions:
                    files.append(item)
            elif item.is_dir():
                for ext in extensions:
                    files.extend(item.rglob(f"*{ext}"))
        return sorted(files)

    def get_clean_files(self, extensions: tuple = ('.wav', '.wv1')) -> List[Path]:
        """Recursively gathers clean audio files from configured directories."""
        return self.get_files(self.clean, extensions)

    def get_noise_files(self, extensions: tuple = ('.wav',)) -> List[Path]:
        """Recursively gathers noise audio files from configured directories."""
        return self.get_files(self.noise, extensions)

    def generate_manifest(self, output_filename: str = "manifest.csv") -> Path:
        """Generates a manifest file containing metadata for all clean and noise audio files."""
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_dir / output_filename

        clean_files = self.get_clean_files()
        noise_files = self.get_noise_files()

        if not clean_files:
            logger.warning("No clean audio files found.")
        if not noise_files:
            logger.warning("No noise audio files found.")

        rows = []

        for clean_path in tqdm(clean_files, desc="Processing files", unit="file"):
            duration = self.audio_manager.get_duration_sec(str(clean_path))

            if duration < self.config['audio']['segment_seconds']:
                continue

            for snr in self.snr_levels:

                noise_path = random.choice(noise_files)
                virtual_mix_id = f"virtual_{clean_path.stem}_{noise_path.stem}_{snr}dB"

                rows.append({
                    SCHEMA[0]: str(clean_path),
                    SCHEMA[1]: str(noise_path),
                    SCHEMA[2]: "",
                    SCHEMA[3]: virtual_mix_id,
                    SCHEMA[4]: duration,
                    SCHEMA[5]: snr,
                    SCHEMA[6]: True,
                    SCHEMA[7]: False,
                    SCHEMA[8]: self.target_sr,
                    SCHEMA[9]: "on-the-fly"
                })

        logger.info(f"Writing {len(rows)} records to CSV...")
        df = pd.DataFrame(rows, columns=SCHEMA)
        df.to_csv(manifest_path, index=False)
        logger.info(f"Manifest successfully saved to {manifest_path}")
        return manifest_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    config_file = "config/metacentrum.yaml"

    logger.info("--- Starting DatasetManager Test ---")
    try:
        manager = DatasetManager(config_path=config_file)

        clean_files = manager.get_clean_files()
        noise_files = manager.get_noise_files()
        logger.info(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files.")

        logger.info("Generating train_manifest.csv...")
        manager.generate_manifest(output_filename="train_manifest.csv")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)