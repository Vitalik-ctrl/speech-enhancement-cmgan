import logging
import random
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
import yaml
from collections import defaultdict

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

        clean_paths = path_config.get("clean_data") or []
        self.clean = [self.base_root / p for p in clean_paths]

        noise_paths = path_config.get("noise_data") or []
        self.noise = [self.scratch_root / p for p in noise_paths]

        ir_paths = path_config.get("ir_data") or []
        self.rir = [self.scratch_root / p for p in ir_paths]

        self.manifest_dir = self.workspace_root / path_config["manifest_dir"]
        self.target_sr = self.config['audio']['sample_rate']
        self.snr_levels = self.config['audio']['target_snr']

    def get_files(self, source: List[Path], extensions: tuple = ('.wav', '.wv1')) -> List[Path]:
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

<<<<<<< HEAD
    def get_clean_files(self, extensions: tuple = ('.wav', '.wv1'), max_per_dir: int = 100) -> List[Path]:
=======
    def get_clean_files(self, extensions: tuple = ('.wav', '.wv1'), max_per_dir: int = 90) -> List[Path]:
        """Recursively gathers clean audio files, limiting the amount taken from each speaker folder."""
>>>>>>> refactor/training-routine
        all_files = self.get_files(self.clean, extensions)

        grouped_files = defaultdict(list)
        for f in all_files:
            grouped_files[f.parent].append(f)

        limited_files = []
        for directory, files in grouped_files.items():
            limited_files.extend(files[:max_per_dir])
        return sorted(limited_files)

    def get_noise_files(self, extensions: tuple = ('.wav',)) -> List[Path]:
        return self.get_files(self.noise, extensions)

    def get_impulse_response_files(self, extensions: tuple = ('.wav',)) -> List[Path]:
        return self.get_files(self.rir, extensions)

    def generate_manifest(self, output_filename: str = "manifest.csv") -> Path:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_dir / output_filename

        clean_files = self.get_clean_files()
        noise_files = self.get_noise_files()
        rir_files = self.get_impulse_response_files()

        if not clean_files:
            logger.warning("No clean audio files found. Generating manifest will likely fail or be empty.")

        valid_mix_types = []
        if noise_files:
            valid_mix_types.append("additive")
        else:
            logger.warning("No additive noise files found. Proceeding without additive noise.")

        if rir_files:
            valid_mix_types.append("reverberation")
        else:
            logger.warning("No impulse response files found. Proceeding without reverberation.")

        if noise_files and rir_files:
            valid_mix_types.append("both")

        if not valid_mix_types:
            valid_mix_types.append("clean")
            logger.warning("No noise or reverb files found. Generating manifest with pure clean audio only.")

        rows = []

        for clean_path in tqdm(clean_files, desc="Processing files", unit="file"):
            duration = self.audio_manager.get_duration_sec(str(clean_path))

            if duration < self.config['audio']['segment_seconds']:
                continue

            snr = random.choice(self.snr_levels)

            mix_type = random.choice(valid_mix_types)

            is_add = mix_type in ["additive", "both"]
            is_rev = mix_type in ["reverberation", "both"]

            noise_path = random.choice(noise_files) if is_add and noise_files else ""
            rir_path = random.choice(rir_files) if is_rev and rir_files else ""

            noise_stem = noise_path.stem if hasattr(noise_path, 'stem') else "none"
            virtual_mix_id = f"virtual_{clean_path.stem}_{noise_stem}_{snr}dB"

            rows.append({
                SCHEMA[0]: str(clean_path),
                SCHEMA[1]: str(noise_path),
                SCHEMA[2]: str(rir_path),
                SCHEMA[3]: virtual_mix_id,
                SCHEMA[4]: duration,
                SCHEMA[5]: snr,
                SCHEMA[6]: is_add,
                SCHEMA[7]: is_rev,
                SCHEMA[8]: self.target_sr,
                SCHEMA[9]: "on-the-fly"
            })

        logger.info(f"Writing {len(rows)} records to CSV...")
        df = pd.DataFrame(rows, columns=SCHEMA)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]

        train_df.to_csv(self.manifest_dir / f"train_{output_filename}", index=False)
        eval_df.to_csv(self.manifest_dir / f"eval_{output_filename}", index=False)

        logger.info(f"Saved {len(train_df)} training and {len(eval_df)} evaluation records.")
        return manifest_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    config_file = "config/metacentrum.yaml"

    logger.info("--- Starting DatasetManager ---")
    try:
        manager = DatasetManager(config_path=config_file)

        clean_files = manager.get_clean_files()
        noise_files = manager.get_noise_files()
        impulse_response_files = manager.get_impulse_response_files()
        logger.info(
            f"Found {len(clean_files)} clean files, {len(noise_files)} noise files and {len(impulse_response_files)} impulse responses.")

<<<<<<< HEAD
        manifest_name = "manifest_mix_source_rir_only_06_04_2026.csv"
=======
        manifest_name = "manifest_wsj07_03_2026.csv"
>>>>>>> refactor/training-routine
        logger.info(f"Generating {manifest_name}...")
        manager.generate_manifest(output_filename=manifest_name)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)