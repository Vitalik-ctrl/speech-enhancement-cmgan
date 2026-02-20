import subprocess
import io
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple
import logging
import yaml

logger = logging.getLogger(__name__)

class AudioManager:
    """Utility class for loading audio files, handling both standard WAV and NIST Sphere (.wv1/.wv2) formats."""

    def __init__(self, sph2pipe_path: str):
        self.sph2pipe_path = sph2pipe_path

    def call_sph2pipe(self, path: str) -> Tuple[np.ndarray, int]:
        """Calls sph2pipe to decode NIST Sphere files and returns audio data and sample rate."""
        cmd = [self.sph2pipe_path, '-f', 'wav', path]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, sr = sf.read(io.BytesIO(proc.stdout))
        return data, sr

    def load_audio(self, path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Loads audio from the given path, handling both standard WAV and NIST Sphere formats."""
        path_str = str(path)

        if path_str.endswith('.wv1') or path_str.endswith('.wv2'):
            data, sr = self.call_sph2pipe(path_str)
        else:
            data, sr = sf.read(path_str)

        if len(data.shape) > 1:
            logging.warning(f"Audio {path} has {data.shape[1]} channels, converting to mono")
            data = np.mean(data, axis=1)

        if sr != target_sr:
            logging.warning(f"Resampling {path} from {sr} Hz to {target_sr} Hz")
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return data, sr

    def get_duration_sec(self, path: str, sr: int = 16000) -> float:
        """Calculates duration of the audio file in seconds, using load_audio to handle both formats."""
        path_str = str(path)

        if not path_str.endswith(('.wv1', '.wv2')):
            try:
                return round(sf.info(path_str).duration, 2)
            except Exception:
                logging.warning(f"sf.info failed for {path}, falling back to load_audio")
                pass

        try:
            data, sr = self.load_audio(path)
            return round(len(data) / sr, 2)
        except Exception as e:
            logging.warning(f"Failed to get duration for {path}: {e}")
            return 0.0


if __name__ == "__main__":
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    config_path = Path("config/metacentrum.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}. Run this from the project root.")
        exit(1)

    workspace_root = Path(config['system']['workspace'])
    scratch_root = Path(config['system']['scratch'])
    system_root = Path(config['system']['root'])

    sph2pipe_exe = workspace_root / config['paths']['sph2pipe']
    clean_dir = system_root / config['paths']['clean_data'][0]
    test_wav = scratch_root / config['paths']['noise_data'][0]

    test_wv1 = next(clean_dir.rglob("*.wv1"), None)

    logger.info("--- Initializing AudioManager ---")
    audio_mgr = AudioManager(sph2pipe_path=str(sph2pipe_exe))

    if test_wv1:
        logger.info(f"--- Testing WV1: {test_wv1.name} ---")
        wv1_dur = audio_mgr.get_duration_sec(str(test_wv1))
        logger.info(f"WV1 Duration: {wv1_dur} seconds")

        wv1_data, wv1_sr = audio_mgr.load_audio(str(test_wv1), target_sr=16000)
        logger.info(f"WV1 Loaded -> Shape: {wv1_data.shape}, SR: {wv1_sr}")
    else:
        logger.warning(f"No .wv1 files found in {clean_dir}")

    if test_wav.exists():
        logger.info(f"--- Testing WAV: {test_wav.name} ---")
        wav_dur = audio_mgr.get_duration_sec(str(test_wav))
        logger.info(f"WAV Duration: {wav_dur} seconds")

        wav_data, wav_sr = audio_mgr.load_audio(str(test_wav), target_sr=16000)
        logger.info(f"WAV Loaded -> Shape: {wav_data.shape}, SR: {wav_sr}")
    else:
        logger.warning(f"Noise file not found at {test_wav}")

    logger.info("--- Tests Complete ---")