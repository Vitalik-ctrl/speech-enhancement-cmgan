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
