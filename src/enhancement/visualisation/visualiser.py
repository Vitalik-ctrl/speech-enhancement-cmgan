import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioVisualizer:
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 100):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.noverlap = self.n_fft - self.hop_length

        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_spectrogram(self, audio: np.ndarray, title: str, output_path: str | Path):
        """Generates and saves a professional spectrogram with a dB colorbar."""
        fig, ax = plt.subplots(figsize=(10, 4))

        Pxx, freqs, bins, im = ax.specgram(
            audio,
            Fs=self.sr,
            NFFT=self.n_fft,
            noverlap=(self.n_fft - self.hop_length),
            cmap='magma',
            scale='dB'
        )

        ax.set_title(f"{title} - Spectrogram", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)

        # cbar = fig.colorbar(im, ax=ax, format='%+2.0f dB')
        # cbar.set_label('Magnitude (dB)', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved spectrogram to {output_path}")

    def plot_error_map(self, clean: np.ndarray, enhanced: np.ndarray, title: str, output_path: str | Path):
        """Generates a Delta Spectrogram (Enhanced - Clean) to expose over/under suppression."""
        fig, ax = plt.subplots(figsize=(10, 4))

        clean_spec, freqs, bins, _ = ax.specgram(clean, Fs=self.sr, NFFT=self.n_fft, noverlap=self.noverlap)
        enh_spec, _, _, _ = ax.specgram(enhanced, Fs=self.sr, NFFT=self.n_fft, noverlap=self.noverlap)
        plt.cla()

        clean_db = 10 * np.log10(clean_spec + 1e-10)
        enh_db = 10 * np.log10(enh_spec + 1e-10)

        error_db = enh_db - clean_db

        im = ax.pcolormesh(bins, freqs, error_db, cmap='RdBu_r', vmin=-30, vmax=30, shading='gouraud')

        ax.set_title(f"{title} (Red = Artifacts, Blue = Lost Speech)", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved error map to {output_path}")

    def plot_attention_map(self, attn_matrix: np.ndarray, title: str, output_path: str | Path):
        """Generates a 2D heatmap of the self-attention weights."""
        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(attn_matrix, cmap='viridis', origin='lower', aspect='auto')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Target Frame (Time)", fontsize=12)
        ax.set_ylabel("Source Context (Time)", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved attention map to {output_path}")

    def generate_all_plots(self, clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray, base_name: str,
                           output_dir: str | Path):
        """Generates exactly 7 independent visualization files."""
        os.makedirs(output_dir, exist_ok=True)
        out_dir = Path(output_dir)

        signals = {
            "Clean": clean,
            "Noisy": noisy,
            "Enhanced": enhanced
        }

        for name, audio in signals.items():
            if audio is not None:
                self.plot_spectrogram(audio, name, out_dir / f"{base_name}_{name.lower()}_spectrogram.png")

        if clean is not None and enhanced is not None:
            self.plot_error_map(
                clean,
                enhanced,
                "Spectrogram Error Map",
                out_dir / f"{base_name}_error_map.png"
            )
