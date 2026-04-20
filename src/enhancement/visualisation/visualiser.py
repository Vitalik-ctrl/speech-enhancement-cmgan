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

        self.vmin = -125
        self.vmax = -30

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('ggplot')

    def plot_spectrogram(self, audio: np.ndarray, title: str, output_path: str | Path):
        """Generates a spectrogram and prints dB stats to find the perfect boundaries."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        else:
            logger.warning(f"Audio for {title} is silent!")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        Pxx, freqs, bins, im = ax.specgram(
            audio,
            Fs=self.sr,
            NFFT=self.n_fft,
            noverlap=self.noverlap,
            cmap='inferno',
            scale='dB',
            vmin=self.vmin,
            vmax=self.vmax
        )

        Pxx_db = 10 * np.log10(Pxx + 1e-10)
        db_min, db_max = np.min(Pxx_db), np.max(Pxx_db)
        db_mean = np.mean(Pxx_db)

        print(f"\n--- {title} Spectrogram Stats ---")
        print(f"Max dB: {db_max:.2f}")
        print(f"Min dB: {db_min:.2f}")
        print(f"Mean dB: {db_mean:.2f}")

        im.set_clim(self.vmin, self.vmax)

        ax.set_xlabel("Time (s)", fontsize=24)
        ax.set_ylabel("Frequency (Hz)", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)

        # cbar = fig.colorbar(im, ax=ax, format='%+2.0f dB')
        # cbar.set_label('Magnitude (dB)', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved spectrogram to {output_path}")

    def plot_waveform(self, audio: np.ndarray, title: str, output_path: str | Path):
        """Generates and saves a time-domain waveform plot."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        else:
            logger.warning(f"Audio for {title} is silent!")
            return

        # times = np.arange(len(audio)) / self.sr

        fig, ax = plt.subplots(figsize=(10, 3))

        ax.plot(audio, color='#007fd3', linewidth=0.3)

        # ax.set_title(f"{title} - Waveform", fontsize=14)
        # ax.set_xlabel("Time (s)", fontsize=12)
        # ax.set_ylabel("Amplitude", fontsize=12)

        ax.set_axis_off()

        ax.set_ylim(-1.05, 1.05)

        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)

    def plot_error_map(self, clean: np.ndarray, enhanced: np.ndarray, title: str, output_path: str | Path):
        """Generates a Delta Spectrogram (Enhanced - Clean) to expose over/under suppression."""
        min_len = min(len(clean), len(enhanced))
        clean, enhanced = clean[:min_len], enhanced[:min_len]

        fig_temp, ax_temp = plt.subplots()
        clean_spec, freqs, bins, _ = ax_temp.specgram(clean, Fs=self.sr, NFFT=self.n_fft, noverlap=self.noverlap)
        enh_spec, _, _, _ = ax_temp.specgram(enhanced, Fs=self.sr, NFFT=self.n_fft, noverlap=self.noverlap)
        plt.close(fig_temp)

        clean_db = 10 * np.log10(clean_spec + 1e-10)
        enh_db = 10 * np.log10(enh_spec + 1e-10)

        error_db = enh_db - clean_db

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.pcolormesh(bins, freqs, error_db, cmap='RdBu_r', vmin=-20, vmax=20, shading='gouraud')

        # ax.set_title(f"{title} (Red = Artifacts, Blue = Lost Speech)", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        # fig.colorbar(im, ax=ax, label='Delta dB')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved error map to {output_path}")

    def plot_attention_map(self, attn_matrix: np.ndarray, title: str, output_path: str | Path):
        """Generates a 2D heatmap of the self-attention weights."""
        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(attn_matrix, cmap='inferno', origin='lower', aspect='auto')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Target Frame (Time)", fontsize=12)
        ax.set_ylabel("Source Context (Time)", fontsize=12)
        fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved attention map to {output_path}")

    def log(self, metrics: dict):
        """Saves a simple text file with metric values for easy reference."""
        log_path = Path(metrics.get("output_dir", ".")) / f"{metrics.get('base_name', 'metrics')}_metrics.txt"
        with open(log_path, "w") as f:
            for key, value in metrics.items():
                if key not in ["output_dir", "base_name"]:
                    f.write(f"{key}: {value}\n")
        logger.info(f"Saved metrics log to {log_path}")

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
                self.plot_waveform(audio, name, out_dir / f"{base_name}_{name.lower()}_waveform.png")

        if clean is not None and enhanced is not None:
            self.plot_error_map(
                clean,
                enhanced,
                "Spectrogram Error Map",
                out_dir / f"{base_name}_error_map.png"
            )
