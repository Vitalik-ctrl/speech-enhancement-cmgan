import torch
import numpy as np
import logging

from enhancement.models.cmgan.utils import power_compress, power_uncompress

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config, model, metrics_calc, device):
        self.config = config
        self.model = model
        self.metrics_calc = metrics_calc
        self.device = device

        self.sr = self.config.get('audio', {}).get('sample_rate', 16000)
        self.n_fft = self.config.get("audio", {}).get("n_fft", 400)
        self.hop = self.config.get("audio", {}).get("hop_length", 100)
        self.window = torch.hamming_window(self.n_fft).to(self.device)

        self.max_chunk_samples = 4 * self.sr

    def enhance_tensor(self, noisy_tensor: torch.Tensor) -> np.ndarray:
        """Passes a noisy PyTorch tensor through the model. Handles chunking for long files."""
        self.model.eval()

        if noisy_tensor.size(-1) <= self.max_chunk_samples:
            return self._forward_pass(noisy_tensor)

        logger.info(f"Audio is long ({noisy_tensor.size(-1)} samples). Processing in chunks...")
        total_samples = noisy_tensor.size(-1)
        enhanced_chunks = []

        with torch.no_grad():
            for start in range(0, total_samples, self.max_chunk_samples):
                end = min(start + self.max_chunk_samples, total_samples)
                chunk = noisy_tensor[:, start:end]

                enhanced_chunk_np = self._forward_pass(chunk)
                enhanced_chunks.append(enhanced_chunk_np)

        final_audio = np.concatenate(enhanced_chunks, axis=-1)
        return final_audio

    def _forward_pass(self, noisy_batch: torch.Tensor) -> np.ndarray:
        """Processes an entire batch of audio at once on the GPU."""
        with torch.no_grad(), torch.autocast(device_type='cuda',
                                             dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
            noisy_batch = noisy_batch.to(self.device)

            c = torch.sqrt(noisy_batch.size(-1) / torch.sum((noisy_batch ** 2.0), dim=-1, keepdim=True))
            noisy_norm = noisy_batch * c

            noisy_spec = torch.stft(noisy_norm, self.n_fft, self.hop, window=self.window, onesided=True,
                                    return_complex=True)
            noisy_spec = torch.view_as_real(noisy_spec)
            noisy_spec = power_compress(noisy_spec)
            noisy_spec = noisy_spec.permute(0, 1, 3, 2)

            est_real, est_imag = self.model(noisy_spec)

            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_spec_complex = torch.complex(est_spec_uncompress[..., 0], est_spec_uncompress[..., 1])

            est_audio = torch.istft(est_spec_complex.transpose(1, 2), self.n_fft, self.hop, window=self.window,
                                    onesided=True)

            enhanced_waveforms = (est_audio / c).float()
            enhanced_np = enhanced_waveforms.cpu().numpy()

            peaks = np.max(np.abs(enhanced_np), axis=-1, keepdims=True)
            mask = peaks > 1.0
            enhanced_np = np.where(mask, enhanced_np / np.where(mask, peaks, 1.0), enhanced_np)

        return enhanced_np

    def score_pair(self, clean_np: np.ndarray, noisy_np: np.ndarray, enhanced_np: np.ndarray,
                   fast_pesq_only: bool = False) -> dict:
        """Calculates metrics. Can skip heavy neural net metrics if fast_pesq_only is True."""

        if fast_pesq_only:
            return {
                'PESQ_WB': self.metrics_calc.pesq_score(clean_np, enhanced_np),
                'Baseline_PESQ': self.metrics_calc.pesq_score(clean_np, noisy_np)
            }

        metrics_dict = self.metrics_calc.compute_all(clean_np, enhanced_np)
        metrics_dict['Baseline_PESQ'] = self.metrics_calc.pesq_score(clean_np, noisy_np)
        metrics_dict['Baseline_ESTOI'] = self.metrics_calc.estoi_score(clean_np, noisy_np)
        metrics_dict['Baseline_SNR'] = self.metrics_calc.snr_db(clean_np, noisy_np)
        metrics_dict['Baseline_DNSMOS_OVRL'] = self.metrics_calc.dnsmos_score(noisy_np)['OVRL'] if self.metrics_calc.dnsmos_session else float('nan')
        metrics_dict['Baseline_DNSMOS_SIG'] = self.metrics_calc.dnsmos_score(noisy_np)['SIG'] if self.metrics_calc.dnsmos_session else float('nan')
        metrics_dict['Baseline_DNSMOS_BAK'] = self.metrics_calc.dnsmos_score(noisy_np)['BAK'] if self.metrics_calc.dnsmos_session else float('nan')
        return metrics_dict
