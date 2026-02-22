import os
import torch
import numpy as np
import pandas as pd
import logging

from enhancement.models.cmgan.utils import power_compress, power_uncompress

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, config, model, dataloader, metrics_calc, device, audio_manager, manifest_path=None):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.metrics_calc = metrics_calc
        self.device = device
        self.audio_manager = audio_manager

        self.output_dir = config.get('evaluation', {}).get('output_dir', 'results/enhanced_audio')
        self.metrics_csv = config.get('evaluation', {}).get('metrics_csv', 'results/metrics.csv')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)

        self.sr = self.config.get('audio', {}).get('sample_rate', 16000)
        self.n_fft = self.config.get("audio", {}).get("n_fft", 400)
        self.hop = self.config.get("audio", {}).get("hop_length", 100)
        self.window = torch.hamming_window(self.n_fft).to(self.device)

        if manifest_path and os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            self.manifest_df = df[df["additive_noise"] == True].reset_index(drop=True)
        else:
            self.manifest_df = None

    def _enhance_batch(self, noisy_tensor):
        """Shared forward pass logic for enhancing a noisy tensor."""
        c = torch.sqrt(noisy_tensor.size(-1) / torch.sum((noisy_tensor ** 2.0), dim=-1, keepdim=True))
        noisy_norm = noisy_tensor * c

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

        enhanced_waveforms = est_audio / c
        return enhanced_waveforms

    def evaluate(self):
        self.model.eval()
        results = []

        logger.info(f"Starting evaluation on {self.device}...")

        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= 5:
                    logger.info("Reached batch limit. Stopping evaluation loop.")
                    break

                noisy, clean = batch
                noisy = noisy.to(self.device)

                logger.info(f"Processing batch {i + 1}/{len(self.dataloader)} with shape {noisy.shape}...")

                enhanced_waveforms = self._enhance_batch(noisy)

                enhanced_np = enhanced_waveforms.cpu().numpy()
                clean_np = clean.cpu().numpy()
                noisy_np = noisy.cpu().numpy()

                batch_size = noisy.size(0)
                filenames = [f"batch_{i}_sample_{b}.wav" for b in range(batch_size)]

                for b in range(len(filenames)):
                    enh_audio = self._normalize_audio(enhanced_np[b])
                    cln_audio = clean_np[b]
                    nsy_audio = noisy_np[b]
                    filename = filenames[b]

                    out_path = os.path.join(self.output_dir, f"enhanced_{filename}")
                    nsy_path = os.path.join(self.output_dir, f"noisy_{filename}")

                    self.audio_manager.save_audio(out_path, enh_audio, self.sr)
                    self.audio_manager.save_audio(nsy_path, nsy_audio, self.sr)

                    logger.info(f"Scoring {filename}...")

                    metrics_dict = self.metrics_calc.compute_all(cln_audio, enh_audio)
                    metrics_dict['Baseline_PESQ'] = self.metrics_calc.pesq_score(cln_audio, nsy_audio)
                    metrics_dict['Baseline_SNR'] = self.metrics_calc.snr_db(cln_audio, nsy_audio)
                    metrics_dict['filename'] = filename

                    if self.manifest_df is not None:
                        global_idx = i * batch_size + b
                        if global_idx < len(self.manifest_df):
                            row = self.manifest_df.iloc[global_idx]
                            metrics_dict['clean_path'] = row['clean_path']
                            metrics_dict['noise_path'] = row['additive_noise_path']
                            metrics_dict['target_snr_db'] = row['snr_db']
                            metrics_dict['virtual_noisy_path'] = row['noisy_path']

                    results.append(metrics_dict)

        self._save_results(results)

    def evaluate_one(self, clean_path, noisy_path, output_path):
        self.model.eval()

        clean_np, _ = self.audio_manager.load_audio(clean_path, target_sr=self.sr)
        noisy_np, _ = self.audio_manager.load_audio(noisy_path, target_sr=self.sr)

        with torch.no_grad():
            noisy_tensor = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0).to(self.device)

            enhanced_waveforms = self._enhance_batch(noisy_tensor)

            enhanced_np = enhanced_waveforms.squeeze(0).cpu().numpy()
            enhanced_np = self._normalize_audio(enhanced_np)

        metrics_dict = self.metrics_calc.compute_all(clean_np, enhanced_np)
        metrics_dict['Baseline_PESQ'] = self.metrics_calc.pesq_score(clean_np, noisy_np)
        metrics_dict['Baseline_SNR'] = self.metrics_calc.snr_db(clean_np, noisy_np)

        logger.info(f"Metrics for {os.path.basename(noisy_path)}: {metrics_dict}")

        self.audio_manager.save_audio(output_path, enhanced_np, self.sr)

    def _normalize_audio(self, audio):
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            return audio / peak
        return audio

    def _save_results(self, results):
        df = pd.DataFrame(results)
        df.to_csv(self.metrics_csv, index=False)
        logger.info(f"Evaluation complete. Saved to {self.metrics_csv}")

        logger.info("\n--- Average Scores ---")
        for col in df.select_dtypes(include=np.number).columns:
            logger.info(f"Mean {col}: {df[col].mean():.4f}")