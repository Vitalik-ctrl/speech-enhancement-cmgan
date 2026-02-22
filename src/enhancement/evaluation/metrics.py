import numpy as np
import torch
import onnxruntime as ort
import logging
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio

logger = logging.getLogger(__name__)

# [Vitalii Varhanik] 11/2025 implementation

class SpeechMetrics:
    def __init__(self, dnsmos_model_path: str = None, sample_rate: int = 16000):
        self.sr = sample_rate

        self.srmr_calc = SpeechReverberationModulationEnergyRatio(fs=self.sr)
        self.dnsmos_session = None
        if dnsmos_model_path:
            try:
                self.dnsmos_session = ort.InferenceSession(dnsmos_model_path)
            except Exception as e:
                logger.warning(f"Could not load DNSMOS model from {dnsmos_model_path}: {e}")
        else:
            logger.warning("No DNSMOS model path provided. DNSMOS metric will be skipped.")

    def compute_all(self, clean: np.ndarray, enhanced: np.ndarray) -> dict:
        length = min(len(clean), len(enhanced))
        clean = clean[:length]
        enhanced = enhanced[:length]

        metrics = {
            'PESQ_WB': self.pesq_score(clean, enhanced),
            'ESTOI': self.estoi_score(clean, enhanced),
            'SNR_dB': self.snr_db(clean, enhanced),
            'SI_SDR_dB': self.si_sdr_db(clean, enhanced),
            'SRMR': self.srmr_score(enhanced),
        }

        if self.dnsmos_session:
            metrics.update(self.dnsmos_score(enhanced))

        return metrics

    def pesq_score(self, clean: np.ndarray, test: np.ndarray) -> float:
        try:
            return pesq(self.sr, clean, test, 'wb')
        except Exception as e:
            logger.error(f"PESQ calculation failed: {e}")
            return float('nan')

    def estoi_score(self, clean: np.ndarray, test: np.ndarray) -> float:
        return stoi(clean, test, self.sr, extended=True)

    def snr_db(self, clean: np.ndarray, test: np.ndarray) -> float:
        Ps = np.mean(clean ** 2)
        Pn = np.mean((clean - test) ** 2) + 1e-12
        return float(10 * np.log10(Ps / Pn))

    def si_sdr_db(self, clean: np.ndarray, est: np.ndarray) -> float:
        s = clean - np.mean(clean)
        sh = est - np.mean(est)
        alpha = np.sum(sh * s) / (np.sum(s ** 2) + 1e-12)
        s_target = alpha * s
        e_noise = sh - s_target
        return float(10 * np.log10((np.sum(s_target ** 2) + 1e-12) / (np.sum(e_noise ** 2) + 1e-12)))

    def srmr_score(self, test: np.ndarray) -> float:
        test_tensor = torch.tensor(test, dtype=torch.float32).unsqueeze(0)
        return float(self.srmr_calc(test_tensor).item())

    def dnsmos_score(self, test: np.ndarray) -> dict:
        if not self.dnsmos_session:
            return {"OVRL": float('nan'), "SIG": float('nan'), "BAK": float('nan')}

        y = test.astype(np.float32)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=0)

        input_info = self.dnsmos_session.get_inputs()[0]
        input_name = input_info.name
        expected_len = input_info.shape[1] if len(input_info.shape) > 1 else len(y[0])

        if y.shape[1] < expected_len:
            y = np.pad(y, ((0, 0), (0, expected_len - y.shape[1])), mode="constant")
        elif y.shape[1] > expected_len:
            y = y[:, :expected_len]

        pred = self.dnsmos_session.run(None, {input_name: y})
        ovr, sig, bak = [float(x) for x in pred[0][0]]
        return {"OVRL": ovr, "SIG": sig, "BAK": bak}
