import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

from enhancement.models.cmgan.generator import TSCNet as Generator
from enhancement.dataset.loader import get_dataloader
from enhancement.evaluation.evaluator import Evaluator
from enhancement.dataset.audio import AudioManager
from enhancement.evaluation.metrics import SpeechMetrics as MetricsManager
from enhancement.visualisation.visualiser import AudioVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_results_csv(results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Evaluation complete. Saved to {csv_path}")
    logger.info("\n--- Average Scores ---")
    for col in df.select_dtypes(include=np.number).columns:
        logger.info(f"Mean {col}: {df[col].mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_clean", type=str, help="Path to clean audio for single testing")
    parser.add_argument("--single_noise", type=str, help="Path to noise audio for single testing")
    parser.add_argument("--snr", type=float, default=0.0, help="SNR level for single testing")
    args = parser.parse_args()

    with open("config/metacentrum.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = Generator().to(device)
    checkpoint_path = "/storage/praha5-elixir/home/varhavit/cmgan_checkpoints/17593823.pbs-m1.metacentrum.cz/17593823/cmgan_epoch_14_valG_0.1261.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    workspace_dir = Path(config.get("system", {}).get("workspace", "."))
    audio_manager = AudioManager(workspace_dir / config.get("paths", {}).get("sph2pipe", ""))
    dnsmos_model = workspace_dir / config.get("paths", {}).get("dnsmos_model", "models/dnsmos_model.onnx")
    metrics_manager = MetricsManager(dnsmos_model_path=dnsmos_model)
    evaluator = Evaluator(config, model, metrics_manager, device)

    visualizer = AudioVisualizer(
        sample_rate=16000,
        n_fft=config.get("audio", {}).get("n_fft", 400),
        hop_length=config.get("audio", {}).get("hop_length", 100)
    )

    sr = config.get('audio', {}).get('sample_rate', 16000)
    output_dir = config.get('evaluation', {}).get('output_dir', 'results/enhanced_audio')
    os.makedirs(output_dir, exist_ok=True)

    if args.single_clean and args.single_noise:

        # SINGLE FILE TESTING
        logger.info(f"Evaluating single mix at {args.snr}dB SNR...")
        clean_np, _ = audio_manager.load_audio(args.single_clean, target_sr=sr)
        noise_np, _ = audio_manager.load_audio(args.single_noise, target_sr=sr)

        noisy_np = audio_manager.mix_at_snr(clean_np, noise_np, args.snr)

        noisy_tensor = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0)
        enhanced_np = evaluator.enhance_tensor(noisy_tensor).squeeze(0)

        # raw_attn = model.TSCB_4.time_conformer.attn.fn.saved_attention
        # single_head_matrix = raw_attn[0].mean(dim=0).numpy()
        # logger.info(f"Attention map: {single_head_matrix}")

        metrics = evaluator.score_pair(clean_np, noisy_np, enhanced_np)
        logger.info(f"Metrics: {metrics}")

        base_name = Path(args.single_clean).stem
        audio_manager.save_audio(f"{output_dir}/{base_name}_snr{args.snr}_enhanced.wav", enhanced_np, sr)
        audio_manager.save_audio(f"{output_dir}/{base_name}_snr{args.snr}_noisy.wav", noisy_np, sr)

        visualizer.generate_all_plots(
            clean=clean_np,
            noisy=noisy_np,
            enhanced=enhanced_np,
            base_name=base_name,
            output_dir=output_dir
        )

        visualizer.plot_attention_map(
            single_head_matrix,
            title="Time-Conformer Self-Attention (Layer 1)",
            output_path=f"{output_dir}/{base_name}_attention.png"
        )

        visualizer.log(metrics)

    else:
        # FULL DATASET EVALUATION
        eval_manifest = workspace_dir / config.get("paths", {}).get("eval_manifest", "manifest/eval_manifest_wsj.csv")
        test_loader = get_dataloader(config, eval_manifest, audio_manager, is_train=False)
        manifest_df = pd.read_csv(eval_manifest)
        manifest_df = manifest_df[manifest_df["additive_noise"] == True].reset_index(drop=True)

        results = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):

                if i >= 10: break

                noisy, clean = batch
                enhanced_waveforms = evaluator.enhance_tensor(noisy)

                enhanced_np = enhanced_waveforms
                clean_np = clean.cpu().numpy()
                noisy_np = noisy.cpu().numpy()

                for b in range(noisy.size(0)):
                    filename = f"batch_{i}_sample_{b}.wav"

                    audio_manager.save_audio(f"{output_dir}/enhanced_{filename}", enhanced_np[b], sr)
                    audio_manager.save_audio(f"{output_dir}/noisy_{filename}", noisy_np[b], sr)

                    metrics_dict = evaluator.score_pair(clean_np[b], noisy_np[b], enhanced_np[b])
                    metrics_dict['filename'] = filename

                    global_idx = i * noisy.size(0) + b
                    if global_idx < len(manifest_df):
                        row = manifest_df.iloc[global_idx]
                        metrics_dict.update({'clean_path': row['clean_path'], 'noise_path': row['additive_noise_path'],
                                             'target_snr_db': row['snr_db']})

                    results.append(metrics_dict)

        metrics_csv = config.get('evaluation', {}).get('metrics_csv', 'results/metrics.csv')
        save_results_csv(results, metrics_csv)


if __name__ == "__main__":
    main()
