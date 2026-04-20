import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import logging
import random
from tqdm import tqdm
import time

from enhancement.models.cmgan.discriminator import batch_pesq
from enhancement.models.cmgan.generator import TSCNet as Generator
from enhancement.dataset.loader import get_dataloader
from enhancement.evaluation.evaluator import Evaluator
from enhancement.dataset.audio import AudioManager
from enhancement.evaluation.metrics import SpeechMetrics as MetricsManager
from enhancement.visualisation.visualiser import AudioVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Turn on deterministic behavior for better comparisons

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def save_results_csv(results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset evaluation to {csv_path}")
    logger.info("\n--- Average Scores ---")
    for col in df.select_dtypes(include=np.number).columns:
        logger.info(f"Mean {col}: {df[col].mean():.4f}")


def evaluate_single_additive(args, audio_manager, evaluator, visualizer, sr, output_dir):
    logger.info(f"Evaluating single additive mix at {args.snr}dB SNR...")
    clean_np, _ = audio_manager.load_audio(args.single_clean, target_sr=sr)
    noise_np, _ = audio_manager.load_audio(args.single_noise, target_sr=sr)

    noisy_np = audio_manager.mix_at_snr(clean_np, noise_np, args.snr)
    noisy_tensor = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0)
    enhanced_np = evaluator.enhance_tensor(noisy_tensor).squeeze(0)

    metrics = evaluator.score_pair(clean_np, noisy_np, enhanced_np)
    logger.info(f"Metrics: {metrics}")

    base_name = Path(args.single_clean).stem
    audio_manager.save_audio(f"{output_dir}/{base_name}_snr{args.snr}_enhanced.wav", enhanced_np, sr)
    audio_manager.save_audio(f"{output_dir}/{base_name}_snr{args.snr}_noisy.wav", noisy_np, sr)

    visualizer.generate_all_plots(clean_np, noisy_np, enhanced_np, base_name, output_dir)
    visualizer.log(metrics, path=output_dir)


def evaluate_single_reverb(args, audio_manager, evaluator, visualizer, sr, output_dir):
    logger.info(f"Evaluating single reverberant mix at {args.snr}dB DRR...")
    clean_np, _ = audio_manager.load_audio(args.single_clean, target_sr=sr)
    rir_np, _ = audio_manager.load_audio(args.single_reverberation, target_sr=sr)

    noisy_np = audio_manager.convolve_at_drr(clean_np, rir_np, drr_db=args.snr)
    noisy_tensor = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0)
    enhanced_np = evaluator.enhance_tensor(noisy_tensor).squeeze(0)

    metrics = evaluator.score_pair(clean_np, noisy_np, enhanced_np)
    logger.info(f"Metrics: {metrics}")

    base_name = f"{Path(args.single_clean).stem}_reverb_drr{args.snr}"
    audio_manager.save_audio(f"{output_dir}/{base_name}_enhanced.wav", enhanced_np, sr)
    audio_manager.save_audio(f"{output_dir}/{base_name}_noisy.wav", noisy_np, sr)

    visualizer.generate_all_plots(clean_np, noisy_np, enhanced_np, base_name, output_dir)
    visualizer.log(metrics, path=output_dir)


def evaluate_dataset(config, audio_manager, evaluator, output_dir, sr, pesq_only=False, fast_mode=False):
    workspace_dir = Path(config.get("system", {}).get("workspace", "."))
    eval_manifest = workspace_dir / config.get("paths", {}).get("final_eval_manifest", "manifest/test_manifest_final.csv")
    test_loader = get_dataloader(config, eval_manifest, audio_manager, is_train=False)

    manifest_df = pd.read_csv(eval_manifest)
    manifest_records = manifest_df.to_dict('records')

    results = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating Dataset")):
            if fast_mode and i >= 10: break

            noisy, clean = batch

            enhanced_waveforms = evaluator._forward_pass(noisy)

            clean_np = clean.cpu().numpy()
            noisy_np = noisy.cpu().numpy()

            if pesq_only:
                clean_list = list(clean_np)
                enhanced_list = list(enhanced_waveforms)

                batch_scores = batch_pesq(clean_list, enhanced_list)

                for b in range(noisy.size(0)):
                    score = batch_scores[b] if batch_scores is not None else float('nan')
                    metrics_dict = {'PESQ_WB': score}

                    global_idx = i * noisy.size(0) + b
                    if global_idx < len(manifest_records):
                        row = manifest_records[global_idx]
                        metrics_dict.update({'target_snr_db': row.get('snr_db', '')})
                    results.append(metrics_dict)

            else:
                for b in range(noisy.size(0)):
                    filename = f"batch_{i}_sample_{b}.wav"

                    if i < 2:
                        audio_manager.save_audio(f"{output_dir}/enhanced_{filename}", enhanced_waveforms[b], sr)
                        audio_manager.save_audio(f"{output_dir}/noisy_{filename}", noisy_np[b], sr)

                    metrics_dict = evaluator.score_pair(clean_np[b], noisy_np[b], enhanced_waveforms[b],
                                                        fast_pesq_only=False)
                    metrics_dict['filename'] = filename

                    global_idx = i * noisy.size(0) + b
                    if global_idx < len(manifest_records):
                        row = manifest_records[global_idx]
                        metrics_dict.update({
                            'clean_path': row.get('clean_path', ''),
                            'target_snr_db': row.get('snr_db', '')
                        })
                    results.append(metrics_dict)

    return results


def track_checkpoints(args, config, model, audio_manager, metrics_manager, device, output_dir, sr, visualizer):
    logger.info(f"Tracking checkpoints from directory: {args.checkpoint_dir}")
    checkpoint_dir = Path(args.checkpoint_dir)

    checkpoints = []
    for cp in checkpoint_dir.glob("*.pth"):
        match = re.search(r'epoch_(\d+)', cp.name)
        if match:
            checkpoints.append((int(match.group(1)), cp))

    checkpoints.sort(key=lambda x: x[0])

    if not checkpoints:
        logger.error("No valid checkpoints found in directory.")
        return

    epoch_stats = []

    for epoch, cp_path in checkpoints:
        logger.info(f"--- Evaluating Epoch {epoch} ---")

        try:
            model.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
        except RuntimeError as e:
            logger.warning(f"Skipping Epoch {epoch}: Checkpoint file is corrupted! ({e})")
            continue

        evaluator = Evaluator(config, model, metrics_manager, device)

        results = evaluate_dataset(config, audio_manager, evaluator, output_dir, sr, pesq_only=True,
                                   fast_mode=args.fast)

        pesq_scores = []
        for r in results:
            if 'PESQ_WB' in r:
                val = r['PESQ_WB']
                if isinstance(val, torch.Tensor):
                    val = val.cpu().item()
                val = (val * 3.5) + 1.0
                pesq_scores.append(val)

        pesq_scores = [p for p in pesq_scores if not np.isnan(p)]

        if pesq_scores:
            epoch_stats.append({
                'epoch': epoch,
                'mean_pesq': np.mean(pesq_scores),
                'max_pesq': np.max(pesq_scores),
                'min_pesq': np.min(pesq_scores)
            })

            # Indentation fixed so it prints progressively!
            logger.info(
                f"Epoch {epoch} Results | Mean PESQ: {epoch_stats[-1]['mean_pesq']:.4f} | Max: {epoch_stats[-1]['max_pesq']:.4f} | Min: {epoch_stats[-1]['min_pesq']:.4f}")

    if epoch_stats:
        home = Path.home()
        plot_dir = home / "cmgan_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = plot_dir / f"checkpoint_pesq_trend_{timestamp}.png"

        visualizer.plot_checkpoint_trend(epoch_stats, str(plot_path))
        logger.info(f"Saved unique plot to: {plot_path}")


def evaluate_single_joint(args, audio_manager, evaluator, visualizer, sr, output_dir):
    logger.info(f"Evaluating Joint (RIR + Noise) at {args.snr}dB SNR/DRR...")
    clean_np, _ = audio_manager.load_audio(args.single_clean, target_sr=sr)
    noise_np, _ = audio_manager.load_audio(args.single_noise, target_sr=sr)
    rir_np, _ = audio_manager.load_audio(args.single_reverberation, target_sr=sr)

    reverb_voice = audio_manager.convolve_at_drr(clean_np, rir_np, drr_db=args.snr)

    noisy_np = audio_manager.mix_at_snr(reverb_voice, noise_np, args.snr)

    noisy_tensor = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0)
    enhanced_np = evaluator.enhance_tensor(noisy_tensor).squeeze(0)

    metrics = evaluator.score_pair(clean_np, noisy_np, enhanced_np)
    logger.info(f"Joint Metrics: {metrics}")

    base_name = f"{Path(args.single_clean).stem}_joint_snr{args.snr}"
    audio_manager.save_audio(f"{output_dir}/{base_name}_enhanced.wav", enhanced_np, sr)
    audio_manager.save_audio(f"{output_dir}/{base_name}_noisy.wav", noisy_np, sr)

    visualizer.generate_all_plots(clean_np, noisy_np, enhanced_np, base_name, output_dir)
    visualizer.log(metrics, path=output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_clean", type=str, help="Path to clean audio")
    parser.add_argument("--single_noise", type=str, help="Path to additive noise audio")
    parser.add_argument("--single_reverberation", type=str, help="Path to RIR audio")
    parser.add_argument("--snr", type=float, default=0.0, help="SNR or DRR level")
    parser.add_argument("--pesq_only", action="store_true", help="Skip heavy metrics for bulk dataset eval")
    parser.add_argument("--checkpoint_dir", type=str, help="Path to directory containing .pth files to plot trend")
    parser.add_argument("--checkpoint", type=str, help="Path to a single .pth model checkpoint to load")
    parser.add_argument("--num_channel", type=int, help="Override model channels (e.g., 64, 96, 128)")

    parser.add_argument("--fast", action="store_true", help="Evaluate only 10 batches per epoch for ultra-fast graphing")
    args = parser.parse_args()

    with open("config/metacentrum.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.num_channel:
        num_channel = args.num_channel
        logger.info(f"Overriding config! Using num_channel: {num_channel}")
    else:
        num_channel = config.get("model", {}).get("num_channel", 64)

    model = Generator(num_channel=num_channel).to(device)

    workspace_dir = Path(config.get("system", {}).get("workspace", "."))
    audio_manager = AudioManager(workspace_dir / config.get("paths", {}).get("sph2pipe", ""))
    dnsmos_model = workspace_dir / config.get("paths", {}).get("dnsmos_model", "models/dnsmos_model.onnx")
    metrics_manager = MetricsManager(dnsmos_model_path=dnsmos_model)

    sr = config.get('audio', {}).get('sample_rate', 16000)
    output_dir = config.get('evaluation', {}).get('output_dir', 'results/enhanced_audio')
    os.makedirs(output_dir, exist_ok=True)

    visualizer = AudioVisualizer(
        sample_rate=sr,
        n_fft=config.get("audio", {}).get("n_fft", 400),
        hop_length=config.get("audio", {}).get("hop_length", 100)
    )

    if args.checkpoint_dir:
        track_checkpoints(args, config, model, audio_manager, metrics_manager, device, output_dir, sr, visualizer)

    else:
        if not args.checkpoint:
            logger.error("You must provide a --checkpoint path unless using --checkpoint_dir")
            return

        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        evaluator = Evaluator(config, model, metrics_manager, device)

        if args.single_clean and args.single_noise and args.single_reverberation:
            evaluate_single_joint(args, audio_manager, evaluator, visualizer, sr, output_dir)

        elif args.single_clean and args.single_noise:
            evaluate_single_additive(args, audio_manager, evaluator, visualizer, sr, output_dir)

        elif args.single_clean and args.single_reverberation:
            evaluate_single_reverb(args, audio_manager, evaluator, visualizer, sr, output_dir)

        else:
            logger.info("No single files provided. Running full dataset evaluation.")
            results = evaluate_dataset(
                config,
                audio_manager,
                evaluator,
                output_dir,
                sr,
                pesq_only=args.pesq_only,
                fast_mode=args.fast
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = args.checkpoint.split("/")[-1].replace(".pth", "")

            default_csv_name = f'{output_dir}/metrics_{checkpoint_name}_{timestamp}.csv'

            csv_path = config.get('evaluation', {}).get('metrics_csv', default_csv_name)

            save_results_csv(results, csv_path)


if __name__ == "__main__":
    main()
