import argparse

import yaml
import torch
from enhancement.models.cmgan.generator import TSCNet as Generator
from enhancement.dataset.loader import get_dataloader
from enhancement.evaluation.evaluator import ModelEvaluator
from enhancement.dataset.audio import AudioManager
from enhancement.evaluation.metrics import SpeechMetrics as MetricsManager

from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():

    ArgsParser = argparse.ArgumentParser()

    ArgsParser.add_argument("--single", type=str, required=False, help="Flag to evaluate a single audio file instead of the whole dataset")
    args = ArgsParser.parse_args()

    marked_single = args.single is not None

    with open("config/metacentrum.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Open configuration file")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")
    model = Generator().to(device)

    checkpoint_path = "checkpoints/17573210/cmgan_epoch_6.pth"

    logger.info(f"checkpoint_path: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    workspace_dir = Path(config.get("system", {}).get("workspace", "."))

    sph2pipe_path = workspace_dir / config.get("paths", {}).get("sph2pipe", "")

    audio_manager = AudioManager(sph2pipe_path)

    train_manifest = workspace_dir / config.get("paths", {}).get("train_manifest", "manifest/train.csv")

    test_loader  = get_dataloader(config, train_manifest, audio_manager, is_train=False)

    logger.info("init metrics manager")
    dnsmos_model_path = Path(config.get("paths", {}).get("dnsmos_model", "models/dnsmos_model.onnx"))
    metrics_manager = MetricsManager(dnsmos_model_path=dnsmos_model_path)

    logger.info("init evaluator")
    evaluator = ModelEvaluator(
        config=config,
        model=model,
        dataloader=test_loader,
        metrics_calc=metrics_manager,
        device=device,
        audio_manager=audio_manager,
        manifest_path=train_manifest
    )

    match marked_single:

        # !!! Exemplarily evaluating a single audio file (Hardcoded paths for demonstration)

        case True:
            logger.info(f"Evaluating single audio file: {args.single}")
            clean_path = Path(
                "/auto/projects-du-praha/CTU_Speech_Lab/data/WSJ/wsj1_CDset/disk1/wsj1/si_tr_s/460/460a010a.wv1")
            noisy_path = Path(
                "/auto/projects-du-praha/CTU_Speech_Lab/scratch/varhavit/mixed_data/1_single_noise_various_snr/460a010a_CAFE_CAFE_1_0dB.wav")
            output_path = Path(
                config.get('evaluation', {}).get('output_dir', 'results/enhanced_audio')) / "460a010a_enhanced.wav"
            evaluator.evaluate_one(clean_path, noisy_path, output_path)
        case False:
            evaluator.evaluate()


if __name__ == "__main__":
    main()
