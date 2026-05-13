# Speech Enhancement using CMGAN

A Conformer-based Metric Generative Adversarial Network (CMGAN) for monaural speech enhancement, with a dynamic on-the-fly mixing pipeline for additive noise and room reverberation. Implementation accompanying the bachelor thesis *Speech Enhancement Using Generative Adversarial Network* (CTU FEE, 2026).

The model operates in the complex Time-Frequency domain. A U-Net generator with a Two-Stage Conformer bottleneck and decoupled (mask + complex) decoders is trained adversarially against a Metric Discriminator that predicts a continuous PESQ-correlated quality score. Training combines clean speech (WSJ, TIMIT, CtuTest, SPEECON, GLOBALPHONE), additive noise (QUT-NOISE), and room impulse responses (OpenAir, SLR28) into unique mixtures generated on every batch.

On the multi-condition test set (additive + reverberation + joint), the released model achieves **PESQ 3.45** (+1.44 over noisy baseline) and **DNSMOS OVRL 3.38** (+0.75).

## Repository Structure

```
speech-enhancement-cmgan/
├── src/enhancement/
│   ├── main.py                       # Training entry point
│   ├── evaluate.py                   # Multi-mode evaluation entry point
│   ├── dataset/
│   │   ├── audio.py                  # AudioManager: I/O, mixing (mix_at_snr, convolve_at_drr)
│   │   ├── manager.py                # DatasetManager: scans corpora, builds CSV manifest
│   │   └── loader.py                 # ModelDataset: on-the-fly stochastic mixing in RAM
│   ├── models/cmgan/
│   │   ├── generator.py              # TSCNet: encoder + decoupled mask/complex decoders
│   │   ├── conformer.py              # Two-Stage Conformer bottleneck (time + frequency)
│   │   ├── discriminator.py          # Metric discriminator (predicts normalized PESQ)
│   │   └── utils.py                  # Power-law compression / uncompression
│   ├── training/
│   │   └── trainer.py                # ModelTrainer: AMP forward, composite loss, optimizer steps
│   ├── evaluation/
│   │   ├── evaluator.py              # Chunked inference for long audio
│   │   └── metrics.py                # SpeechMetrics: PESQ, ESTOI, SI-SDR, SRMR, DNSMOS
│   └── visualisation/
│       └── visualiser.py             # Spectrograms, delta-spectrograms, PESQ-vs-SNR plots
├── config/
│   └── metacentrum.yaml              # Paths, audio params, model, training hyperparameters
├── pyproject.toml
└── README.md
```

A PBS submission template (`metacentrum_template.sh`) for the CESNET MetaCentrum HPC cluster is available on the `core/training_architecture` branch under `templates/hpc/`.

## Setup

### Requirements

- Python 3.10
- PyTorch ≥ 2.0 with CUDA (training requires GPU; CPU works for inference only)
- ~50 GB scratch space for manifests, checkpoints, and processed audio

### Installation

```bash
git clone https://github.com/Vitalik-ctrl/speech-enhancement-cmgan.git
cd speech-enhancement-cmgan
pip install -e .
```

Key dependencies (declared in `pyproject.toml`): `torch`, `torchaudio`, `librosa`, `soundfile`, `pesq`, `pystoi`, `torchmetrics`, `onnxruntime`, `pandas`, `pyyaml`, `tqdm`.

### sph2pipe (required for WSJ)

The WSJ corpus ships in legacy NIST Sphere format (`.wv1`) compressed with the deprecated `shorten` algorithm. `AudioManager` shells out to `sph2pipe` to decode these on-the-fly. Build it from source and place the binary at the path configured in `paths.sph2pipe` (default: `tools/sph2pipe/sph2pipe` relative to `system.workspace`):

```bash
cd $WORKSPACE/tools/sph2pipe
gcc -o sph2pipe *.c -lm
```

If you don't use WSJ, the `sph2pipe` binary is not strictly required — but the manifest builder will still try to resolve the path.

### DNSMOS model

Download the DNSMOS ONNX model and place it at the path configured in `paths.dnsmos_model` (default: `models/dnsmos_model.onnx` relative to `system.workspace`). Without this file the DNSMOS metric is silently skipped during evaluation.

## Configuration

All paths, audio parameters, and hyperparameters live in `config/metacentrum.yaml`. The config uses **three filesystem roots** to match the MetaCentrum storage layout:

- **`system.root`** — read-only persistent storage where raw corpora live (clean speech)
- **`system.scratch`** — fast scratch storage (noise and RIR datasets)
- **`system.workspace`** — read-write workspace (manifests, sph2pipe binary, checkpoints, DNSMOS model)

Paths under `paths.clean_data` are resolved against `system.root`. Paths under `paths.noise_data` and `paths.ir_data` are resolved against `system.scratch`. Manifest, sph2pipe, DNSMOS, and save paths are resolved against `system.workspace`.

Key sections of the config:

| Section | Key fields |
|---|---|
| `system` | `root`, `scratch`, `workspace` |
| `paths` | `sph2pipe`, `clean_data`, `noise_data`, `ir_data`, `manifest_dir`, `train_manifest`, `eval_manifest`, `final_eval_manifest`, `dnsmos_model`, `save_dir` |
| `audio` | `sample_rate` (16000), `n_fft` (400), `hop_length` (100), `segment_seconds` (2.0), `target_snr` (list of dB levels for manifest generation) |
| `training` | `batch_size` (12), `num_workers`, `loss_weights` (`[0.1, 0.9, 0.2, 0.05]` for complex/magnitude/time/adversarial), `epochs` (120), `decay_epoch` (30), `init_lr` (5e-4) |
| `model` | `num_channel` (128) |

The discriminator learning rate is set to **1.6 × `init_lr`** inside the trainer (not configurable separately).

To train one of the model variants (additive specialist, RIR specialist, language-restricted models, Generalist Model, etc.), change only the entries under `paths.clean_data`, `paths.noise_data`, and `paths.ir_data` — architecture and hyperparameters may be identical across all trained variants.

## End-to-End Workflow

### 1. Prepare raw corpora

Arrange the corpora on disk under the three roots configured in the YAML. A typical layout under `system.root` (clean) and `system.scratch` (noise/RIR):

```
${SYSTEM_ROOT}/
├── data/WSJ/wsj1/si_tr_s/...
├── data/timit/timit/train/...
├── data/CtuTest/SHORT/...
├── data/GLOBALPHONE/CZ/adc/...
└── data/cedmo/SPEECON/WAV/ADULT1CS/...

${SYSTEM_SCRATCH}/
├── data/additive_noises/QUT_NOISES/...
└── data/convolutional_noises/...        # OpenAir + SLR28 (exclude synthetic 2D-membrane RIRs)
```

Edit `paths.clean_data`, `paths.noise_data`, and `paths.ir_data` in the YAML to point at these directories (entries are lists; comment any out to exclude them from a run).

### 2. Build the dataset manifest

`DatasetManager` recursively scans the configured directories, caps each speaker subdirectory at 95 files, and emits a CSV manifest that maps each clean utterance to a randomly chosen noise file, RIR, target SNR (drawn from `audio.target_snr`), and degradation flags. Utterances shorter than `audio.segment_seconds` are dropped. The full set is shuffled (seed 42) and split 90 % train / 10 % validation; the test partition is a separate manifest you point at via `paths.final_eval_manifest`.

Manifest schema:

```
clean_path, additive_noise_path, convolutional_noise_path, noisy_path,
length_sec, snr_db, additive_noise, convolutional_noise, sr, remarks
```

Run the manifest builder from the project root:

```bash
cd src/enhancement/dataset
python manager.py
```

Note: `manager.py` currently uses a flat import (`from audio import AudioManager`) and a hardcoded config path (`config/metacentrum.yaml`) and output filename. To customize either, edit the `if __name__ == "__main__":` block at the bottom of the file — change `config_file` and `manifest_name` there. Outputs are written to `${WORKSPACE}/${paths.manifest_dir}/train_<filename>` and `eval_<filename>`.

The manifest only fixes *which* clean/noise/RIR files are paired and the target SNR. The actual mixing — random cropping to 2.0 s, scenario selection (additive / reverberation / both), reverb intensity, SNR draw, peak normalization — happens in `ModelDataset` at batch construction time, so no two epochs see the same mixture.

After generation, update `paths.train_manifest` and `paths.eval_manifest` in the YAML to point at the new files.

### 3. Train

#### Local / single-GPU

```bash
python -m enhancement.main --config config/metacentrum.yaml
```

(Make sure `src/` is on `PYTHONPATH`, e.g. `export PYTHONPATH=$PWD/src:$PYTHONPATH`.)

The trainer initializes the generator (`TSCNet`, 128 channels) and discriminator, uses AdamW with generator LR `init_lr` and discriminator LR `1.6 × init_lr`, StepLR scheduler (γ=0.5 every `decay_epoch` epochs), and Automatic Mixed Precision via `torch.cuda.amp.GradScaler`. Validation runs after every epoch and the latest checkpoint is written to `paths.save_dir`.

On an H100 NVL (94 GB), 100 epochs takes roughly 42 hours at batch size 4.

#### On MetaCentrum (PBS)

The repository ships with a PBS submission template that handles cluster-specific concerns (per-node project-dir discovery, walltime-aware checkpoint rescue, scratch redirection):

```bash
qsub templates/hpc/metacentrum_template.sh
```

What the script does:

- Searches the known `CTU_Speech_Lab` workspace mirrors for the project directory and `cd`s into it.
- Uses the project Python env at `${WORKSPACE_DIR}/envs/audio_env/bin/python`.
- Writes a per-job runtime config to `${SCRATCHDIR}/metacentrum.runtime.yaml`, with `system.root`, `system.workspace`, `system.scratch`, and `paths.save_dir` rewritten to match the assigned node.
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent fragmentation on long jobs.
- Installs an `EXIT`/`TERM` trap that copies all checkpoints from local scratch to `${WORKSPACE}/rescued_checkpoints/${PBS_JOBID}/` before the node is reclaimed (covers walltime hits).
- Tees output to `${HOME}/pbs_logs/cmgan/`.

Adjust `#PBS -l` resources, `walltime`, and the `-M` email at the top of the script for your queue.

### 4. Evaluate

All evaluation modes go through `evaluate.py`. The script **reads `config/metacentrum.yaml` from the current working directory** — run it from the project root, not from `src/`.

`evaluate.py` does not take `--config`; it auto-routes based on which flags are provided.

#### Full test-set benchmark

Runs the trained model on the manifest at `paths.final_eval_manifest` and writes a per-utterance CSV with PESQ, ESTOI, SI-SDR, SNR, SRMR, and DNSMOS (SIG/BAK/OVRL):

```bash
python -m enhancement.evaluate --checkpoint path/to/checkpoint.pth
```

Add `--pesq_only` to compute PESQ only (much faster for large manifests). Add `--fast` to limit to 10 batches (useful for sanity checks). Add `--num_channel 128` if the checkpoint was trained with a non-default channel count.

Output CSV path is auto-generated under `evaluation.output_dir` (or `evaluation.metrics_csv` if set in the YAML) and includes the checkpoint name and timestamp.

#### Checkpoint trend

Scans a directory of `epoch_*.pth` files, evaluates each on the test manifest, and saves a PESQ-vs-epoch trend plot to `~/cmgan_plots/`:

```bash
python -m enhancement.evaluate --checkpoint_dir path/to/checkpoints/
```

Add `--fast` to limit each epoch to 10 batches (recommended — this mode is slow otherwise).

#### Single-file scenarios

The evaluator auto-detects the scenario from which file arguments you pass.

Additive only (clean + noise at given SNR):

```bash
python -m enhancement.evaluate \
    --checkpoint path/to/checkpoint.pth \
    --single_clean path/to/clean.wav \
    --single_noise path/to/noise.wav \
    --snr 5
```

Reverberation only (clean + RIR at given DRR):

```bash
python -m enhancement.evaluate \
    --checkpoint path/to/checkpoint.pth \
    --single_clean path/to/clean.wav \
    --single_reverberation path/to/rir.wav \
    --snr 5
```

Joint (clean + noise + RIR):

```bash
python -m enhancement.evaluate \
    --checkpoint path/to/checkpoint.pth \
    --single_clean path/to/clean.wav \
    --single_noise path/to/noise.wav \
    --single_reverberation path/to/rir.wav \
    --snr 5
```

Each single-file mode writes the noisy mixture and the enhanced output as `.wav` files under `evaluation.output_dir`. Inference uses chunked processing (4 s segments at 16 kHz) under AMP, so arbitrarily long files are handled without OOM, and a final peak-normalization step prevents clipping on save.

### 5. Deployment (optional)

For deployment outside the PyTorch ecosystem, the generator can be exported to ONNX and run via `onnxruntime` (CUDA or CPU provider).

## Results

Trained on WSJ + TIMIT + CtuTest with QUT-NOISE and OpenAir/SLR28 RIRs, evaluated on the held-out multi-condition test set:

| Metric | Noisy | Enhanced | Δ |
|---|---|---|---|
| PESQ | 2.01 | 3.45 | +1.44 |
| ESTOI | 0.71 | 0.91 | +0.20 |
| SI-SDR (dB) | 13.46 | 16.81 | +3.35 |
| DNSMOS OVRL | 2.63 | 3.38 | +0.75 |
| DNSMOS SIG | 2.70 | 3.99 | +1.29 |
| DNSMOS BAK | 2.17 | 3.06 | +0.89 |

## Acknowledgments

The CMGAN architecture (encoder, Two-Stage Conformer, decoupled decoders, metric discriminator) is adapted from the reference implementation by Abdulatif, Cao, and Yang (2024); each adapted source file carries an acknowledgment header.

## Author

Vitalii Varhanik — Bachelor Thesis, Czech Technical University in Prague, Faculty of Electrical Engineering, Department of Cybernetics. Supervised by Doc. Ing. Petr Pollák, CSc.
