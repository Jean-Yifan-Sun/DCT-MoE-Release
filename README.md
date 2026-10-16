# DCT-MoE

Code for **DCT-MoE: Frequency-Domain Medical Image Generation with Mixture-of-Experts**, BMVC 2026.

DCT-MoE models grayscale images directly in a block-wise DCT representation. Four spatially adjacent blocks form one token. A U-ViT predicts diffusion noise, with probability-weighted top-2 routing among four experts in its first and last blocks. Images are reconstructed by inverse DCT. This release contains the DCT + MoE method, training, ODE sampling and evaluation. Checkpoints and datasets are not distributed.

## Installation

Use Python 3.10 or newer in a dedicated environment. Install a PyTorch build appropriate for your GPU, then from this directory:

```bash
python -m pip install -e .
# Optional FID and LPIPS evaluation:
python -m pip install -e '.[eval]'
```

Core training does not require torchvision, pretrained image encoders, experiment-tracking accounts, or evaluation-model downloads. Evaluation downloads InceptionV3 and AlexNet weights on first use if not already cached. The tested runtime and validation scope are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

## Data and configurations

Training consumes prepared 8-bit image folders, recursively. Each image must already have the exact resolution below; images are converted to grayscale as in the original loader. Training applies horizontal flips and does not resize or rescale intensities. Point `--data` at the image folder itself, not a parent containing multiple dataset versions or train/test splits.

| Configuration | Resolution | DCT block | Retained coefficients per block | Tokens × features | DC bound |
| --- | --- | --- | --- | --- | --- |
| `configs/acdc.json` | 96 × 96 | 4 × 4 | 16 (`m=0`) | 144 × 64 | 502.5 |
| `configs/echonet.json` | 112 × 112 | 4 × 4 | 12 (`m=4`) | 196 × 48 | 512.0 |
| `configs/nih.json` | 256 × 256 | 8 × 8 | 16 (`m=48`) | 256 × 64 | 1016.0 |

`configs/acdc_keep12_ablation.json` and `configs/nih_keep48_ablation.json` retain the coefficient-truncation comparisons from the paper. The main configurations use four experts and top-2 routing; expert-count ablations can be run by editing `model.MoE.num_experts` to 6 or 8. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for configuration provenance and author-confirmed implementation decisions.

[docs/DATA.md](docs/DATA.md) describes the preprocessing sources and dataset layout. For ACDC, install the preprocessing extra and run the rewritten notebook pipeline:

```bash
python -m pip install -e '.[preprocess]'
python scripts/preprocess_acdc.py --input /path/to/ACDC/database \
  --output /path/to/ACDC/prepared --config configs/preprocess_acdc.json --splits training
```

For EchoNet, use `bash scripts/extract_frames_from_videos.sh /path/to/Videos /path/to/images` with FFmpeg installed. The original NIH preparation command is:

```bash
python scripts/preprocess_nih_train.py \
  --dataset-root /path/to/NIH/raw \
  --output-dir /path/to/NIH/nih_256_train --workers 8
```

## Training

```bash
python -m dct_moe.train \
  --config configs/acdc.json --data /path/to/ACDC/images \
  --output runs/acdc --workers 8
```

For the original global batch size of 512 on two GPUs:

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision no \
  -m dct_moe.train --config configs/acdc.json \
  --data /path/to/ACDC/images --output runs/acdc --workers 8
```

Training uses AdamW, linear learning-rate warmup, EMA, the original SNR-adjusted continuous VP schedule and entropy-weighted noise MSE plus router load balancing. FP32 is used, matching the original launch configuration. The configured batch size is global and must divide the number of processes. Intermediate checkpoints are saved every 25,000 optimizer steps and at the stopping point. No evaluation is run implicitly during training.

To stop early for a smoke test without changing the saved config, add `--max-steps 2`. For small GPUs, reduce `train.batch_size` in a copied JSON config; this changes the experimental setting. `model.use_checkpoint=true` enables activation checkpointing with consistent router RNG on recomputation. CPU validation is available with `--cpu`.

```bash
python -m dct_moe.train --config configs/acdc.json \
  --data /path/to/ACDC/images --output runs/acdc \
  --resume runs/acdc/step_0025000
```

Resume restores model, optimizer, scheduler, EMA, RNG and position within the data epoch. Use the same config, prepared files and process count. Filenames are checked, but changed image contents are not detected. Do not modify training data between runs. Checkpoints contain private data filenames in the run manifest, so review run artifacts separately before sharing them.

## Sampling

```bash
python -m dct_moe.sample --checkpoint runs/acdc/step_0500000 \
  --output samples/acdc --count 50000 --batch-size 32 --steps 50 --seed 1234
```

This runs the original DPM-Solver-fast ODE with **50 model evaluations (NFEs)** and writes numbered grayscale PNGs and `sampling.json`. The final partial batch is supported. The output directory must be new or empty. Outputs repeat for the same checkpoint, seed, batch size, device and software environment; changing batch size can change the random stream and numerical results.

Legacy EMA state dictionaries are also supported with an explicit matching config:

```bash
python -m dct_moe.sample --checkpoint /path/to/nnet_ema.pth \
  --config configs/acdc_keep12_ablation.json --output samples/legacy
```

State dictionaries load strictly. Match the coefficient count and model dimensions to the checkpoint. The release uses the author-corrected probability-weighted router; legacy hard-sum weights remain loadable but produce different predictions under this corrected forward pass. Legacy optimizer-state resume is not supported; resume uses the release checkpoint format.

## Evaluation

```bash
python -m dct_moe.evaluate --samples samples/acdc \
  --real /path/to/ACDC/reference_images --output runs/acdc/metrics.json
```

FID uses all images in a flat generated-image directory and the standard TensorFlow-compatible InceptionV3 implementation from `pytorch-fid`; the paper uses 50,000 generated images. You can pass a real-data `.npz` statistics file with `--metrics fid`.

LPIPS uses **AlexNet v0.1**, matching the old code. It pairs the first 2,000 lexicographically sorted generated and real image paths. These unconditional samples are not anatomically paired. If a pair differs in size, both images are resized to 224 × 224, matching the old evaluation helper. Specify `--metrics lpips --lpips-pairs N` to evaluate a different count. Missing pairs fail explicitly.

The release preserves the available implementation; it does not claim that running these commands reproduces the reported scores without the exact training data, reference selection and original weights. See the reproducibility notes before comparing results.

## Tests

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
# Optional numerical comparison with an author checkout:
DCT_MOE_LEGACY_ROOT=/path/to/DCT-MoE-paper \
  OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The tests cover DCT order and round trips, truncation, router gradients, checkpoint recomputation, deterministic sampling/NFE, training/resume equivalence and public CLI workflows. LPIPS tests run when optional evaluation dependencies are installed; the identity test needs cached/downloadable AlexNet weights.

## Acknowledgements and license

The backbone derives from U-ViT; the frequency representation follows the grayscale adaptation of DCTdiff. The retained DPM-Solver implementation includes the research code's SNR adjustment. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The supplied MIT license and copyright notice are retained in [LICENSE](LICENSE).
