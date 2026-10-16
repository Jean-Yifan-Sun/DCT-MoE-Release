# Release validation

Validation was performed on 16 October 2026 using the author's Huiping-lab machine, in an isolated temporary release copy. The original research checkout and datasets were read as references and were not modified. Temporary synthetic fixtures and test outputs were kept outside the original project.

## Environment

- Linux, Python 3.12
- NVIDIA RTX 6000 Ada Generation (one GPU)
- PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130
- Accelerate 1.13.0, NumPy 2.4.4
- FFmpeg 8.0.1 for frame extraction
- Existing OpenCV, Pillow, nibabel, pytorch-fid and LPIPS packages from the author's environment

The version floors in `pyproject.toml` are API requirements, not an exhaustive compatibility matrix. This existing environment was used for runtime checks; a fresh dependency installation across all supported Python/PyTorch versions has not been tested.

## Checks

- 22 unittest cases passed, with optional legacy parity, evaluation and preprocessing tests enabled.
- Legacy DCT token order and normalization across 4×4 and 8×8 blocks and retained coefficient settings.
- Strict loading of the original model state-dict structure and backbone prediction/gradient comparisons with the same author-approved probability weighting applied to the reference router. This is intentionally not equality with the unmodified hard-sum model. The weighted diffusion objective is compared separately against the legacy helper.
- DCT round trip, high-frequency truncation, malformed input handling, selected softmax probability weights and gradients, propagation of nondefault noise settings to both MoE blocks, and activation-checkpoint RNG preservation. A full-resolution regression also pins the legacy float64 IDCT and float32 pixel buffer, preventing single-level uint8 rounding differences.
- DPM-Solver repeatability with fixed initial noise, finite outputs and requested NFE counting.
- Synthetic image-folder training, checkpoint save, resume and sampling with a partial final batch. CPU resumed and uninterrupted EMA weights compare exactly in the test fixture.
- Full-size ACDC, EchoNet and NIH models: one GPU forward/backward/AdamW step and five-NFE sampling each. Parameter counts were respectively 163,173,184; 163,188,528; and 163,259,200. Peak allocated memory in this batch-one check was approximately 3.1 GiB; this is not a training memory estimate for batch size 512.
- GPU CLI training, checkpoint save/resume and image generation on a small synthetic config.
- Two-process CPU/Gloo training, checkpointing and resume across an epoch boundary. GPU multi-process execution could not be tested on this single-GPU machine.
- ACDC prepared-data audit: all 25,022 existing training PNGs compare pixel-exactly with the recovered ED whole-heart crop and epsilon-normalization profile. Empty-label slices are omitted before flattened indexing; the one constant-black candidate is skipped. The raw-volume audit was completed in two ranges after a transient gzip CRC read error; the remaining range re-read successfully.
- Public ACDC CLI export for two real source patients: 541 PNGs matched the original pixels and filename set exactly, including constant-slice exclusion.
- Synthetic ACDC volumes validate crop positions, boundary clipping, normalization, missing-mask fallback and filtered-slice indexing.
- EchoNet wrapper on an 11-frame synthetic AVI: exactly three 112×112 JPEGs, corresponding to frames 0, 5 and 10.
- Original NIH preprocessing command on a synthetic shard/list: grayscale PNG at 256×256.
- FID and LPIPS evaluation CLI with identical synthetic reference/generated folders and uppercase PNG extensions: LPIPS 0.0 and FID approximately -0.000043 (floating-point roundoff around zero with a singular three-image covariance). This checks execution, not the paper's metrics. Standard LPIPS torchvision deprecation warnings and the expected small-sample FID covariance warning were emitted.
- Installable wheel build without resolving or reinstalling dependencies.

## Limits

No 500,000-step training runs or 50,000-sample paper evaluations were performed. No original trained weights are included. Validation confirms extraction fidelity and execution behavior; it does not re-establish the reported generation scores. Dataset split selection, caching history and reference selection can affect reproductions as explained in `REPRODUCIBILITY.md`.
