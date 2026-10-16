# Implementation provenance and reproducibility

The extraction reference is the supplied `DCT-MoE-paper` research archive, with the author-approved routing and noise corrections described below. This release extracts the unconditional grayscale DCT + first/last MoE path and replaces its experiment-management plumbing. It does not include checkpoints.

## Retained numerical behavior

- OpenCV orthonormal block DCT after subtracting 128 from 8-bit grayscale pixels.
- Spatial grouping: top-left, top-right, bottom-left, bottom-right blocks; zigzag frequency order within each block; truncate before flattening.
- Global DC-bound normalization divides training coefficients by the maximum absolute training DC coefficient, targeting [-1, 1]. Division is not a hard clip. Separately, the legacy decoder clamps generated normalized coefficients to [-2, 2] before denormalization; that inference safeguard does not define the training normalization range. High frequencies are zero-filled, inverse DCT adds 128, then pixels are clipped to [0, 255] and cast to uint8.
- The original grayscale MoE U-ViT parameter names and dimensions, including its time token and long skip connections.
- Noisy top-k router with selected softmax probability weights and a straight-through selection mask. Selected probabilities are not renormalized: output is the sum of P_i(x) E_i(x) over the selected experts. Load balancing retains the legacy hard dispatch fractions and average router probabilities; auxiliary losses are averaged over the two MoE blocks. Both routers use the configured noise scale during training, and disable noise in evaluation.
- The SNR scale of 4, linear VP beta endpoints 0.1 and 20, and model time labels `999 * t`.
- Entropy-weighted MSE, normalizing retained entropy weights to mean one and repeating the frequency vector for all four blocks. Auxiliary loss weight 0.001.
- AdamW learning rate 0.0002, betas (0.99, 0.99), weight decay 0.03, 5,000 warmup steps, EMA rate 0.9999 and 500,000 optimizer steps at global batch size 512.

## Config source mapping

| Release config | Research source | Selection |
| --- | --- | --- |
| `acdc.json` | `configs_shift/acdc_wholeheart_uncond_uvit_MoE_greyscale_mid_4by4_low16.py` | All 16 coefficients, paper main result |
| `acdc_keep12_ablation.json` | `configs_shift/acdc_wholeheart_uncond_uvit_MoE_greyscale_mid_4by4.py` | 12 coefficients, paper truncation ablation |
| `echonet.json` | `configs_shift/echonet_dynamic_uncond_uvit_MoE_greyscale_mid_4by4.py` | 12 coefficients, available research config |
| `nih.json` | `configs_shift/nih_uncond_uvit_MoE_greyscale_mid_8by8.py` | Author-confirmed main setting: 16 retained (`m=48`); original run config is outside this archive |
| `nih_keep48_ablation.json` | same NIH source | Original expression `8**2 - 16` retains 48; paper `m=16` ablation |

Only parameters actually used by this method are retained. Absolute cluster paths, cache settings, unused normalization statistics and unrelated evaluation settings are removed. Dataset roots are CLI arguments. The sampling default changes from the legacy Euler/100-step configuration to the paper's DPM-Solver/50-NFE setting.

## Author-confirmed implementation and camera-ready decisions

1. **Block count:** eight encoder blocks, one middle block, eight decoder blocks: 17 total. `depth=16` counts the paired encoder/decoder blocks. The author will clarify the middle block in the paper.
2. **Probability weighting:** the release now uses selected softmax probabilities, matching Equation 5, rather than the archive's hard sum of expert outputs. The STE selection surrogate and existing load-balancing loss are retained.
3. **Metric backbones:** AlexNet v0.1 for LPIPS and InceptionV3 for FID. The author will correct the paper's metrics description.
4. **NIH:** 16 retained coefficients is the confirmed main setting. Its run configuration existed elsewhere and is absent from this archive. The 48-retained archive config is retained only as an explicitly named ablation.
5. **Diffusion defaults:** entropy-weighted MSE and SNR scaling are inherited DCTdiff defaults and remain enabled.
6. **ACDC:** the existing ED whole-heart crop, empty-label-slice filtering and per-crop epsilon normalization are confirmed correct.
7. **Router noise:** the configured noise scale now applies to both MoE blocks. The archive's omission for the last block was a coding mistake. Supplied configs use 0.01 for both.

The probability-weighting correction intentionally changes predictions relative to the unmodified archive. Parameter names and tensor shapes remain compatible with legacy state dictionaries, but loading old weights does not reproduce the old hard-sum model's samples. The same caveat applies to resuming checkpoints produced before this correction; use a new training run for the corrected method. Paper scores have not been re-evaluated after the correction.

## Deliberate release plumbing changes

The data loader reads prepared images on demand instead of index-addressed cached tensors, avoiding stale cache/image associations. Horizontal flip choices are deterministic by seed, epoch and sample index, so they can be resumed with workers. This differs from the old cache pipeline, which freezes the augmentation used while caching. It is not a guarantee of the original training trajectory.

Training/resume uses Accelerate state and validates configs, process count and the image filename list. It rejects partial/incompatible model weights rather than silently continuing with missing keys. The inference command accepts legacy EMA state dictionaries with an explicit matching config, but does not import legacy optimizer state.

Activation checkpointing uses non-reentrant recomputation with RNG preservation; the old MoE checkpoint path disabled RNG preservation despite noisy routing. Inference respects no-grad rather than enabling autograd inside expert evaluation. These changes keep the method while fixing execution issues.

Evaluation is an explicit command. The release does not implicitly generate 50,000 images during training, does not delete samples, and does not report zero when inputs are missing. `pytorch-fid` provides the same standard TensorFlow-compatible InceptionV3 feature implementation used by the research helper; LPIPS preserves the old AlexNet and resizing/pairing rules.

## Reproducing scores

Exact score reproduction requires the original prepared images, frame/slice filtering, reference-image selection, saved experiment config, weights and sampling environment. Those artifacts are not all contained in this release. The reported Table 4 scores have not been regenerated as part of release validation; smoke tests establish implementation behavior, not trained image quality.
