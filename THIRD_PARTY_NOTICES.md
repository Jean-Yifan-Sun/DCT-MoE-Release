# Sources and acknowledgements

- **U-ViT**: Fan Bao and contributors, [baofff/U-ViT](https://github.com/baofff/U-ViT). The supplied research archive includes an MIT license with Copyright (c) 2022 Fan Bao; that notice is preserved verbatim in `LICENSE`. The model retains its backbone structure and parameter names.
- **DCTdiff**: the grayscale tokenization and SNR-adjusted solver path were adapted in the supplied research code from the DCTdiff approach cited in the paper (Mang Ning et al., ICML 2025). This release extracts that grayscale path. No RGB models or VAE code are included.
- **DPM-Solver**: Cheng Lu et al., [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver). `dct_moe/dpm_solver.py` retains the solver bundled in the research archive, including its dataset SNR adjustment. Only an informational constructor print is removed.
- **timm**: the small dense MLP derives from the helper identified as timm 0.3.2 in the research archive. `nn.init.trunc_normal_` replaces its copied initialization helper.
- **EchoNet-Synthetic**: Hadrien Reynaud and collaborators, [repository](https://github.com/HReynaud/EchoNet-Synthetic). `scripts/extract_frames_from_videos.sh` is a portable rewrite of their preprocessing command, preserving the frame-selection and grayscale FFmpeg filter. Reference checkout: `87189437cce990daf195f54121f5d4707a5f3be6`. It is not a copy of their model or training code.
- **NIH preprocessing**: `scripts/preprocess_nih_train.py` is the author's original script recovered from the shared NIH dataset's `raw` directory, copied unchanged.
- **ACDC preprocessing**: `scripts/preprocess_acdc.py` is the author's original script that writes the author's crop helpers and the unconditional cine/slice export described. It omits segmentation datasets, RGB channel replication, plotting and scratch cells.

Optional evaluation uses `pytorch-fid` and `lpips` as dependencies instead of vendoring their source. Their licenses and model-weight terms remain with their respective distributions. Dataset access and redistribution follow the original dataset providers' terms; no dataset images or medical volumes are bundled here.
