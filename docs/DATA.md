# Data preparation

No raw data, patient images or checkpoints are distributed. Run preprocessing on your own authorized copies of the datasets. Output images are 8-bit; the DCT image loader performs grayscale conversion and horizontal flips, but does not crop, resize, or change intensity scaling.

## ACDC

Install the preprocessing extra, then point the command at the original ACDC database containing `training/patientNNN/` folders, each with `Info.cfg`, a 4D cine NIfTI and annotated frame NIfTIs:

```bash
python -m pip install -e '.[preprocess]'
python scripts/preprocess_acdc.py \
  --input /path/to/ACDC/database --output /path/to/ACDC/prepared \
  --config configs/preprocess_acdc.json --splits training
```

Use `/path/to/ACDC/prepared/images` as training `--data`. The output is a flat PNG folder plus a preprocessing manifest recording frame/slice identities, settings and skipped crops. The training profile was checked pixel-for-pixel against all 25,022 existing prepared training PNGs. It is distinct from the exploratory labeled-data examples in `get_data_ready.ipynb`.

The rewrite retains nibabel's native array orientation and the notebook's bounding-box-center crop, rather than changing to a voxel centroid or resampling in physical space. The training profile uses ED ground truth and the bounding box of all nonzero labels in each axial slice. It retains slices with a nonempty mask, exports every cine timepoint in time-major order, and numbers only retained slices. Constant crops are skipped while keeping their filename index. Intensities use `(crop - min) / (max - min + 1e-8) * 255` followed by uint8 truncation; this reproduces the prepared pixels, including their maximum value of 254. Grayscale PNG output avoids the notebook's redundant replication into three identical channels. Boundary crops retain the original clipping behavior; the supplied profile skips invalid crops instead of padding or resizing them silently. In the audited training data, one constant-black candidate (`patient-82-frame258.png`) is excluded; its numeric filename remains unused, matching the original folder.

For the author-described alternative that keeps empty slices, set `empty_slices` to `fallback`; it uses reference slice 0 when the corresponding slice mask is empty. If that reference also has no mask, the supplied invalid-crop policy records and skips it.

The notebook's `label_value=2` examples and NumPy normalization can be selected by copying the preprocessing JSON and changing `label_value` to 2, `reference_frame` to `ES`, and `normalization` to `numpy`. Set `skip_constant` to false to retain constant images. These settings can change the dataset and should not be presented as reproducing the supplied training images. A constant-valued crop becomes a zero image, avoiding a division by zero.

Only the unconditional image-generation preprocessing is included. Segmentation-label exports, labeled-training splits, the superseded `get_npy.ipynb` volume export, and visualization cells are excluded.

## EchoNet-Dynamic

Install FFmpeg separately and ensure `ffmpeg` is on PATH. Use the original EchoNet-Dynamic `Videos` folder:

```bash
WORKERS=8 bash scripts/extract_frames_from_videos.sh \
  /path/to/EchoNet-Dynamic/Videos /path/to/EchoNet-Dynamic/images
```

The command preserves the [EchoNet-Synthetic extraction recipe](https://github.com/HReynaud/EchoNet-Synthetic/blob/87189437cce990daf195f54121f5d4707a5f3be6/scripts/extract_frames_from_videos.sh): select zero-based frames 0, 5, 10, ...; apply `format=gray,format=yuv420p`; save `VIDEO_00001.jpg`, etc. The default parallelism is 32 and can be changed with `WORKERS`. The release wrapper validates inputs and rejects nonempty output folders. It does not crop or resize: the supplied EchoNet videos are expected to yield 112 × 112 frames, as in the old image folder. The loader converts the JPEG's stored RGB channels to grayscale.

This command extracts every matching AVI below the input folder. If your experiment uses a subset, prepare the input folder accordingly. It does not infer a patient or video split from `FileList.csv`. Use unique video basenames to avoid flat-output collisions. Train with `configs/echonet.json` and `--data /path/to/EchoNet-Dynamic/images`.

## NIH Chest X-ray

The original script is included unchanged. The expected layout is:

```text
NIH/raw/
  train_val_list.txt
  images_001/images/*.png
  ...
  images_012/images/*.png
```

```bash
python scripts/preprocess_nih_train.py --dataset-root /path/to/NIH/raw \
  --output-dir /path/to/NIH/nih_256_train --workers 8
```

It selects filenames from `train_val_list.txt`, converts to grayscale and resizes to 256 × 256 using Pillow LANCZOS interpolation. Missing listed images fail explicitly. Train with `configs/nih.json` and `--data /path/to/NIH/nih_256_train`.

## Reference images

FID accepts a flat reference-image folder or existing `.npz` statistics containing `mu` and `sigma`. LPIPS requires images and pairs the first requested number of lexicographically sorted generated and real paths. Reference selection affects results; it is not recovered from the training configs. Keep preprocessing manifests alongside private datasets rather than publishing them automatically.
