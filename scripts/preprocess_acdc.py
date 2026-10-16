"""Prepare grayscale ACDC cine slices using the get_data_ready.ipynb crop.

Keep nibabel's array orientation. Each axial slice uses a label bounding-box
center; missing slice masks use reference slice 0, as in the author's pipeline.
"""
import argparse
import configparser
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def crop_by_mask(image, label, size=96, label_value=None):
    if image.ndim != 2 or image.shape != label.shape or size < 2 or size % 2:
        raise ValueError('Expected matching 2D image/mask arrays and a positive even crop size')
    mask = label > 0 if label_value is None else label == label_value
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        raise ValueError('Reference slice contains no requested heart label')
    center = (coordinates.min(axis=0) + coordinates.max(axis=0)) // 2
    start = np.maximum(center - size // 2, 0)
    end = np.minimum(center + size // 2, image.shape)
    return image[start[0]:end[0], start[1]:end[1]]


def normalize_crop(image, method='minmax_eps', epsilon=1e-8):
    # nibabel.get_fdata() supplies float64. Preserve it: OpenCV's scale/shift
    # rounding before uint8 casting differs slightly from NumPy division.
    image = np.asarray(image, dtype=np.float64)
    if not image.size or not np.isfinite(image).all():
        raise ValueError('Crop must be nonempty and finite')
    if method == 'minmax_eps':
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('normalization epsilon must be positive and finite')
        shifted = image - image.min()
        return (shifted / (shifted.max() + epsilon) * 255).astype(np.uint8)
    if method == 'opencv':
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if method == 'numpy':
        image = image - image.min()
        maximum = image.max()
        return (image / maximum * 255).astype(np.uint8) if maximum else np.zeros(image.shape, dtype=np.uint8)
    raise ValueError('normalization must be minmax_eps, opencv or numpy')


def export_patient(patient_dir, output_dir, *, reference_frame, label_value=None,
                   normalization='minmax_eps', invalid_crop='error', size=96,
                   empty_slices='fallback', skip_constant=False, normalization_epsilon=1e-8):
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError('Install preprocessing dependencies: pip install -e ".[preprocess]"') from error
    patient_dir, output_dir = Path(patient_dir), Path(output_dir)
    info = configparser.ConfigParser()
    info.read_string('[DEFAULT]\n' + (patient_dir / 'Info.cfg').read_text())
    frame = info.getint('DEFAULT', reference_frame)
    prefix = patient_dir.name
    cine_path = patient_dir / f'{prefix}_4d.nii.gz'
    label_path = patient_dir / f'{prefix}_frame{frame:02d}_gt.nii.gz'
    for path in (cine_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cine = nib.load(cine_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    if cine.ndim != 4 or label.ndim != 3 or cine.shape[:3] != label.shape:
        raise ValueError(f'{prefix}: expected aligned 4D cine and 3D reference label')
    if invalid_crop not in ('error', 'skip'):
        raise ValueError('invalid_crop must be error or skip')
    masks = label > 0 if label_value is None else label == label_value
    patient_id = int(prefix.removeprefix('patient'))
    output_dir.mkdir(parents=True, exist_ok=True)
    if empty_slices not in ('skip', 'fallback'):
        raise ValueError('empty_slices must be skip or fallback')
    selected_slices = [z for z in range(cine.shape[2]) if empty_slices == 'fallback' or masks[:, :, z].any()]
    if not selected_slices:
        raise ValueError(f'{prefix}: no slices contain the requested reference mask')
    records = []
    for time in range(cine.shape[3]):
        for slice_position, axial_slice in enumerate(selected_slices):
            index = time * len(selected_slices) + slice_position + 1
            filename = f'patient-{patient_id}-frame{index}.png'
            fallback = not masks[:, :, axial_slice].any()
            reference_slice = 0 if fallback else axial_slice
            record = dict(file=filename, patient=prefix, time=time, axial_slice=axial_slice,
                          reference_frame=reference_frame, reference_slice=reference_slice,
                          fallback_to_slice0=fallback)
            try:
                crop = crop_by_mask(cine[:, :, axial_slice, time], label[:, :, reference_slice], size, label_value)
                if crop.shape != (size, size):
                    raise ValueError(f'Edge-clipped crop has shape {crop.shape}, expected {(size, size)}')
            except ValueError as error:
                if invalid_crop == 'error':
                    raise ValueError(f'{prefix}, time={time}, slice={axial_slice}: {error}') from error
                records.append(dict(record, skipped=str(error)))
                continue
            if skip_constant and np.ptp(crop) == 0:
                records.append(dict(record, skipped='constant intensity crop'))
                continue
            target = output_dir / filename
            if target.exists():
                raise FileExistsError(target)
            Image.fromarray(normalize_crop(crop, normalization, normalization_epsilon)).save(target)
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Database containing training/ and testing/')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--config', type=Path, required=True, help='Preprocessing JSON settings')
    parser.add_argument('--splits', nargs='+', choices=['training', 'testing'], default=['training'])
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be new or empty')
    patients = []
    for split in args.splits:
        folder = args.input / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        patients.extend(sorted(p for p in folder.glob('patient*') if p.is_dir()))
    if not patients:
        raise ValueError('No patient folders found')
    if len({p.name for p in patients}) != len(patients):
        raise ValueError('Patient identifiers must be unique across selected splits')
    records = []
    for patient in patients:
        records.extend(export_patient(patient, args.output / 'images', **settings))
        print(f'Processed {patient.name}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'preprocessing.json').write_text(json.dumps(dict(settings=settings,
        splits=args.splits, records=records), indent=2) + '\n')
    skipped = sum('skipped' in r for r in records)
    print(f'Saved {len(records) - skipped} PNGs; skipped {skipped} invalid crops. See preprocessing.json.')


if __name__ == '__main__':
    main()
