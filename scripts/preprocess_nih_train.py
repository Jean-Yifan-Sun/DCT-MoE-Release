from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NIH Chest X-ray train/val images into 256x256 grayscale PNGs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the NIH dataset root containing train_val_list.txt and images_001...images_012.",
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help="Optional path to the filename list. Defaults to <dataset-root>/train_val_list.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <dataset-root>/nih_256_train.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Output image size. Default: 256.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker processes. Default: number of CPU cores.",
    )
    return parser.parse_args()


def read_filenames(list_file: Path) -> list[str]:
    with list_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_image_index(dataset_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for shard in range(1, 13):
        image_dir = dataset_root / f"images_{shard:03d}" / "images"
        if not image_dir.is_dir():
            continue

        for image_path in image_dir.glob("*.png"):
            image_index[image_path.name] = image_path

    return image_index


def preprocess_one(task: tuple[Path, Path, int]) -> str:
    src_path, dst_path, size = task
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as image:
        processed = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        processed.save(dst_path)

    return dst_path.name


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    list_file = args.list_file.resolve() if args.list_file else dataset_root / "train_val_list.txt"
    output_dir = args.output_dir.resolve() if args.output_dir else dataset_root / "nih_256_train"

    filenames = read_filenames(list_file)
    image_index = build_image_index(dataset_root)

    missing_files = [name for name in filenames if name not in image_index]
    if missing_files:
        preview = ", ".join(missing_files[:10])
        raise FileNotFoundError(
            f"Could not locate {len(missing_files)} files listed in {list_file}. "
            f"Examples: {preview}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(image_index[name], output_dir / name, args.size) for name in filenames]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(
            tqdm(
                executor.map(preprocess_one, tasks),
                total=len(tasks),
                desc="Preprocessing NIH train images",
            )
        )

    print(f"Saved {len(tasks)} images to {output_dir}")


if __name__ == "__main__":
    main()
