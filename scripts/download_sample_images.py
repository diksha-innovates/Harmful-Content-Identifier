#!/usr/bin/env python3
"""Download a small number of sample images per category for local inspection.

This is for manually sanity-checking what the source datasets look like; it
is NOT used by training (training streams images directly via
`datasets.load_dataset`). Images are written to images/samples/<category>/,
which is gitignored - nothing this script downloads is meant to be committed.

Usage:
    python scripts/download_sample_images.py --per-category 5
"""
import argparse
from pathlib import Path

from datasets import load_dataset

CATEGORY_TO_DATASET = {
    "nsfw_darkyman": "DarkyMan/nsfw-image-classification",
    "nsfw_x1101": "x1101/nsfw-full",
    "gore": "NeuralShell/Gore-Blood-Dataset-v1.0",
    "violence": "rtb1271/human_fight_detection",
}


def download_samples(category: str, hf_id: str, count: int, out_dir: Path) -> None:
    dataset = load_dataset(hf_id, split="train", streaming=True)
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for example in dataset:
        if saved >= count:
            break
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(category_dir / f"{saved:03d}.jpg")
        saved += 1

    print(f"Saved {saved} images to {category_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--out-dir", default="images/samples")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORY_TO_DATASET),
        default=list(CATEGORY_TO_DATASET),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for category in args.categories:
        download_samples(category, CATEGORY_TO_DATASET[category], args.per_category, out_dir)


if __name__ == "__main__":
    main()
