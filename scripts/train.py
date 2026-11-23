#!/usr/bin/env python3
"""CLI entrypoint for training the harmful content classifier.

Usage:
    python scripts/train.py --epochs 10 --batch-size 16 --lr 0.001
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmful_content_identifier.config import TrainConfig
from harmful_content_identifier.train import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        image_size=args.image_size,
    )
    run(cfg)


if __name__ == "__main__":
    main()
