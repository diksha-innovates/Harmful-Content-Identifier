"""Central configuration: dataset sources, label mapping, and training hyperparameters.

Keeping this in one place means every module (data loading, training, evaluation)
agrees on the same class indices without re-deriving them.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    label: int
    # Column holding a pre-existing label in the source dataset that must be
    # dropped before we overwrite it with our own `label`, if any.
    existing_label_column: str = None


CLASS_NAMES: List[str] = [
    "unused",       # 0 - reserved, not populated by the current pipeline
    "nsfw_1",       # 1 - DarkyMan/nsfw-image-classification
    "nsfw_2",       # 2 - x1101/nsfw-full
    "gore",         # 3 - NeuralShell/Gore-Blood-Dataset-v1.0
    "violence",     # 4 - rtb1271/human_fight_detection
]

DATASET_SPECS = {
    "DarkyMan/nsfw-image-classification": DatasetSpec(
        hf_id="DarkyMan/nsfw-image-classification", label=1
    ),
    "x1101/nsfw-full": DatasetSpec(
        hf_id="x1101/nsfw-full", label=2
    ),
    "NeuralShell/Gore-Blood-Dataset-v1.0": DatasetSpec(
        hf_id="NeuralShell/Gore-Blood-Dataset-v1.0", label=3, existing_label_column="label"
    ),
    "rtb1271/human_fight_detection": DatasetSpec(
        hf_id="rtb1271/human_fight_detection", label=4, existing_label_column="labels"
    ),
}

NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class TrainConfig:
    num_classes: int = NUM_CLASSES
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-3
    val_split: float = 0.2
    seed: int = 42
    image_size: int = 224
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
