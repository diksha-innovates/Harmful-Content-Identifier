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
    # Name of the image column in the source dataset, if not "image".
    image_column: str = "image"
    # Cap on how many rows to keep from this dataset (applied before the
    # train/validation split), to avoid one source dataset dwarfing the rest.
    max_samples: int = None


CLASS_NAMES: List[str] = [
    "safe",         # 0 - uoft-cs/cifar10 (benign negative class)
    "nsfw_1",       # 1 - DarkyMan/nsfw-image-classification
    "nsfw_2",       # 2 - x1101/nsfw-full
    "gore",         # 3 - NeuralShell/Gore-Blood-Dataset-v1.0
    "violence",     # 4 - rtb1271/human_fight_detection
]

# The original pipeline never populated class 0, so the model could never
# predict "this image is fine" - every input was forced into one of four
# harmful buckets. We add a small benign dataset as a real negative class so
# precision/recall against actual safe content become meaningful.
DATASET_SPECS = {
    "uoft-cs/cifar10": DatasetSpec(
        hf_id="uoft-cs/cifar10", label=0, image_column="img", max_samples=600
    ),
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
    metrics_dir: str = "metrics"
    # Weight the loss inversely to class frequency so the small datasets
    # (e.g. 398 gore images) aren't drowned out by larger ones.
    use_class_weighted_loss: bool = True
    # Stop training if validation macro-F1 hasn't improved for this many epochs.
    early_stopping_patience: int = 3
    lr_scheduler_patience: int = 2
