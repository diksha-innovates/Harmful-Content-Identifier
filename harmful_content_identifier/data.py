"""Dataset loading, labeling, and combination.

Mirrors the logic that previously lived inline in the notebook: each source
dataset is loaded from the Hugging Face Hub, stamped with a single project-wide
label, split into train/validation, and concatenated into one combined dataset.
No images are scraped from the web - everything is pulled through
`datasets.load_dataset` at run time and is never committed to this repo.
"""
from collections import Counter
from typing import Dict, Tuple

import torch
from datasets import ClassLabel, DatasetDict, concatenate_datasets, load_dataset
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from .config import DATASET_SPECS, NUM_CLASSES, DatasetSpec

# Some Hugging Face images are truncated; let PIL load them anyway instead of raising.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _stamp_label(dataset: DatasetDict, spec: DatasetSpec, num_classes: int) -> DatasetDict:
    """Drop any pre-existing label column and assign this project's label to every row."""
    if spec.existing_label_column and spec.existing_label_column in dataset["train"].features:
        dataset = dataset.remove_columns(spec.existing_label_column)

    dataset = dataset.map(lambda example: {"label": spec.label})

    new_features = dataset["train"].features.copy()
    new_features["label"] = ClassLabel(num_classes=num_classes)
    dataset = dataset.cast(new_features)
    return dataset


def load_and_split(spec: DatasetSpec, val_split: float, seed: int) -> DatasetDict:
    """Load one dataset from the Hub and produce a train/validation split."""
    dataset = load_dataset(spec.hf_id)
    train_split = dataset["train"]

    if spec.image_column != "image":
        train_split = train_split.rename_column(spec.image_column, "image")

    if spec.max_samples is not None and spec.max_samples < len(train_split):
        train_split = train_split.shuffle(seed=seed).select(range(spec.max_samples))

    split = train_split.train_test_split(test_size=val_split, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def load_all_datasets(
    val_split: float = 0.2, seed: int = 42, num_classes: int = NUM_CLASSES
) -> Dict[str, DatasetDict]:
    """Load and label every dataset configured in `config.DATASET_SPECS`."""
    processed = {}
    for name, spec in DATASET_SPECS.items():
        dataset = load_and_split(spec, val_split, seed)
        processed[name] = _stamp_label(dataset, spec, num_classes)
    return processed


def combine_datasets(datasets: Dict[str, DatasetDict]) -> Tuple:
    """Concatenate every dataset's train split together, and every validation split together."""
    combined_train = concatenate_datasets([d["train"] for d in datasets.values()])
    combined_validation = concatenate_datasets([d["validation"] for d in datasets.values()])
    return combined_train, combined_validation


def compute_class_weights(combined_train, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, for use in a weighted CrossEntropyLoss.

    Our source datasets range from ~190 images (violence) to tens of thousands
    (cifar10, nsfw sets), so an unweighted loss would let the model ignore the
    small classes almost entirely while still minimizing average loss.
    """
    counts = Counter(combined_train["label"])
    total = sum(counts.values())
    weights = [
        total / (num_classes * counts.get(c, 1)) for c in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Transform with light augmentation, used only on the training split."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform, used for validation/inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class CustomImageDataset(Dataset):
    """Wraps a Hugging Face image dataset for use with a PyTorch DataLoader."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label
