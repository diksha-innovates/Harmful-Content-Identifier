"""Training loop with proper evaluation metrics, class balancing, and early stopping."""
import csv
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, TrainConfig
from .data import (
    CustomImageDataset,
    combine_datasets,
    compute_class_weights,
    get_eval_transform,
    get_train_transform,
    load_all_datasets,
)
from .metrics import classification_report_text, compute_metrics, plot_confusion_matrix, save_metrics
from .model import SimpleCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    datasets = load_all_datasets(val_split=cfg.val_split, seed=cfg.seed, num_classes=cfg.num_classes)
    combined_train, combined_validation = combine_datasets(datasets)

    class_weights = compute_class_weights(combined_train, cfg.num_classes)

    train_dataset = CustomImageDataset(combined_train, transform=get_train_transform(cfg.image_size))
    validation_dataset = CustomImageDataset(
        combined_validation, transform=get_eval_transform(cfg.image_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, validation_loader, class_weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer = None,
) -> Tuple[float, list, list]:
    """Run one pass over `loader`. Pass `optimizer` to train, omit it to evaluate."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return running_loss / total, all_preds, all_labels


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
) -> dict:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg.lr_scheduler_patience
    )

    log_path = Path(cfg.log_dir) / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    final_val_metrics = None

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1"]
        )

        for epoch in range(cfg.num_epochs):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_preds, val_labels = run_epoch(model, validation_loader, criterion)

            val_metrics = compute_metrics(val_labels, val_preds, CLASS_NAMES)
            val_accuracy = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)

            scheduler.step(val_metrics["macro_f1"])

            print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
            print(f"  train_loss={train_loss:.4f}")
            print(
                f"  val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )
            print(classification_report_text(val_labels, val_preds, CLASS_NAMES))

            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_accuracy,
                    val_metrics["macro_precision"],
                    val_metrics["macro_recall"],
                    val_metrics["macro_f1"],
                ]
            )
            log_file.flush()

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                epochs_without_improvement = 0
                final_val_metrics = val_metrics
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.early_stopping_patience:
                    print(f"No val macro-F1 improvement for {cfg.early_stopping_patience} epochs, stopping early.")
                    break

    metrics_path = Path(cfg.metrics_dir) / "final_validation_metrics.json"
    save_metrics(final_val_metrics, str(metrics_path))
    plot_confusion_matrix(
        final_val_metrics["confusion_matrix"], CLASS_NAMES, str(Path(cfg.metrics_dir) / "confusion_matrix.png")
    )
    return final_val_metrics


def run(cfg: TrainConfig = None) -> nn.Module:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    train_loader, validation_loader, class_weights = build_loaders(cfg)

    model = SimpleCNN(num_classes=cfg.num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights if cfg.use_class_weighted_loss else None)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    fit(model, train_loader, validation_loader, criterion, optimizer, cfg)
    return model
