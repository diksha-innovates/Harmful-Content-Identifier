"""Training loop."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import CustomImageDataset, combine_datasets, get_transform, load_all_datasets
from .model import SimpleCNN


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    datasets = load_all_datasets(val_split=cfg.val_split, seed=cfg.seed, num_classes=cfg.num_classes)
    combined_train, combined_validation = combine_datasets(datasets)

    transform = get_transform(cfg.image_size)
    train_dataset = CustomImageDataset(combined_train, transform=transform)
    validation_dataset = CustomImageDataset(combined_validation, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, validation_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> None:
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = running_corrects.double() / total_train

        model.eval()
        val_running_loss, val_running_corrects, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_epoch_loss = val_running_loss / total_val
        val_epoch_acc = val_running_corrects.double() / total_val

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        print("-" * 20)


def run(cfg: TrainConfig = None) -> nn.Module:
    cfg = cfg or TrainConfig()
    train_loader, validation_loader = build_loaders(cfg)

    model = SimpleCNN(num_classes=cfg.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_model(model, train_loader, validation_loader, criterion, optimizer, cfg.num_epochs)
    return model
