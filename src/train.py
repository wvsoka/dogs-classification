import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    HEAD_EPOCHS,
    FINETUNE_EPOCHS,
    HEAD_LR,
    FINETUNE_LR,
    WEIGHT_DECAY,
    PATIENCE,
    SEED,
)
from dataset import get_fold_dataloaders
from models import build_model, unfreeze_all, count_parameters
from utils import set_seed, get_device, ensure_dirs, EarlyStopping, Timer, save_json


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    if train:
        context = torch.enable_grad()
    else:
        context = torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_log_header(log_path: Path):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase",
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "lr",
        ])


def append_log(log_path: Path, row: list):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    model_name,
    fold,
    phase_name,
    num_epochs,
    log_path,
    checkpoint_path,
):
    early_stopping = EarlyStopping(patience=PATIENCE, mode="min")

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        append_log(log_path, [
            phase_name,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        ])

        print(
            f"[{model_name} | fold {fold} | {phase_name}] "
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        improved = early_stopping.step(val_loss)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "fold": fold,
                    "phase": phase_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )

        if early_stopping.should_stop:
            print(f"Early stopping w fazie {phase_name}, epoka {epoch}.")
            break

    return {
        "phase": phase_name,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    set_seed(SEED)

    ensure_dirs(CHECKPOINT_DIR, LOG_DIR)

    device = get_device()
    print("Device:", device)

    model_name = args.model
    fold = args.fold

    train_loader, val_loader = get_fold_dataloaders(fold=fold, n_splits=5)

    model = build_model(model_name=model_name, freeze=True)
    model = model.to(device)

    print("Parametry po zamrożeniu:")
    print(count_parameters(model))

    criterion = nn.CrossEntropyLoss()

    log_path = LOG_DIR / f"{model_name}_fold{fold}_train_log.csv"
    save_log_header(log_path)

    head_checkpoint_path = CHECKPOINT_DIR / f"{model_name}_fold{fold}_best_head.pt"
    finetune_checkpoint_path = CHECKPOINT_DIR / f"{model_name}_fold{fold}_best_finetune.pt"

    with Timer() as timer:
        # Faza 1: tylko głowica
        head_optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
        )

        head_scheduler = ReduceLROnPlateau(
            head_optimizer,
            mode="min",
            factor=0.1,
            patience=3,
        )

        head_summary = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=head_optimizer,
            scheduler=head_scheduler,
            device=device,
            model_name=model_name,
            fold=fold,
            phase_name="head",
            num_epochs=HEAD_EPOCHS,
            log_path=log_path,
            checkpoint_path=head_checkpoint_path,
        )

        # Wczytujemy najlepszą głowicę przed fine-tuningiem
        checkpoint = torch.load(head_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Faza 2: fine-tuning całego modelu
        unfreeze_all(model)

        print("Parametry po odmrożeniu:")
        print(count_parameters(model))

        finetune_optimizer = AdamW(
            model.parameters(),
            lr=FINETUNE_LR,
            weight_decay=WEIGHT_DECAY,
        )

        finetune_scheduler = ReduceLROnPlateau(
            finetune_optimizer,
            mode="min",
            factor=0.1,
            patience=3,
        )

        finetune_summary = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=finetune_optimizer,
            scheduler=finetune_scheduler,
            device=device,
            model_name=model_name,
            fold=fold,
            phase_name="finetune",
            num_epochs=FINETUNE_EPOCHS,
            log_path=log_path,
            checkpoint_path=finetune_checkpoint_path,
        )

    summary = {
        "model": model_name,
        "fold": fold,
        "head": head_summary,
        "finetune": finetune_summary,
        "training_time_seconds": timer.interval,
        "device": str(device),
        "parameters": count_parameters(model),
    }

    save_json(summary, LOG_DIR / f"{model_name}_fold{fold}_summary.json")

    print("Zapisano log:", log_path)
    print("Zapisano checkpoint:", finetune_checkpoint_path)
    print("Czas treningu:", timer.interval, "s")


if __name__ == "__main__":
    main()