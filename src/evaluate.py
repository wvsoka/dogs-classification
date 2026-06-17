import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    METRICS_DIR,
    MODELS,
    NUM_CLASSES,
    SEED,
)
from dataset import get_final_train_test_dataloaders
from models import build_model
from utils import set_seed, get_device


def parse_folds(fold_arg: str):
    if fold_arg == "all":
        return list(range(5))

    if "-" in fold_arg:
        start, end = fold_arg.split("-")
        return list(range(int(start), int(end) + 1))

    return [int(fold_arg)]


def parse_models(model_arg: str):
    if model_arg == "all":
        return MODELS

    return [model_arg]


def evaluate_checkpoint(model_name: str, fold: int, device):
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_fold{fold}_best_finetune.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {checkpoint_path}")

    print("\n" + "=" * 80)
    print(f"EVALUATE: model={model_name}, fold={fold}")
    print("=" * 80)

    _, test_loader = get_final_train_test_dataloaders()

    model = build_model(model_name=model_name, freeze=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{model_name} fold {fold}"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    top5_acc = top_k_accuracy_score(
        y_true,
        y_prob,
        k=5,
        labels=list(range(NUM_CLASSES)),
    )

    try:
        roc_auc_macro = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=list(range(NUM_CLASSES)),
        )

        roc_auc_weighted = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="weighted",
            labels=list(range(NUM_CLASSES)),
        )
    except ValueError as e:
        print("Nie udało się policzyć ROC-AUC:", e)
        roc_auc_macro = None
        roc_auc_weighted = None

    metrics = {
        "model": model_name,
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "top5_accuracy": top5_acc,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
    }

    output_dir = METRICS_DIR / "test"
    pred_dir = output_dir / "predictions"
    report_dir = output_dir / "reports"
    cm_dir = output_dir / "confusion_matrices"

    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{model_name}_fold{fold}_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    pred_df = pd.DataFrame({
        "image_index": np.arange(len(y_true)),
        "true_label": y_true,
        "pred_label": y_pred,
        "correct": y_true == y_pred,
    })

    pred_path = pred_dir / f"{model_name}_fold{fold}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    prob_df = pd.DataFrame(
        y_prob,
        columns=[f"class_{i}" for i in range(NUM_CLASSES)],
    )
    prob_df.insert(0, "image_index", np.arange(len(y_true)))
    prob_df.insert(1, "true_label", y_true)
    prob_df.insert(2, "pred_label", y_pred)

    prob_path = pred_dir / f"{model_name}_fold{fold}_test_probabilities.csv"
    prob_df.to_csv(prob_path, index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        output_dict=True,
        zero_division=0,
    )

    report_path = report_dir / f"{model_name}_fold{fold}_classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    cm_path = cm_dir / f"{model_name}_fold{fold}_confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_path, index=False)

    print(f"accuracy:          {accuracy:.4f}")
    print(f"balanced_accuracy: {balanced_acc:.4f}")
    print(f"macro F1:          {f1_macro:.4f}")
    print(f"weighted F1:       {f1_weighted:.4f}")
    print(f"top-5 accuracy:    {top5_acc:.4f}")
    print(f"ROC-AUC macro:     {roc_auc_macro}")

    return metrics


def summarize_results(all_metrics):
    output_dir = METRICS_DIR / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_metrics)

    raw_path = output_dir / "test_results_raw.csv"
    df.to_csv(raw_path, index=False)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "top5_accuracy",
        "roc_auc_macro",
        "roc_auc_weighted",
    ]

    summary_rows = []

    for model_name, group in df.groupby("model"):
        row = {"model": model_name}

        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_path = output_dir / "test_results_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("PODSUMOWANIE TEST SET")
    print("=" * 80)

    print(summary)

    print("\nZapisano:")
    print(raw_path)
    print(summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="mobilenet_v2 / efficientnet_b0 / resnet18 / all")
    parser.add_argument("--fold", type=str, required=True, help="0 / 1 / 2 / 3 / 4 / all / np. 1-4")
    args = parser.parse_args()

    set_seed(SEED)

    device = get_device()
    print("Device:", device)

    models_to_run = parse_models(args.model)
    folds_to_run = parse_folds(args.fold)

    all_metrics = []

    for model_name in models_to_run:
        for fold in folds_to_run:
            metrics = evaluate_checkpoint(
                model_name=model_name,
                fold=fold,
                device=device,
            )
            all_metrics.append(metrics)

    summarize_results(all_metrics)


if __name__ == "__main__":
    main()