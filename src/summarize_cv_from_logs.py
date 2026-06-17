from pathlib import Path

import pandas as pd

from config import LOG_DIR, METRICS_DIR, MODELS


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name in MODELS:
        for fold in range(5):
            log_path = LOG_DIR / f"{model_name}_fold{fold}_train_log.csv"

            if not log_path.exists():
                print(f"Brak pliku: {log_path}")
                continue

            df = pd.read_csv(log_path)

            # Interesuje nas najlepszy checkpoint z fazy fine-tune,
            # czyli epoka z najniższym val_loss.
            ft = df[df["phase"] == "finetune"].copy()

            if ft.empty:
                print(f"Brak fazy fine-tune w: {log_path}")
                continue

            best_row = ft.loc[ft["val_loss"].idxmin()]

            rows.append({
                "model": model_name,
                "fold": fold,
                "best_epoch": int(best_row["epoch"]),
                "train_loss": float(best_row["train_loss"]),
                "train_acc": float(best_row["train_acc"]),
                "val_loss": float(best_row["val_loss"]),
                "val_acc": float(best_row["val_acc"]),
                "lr": float(best_row["lr"]),
            })

    raw = pd.DataFrame(rows)

    raw_path = METRICS_DIR / "cv_results_from_logs_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby("model")
        .agg(
            mean_val_loss=("val_loss", "mean"),
            std_val_loss=("val_loss", "std"),
            mean_val_acc=("val_acc", "mean"),
            std_val_acc=("val_acc", "std"),
            mean_train_acc=("train_acc", "mean"),
            std_train_acc=("train_acc", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            std_best_epoch=("best_epoch", "std"),
        )
        .reset_index()
    )

    summary_path = METRICS_DIR / "cv_results_from_logs_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nWyniki najlepszych epok fine-tune:")
    print(raw)

    print("\nPodsumowanie 5CV:")
    print(summary)

    print("\nZapisano:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()