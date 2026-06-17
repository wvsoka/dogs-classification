import json
from pathlib import Path

import pandas as pd

from config import LOG_DIR, METRICS_DIR, MODELS


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name in MODELS:
        for fold in range(5):
            summary_path = LOG_DIR / f"{model_name}_fold{fold}_summary.json"

            if not summary_path.exists():
                print(f"Brak pliku: {summary_path}")
                continue

            summary = load_summary(summary_path)

            row = {
                "model": model_name,
                "fold": fold,
                "head_best_val_loss": summary["head"]["best_val_loss"],
                "head_best_epoch": summary["head"]["best_epoch"],
                "finetune_best_val_loss": summary["finetune"]["best_val_loss"],
                "finetune_best_epoch": summary["finetune"]["best_epoch"],
                "training_time_seconds": summary["training_time_seconds"],
                "training_time_minutes": summary["training_time_seconds"] / 60,
                "device": summary["device"],
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("Nie znaleziono żadnych wyników.")
        return

    raw_path = METRICS_DIR / "cv_results_raw.csv"
    df.to_csv(raw_path, index=False)

    summary_df = (
        df.groupby("model")
        .agg(
            mean_val_loss=("finetune_best_val_loss", "mean"),
            std_val_loss=("finetune_best_val_loss", "std"),
            mean_training_time_min=("training_time_minutes", "mean"),
            std_training_time_min=("training_time_minutes", "std"),
            mean_best_epoch=("finetune_best_epoch", "mean"),
            std_best_epoch=("finetune_best_epoch", "std"),
        )
        .reset_index()
    )

    summary_path = METRICS_DIR / "cv_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nWyniki pojedynczych foldów:")
    print(df)

    print("\nPodsumowanie 5CV:")
    print(summary_df)

    print("\nZapisano:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()