import pandas as pd
import matplotlib.pyplot as plt

from config import LOG_DIR, OUTPUT_DIR, MODELS


def plot_metric(model_name: str, fold: int, metric_train: str, metric_val: str, ylabel: str, output_name: str):
    log_path = LOG_DIR / f"{model_name}_fold{fold}_train_log.csv"

    if not log_path.exists():
        print(f"Brak pliku: {log_path}")
        return

    df = pd.read_csv(log_path)

    df["global_epoch"] = range(1, len(df) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(df["global_epoch"], df[metric_train], marker="o", label=metric_train)
    plt.plot(df["global_epoch"], df[metric_val], marker="o", label=metric_val)

    phase_change = len(df[df["phase"] == "head"])
    plt.axvline(phase_change + 0.5, linestyle="--", linewidth=1)
    plt.text(phase_change + 0.6, plt.ylim()[1] * 0.95, "fine-tuning", va="top")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{model_name}, fold {fold}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_dir = OUTPUT_DIR / "figures" / "training_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Zapisano:", output_path)


def main():
    for model_name in MODELS:
        for fold in range(5):
            plot_metric(
                model_name=model_name,
                fold=fold,
                metric_train="train_loss",
                metric_val="val_loss",
                ylabel="Loss",
                output_name=f"{model_name}_fold{fold}_loss.png",
            )

            plot_metric(
                model_name=model_name,
                fold=fold,
                metric_train="train_acc",
                metric_val="val_acc",
                ylabel="Accuracy",
                output_name=f"{model_name}_fold{fold}_accuracy.png",
            )


if __name__ == "__main__":
    main()