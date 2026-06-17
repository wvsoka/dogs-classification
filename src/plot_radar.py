import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import METRICS_DIR, OUTPUT_DIR


def main():
    input_path = METRICS_DIR / "test" / "test_results_summary.csv"
    output_dir = OUTPUT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    metrics = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "f1_macro_mean",
        "f1_weighted_mean",
        "top5_accuracy_mean",
        "roc_auc_macro_mean",
    ]

    labels = [
        "Accuracy",
        "Balanced acc",
        "Macro F1",
        "Weighted F1",
        "Top-5 acc",
        "ROC-AUC",
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)
    ax.set_title("Comparison of models on test metrics", pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    output_path = output_dir / "radar_test_metrics.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Zapisano wykres radarowy:", output_path)


if __name__ == "__main__":
    main()