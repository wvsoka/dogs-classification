import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import METRICS_DIR, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    cm_path = (
        METRICS_DIR
        / "test"
        / "confusion_matrices"
        / f"{args.model}_fold{args.fold}_confusion_matrix.csv"
    )

    if not cm_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {cm_path}")

    cm = pd.read_csv(cm_path).values

    if args.normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    output_dir = OUTPUT_DIR / "figures" / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix: {args.model}, fold {args.fold}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    plt.tight_layout()

    suffix = "normalized" if args.normalize else "raw"
    output_path = output_dir / f"{args.model}_fold{args.fold}_confusion_matrix_{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Zapisano:", output_path)


if __name__ == "__main__":
    main()