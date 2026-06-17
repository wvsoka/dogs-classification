import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from config import METRICS_DIR, MODELS


def load_per_class_f1(model_name: str, fold: int) -> pd.Series:
    path = (
        METRICS_DIR
        / "test"
        / "reports"
        / f"{model_name}_fold{fold}_classification_report.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    df = pd.read_csv(path, index_col=0)

    # W classification_report klasy są zapisane jako "0", "1", ..., "119".
    class_rows = [str(i) for i in range(120)]

    f1 = df.loc[class_rows, "f1-score"]
    f1.index = f1.index.astype(int)

    return f1


def holm_bonferroni(p_values: dict) -> pd.DataFrame:
    """
    Korekta Holm-Bonferroni dla porównań parami.
    """
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)

    rows = []

    for rank, ((comparison, p), i) in enumerate(zip(items, range(m)), start=1):
        adjusted_alpha = 0.05 / (m - i)
        significant = p <= adjusted_alpha

        rows.append({
            "comparison": comparison,
            "p_value": p,
            "rank": rank,
            "holm_alpha": adjusted_alpha,
            "significant": significant,
        })

    return pd.DataFrame(rows)


def main():
    output_dir = METRICS_DIR / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Liczymy średnie per-class F1 po 5 foldach dla każdego modelu.
    model_f1 = {}

    for model_name in MODELS:
        fold_f1_values = []

        for fold in range(5):
            f1 = load_per_class_f1(model_name, fold)
            fold_f1_values.append(f1)

        f1_df = pd.concat(fold_f1_values, axis=1)
        f1_df.columns = [f"fold_{i}" for i in range(5)]

        mean_f1 = f1_df.mean(axis=1)
        std_f1 = f1_df.std(axis=1)

        model_f1[model_name] = mean_f1

        for class_id in mean_f1.index:
            rows.append({
                "model": model_name,
                "class_id": class_id,
                "mean_f1": mean_f1.loc[class_id],
                "std_f1": std_f1.loc[class_id],
            })

    per_class_df = pd.DataFrame(rows)
    per_class_path = output_dir / "per_class_f1_mean_std.csv"
    per_class_df.to_csv(per_class_path, index=False)

    # Przygotowanie danych do testu Friedmana.
    data = [model_f1[model].values for model in MODELS]

    friedman_stat, friedman_p = friedmanchisquare(*data)

    friedman_result = {
        "test": "Friedman test",
        "statistic": friedman_stat,
        "p_value": friedman_p,
        "models": MODELS,
        "n_classes": 120,
        "metric": "mean per-class F1 across 5 folds",
    }

    friedman_path = output_dir / "friedman_test.json"
    pd.Series(friedman_result).to_json(friedman_path, indent=4)

    print("\nFriedman test:")
    print(f"statistic = {friedman_stat:.6f}")
    print(f"p-value   = {friedman_p:.10f}")

    # Testy Wilcoxona parami.
    pairwise_p = {}
    pairwise_rows = []

    for model_a, model_b in itertools.combinations(MODELS, 2):
        x = model_f1[model_a].values
        y = model_f1[model_b].values

        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

        comparison = f"{model_a} vs {model_b}"
        pairwise_p[comparison] = p

        pairwise_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "wilcoxon_statistic": stat,
            "p_value": p,
            "mean_f1_a": np.mean(x),
            "mean_f1_b": np.mean(y),
            "mean_difference_a_minus_b": np.mean(x - y),
        })

    pairwise_df = pd.DataFrame(pairwise_rows)

    holm_df = holm_bonferroni(pairwise_p)

    pairwise_path = output_dir / "wilcoxon_pairwise_tests.csv"
    holm_path = output_dir / "holm_bonferroni_correction.csv"

    pairwise_df.to_csv(pairwise_path, index=False)
    holm_df.to_csv(holm_path, index=False)

    print("\nWilcoxon pairwise tests:")
    print(pairwise_df)

    print("\nHolm-Bonferroni correction:")
    print(holm_df)

    print("\nZapisano:")
    print(per_class_path)
    print(friedman_path)
    print(pairwise_path)
    print(holm_path)


if __name__ == "__main__":
    main()