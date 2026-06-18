from pathlib import Path
import random

import numpy as np
import pandas as pd
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "Images"
LISTS_DIR = DATASET_DIR / "lists"

TRAIN_LIST = LISTS_DIR / "train_list.mat"
TEST_LIST = LISTS_DIR / "test_list.mat"

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def matlab_cell_to_str(value) -> str:
    while isinstance(value, (list, tuple, np.ndarray)):
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return ""
            value = value.flatten()[0]
        else:
            value = value[0]

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def load_stanford_list(mat_path: Path, split_name: str) -> pd.DataFrame:
    data = scipy.io.loadmat(mat_path)

    file_list = data["file_list"]
    labels = data["labels"].squeeze()

    rows = []

    for i in range(len(labels)):
        rel_path = matlab_cell_to_str(file_list[i])
        label_id = int(labels[i])

        class_dir = Path(rel_path).parts[0]

        if "-" in class_dir:
            synset, breed = class_dir.split("-", 1)
        else:
            synset, breed = class_dir, class_dir

        breed_clean = breed.replace("_", " ")

        image_path = IMAGES_DIR / rel_path

        rows.append({
            "split": split_name,
            "rel_path": rel_path,
            "label_id_original": label_id,
            "label_id_zero_based": label_id - 1,
            "class_dir": class_dir,
            "synset": synset,
            "breed": breed_clean,
            "exists": image_path.exists(),
        })

    return pd.DataFrame(rows)


def collect_image_sizes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for idx, row in df.iterrows():
        # img_path = Path(row["image_path"])
        img_path = IMAGES_DIR / row["rel_path"]

        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size

            rows.append({
                "split": row["split"],
                "rel_path": row["rel_path"],
                "label_id_original": row["label_id_original"],
                "breed": row["breed"],
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height != 0 else np.nan,
            })

        except Exception as e:
            print(f"Nie udało się odczytać obrazu: {img_path} | {e}")

        if (idx + 1) % 2000 == 0:
            print(f"Przetworzono {idx + 1} obrazów...")

    return pd.DataFrame(rows)


def save_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        df.pivot_table(
            index=["label_id_original", "breed", "class_dir", "synset"],
            columns="split",
            values="rel_path",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    if "train" not in distribution.columns:
        distribution["train"] = 0
    if "test" not in distribution.columns:
        distribution["test"] = 0

    distribution["total"] = distribution["train"] + distribution["test"]
    distribution = distribution.sort_values("total", ascending=False)

    output_path = TABLES_DIR / "class_distribution.csv"
    distribution.to_csv(output_path, index=False)

    return distribution


def save_dataset_summary(df: pd.DataFrame, distribution: pd.DataFrame, image_sizes: pd.DataFrame) -> pd.DataFrame:
    total_counts = distribution["total"]

    summary = {
        "number_of_classes": df["label_id_original"].nunique(),
        "number_of_images_total": len(df),
        "number_of_images_train": int((df["split"] == "train").sum()),
        "number_of_images_test": int((df["split"] == "test").sum()),
        "min_images_per_class": int(total_counts.min()),
        "max_images_per_class": int(total_counts.max()),
        "mean_images_per_class": float(total_counts.mean()),
        "std_images_per_class": float(total_counts.std()),
        "imbalance_ratio_max_min": float(total_counts.max() / total_counts.min()),
    }

    if not image_sizes.empty:
        summary.update({
            "min_width": int(image_sizes["width"].min()),
            "max_width": int(image_sizes["width"].max()),
            "mean_width": float(image_sizes["width"].mean()),
            "min_height": int(image_sizes["height"].min()),
            "max_height": int(image_sizes["height"].max()),
            "mean_height": float(image_sizes["height"].mean()),
            "mean_aspect_ratio": float(image_sizes["aspect_ratio"].mean()),
        })

    summary_df = pd.DataFrame([summary])
    output_path = TABLES_DIR / "dataset_summary.csv"
    summary_df.to_csv(output_path, index=False)

    return summary_df


def plot_class_distribution_histogram(distribution: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(distribution["total"], bins=20, edgecolor="black")
    plt.xlabel("Liczba obrazów w klasie")
    plt.ylabel("Liczba klas")
    plt.title("Rozkład liczby obrazów na klasę w Stanford Dogs Dataset")
    plt.tight_layout()

    output_path = FIGURES_DIR / "class_distribution_histogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_bottom_classes(distribution: pd.DataFrame, n: int = 10):
    bottom = distribution.sort_values("total", ascending=True).head(n)
    top = distribution.sort_values("total", ascending=False).head(n)

    combined = pd.concat([bottom, top]).drop_duplicates(subset=["breed"])
    combined = combined.sort_values("total", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(combined["breed"], combined["total"])
    plt.xlabel("Liczba obrazów")
    plt.ylabel("Rasa")
    plt.title(f"{n} najmniej i {n} najbardziej licznych klas")
    plt.tight_layout()

    output_path = FIGURES_DIR / "top_bottom_classes.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_image_size_scatter(image_sizes: pd.DataFrame):
    if image_sizes.empty:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(image_sizes["width"], image_sizes["height"], alpha=0.35, s=8)
    plt.xlabel("Szerokość obrazu [px]")
    plt.ylabel("Wysokość obrazu [px]")
    plt.title("Rozkład rozmiarów obrazów")
    plt.tight_layout()

    output_path = FIGURES_DIR / "image_size_scatter.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_example_images(df: pd.DataFrame, n_classes: int = 12, seed: int = 42):
    random.seed(seed)

    breeds = sorted(df["breed"].unique())
    selected_breeds = random.sample(breeds, min(n_classes, len(breeds)))

    selected_rows = []

    for breed in selected_breeds:
        breed_df = df[df["breed"] == breed]
        row = breed_df.sample(1, random_state=seed).iloc[0]
        selected_rows.append(row)

    cols = 4
    rows = int(np.ceil(len(selected_rows) / cols))

    plt.figure(figsize=(12, 3 * rows))

    for i, row in enumerate(selected_rows):
        # img_path = Path(row["image_path"])
        img_path = IMAGES_DIR / row["rel_path"]

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(row["breed"], fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    output_path = FIGURES_DIR / "example_images_grid.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    print("ANALIZA STANFORD DOGS DATASET")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu obrazów: {IMAGES_DIR}")

    if not TRAIN_LIST.exists():
        raise FileNotFoundError(f"Nie znaleziono train_list.mat: {TRAIN_LIST}")

    if not TEST_LIST.exists():
        raise FileNotFoundError(f"Nie znaleziono test_list.mat: {TEST_LIST}")

    train_df = load_stanford_list(TRAIN_LIST, "train")
    test_df = load_stanford_list(TEST_LIST, "test")

    df = pd.concat([train_df, test_df], ignore_index=True)

    all_files_path = TABLES_DIR / "all_files_with_splits.csv"
    df.to_csv(all_files_path, index=False)

    missing = df[~df["exists"]]
    if len(missing) > 0:
        print("UWAGA: Niektóre obrazy z list .mat nie istnieją na dysku.")
        print(missing[["split", "rel_path"]].head())
    else:
        print("Wszystkie obrazy z list .mat istnieją na dysku.")

    distribution = save_class_distribution(df)
    image_sizes = collect_image_sizes(df)

    image_sizes_path = TABLES_DIR / "image_sizes.csv"
    image_sizes.to_csv(image_sizes_path, index=False)

    summary_df = save_dataset_summary(df, distribution, image_sizes)

    plot_class_distribution_histogram(distribution)
    plot_top_bottom_classes(distribution, n=10)
    plot_image_size_scatter(image_sizes)
    plot_example_images(df, n_classes=12, seed=42)

    print("\nPODSUMOWANIE")
    print(summary_df.T)

    print("\nNajmniej liczne klasy:")
    print(distribution.sort_values("total", ascending=True).head(10)[["breed", "train", "test", "total"]])

    print("\nNajbardziej liczne klasy:")
    print(distribution.sort_values("total", ascending=False).head(10)[["breed", "train", "test", "total"]])


if __name__ == "__main__":
    main()