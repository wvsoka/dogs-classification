import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import CHECKPOINT_DIR, METRICS_DIR, OUTPUT_DIR, SEED
from dataset import StanfordDogsDataset, get_transforms
from models import build_model
from utils import get_device, set_seed


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), class_idx


def get_label_name_map():
    """
    Buduje mapowanie label_id -> nazwa rasy na podstawie ścieżek train split.
    """
    ds = StanfordDogsDataset(split="train", transform=None)

    label_to_name = {}

    for rel_path, label in zip(ds.paths, ds.labels):
        folder_name = Path(rel_path).parts[0]  # np. n02085620-Chihuahua
        if "-" in folder_name:
            breed_name = folder_name.split("-", 1)[1]
        else:
            breed_name = folder_name

        breed_name = breed_name.replace("_", " ")
        label_to_name[label] = breed_name

    return label_to_name


def overlay_cam_on_image(pil_img, cam_array, alpha=0.4):
    """
    Nakłada heatmapę Grad-CAM na oryginalny obraz PIL.
    """
    img = pil_img.convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0

    cam_uint8 = (cam_array * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_uint8).resize(img.size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_pil).astype(np.float32) / 255.0

    heatmap = cm.jet(cam_resized)[..., :3]  # RGB

    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def load_predictions_for_all_folds(model_name="resnet18", n_folds=5):
    pred_dir = METRICS_DIR / "test" / "predictions"

    dfs = []

    for fold in range(n_folds):
        path = pred_dir / f"{model_name}_fold{fold}_test_predictions.csv"

        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

        df = pd.read_csv(path)
        df = df.rename(columns={
            "pred_label": f"pred_fold{fold}",
            "correct": f"correct_fold{fold}",
        })

        dfs.append(df[["image_index", "true_label", f"pred_fold{fold}", f"correct_fold{fold}"]])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["image_index", "true_label"], how="inner")

    correct_cols = [f"correct_fold{i}" for i in range(n_folds)]
    merged["correct_count"] = merged[correct_cols].sum(axis=1)

    return merged


def select_examples(df, n_correct=3, n_wrong=3, random_state=42):
    """
    Wybiera przykłady z różnych klas:
    - obrazy poprawnie klasyfikowane przez wszystkie 5 foldów,
    - obrazy błędnie klasyfikowane przez większość foldów.

    """
    all_correct = df[df["correct_count"] == 5].copy()
    mostly_wrong = df[df["correct_count"] <= 1].copy()

    selected_correct = (
        all_correct
        .drop_duplicates(subset=["true_label"])
        .sample(
            n=min(n_correct, all_correct["true_label"].nunique()),
            random_state=random_state,
        )
    )

    selected_wrong = (
        mostly_wrong
        .drop_duplicates(subset=["true_label"])
        .sample(
            n=min(n_wrong, mostly_wrong["true_label"].nunique()),
            random_state=random_state + 1,
        )
    )

    selected_correct["category"] = "correct"
    selected_wrong["category"] = "wrong"

    selected = pd.concat([selected_correct, selected_wrong], axis=0).copy()

    return selected


def load_resnet18_fold_model(fold, device):
    checkpoint_path = CHECKPOINT_DIR / f"resnet18_fold{fold}_best_finetune.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {checkpoint_path}")

    model = build_model("resnet18", freeze=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint_path


def create_gradcam_figure_for_image(
    image_index,
    category,
    raw_dataset,
    eval_dataset,
    label_to_name,
    output_dir,
    device,
):
    raw_img, true_label = raw_dataset[image_index]
    input_tensor, _ = eval_dataset[image_index]

    input_tensor = input_tensor.unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, 6, figsize=(24, 5))

    axes[0].imshow(raw_img)
    axes[0].set_title(f"Original\nTrue: {label_to_name[true_label]}")
    axes[0].axis("off")

    summary_rows = []

    for fold in range(5):
        model, checkpoint_path = load_resnet18_fold_model(fold, device)

        # target layer dla ResNet-18
        target_layer = model.layer4
        gradcam = GradCAM(model, target_layer)

        cam, pred_label = gradcam.generate(input_tensor)
        gradcam.remove_hooks()

        overlay = overlay_cam_on_image(raw_img, cam, alpha=0.4)

        pred_name = label_to_name.get(pred_label, str(pred_label))
        true_name = label_to_name.get(true_label, str(true_label))
        correct = (pred_label == true_label)

        axes[fold + 1].imshow(overlay)
        axes[fold + 1].set_title(
            f"Fold {fold}\nPred: {pred_name}\n"
            f"{'correct' if correct else 'wrong'}"
        )
        axes[fold + 1].axis("off")

        summary_rows.append({
            "image_index": image_index,
            "category": category,
            "fold": fold,
            "checkpoint": str(checkpoint_path),
            "true_label": int(true_label),
            "true_name": true_name,
            "pred_label": int(pred_label),
            "pred_name": pred_name,
            "correct": bool(correct),
        })

        del model

    fig.suptitle(f"Grad-CAM comparison for image_index={image_index} ({category})", fontsize=16)
    plt.tight_layout()

    image_output_dir = output_dir / category
    image_output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = image_output_dir / f"image_{image_index}_gradcam_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return summary_rows, fig_path


def main():
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    output_dir = OUTPUT_DIR / "figures" / "gradcam_resnet18_all_folds"
    output_dir.mkdir(parents=True, exist_ok=True)

    label_to_name = get_label_name_map()

    merged_preds = load_predictions_for_all_folds(model_name="resnet18", n_folds=5)

    selected = select_examples(merged_preds, n_correct=3, n_wrong=3)

    if selected.empty:
        print("Nie udało się wybrać przykładów.")
        return

    selected_path = output_dir / "selected_examples.csv"
    selected.to_csv(selected_path, index=False)
    print("Zapisano:", selected_path)

    _, eval_transform = get_transforms()
    raw_dataset = StanfordDogsDataset(split="test", transform=None)
    eval_dataset = StanfordDogsDataset(split="test", transform=eval_transform)

    all_summary_rows = []

    for _, row in selected.iterrows():
        image_index = int(row["image_index"])
        category = row["category"]

        summary_rows, fig_path = create_gradcam_figure_for_image(
            image_index=image_index,
            category=category,
            raw_dataset=raw_dataset,
            eval_dataset=eval_dataset,
            label_to_name=label_to_name,
            output_dir=output_dir,
            device=device,
        )

        all_summary_rows.extend(summary_rows)
        print("Zapisano figurę:", fig_path)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_csv_path = output_dir / "gradcam_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("Zapisano:", summary_csv_path)
    print("Gotowe.")


if __name__ == "__main__":
    main()