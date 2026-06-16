from pathlib import Path
from typing import Tuple, List

import scipy.io
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import IMAGES_DIR, TRAIN_LIST, TEST_LIST, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


def _matlab_cell_to_str(x) -> str:
    while isinstance(x, (list, tuple)) or hasattr(x, "shape"):
        try:
            x = x[0]
        except Exception:
            break
        if isinstance(x, str):
            return x
    return str(x)


def load_stanford_list(mat_path: Path) -> Tuple[List[str], List[int]]:
    data = scipy.io.loadmat(mat_path)

    file_list = data["file_list"]
    labels = data["labels"].squeeze()

    paths = []
    y = []

    for i in range(len(labels)):
        rel_path = _matlab_cell_to_str(file_list[i])
        label = int(labels[i]) - 1  # MATLAB labels: 1-120, PyTorch: 0-119

        paths.append(rel_path)
        y.append(label)

    return paths, y


class StanfordDogsDataset(Dataset):
    def __init__(self, split: str, transform=None):
        if split not in {"train", "test"}:
            raise ValueError("split musi być 'train' albo 'test'.")

        self.split = split
        self.transform = transform

        mat_path = TRAIN_LIST if split == "train" else TEST_LIST
        self.paths, self.labels = load_stanford_list(mat_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        label = self.labels[idx]

        img_path = IMAGES_DIR / rel_path

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, test_transform


def get_dataloaders():
    train_transform, test_transform = get_transforms()

    train_dataset = StanfordDogsDataset(split="train", transform=train_transform)
    test_dataset = StanfordDogsDataset(split="test", transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset