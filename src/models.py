import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def build_model(model_name: str, num_classes: int = NUM_CLASSES, freeze: bool = True) -> nn.Module:
    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)

        if freeze:
            freeze_backbone(model)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)

        if freeze:
            freeze_backbone(model)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        if freeze:
            freeze_backbone(model)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Nieznany model: {model_name}")

    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total,
        "trainable_params": trainable,
    }