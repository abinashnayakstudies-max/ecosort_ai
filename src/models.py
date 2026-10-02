"""
models.py - the three model families we compare.

1. BaselineCNN       small CNN trained from scratch
2. resnet18          ImageNet-pretrained, last layer replaced (transfer learning)
3. mobilenet_v3_small / efficientnet_b0   pretrained, light-weight alternatives
"""
import torch
import torch.nn as nn
from torchvision import models as tv

from src.config import NUM_CLASSES

SUPPORTED = ["baseline", "resnet18", "mobilenet_v3_small", "efficientnet_b0"]


class BaselineCNN(nn.Module):
    """
    4 blocks of [Conv -> BatchNorm -> ReLU -> MaxPool], then
    global average pooling -> Dropout -> Fully connected (4 outputs).

    Conv:  learns small filters (edges -> textures -> shapes).
    ReLU:  non-linearity, so stacked layers can learn more than a straight line.
    Pool:  halves height/width -> fewer numbers, some shift tolerance.
    Dropout: randomly zeroes activations while training -> less overfitting.
    Output: raw scores ("logits"); softmax is applied later, in the loss / predict.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in (32, 64, 128, 256):
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace=False),        # not inplace: Grad-CAM hooks it
                       nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)           # works for any image size
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes=NUM_CLASSES, pretrained=True):
    """Return a model whose output layer has `num_classes` units."""
    if name == "baseline":
        return BaselineCNN(num_classes)
    if name == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)                  # new head
    elif name == "mobilenet_v3_small":
        m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED}")
    return m


def head_parameters(model, name):
    """Parameters of the NEW classification layer (trained first, backbone frozen)."""
    if name == "baseline":
        head = model.classifier
    elif name == "resnet18":
        head = model.fc
    elif name in ("mobilenet_v3_small", "efficientnet_b0"):
        head = model.classifier
    else:
        raise ValueError(f"Unknown model: {name}")
    return list(head.parameters())


def set_backbone_trainable(model, name, trainable: bool):
    """Freeze/unfreeze everything except the new head."""
    head_ids = {id(p) for p in head_parameters(model, name)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def gradcam_target_layer(model, name):
    """Last convolutional feature map: it still has spatial layout but the
    richest semantics, which is why Grad-CAM uses it."""
    if name == "baseline":
        return model.features[-2]          # ReLU before the last MaxPool
    if name == "resnet18":
        return model.layer4[-1]
    return model.features[-1]              # mobilenet / efficientnet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
