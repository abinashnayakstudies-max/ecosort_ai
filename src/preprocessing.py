"""
preprocessing.py - everything that turns a raw photo into a model input tensor.

EVAL / INFERENCE pipeline (deterministic, used for val, test, app, webcam):
    Resize -> ToTensor (0..255 -> 0..1, HWC -> CHW) -> Normalize (ImageNet mean/std)

TRAIN pipeline adds RANDOM augmentation. Every augmentation is chosen for a
reason specific to waste photos; ones we deliberately DID NOT use are listed too.
"""
from torchvision import transforms as T
from src.config import IMAGENET_MEAN, IMAGENET_STD, DEFAULT_IMG_SIZE


def get_eval_transform(img_size: int = DEFAULT_IMG_SIZE):
    """No randomness. We squash to img_size x img_size instead of center-cropping
    so an item near the edge of the photo is never cut off."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(img_size: int = DEFAULT_IMG_SIZE, augment: bool = True):
    if not augment:                      # used for the augmentation ablation study
        return get_eval_transform(img_size)
    return T.Compose([
        # RANDOM CROP (zoom 70-100% of the area): phone photos differ in distance
        # and framing; forces the model to recognise partially visible items.
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        # HORIZONTAL FLIP: a bottle or a phone is still the same class mirrored.
        T.RandomHorizontalFlip(p=0.5),
        # SMALL ROTATION (+-15 deg): waste is thrown on the floor at any tilt, but
        # small angles keep the object recognisable. (No vertical flip: upside-down
        # scenes with gravity/table cues are unnatural and add nothing.)
        T.RandomRotation(15),
        # BRIGHTNESS / CONTRAST (+-30%): indoor/outdoor/night lighting varies a lot
        # in real photos. Small hue shift is NOT used: colour is a real clue
        # (green organic waste, blue circuit boards).
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Undo Normalize so a tensor can be shown as an image (used by Grad-CAM/plots)."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
