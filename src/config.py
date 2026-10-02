"""
config.py - constants shared by EVERY file in the project.

Why one place? If the class order differed between training and the app,
the model would say "wet" when it means "dry". A single source of truth
prevents that whole category of bug.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "raw"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# ORDER MATTERS: index 0 -> "dry", 1 -> "wet", ... The model outputs 4 numbers
# and the i-th number is the score of CLASS_NAMES[i].
CLASS_NAMES = ["dry", "wet", "recyclable", "e_waste"]
DISPLAY_NAMES = {"dry": "Dry Waste", "wet": "Wet Waste",
                 "recyclable": "Recyclable Waste", "e_waste": "E-Waste"}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)   # statistics the pretrained models expect
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png"}

LOW_CONFIDENCE = 0.60   # below this we warn the user (tune it on validation data)
SEED = 42
