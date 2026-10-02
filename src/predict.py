"""
predict.py - load a checkpoint and classify ONE image.

    python src/predict.py path/to/photo.jpg [--ckpt checkpoints/resnet18/best.pt]

Confidence = the softmax probability of the predicted class.
  softmax(z_i) = exp(z_i) / sum_j exp(z_j)   -> 4 positive numbers summing to 1.
It is NOT a guarantee of correctness: an over-confident network can be wrong,
and it must still pick one of the 4 classes even for a photo of a cat.
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from src.config import ROOT, CKPT_DIR, RESULTS_DIR, CLASS_NAMES, DISPLAY_NAMES, IMG_EXTS, LOW_CONFIDENCE
from src.models import build_model
from src.preprocessing import get_eval_transform


class InvalidImageError(ValueError):
    """Raised for empty / unsupported / corrupt uploads (shown nicely in the app)."""


def load_image_safe(source, filename=None, max_mb=15) -> Image.Image:
    """source: path, bytes, or file-like object. Returns an RGB PIL image or raises."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    if name and Path(name).suffix.lower() not in IMG_EXTS:
        raise InvalidImageError("Unsupported file type. Please upload a JPG, JPEG or PNG image.")
    try:
        if isinstance(source, (str, Path)):
            data = Path(source).read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = source.read()
    except OSError as e:
        raise InvalidImageError(f"Could not read the file: {e}")
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise InvalidImageError(f"File is larger than {max_mb} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()                                   # forces full decode -> catches corruption
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise InvalidImageError("The file is not a valid image or is corrupted.")
    if min(img.size) < 32:
        raise InvalidImageError("Image is too small (minimum 32 pixels per side).")
    return img.convert("RGB")


def load_checkpoint(path=None, device="cpu"):
    """Returns (model, meta). If path is None uses results/selected_model.json, else first checkpoint."""
    if path is None:
        sel = RESULTS_DIR / "selected_model.json"
        if sel.exists():
            path = Path(json.loads(sel.read_text())["checkpoint"])
            if not path.is_absolute():
                path = ROOT / path            # stored relative to the project root
        else:
            found = sorted(CKPT_DIR.glob("*/best.pt"))
            if not found:
                raise FileNotFoundError("No trained model found. Run: python src/train.py --model resnet18")
            path = found[0]
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], pretrained=False)   # architecture only, weights come from ckpt
    model.load_state_dict(ck["state_dict"])
    for p in model.parameters():
        p.requires_grad = True                                 # Grad-CAM needs gradients
    model.to(device).eval()
    meta = {k: ck[k] for k in ["model_name", "class_names", "img_size", "best_val_macro_f1", "augment"] if k in ck}
    meta["checkpoint"] = str(path)
    return model, meta


def probabilities_from_logits(logits):
    return F.softmax(logits, dim=1)


def summarize(probs, class_names=CLASS_NAMES, low_conf=LOW_CONFIDENCE):
    """probs: 1-D tensor of 4 probabilities -> result dict (pure function, easy to unit-test)."""
    p = probs.detach().cpu().flatten()
    top_p, top_i = torch.topk(p, k=min(3, len(p)))
    top3 = [(class_names[i], float(v)) for v, i in zip(top_p, top_i)]
    conf = float(top_p[0])
    margin = float(top_p[0] - top_p[1]) if len(p) > 1 else 1.0
    warning = None
    if conf < low_conf:
        warning = "Low confidence - the model is unsure. Try a clearer photo, or check the alternatives."
    elif margin < 0.15:
        warning = "The top two classes are close - treat this result with caution."
    return dict(predicted_class=class_names[int(top_i[0])], confidence=conf, top3=top3,
                probabilities=[float(v) for v in p], warning=warning)


@torch.no_grad()
def predict_image(model, pil_image, img_size, device="cpu", class_names=CLASS_NAMES):
    x = get_eval_transform(img_size)(pil_image).unsqueeze(0).to(device)
    return summarize(probabilities_from_logits(model(x))[0], class_names)


def format_result(r):
    lines = [f"Predicted class:\n{DISPLAY_NAMES[r['predicted_class']]}", "",
             f"Confidence:\n{r['confidence'] * 100:.1f}%", "", "Top alternatives:"]
    lines += [f"{i}. {DISPLAY_NAMES[c]} — {p * 100:.1f}%" for i, (c, p) in enumerate(r["top3"], 1)]
    if r["warning"]:
        lines += ["", "WARNING: " + r["warning"]]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    model, meta = load_checkpoint(a.ckpt)
    print(format_result(predict_image(model, load_image_safe(a.image), meta["img_size"])))
