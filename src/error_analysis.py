"""
error_analysis.py - list misclassified TEST images and draft the failure log.

    python src/error_analysis.py --run resnet18 --n 30

Outputs (results/error_analysis/<run>/):
  errors.csv, error_grid_*.png (image + actual/pred/conf/top-3), and ../../failure_log.md

HONESTY NOTE: the "likely reason" column is only an AUTOMATIC HINT from simple
image statistics (brightness, blur, known confusable pairs). You must LOOK at each
image and edit the reason yourself - this is required for the viva.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import RESULTS_DIR, ROOT

CONFUSABLE = {
    frozenset(["dry", "recyclable"]): "Similar visual appearance / ambiguous definition (e.g. plastic vs non-recyclable plastic, paper vs tissue)",
    frozenset(["dry", "wet"]): "Similar texture/colour (e.g. soiled paper or cardboard vs food waste)",
    frozenset(["recyclable", "wet"]): "Food residue on recyclable packaging",
    frozenset(["e_waste", "dry"]): "E-waste part looks like generic plastic/metal debris",
    frozenset(["e_waste", "recyclable"]): "Metal/plastic casing of a device looks like a recyclable item",
    frozenset(["e_waste", "wet"]): "Unusual: check for label noise or clutter",
}
FIXES = {"dark": "Add brightness augmentation / more low-light training photos",
         "blurry": "Add blur augmentation; collect sharper images",
         "confusable": "Add more examples of the boundary cases; clarify the class definition",
         "clutter": "Add cluttered-background training images; consider object detection/cropping first",
         "unknown": "Inspect the image manually; check for label noise"}


def hints(path, actual, pred):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reasons, fix = [], "unknown"
    if gray.mean() < 70:
        reasons.append("poor lighting (dark image)"); fix = "dark"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 60:
        reasons.append("blurry / low detail"); fix = "blurry"
    pair = CONFUSABLE.get(frozenset([actual, pred]))
    if pair:
        reasons.append(pair); fix = fix if reasons[:-1] else "confusable"
    edges = cv2.Canny(gray, 100, 200).mean() / 255
    if edges > 0.12:
        reasons.append("busy background / clutter (many edges)"); fix = fix if len(reasons) > 1 else "clutter"
    return "; ".join(reasons) or "unclear - inspect manually", FIXES[fix]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    df = pd.read_csv(RESULTS_DIR / a.run / "predictions.csv")
    err = df[~df.correct].sort_values("confidence", ascending=False).head(a.n).reset_index(drop=True)
    total_err = int((~df.correct).sum())
    print(f"{total_err} misclassified of {len(df)} test images; analysing top {len(err)} by confidence.")
    if len(err) < 20:
        print("WARNING: fewer than 20 errors exist. Requirement is >=20: also add real-world failures "
              "from real_world_eval.py to the failure log.")

    out = RESULTS_DIR / "error_analysis" / a.run
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in err.iterrows():
        reason, fix = hints(r.path, r.actual, r.predicted)
        rows.append(dict(image_id=Path(r.path).name, actual=r.actual, predicted=r.predicted,
                         confidence=round(r.confidence, 3), top3=r.top3,
                         auto_reason_hint=reason, possible_improvement=fix, path=r.path))
    res = pd.DataFrame(rows)
    res.to_csv(out / "errors.csv", index=False)

    for page, start in enumerate(range(0, len(res), 12)):
        chunk = res.iloc[start:start + 12]
        fig, axes = plt.subplots(3, 4, figsize=(14, 11))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), chunk.iterrows()):
            ax.imshow(Image.open(r.path))
            ax.set_title(f"{r.image_id}\nactual: {r.actual} | pred: {r.predicted} ({r.confidence:.0%})",
                         fontsize=8)
        plt.tight_layout(); plt.savefig(out / f"error_grid_{page + 1}.png", dpi=110); plt.close()

    lines = [f"# Failure Log ({a.run})", "",
             f"Test images misclassified: {total_err} of {len(df)}. Listed: the {len(res)} most confident errors.", "",
             "> The *Likely reason (auto hint)* column comes from simple image statistics. "
             "**Replace it with your own judgement after viewing each image** (see results/error_analysis).", "",
             "| Image ID | Actual | Predicted | Confidence | Top-3 | Likely reason (auto hint) | Your reason (edit) | Possible improvement |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r.image_id} | {r.actual} | {r.predicted} | {r.confidence:.1%} | {r.top3} | "
                     f"{r.auto_reason_hint} | TODO | {r.possible_improvement} |")
    (ROOT / "failure_log.md").write_text("\n".join(lines) + "\n")
    print("Wrote failure_log.md and", out)


if __name__ == "__main__":
    main()
