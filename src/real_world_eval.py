"""
real_world_eval.py - evaluate on YOUR OWN phone photos, reported SEPARATELY from the test set.

1. Put photos in real_world/images/
2. Fill real_world/labels.csv (template is created if missing) with columns:
   filename,true_class,lighting,background,angle,distance,occlusion,clutter
   (tags are free text, e.g. lighting=dim/bright/mixed, occlusion=none/partial ...)
3. python src/real_world_eval.py [--ckpt checkpoints/resnet18/best.pt]

Real photos differ from dataset photos (domain shift), so expect lower numbers.
Never use these images for training or model selection.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import classification_report

from src.config import ROOT, RESULTS_DIR, CLASS_NAMES
from src.predict import load_checkpoint, load_image_safe, predict_image

TAGS = ["lighting", "background", "angle", "distance", "occlusion", "clutter"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dir", default=str(ROOT / "real_world"))
    a = ap.parse_args()
    d = Path(a.dir)
    lab = d / "labels.csv"
    if not lab.exists():
        pd.DataFrame(columns=["filename", "true_class"] + TAGS).to_csv(lab, index=False)
        raise SystemExit(f"Created template {lab}. Add your photos + rows, then re-run.")
    df = pd.read_csv(lab)
    if df.empty:
        raise SystemExit("labels.csv has no rows yet.")
    model, meta = load_checkpoint(a.ckpt)
    rows = []
    for _, r in df.iterrows():
        res = predict_image(model, load_image_safe(d / "images" / r.filename), meta["img_size"])
        rows.append(dict(r, predicted=res["predicted_class"], confidence=res["confidence"],
                         correct=res["predicted_class"] == r.true_class,
                         top3=";".join(f"{c}:{p:.2f}" for c, p in res["top3"])))
    out = pd.DataFrame(rows)
    o = RESULTS_DIR / "real_world"
    o.mkdir(parents=True, exist_ok=True)
    out.to_csv(o / "real_world_predictions.csv", index=False)
    report = [f"# Real-world results ({meta['model_name']}) - separate from the test set", "",
              f"Photos: {len(out)}   Accuracy: {out.correct.mean():.1%}", "",
              "```", classification_report(out.true_class, out.predicted, labels=CLASS_NAMES, zero_division=0), "```"]
    for t in TAGS:
        if t in out and out[t].notna().any():
            report += ["", f"## Accuracy by {t}", out.groupby(t).correct.agg(["mean", "count"]).round(3).to_string()]
    (o / "real_world_report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
