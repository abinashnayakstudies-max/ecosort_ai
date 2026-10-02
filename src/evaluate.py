"""
evaluate.py - the ONE place where the frozen TEST set is used.

    python src/evaluate.py                       # evaluates every checkpoints/*/best.pt
    python src/evaluate.py --runs baseline resnet18

For each model (identical split, identical preprocessing, identical code):
  accuracy, per-class precision/recall/F1, macro & weighted F1, confusion matrix,
  classification report, per-image predictions, CPU latency, size.
Then builds results/model_comparison.{csv,md}.

MODEL SELECTION uses VALIDATION macro-F1 (recorded at training time) plus
deployment limits (latency, size) - NOT the test set, otherwise the test set
would stop being an unbiased estimate.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, f1_score)
from PIL import Image

from src.config import CKPT_DIR, RESULTS_DIR, CLASS_NAMES, DATA_DIR, ROOT
from src.dataset import make_loader
from src.predict import load_checkpoint
from src.preprocessing import get_eval_transform
from src.models import count_parameters


def plot_confusion(cm, path, title):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


@torch.no_grad()
def collect_predictions(model, loader, ds, device):
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
    probs = torch.cat(probs).numpy()
    y_true = np.array([l for _, l in ds.samples])   # loader is not shuffled for eval splits
    return y_true, probs


@torch.no_grad()
def measure_latency_ms(model, sample_paths, img_size, n=30):
    """Median CPU time for ONE image, including preprocessing (what a user waits for)."""
    model = model.to("cpu").eval()
    tf = get_eval_transform(img_size)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    times = []
    for i in range(n + 5):
        img = Image.open(sample_paths[i % len(sample_paths)]).convert("RGB")
        t = time.perf_counter()
        model(tf(img).unsqueeze(0))
        if i >= 5:                                   # first 5 = warm-up, ignored
            times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def evaluate_run(run, device, args):
    ckpt = CKPT_DIR / run / "best.pt"
    model, meta = load_checkpoint(ckpt, device)
    ds, loader = make_loader("test", meta["img_size"], 64, False, args.num_workers, data_dir=args.data_dir)
    y_true, probs = collect_predictions(model, loader, ds, device)
    y_pred = probs.argmax(1)
    labels = list(range(4))

    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = RESULTS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, out / "confusion_matrix.png", f"{run} - test set")
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # per-image predictions (used by error analysis)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    pd.DataFrame(dict(
        path=[str(pth) for pth, _ in ds.samples],
        actual=[CLASS_NAMES[i] for i in y_true], predicted=[CLASS_NAMES[i] for i in y_pred],
        confidence=probs.max(1),
        top3=[";".join(f"{CLASS_NAMES[j]}:{probs[k, j]:.3f}" for j in top3[k]) for k in range(len(y_true))],
        correct=(y_true == y_pred))).to_csv(out / "predictions.csv", index=False)

    test_hash = hashlib.sha256("\n".join(sorted(Path(pth).name for pth, _ in ds.samples)).encode()).hexdigest()[:16]
    size_mb = (CKPT_DIR / run / "best.pt").stat().st_size / 1e6
    metrics = dict(
        run=run, model=meta["model_name"], augment=meta["augment"], n_test=len(y_true),
        test_set_fingerprint=test_hash,                 # identical for every model = same frozen test set
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(p.mean()), macro_recall=float(r.mean()),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class={c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(sup[i]))
                   for i, c in enumerate(CLASS_NAMES)},
        confusion_matrix=cm.tolist(),
        params=count_parameters(model), size_mb=round(size_mb, 2),
        cpu_latency_ms=round(measure_latency_ms(model, [pth for pth, _ in ds.samples], meta["img_size"]), 1),
        best_val_macro_f1=meta.get("best_val_macro_f1"))
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{run:24s} acc {metrics['accuracy']:.4f} macroF1 {metrics['macro_f1']:.4f} "
          f"{metrics['cpu_latency_ms']} ms  {metrics['size_mb']} MB")
    return metrics


def build_comparison(all_metrics, max_ms, max_mb):
    rows = [dict(Model=m["run"], Augmentation="yes" if m["augment"] else "no", Accuracy=m["accuracy"],
                 **{"Macro Precision": m["macro_precision"], "Macro Recall": m["macro_recall"],
                    "Macro F1": m["macro_f1"], "Weighted F1": m["weighted_f1"],
                    "Params (M)": m["params"] / 1e6, "Size (MB)": m["size_mb"],
                    "CPU ms/img": m["cpu_latency_ms"], "Val Macro F1": m["best_val_macro_f1"]})
            for m in all_metrics]
    df = pd.DataFrame(rows)
    df.round(4).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    (RESULTS_DIR / "model_comparison.md").write_text(df.round(4).to_markdown(index=False)
                                                     if _has_tabulate() else df.round(4).to_string(index=False))
    print("\n", df.round(4).to_string(index=False))

    # selection: augmentation-ablation runs are excluded; rule uses VALIDATION F1 + deployment limits
    cand = [m for m in all_metrics if m["augment"] and "smoke" not in m["run"]
            and m["cpu_latency_ms"] <= max_ms and m["size_mb"] <= max_mb]
    if not cand:
        print("No model satisfies the latency/size limits - relax --max_ms / --max_mb.")
        return
    best = max(cand, key=lambda m: m["best_val_macro_f1"] or 0)
    (RESULTS_DIR / "selected_model.json").write_text(json.dumps(dict(
        run=best["run"], checkpoint=(CKPT_DIR / best["run"] / "best.pt").relative_to(ROOT).as_posix(),  # relative -> works after deployment
        rule=f"highest validation macro-F1 among models with <= {max_ms} ms/img and <= {max_mb} MB",
        limits=dict(max_ms=max_ms, max_mb=max_mb)), indent=2))
    print(f"\nSelected model: {best['run']} (rule: best VAL macro-F1 within speed/size limits)")


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_ms", type=float, default=200, help="CPU latency limit for selection")
    ap.add_argument("--max_mb", type=float, default=100, help="model size limit for selection")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = a.runs or sorted(p.parent.name for p in CKPT_DIR.glob("*/best.pt"))
    if not runs:
        raise SystemExit("No checkpoints found. Train first (python src/train.py --model all).")
    all_m = [evaluate_run(r, device, a) for r in runs]
    assert len({m["test_set_fingerprint"] for m in all_m}) == 1, "Models were NOT tested on the same set!"
    build_comparison(all_m, a.max_ms, a.max_mb)


if __name__ == "__main__":
    main()
