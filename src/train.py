"""
train.py - reusable training pipeline for ALL models.

    python src/train.py --model baseline
    python src/train.py --model resnet18 --epochs 15
    python src/train.py --model all                # baseline + resnet18 + mobilenet_v3_small
    python src/train.py --model resnet18 --no_augment   # augmentation ablation
    python src/train.py --model baseline --smoke        # 1-epoch pipeline check on tiny subset

Per epoch: train on `train`, measure loss/accuracy/macro-F1 on `val`.
The checkpoint with the best VAL macro-F1 is saved. The test set is never
touched here (it is only used by evaluate.py) -> the test set stays frozen.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.config import CKPT_DIR, RESULTS_DIR, CLASS_NAMES, DATA_DIR, SEED
from src.dataset import make_loader, get_class_weights
from src.models import (build_model, set_backbone_trainable, count_parameters)


def set_seed(seed=SEED):
    """Make runs repeatable (same seed -> same shuffling/init on the same machine)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(name)


def make_scheduler(name, opt, epochs):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.3)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=2)
    return None


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. If optimizer is given -> training, else evaluation."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += logits.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    n = len(labels)
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    macro_f1 = float(f1_score(labels, preds, average="macro", labels=range(len(CLASS_NAMES)), zero_division=0))
    return total_loss / n, acc, macro_f1


def plot_history(hist, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(hist["train_loss"]) + 1)
    ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_title(f"{title} - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train"); ax[1].plot(ep, hist["val_acc"], label="val")
    ax[1].set_title(f"{title} - accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def train_one(cfg):
    set_seed(cfg["seed"])
    device = get_device()
    run = cfg["run_name"]
    ckpt_dir, res_dir = CKPT_DIR / run, RESULTS_DIR / run
    ckpt_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run} on {device} ===")

    mpc = 40 if cfg["smoke"] else None
    train_ds, train_loader = make_loader("train", cfg["img_size"], cfg["batch_size"], cfg["augment"],
                                         cfg["num_workers"], mpc, cfg["data_dir"], cfg["seed"])
    _, val_loader = make_loader("val", cfg["img_size"], cfg["batch_size"], False,
                                cfg["num_workers"], mpc, cfg["data_dir"], cfg["seed"])

    pretrained = cfg["model"] != "baseline"
    model = build_model(cfg["model"], pretrained=pretrained).to(device)
    weights = get_class_weights(train_ds.class_counts()).to(device) if cfg["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights)      # = softmax + negative log-likelihood
    optimizer = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"])
    scheduler = make_scheduler(cfg["scheduler"], optimizer, cfg["epochs"])

    if pretrained and cfg["freeze_epochs"] > 0:          # stage 1: train only the new head
        set_backbone_trainable(model, cfg["model"], False)

    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1", "lr"]}
    best_f1, bad_epochs, best_epoch = -1.0, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        if pretrained and epoch == cfg["freeze_epochs"] + 1 and cfg["freeze_epochs"] > 0:
            set_backbone_trainable(model, cfg["model"], True)   # stage 2: fine-tune everything
            print("Backbone unfrozen (fine-tuning).")
        tl, ta, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va, vf = run_epoch(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        for k, v in zip(hist, [tl, vl, ta, va, vf, lr_now]):
            hist[k].append(v)
        print(f"epoch {epoch:02d} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f} "
              f"macroF1 {vf:.3f} | lr {lr_now:.2e}")
        if scheduler is not None:
            scheduler.step(vf) if cfg["scheduler"] == "plateau" else scheduler.step()

        if vf > best_f1:                                  # best = highest VAL macro-F1
            best_f1, best_epoch, bad_epochs = vf, epoch, 0
            torch.save(dict(state_dict=model.state_dict(), model_name=cfg["model"],
                            class_names=CLASS_NAMES, img_size=cfg["img_size"],
                            best_val_macro_f1=vf, best_val_acc=va, best_epoch=epoch,
                            augment=cfg["augment"], config=cfg), ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
            if cfg["patience"] and bad_epochs >= cfg["patience"]:
                print(f"Early stopping (no val improvement for {cfg['patience']} epochs).")
                break

    cfg_out = dict(cfg, params=count_parameters(model), train_seconds=round(time.time() - t0, 1),
                   best_epoch=best_epoch, best_val_macro_f1=best_f1, device=str(device),
                   torch=torch.__version__)
    (res_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
    (res_dir / "history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, res_dir / "training_curves.png", run)
    print(f"Best val macro-F1 {best_f1:.4f} at epoch {best_epoch}. Saved {ckpt_dir / 'best.pt'}")


DEFAULT_LR = {"baseline": 1e-3, "resnet18": 3e-4, "mobilenet_v3_small": 5e-4, "efficientnet_b0": 3e-4}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline",
                    help="baseline | resnet18 | mobilenet_v3_small | efficientnet_b0 | all")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default depends on model")
    ap.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau", "none"])
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224, help="use 128 for faster CPU runs")
    ap.add_argument("--freeze_epochs", type=int, default=2, help="epochs with frozen backbone (pretrained only)")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience, 0 = off")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny 1-epoch run to test the pipeline")
    return ap.parse_args()


def main():
    a = parse_args()
    models = ["baseline", "resnet18", "mobilenet_v3_small"] if a.model == "all" else [a.model]
    for m in models:
        run = a.run_name or (m + ("_noaug" if a.no_augment else ""))
        if a.smoke:
            run += "_smoke"
        cfg = dict(model=m, run_name=run, epochs=1 if a.smoke else a.epochs, batch_size=a.batch_size,
                   lr=a.lr or DEFAULT_LR[m], optimizer=a.optimizer, scheduler=a.scheduler,
                   weight_decay=a.weight_decay, img_size=a.img_size, freeze_epochs=a.freeze_epochs,
                   patience=a.patience, augment=not a.no_augment, class_weights=not a.no_class_weights,
                   num_workers=a.num_workers, seed=a.seed, data_dir=a.data_dir, smoke=a.smoke)
        if a.smoke:
            cfg["freeze_epochs"] = 0
        train_one(cfg)


if __name__ == "__main__":
    main()
