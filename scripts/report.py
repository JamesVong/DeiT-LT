#!/usr/bin/env python3
"""
report.py — DeiT-LT training results report

Scans all deit_out_*_if100 directories and prints a comparison table using
the AVG metric (CLS+DIST)/2, consistent with Table 6 of the DeiT-LT paper.

T4 (Decoupled Fine-tuning) is evaluated by running the fine-tuned checkpoint
through the full val set with averaged CLS+DIST logits — the same method used
during training. Result is cached next to the checkpoint so subsequent runs
are instant.

Usage:
    python scripts/report.py                         # CIFAR-10 LT (default)
    python scripts/report.py --dataset CIFAR100LT    # CIFAR-100 LT
    python scripts/report.py --root /path/           # explicit root
    python scripts/report.py --verbose               # also show CLS and DIST rows
    python scripts/report.py --reeval                # force re-run T4 eval
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Repo root on sys.path so deit_models / datasets are importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import deit_models
from datasets import build_dataset


# ---------------------------------------------------------------------------
# Paper baselines per dataset (Table 6, DeiT-LT Base B, AVG metric)
# ---------------------------------------------------------------------------
_PAPER_BASELINES = {
    "CIFAR10LT":  {"overall": 87.50, "head": 94.50, "mid": 84.10, "tail": 85.00},
    "CIFAR100LT": {"overall": 55.60, "head": 72.80, "mid": 55.40, "tail": 31.40},
}

_DATASET_LABEL = {
    "CIFAR10LT":  "CIFAR-10 LT",
    "CIFAR100LT": "CIFAR-100 LT",
}

RUN_ORDER = ["ce", "logit_adj", "balsoftmax", "classsmooth"]

RUN_LABELS = {
    "ce":          "CE (Cross-Entropy) baseline",
    "logit_adj":   "T1: Logit Adjustment (tau=1.0)",
    "balsoftmax":  "T2: Balanced Softmax",
    "classsmooth": "T3: Class-Aware Label Smoothing",
}

# Default head/tail split indices per dataset
_DEFAULT_CATEGORIES = {
    "CIFAR10LT":  [3, 7],
    "CIFAR100LT": [36, 71],
    "IMAGENETLT": [390, 835],
    "INAT18":     [842, 4543],
}

W_LABEL = 40
W_NUM   = 9


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def find_best_ckpt(outdir):
    hits = sorted(glob.glob(
        os.path.join(outdir, "**", "*_best_checkpoint.pth"), recursive=True))
    return hits[-1] if hits else None


def find_rolling_ckpt(outdir):
    hits = [f for f in sorted(glob.glob(
        os.path.join(outdir, "**", "*_checkpoint.pth"), recursive=True))
        if "_best_" not in f and "_epoch_" not in f and "_DRW_" not in f]
    return hits[-1] if hits else None


def load_ckpt_metrics(path):
    c = torch.load(path, map_location="cpu")
    return {
        "epoch":        c.get("epoch", -1),
        "overall":      c.get("best_acc_avg",   0.0),
        "head":         c.get("head_acc_avg",   0.0),
        "mid":          c.get("med_acc_avg",    0.0),
        "tail":         c.get("tail_acc_avg",   0.0),
        "overall_cls":  c.get("best_acc_cls",   0.0),
        "head_cls":     c.get("head_acc_cls",   0.0),
        "mid_cls":      c.get("med_acc_cls",    0.0),
        "tail_cls":     c.get("tail_acc_cls",   0.0),
        "overall_dist": c.get("best_acc_dist",  0.0),
        "head_dist":    c.get("head_acc_dist",  0.0),
        "mid_dist":     c.get("med_acc_dist",   0.0),
        "tail_dist":    c.get("tail_acc_dist",  0.0),
    }


def run_status(outdir, total_epochs=1200):
    rolling = find_rolling_ckpt(outdir)
    if rolling is None:
        return "not_started", -1
    try:
        ep = torch.load(rolling, map_location="cpu").get("epoch", -1)
    except Exception:
        return "not_started", -1
    return ("complete" if ep >= total_epochs - 1 else "in_progress"), ep


# ---------------------------------------------------------------------------
# T4 evaluation — full AVG inference on the fine-tuned checkpoint
# ---------------------------------------------------------------------------
def _eval_cache_path(ft_ckpt_path):
    base = os.path.splitext(ft_ckpt_path)[0]
    return base + "_eval.json"


def _run_eval(ft_ckpt_path):
    """Load the fine-tuned model and run full val-set evaluation.

    Returns a dict with overall/head/mid/tail for AVG, CLS, and DIST.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load fine-tune checkpoint
    ft_ckpt = torch.load(ft_ckpt_path, map_location="cpu")
    stage1_path = ft_ckpt.get("stage1_checkpoint")
    if not stage1_path or not os.path.exists(stage1_path):
        raise FileNotFoundError(
            f"stage1_checkpoint not found: {stage1_path}")

    # 2. Load stage-1 checkpoint to recover original training args
    stage1_ckpt = torch.load(stage1_path, map_location="cpu")
    train_args  = stage1_ckpt["args"]

    # Patch categories if missing from old checkpoints
    if not getattr(train_args, "categories", None):
        train_args.categories = _DEFAULT_CATEGORIES.get(
            train_args.data_set, [])

    # 3. Rebuild model from stage-1 arch args, load fine-tuned weights
    model = deit_models.__dict__[train_args.model](
        pretrained=False,
        num_classes=train_args.nb_classes,
        drop_rate=getattr(train_args, "drop", 0.0),
        drop_path_rate=getattr(train_args, "drop_path", 0.1),
        mask_attn=getattr(train_args, "mask_attn", False),
        early_stopping=getattr(train_args, "early_stopping", False),
    )
    model.load_state_dict(ft_ckpt["model"])
    model.to(device).eval()

    # 4. Build val dataset using training args
    dataset_val, _ = build_dataset(is_train=False, args=train_args)
    val_loader = DataLoader(
        dataset_val,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 5. Inference — average CLS and DIST logits, collect all predictions
    all_preds_avg  = []
    all_preds_cls  = []
    all_preds_dist = []
    all_targets    = []

    with torch.no_grad():
        for batch in val_loader:
            imgs, targets = batch[0].to(device), batch[1]
            with torch.cuda.amp.autocast():
                out = model(imgs)
            out_cls, out_dist = out
            out_avg = (out_cls + out_dist) / 2

            all_preds_avg.extend(out_avg.argmax(1).cpu().numpy())
            all_preds_cls.extend(out_cls.argmax(1).cpu().numpy())
            all_preds_dist.extend(out_dist.argmax(1).cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds_avg  = np.array(all_preds_avg)
    all_preds_cls  = np.array(all_preds_cls)
    all_preds_dist = np.array(all_preds_dist)
    all_targets    = np.array(all_targets)
    C = train_args.nb_classes

    # 6. Per-class accuracy then head/mid/tail splits
    def split_metrics(preds):
        per_class = np.array([
            (preds[all_targets == c] == c).mean() * 100.0
            if (all_targets == c).any() else 0.0
            for c in range(C)
        ])
        h_end, t_start = train_args.categories
        return {
            "overall": float((preds == all_targets).mean() * 100.0),
            "head":    float(per_class[:h_end].mean()),
            "mid":     float(per_class[h_end:t_start].mean()),
            "tail":    float(per_class[t_start:].mean()),
        }

    avg_m  = split_metrics(all_preds_avg)
    cls_m  = split_metrics(all_preds_cls)
    dist_m = split_metrics(all_preds_dist)

    return {
        "overall":      avg_m["overall"], "head":      avg_m["head"],
        "mid":          avg_m["mid"],     "tail":      avg_m["tail"],
        "overall_cls":  cls_m["overall"], "head_cls":  cls_m["head"],
        "mid_cls":      cls_m["mid"],     "tail_cls":  cls_m["tail"],
        "overall_dist": dist_m["overall"],"head_dist": dist_m["head"],
        "mid_dist":     dist_m["mid"],    "tail_dist": dist_m["tail"],
    }


def get_t4_metrics(ft_ckpt_path, force_reeval=False):
    """Return T4 eval metrics, using cache if available."""
    cache = _eval_cache_path(ft_ckpt_path)
    if not force_reeval and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    print("  [T4] Running full eval on fine-tuned checkpoint "
          f"(cache: {cache}) ...")
    m = _run_eval(ft_ckpt_path)
    with open(cache, "w") as f:
        json.dump(m, f, indent=2)
    return m


# ---------------------------------------------------------------------------
# Formatting — spaces only, fixed widths
# ---------------------------------------------------------------------------
def _n(v):
    if isinstance(v, (int, float)):
        return f"{v:{W_NUM}.2f}"
    return f"{'--':{W_NUM}}"


def _delta(v, ref):
    if isinstance(v, (int, float)) and ref is not None:
        return f"  {v - ref:>+6.2f}"
    return f"  {'':>6}"


def print_header():
    cols = (f"{'Overall':>{W_NUM}}  {'Head':>{W_NUM}}  "
            f"{'Mid':>{W_NUM}}  {'Tail':>{W_NUM}}  {'D-Ovr':>6}")
    print(f"  {'Run':<{W_LABEL}}  {cols}")
    print("  " + "-" * (W_LABEL + W_NUM * 4 + 20))


def print_row(label, overall, head, mid, tail, ref=None, note=""):
    nums = f"{_n(overall)}  {_n(head)}  {_n(mid)}  {_n(tail)}"
    d    = _delta(overall, ref)
    note_str = f"  {note}" if note else ""
    print(f"  {label:<{W_LABEL}}  {nums}{d}{note_str}")


def print_sub_row(tag, overall, head, mid, tail):
    label = f"    {tag}"
    nums  = f"{_n(overall)}  {_n(head)}  {_n(mid)}  {_n(tail)}"
    print(f"  {label:<{W_LABEL}}  {nums}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default="CIFAR10LT",
                        choices=list(_PAPER_BASELINES.keys()),
                        help="Dataset to report on (default: CIFAR10LT)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show CLS and DIST sub-rows")
    parser.add_argument("--reeval", action="store_true",
                        help="Force re-run T4 eval even if cache exists")
    parser.add_argument("--total-epochs", type=int, default=1200)
    args = parser.parse_args()

    root   = os.path.abspath(args.root)
    paper  = _PAPER_BASELINES[args.dataset]
    ds_lbl = _DATASET_LABEL.get(args.dataset, args.dataset)

    print()
    print("=" * 78)
    print(f"  DeiT-LT  --  {ds_lbl}  IF=100  |  metric: AVG = (CLS + DIST) / 2")
    print("=" * 78)
    print_header()

    # Paper baseline
    print_row("Paper DeiT-LT (Table 6)",
              paper["overall"], paper["head"], paper["mid"], paper["tail"],
              note="<-- paper")
    print()

    # Our trained runs in explicit order
    for tag in RUN_ORDER:
        outdir = os.path.join(root, f"deit_out_{tag}_if100")
        label  = RUN_LABELS[tag]

        if not os.path.isdir(outdir):
            print_row(label, None, None, None, None, note="dir not found")
            continue

        status, cur_ep = run_status(outdir, args.total_epochs)
        best_ckpt = find_best_ckpt(outdir)

        if status == "not_started":
            print_row(label, None, None, None, None, note="not started")
            continue

        if best_ckpt is None:
            pct  = 100.0 * (cur_ep + 1) / args.total_epochs
            print_row(label, None, None, None, None,
                      note=f"in progress  ep {cur_ep}/{args.total_epochs-1} ({pct:.0f}%)")
            continue

        try:
            m = load_ckpt_metrics(best_ckpt)
        except Exception as e:
            print_row(label, None, None, None, None, note=f"ERROR: {e}")
            continue

        note = ""
        if status == "in_progress":
            pct  = 100.0 * (cur_ep + 1) / args.total_epochs
            note = f"ep {cur_ep}/{args.total_epochs-1} ({pct:.0f}%)"

        print_row(label, m["overall"], m["head"], m["mid"], m["tail"],
                  ref=paper["overall"], note=note)

        if args.verbose:
            print_sub_row("CLS token",  m["overall_cls"], m["head_cls"],
                          m["mid_cls"],  m["tail_cls"])
            print_sub_row("DIST token", m["overall_dist"], m["head_dist"],
                          m["mid_dist"], m["tail_dist"])

    # T4: find fine-tuned checkpoint and run / load cached eval
    print()
    ft_ckpts = sorted(glob.glob(
        os.path.join(root, "checkpoints", "deit_lt_finetuned_*.pth")))

    if not ft_ckpts:
        print_row("T4: Decoupled Classifier Fine-tuning",
                  None, None, None, None, note="not run yet")
    else:
        ft_path = ft_ckpts[-1]
        try:
            m = get_t4_metrics(ft_path, force_reeval=args.reeval)
            print_row("T4: Decoupled Classifier Fine-tuning",
                      m["overall"], m["head"], m["mid"], m["tail"],
                      ref=paper["overall"])
            if args.verbose:
                print_sub_row("CLS token (retrained)",
                              m["overall_cls"], m["head_cls"],
                              m["mid_cls"],     m["tail_cls"])
                print_sub_row("DIST token (unchanged)",
                              m["overall_dist"], m["head_dist"],
                              m["mid_dist"],     m["tail_dist"])
        except Exception as e:
            print_row("T4: Decoupled Classifier Fine-tuning",
                      None, None, None, None, note=f"ERROR: {e}")

    print()
    print("  D-Ovr = delta vs paper Overall")
    print("  AVG   = average of CLS and DIST logits at inference")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
