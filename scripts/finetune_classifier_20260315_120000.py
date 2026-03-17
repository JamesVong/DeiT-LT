"""
Technique 4: Two-Stage Decoupled Training — Stage-2 Classifier Fine-tuning.

Stage 1 : Run main.py (or scripts/train_lt_*.py) for full DeiT-LT training.
Stage 2 : This script.  Loads the stage-1 checkpoint, freezes the entire
          backbone, and fine-tunes only the two classifier heads
          (model.head, model.head_dist) with a class-balanced
          WeightedRandomSampler.

Usage
-----
python scripts/finetune_classifier_20260315_120000.py \\
    --checkpoint deit_out_ce/<run>_best_checkpoint.pth \\
    --finetune-epochs 10

The script is deliberately NOT run under torchrun — it is a single-GPU
stage-2 job and does not need DDP.  The backbone was already trained with
DDP in stage 1; the checkpoint stores the unwrapped model.state_dict().
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── repo imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deit_models
from datasets import build_dataset

# ── timestamp for all output files produced by this run ─────────────────────
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── logging setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

_log_file = os.path.join("logs", f"finetune_{_TS}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Arg parser
# ─────────────────────────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage-2 classifier fine-tuning for DeiT-LT (Technique 4)"
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to stage-1 DeiT-LT .pth checkpoint file.",
    )
    p.add_argument(
        "--finetune-epochs", type=int, default=10,
        help="Number of fine-tuning epochs (default: 10).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for fine-tuning dataloader (default: 128).",
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4).",
    )
    p.add_argument(
        "--lr", type=float, default=0.01,
        help="SGD learning rate (default: 0.01).",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Model reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def build_model_from_ckpt_args(ckpt_args) -> torch.nn.Module:
    """Rebuild the DeiT model using the namespace stored in ckpt['args']."""
    model = deit_models.__dict__[ckpt_args.model](
        pretrained=False,
        num_classes=ckpt_args.nb_classes,
        drop_rate=getattr(ckpt_args, "drop", 0.0),
        drop_path_rate=getattr(ckpt_args, "drop_path", 0.1),
        mask_attn=getattr(ckpt_args, "mask_attn", False),
        early_stopping=getattr(ckpt_args, "early_stopping", False),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ─────────────────────────────────────────────────────────────────────────────

_FREEZE_SUBSTRINGS = ("patch_embed", "blocks", "norm")


def freeze_backbone(model_without_ddp: torch.nn.Module) -> None:
    """Freeze all parameters except model.head and model.head_dist.

    Strategy: freeze everything first, then re-enable the two classifier heads.
    This ensures cls_token, pos_embed, dist_token, and all backbone layers
    (patch_embed, blocks, norm) are frozen — matching the repo's layer naming
    convention where _FREEZE_SUBSTRINGS covers the main blocks but not the
    learnable positional / class tokens.
    """
    # Step 1: freeze everything.
    for param in model_without_ddp.parameters():
        param.requires_grad_(False)

    # Step 2: unfreeze head and head_dist only.
    for param in model_without_ddp.head.parameters():
        param.requires_grad_(True)
    for param in model_without_ddp.head_dist.parameters():
        param.requires_grad_(True)

    frozen    = sum(p.numel() for p in model_without_ddp.parameters()
                    if not p.requires_grad)
    trainable = sum(p.numel() for p in model_without_ddp.parameters()
                    if p.requires_grad)
    logger.info(
        "[FINETUNE] Froze %d params (backbone). Trainable: %d params (heads).",
        frozen, trainable,
    )


def get_trainable_params(model_without_ddp: torch.nn.Module):
    """Return only the parameters of model.head and model.head_dist."""
    head_params      = list(model_without_ddp.head.parameters())
    head_dist_params = list(model_without_ddp.head_dist.parameters())
    return head_params + head_dist_params


# ─────────────────────────────────────────────────────────────────────────────
# Balanced WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────────

def build_balanced_sampler(dataset, cls_num_list):
    """Build a WeightedRandomSampler that draws N_max samples per class per epoch.

    Each class has equal expected representation regardless of its size in
    the original long-tailed distribution.

    Parameters
    ----------
    dataset      : training dataset (must have .targets attribute).
    cls_num_list : list[int] from dataset.get_cls_num_list().

    Returns
    -------
    sampler      : WeightedRandomSampler
    N_max        : int, max class count (samples per class per epoch).
    """
    N_max    = max(cls_num_list)
    C        = len(cls_num_list)
    # weight[i] = N_max / count[class_of_i]  →  each class draws N_max in expectation
    weights  = [N_max / cls_num_list[t] for t in dataset.targets]
    sampler  = WeightedRandomSampler(
        weights     = weights,
        num_samples = N_max * C,     # exactly N_max * C draws per epoch
        replacement = True,
    )
    logger.info(
        "[FINETUNE] WeightedRandomSampler: N_max=%d, classes=%d, "
        "total samples/epoch=%d.",
        N_max, C, N_max * C,
    )
    return sampler, N_max


# ─────────────────────────────────────────────────────────────────────────────
# Batch distribution monitor (first 5 batches)
# ─────────────────────────────────────────────────────────────────────────────

def check_batch_distribution(targets_cpu: torch.Tensor,
                              num_classes: int,
                              batch_idx: int) -> None:
    """Log class distribution for the first 5 batches; warn on >5% deviation."""
    if batch_idx >= 5:
        return

    counts   = torch.bincount(targets_cpu, minlength=num_classes).float()
    freq     = counts / counts.sum()
    expected = 1.0 / num_classes

    logger.info(
        "[FINETUNE] Batch %d class distribution: %s",
        batch_idx,
        [f"{f:.3f}" for f in freq.tolist()],
    )

    deviations = (freq - expected).abs()
    if (deviations > 0.05).any():
        bad = deviations.argmax().item()
        logger.warning(
            "[FINETUNE] Batch %d: class %d deviates %.1f%% from expected %.1f%%.",
            batch_idx, bad,
            deviations[bad].item() * 100,
            expected * 100,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-class accuracy evaluation (CLS head only)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cls_head(model_without_ddp: torch.nn.Module,
                      data_loader,
                      device: torch.device,
                      categories,
                      num_classes: int):
    """Evaluate the CLS head (model.head) and return head/mid/tail/overall accuracy.

    Parameters
    ----------
    categories : [head_end, tail_start] indices, e.g. [3, 7] for CIFAR-10 LT.
                 head = classes [0, head_end), mid = [head_end, tail_start),
                 tail = [tail_start, C).

    Returns
    -------
    head_acc, mid_acc, tail_acc, overall_acc : floats in [0, 100].
    per_class_acc                            : list of per-class accuracies.
    """
    model_without_ddp.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            # Forward through the full model to get CLS token output.
            out = model_without_ddp(samples)
            # out may be a tuple (cls_logits, dist_logits) for distilled models.
            cls_logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = cls_logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(targets.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    correct     = all_preds.eq(all_targets)

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = all_targets == c
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(correct[mask].float().mean().item() * 100.0)

    def _split_acc(indices):
        mask = torch.zeros(len(all_targets), dtype=torch.bool)
        for i in indices:
            mask |= (all_targets == i)
        if mask.sum() == 0:
            return 0.0
        return correct[mask].float().mean().item() * 100.0

    head_end    = categories[0]
    tail_start  = categories[1]
    head_acc    = _split_acc(range(0,          head_end))
    mid_acc     = _split_acc(range(head_end,   tail_start))
    tail_acc    = _split_acc(range(tail_start, num_classes))
    overall_acc = correct.float().mean().item() * 100.0

    model_without_ddp.train()
    return head_acc, mid_acc, tail_acc, overall_acc, per_class_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    parser = get_parser()
    args   = parser.parse_args()

    set_seed(args.seed)
    logger.info("[FINETUNE] Args: %s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[FINETUNE] Using device: %s", device)

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    logger.info("[FINETUNE] Loading stage-1 checkpoint: %s", args.checkpoint)
    ckpt      = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]  # original training namespace
    logger.info("[FINETUNE] Stage-1 args: %s", ckpt_args)

    # Ensure categories is present (older checkpoints may lack it).
    if not hasattr(ckpt_args, "categories") or not ckpt_args.categories:
        _cats = {
            "CIFAR10LT":  [3, 7],
            "CIFAR100LT": [36, 71],
            "IMAGENETLT": [390, 835],
            "INAT18":     [842, 4543],
        }
        ckpt_args.categories = _cats.get(ckpt_args.data_set, [])
        logger.info(
            "[FINETUNE] categories not in checkpoint; using default %s for %s.",
            ckpt_args.categories, ckpt_args.data_set,
        )

    # ── 2. Rebuild dataset ────────────────────────────────────────────────────
    logger.info("[FINETUNE] Building dataset: %s", ckpt_args.data_set)
    if ckpt_args.data_set in ("INAT18", "IMAGENETLT"):
        dataset_train, _ = build_dataset(is_train=True,  args=ckpt_args)
        dataset_val,   _ = build_dataset(
            is_train=False, args=ckpt_args,
            class_map=dataset_train.class_map,
        )
    else:
        dataset_train, _ = build_dataset(is_train=True,  args=ckpt_args)
        dataset_val,   _ = build_dataset(is_train=False, args=ckpt_args)

    cls_num_list = dataset_train.get_cls_num_list()
    num_classes  = ckpt_args.nb_classes
    logger.info("[FINETUNE] num_classes=%d, cls_num_list=%s", num_classes, cls_num_list)

    # Val loader (standard sequential).
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── 3. Rebuild model, load weights ───────────────────────────────────────
    logger.info("[FINETUNE] Rebuilding model architecture: %s", ckpt_args.model)
    model = build_model_from_ckpt_args(ckpt_args)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # DDP unwrap (not actually wrapped here, but kept for correctness contract).
    model_without_ddp = model.module if hasattr(model, "module") else model

    # ── 4. Evaluate stage-1 baseline ─────────────────────────────────────────
    logger.info("[FINETUNE] Evaluating stage-1 baseline …")
    s1_head, s1_mid, s1_tail, s1_overall, _ = evaluate_cls_head(
        model_without_ddp, val_loader, device,
        ckpt_args.categories, num_classes,
    )
    logger.info(
        "[FINETUNE] Stage-1 | head=%.2f  mid=%.2f  tail=%.2f  overall=%.2f",
        s1_head, s1_mid, s1_tail, s1_overall,
    )

    # ── 5. Freeze backbone, keep only heads trainable ────────────────────────
    freeze_backbone(model_without_ddp)

    # Sanity-check: only head and head_dist are trainable.
    for name, param in model_without_ddp.named_parameters():
        if param.requires_grad:
            logger.info("[FINETUNE] Trainable: %s  shape=%s", name, list(param.shape))

    # ── 6. Build balanced train loader ───────────────────────────────────────
    sampler, N_max = build_balanced_sampler(dataset_train, cls_num_list)
    train_loader   = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── 7. Optimizer (SGD, heads only) ───────────────────────────────────────
    trainable_params = get_trainable_params(model_without_ddp)
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # ── 8. Fine-tuning loop ───────────────────────────────────────────────────
    best_overall = 0.0
    best_head    = 0.0
    best_tail    = 0.0
    best_mid     = 0.0

    for epoch in range(1, args.finetune_epochs + 1):
        model_without_ddp.train()
        running_loss = 0.0

        for batch_idx, (samples, targets) in enumerate(train_loader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Log + check first 5 batch class distributions.
            check_batch_distribution(targets.cpu(), num_classes, batch_idx)

            out        = model_without_ddp(samples)
            cls_logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = criterion(cls_logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        # Per-epoch eval.
        h, m, t, ov, _ = evaluate_cls_head(
            model_without_ddp, val_loader, device,
            ckpt_args.categories, num_classes,
        )

        logger.info(
            "[FINETUNE] epoch=%d/%d  head_acc=%.2f  tail_acc=%.2f  "
            "overall=%.2f  avg_loss=%.4f  lr=%s",
            epoch, args.finetune_epochs, h, t, ov, avg_loss, current_lr,
        )

        if ov > best_overall:
            best_overall = ov
            best_head    = h
            best_mid     = m
            best_tail    = t

    # ── 9. Save stage-2 checkpoint ───────────────────────────────────────────
    out_ckpt_path = os.path.join(
        "checkpoints", f"deit_lt_finetuned_{_TS}.pth"
    )
    torch.save(
        {
            "model": model_without_ddp.state_dict(),
            "epoch": args.finetune_epochs,
            "args":  args,
            "stage1_checkpoint": args.checkpoint,
            "head_acc":    best_head,
            "mid_acc":     best_mid,
            "tail_acc":    best_tail,
            "overall_acc": best_overall,
        },
        out_ckpt_path,
    )
    logger.info("[FINETUNE] Stage-2 checkpoint saved to %s", out_ckpt_path)

    # ── 10. Write JSON summary ────────────────────────────────────────────────
    head_delta    = best_head    - s1_head
    mid_delta     = best_mid     - s1_mid
    tail_delta    = best_tail    - s1_tail
    overall_delta = best_overall - s1_overall

    summary = {
        "stage1_checkpoint":  args.checkpoint,
        "stage2_checkpoint":  out_ckpt_path,
        "epochs":             args.finetune_epochs,
        "head_delta":         round(head_delta,    4),
        "mid_delta":          round(mid_delta,     4),
        "tail_delta":         round(tail_delta,    4),
        "overall_delta":      round(overall_delta, 4),
        "stage1_head_acc":    round(s1_head,    2),
        "stage1_mid_acc":     round(s1_mid,     2),
        "stage1_tail_acc":    round(s1_tail,    2),
        "stage1_overall_acc": round(s1_overall, 2),
        "stage2_head_acc":    round(best_head,    2),
        "stage2_mid_acc":     round(best_mid,     2),
        "stage2_tail_acc":    round(best_tail,    2),
        "stage2_overall_acc": round(best_overall, 2),
        "overall_acc":        round(best_overall, 2),  # for run.sh comparison table
        "head_acc":           round(best_head,    2),
        "mid_acc":            round(best_mid,     2),
        "tail_acc":           round(best_tail,    2),
    }

    summary_path = os.path.join("logs", f"finetune_summary_{_TS}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[FINETUNE] Summary written to %s", summary_path)

    # Emit ERROR log if any delta is negative.
    for key, delta in [("head", head_delta), ("mid",  mid_delta),
                       ("tail", tail_delta), ("overall", overall_delta)]:
        if delta < 0:
            logger.error(
                "[FINETUNE] Negative delta for %s: %.4f  "
                "(stage2=%.2f < stage1=%.2f).  "
                "Fine-tuning degraded this split.",
                key, delta,
                summary[f"stage2_{key}_acc"],
                summary[f"stage1_{key}_acc"],
            )

    logger.info(
        "[FINETUNE] Done.  Δhead=%.2f  Δmid=%.2f  Δtail=%.2f  Δoverall=%.2f",
        head_delta, mid_delta, tail_delta, overall_delta,
    )


if __name__ == "__main__":
    main()
