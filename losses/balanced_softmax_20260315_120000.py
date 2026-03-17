"""
Technique 2: Balanced Softmax Loss for Long-Tailed Recognition.

Reference: Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition"
           (NeurIPS 2020).

Mathematical equivalence
------------------------
Balanced Softmax is equivalent to Logit Adjustment (Menon et al., ICLR 2021)
with tau=1 when class_counts sums to N (total training samples).  Both methods
add a class-frequency term to the raw logits before cross-entropy:

    Logit Adj (tau=1):  adjusted_i = logit_i + log(n_i / N)
    Balanced Softmax:   adjusted_i = logit_i + log(n_i)

The difference of log(N) is a constant offset shared across all classes and
therefore cancels inside the softmax / cross-entropy, making the two losses
numerically identical when tau=1.

Usage
-----
Pre-compute class_counts ONCE before training:

    cls_num_list  = dataset_train.get_cls_num_list()
    class_counts  = torch.tensor(cls_num_list, dtype=torch.float32)

Then pass class_counts to BalancedSoftmaxCriterion (or balanced_softmax_loss).
Do NOT recompute per batch.
"""

import csv
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Seeding helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed torch, numpy, and random for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Core loss function
# ---------------------------------------------------------------------------

def balanced_softmax_loss(logits: torch.Tensor,
                          targets: torch.Tensor,
                          class_counts: torch.Tensor) -> torch.Tensor:
    """Balanced Softmax cross-entropy loss.

    Mathematically equivalent to Logit Adjustment with tau=1 when
    class_counts sums to N (total training samples).  Adds log(n_y) to the
    logit of each sample's ground-truth class y before cross-entropy, which
    up-weights rare classes implicitly without changing the model output.

    Parameters
    ----------
    logits       : (B, C) raw model output (CLS token).
    targets      : (B,)   integer class labels.
    class_counts : (C,)   per-class sample counts. Must be pre-computed once
                          from dataset_train.get_cls_num_list() and converted
                          via torch.tensor(..., dtype=torch.float32).

    Returns
    -------
    Scalar balanced softmax cross-entropy loss.

    Notes
    -----
    Zero entries in class_counts are clamped to 1 before taking log to avoid
    -inf.  A WARNING is emitted before clamping if any zeros are detected.
    """
    # Warn before clamping so the caller knows the data has empty classes.
    if (class_counts == 0).any():
        zero_classes = (class_counts == 0).nonzero(as_tuple=True)[0].tolist()
        logger.warning(
            "[BALANCED_SOFTMAX] class_counts contains zeros for class indices %s. "
            "Clamping to 1 to avoid log(0). Check your dataset / cls_num_list.",
            zero_classes,
        )

    freq = class_counts.clamp(min=1).to(logits.device)
    adjusted_logits = logits + torch.log(freq)
    return F.cross_entropy(adjusted_logits, targets)


# ---------------------------------------------------------------------------
# Head / mid / tail split helper
# ---------------------------------------------------------------------------

def get_head_mid_tail_indices(class_counts: torch.Tensor):
    """Split class indices into head, mid, and tail thirds by frequency.

    Parameters
    ----------
    class_counts : (C,) tensor of per-class sample counts (descending order
                    assumed, as produced by standard LT dataset builders).

    Returns
    -------
    head_idx, mid_idx, tail_idx : lists of class indices.
    """
    C = len(class_counts)
    third = C // 3
    head_idx = list(range(0, third))
    mid_idx  = list(range(third, 2 * third))
    tail_idx = list(range(2 * third, C))
    return head_idx, mid_idx, tail_idx


def per_split_accuracy(preds: torch.Tensor,
                       targets: torch.Tensor,
                       head_idx, mid_idx, tail_idx):
    """Compute per-split and overall top-1 accuracy.

    Parameters
    ----------
    preds   : (N,) predicted class indices.
    targets : (N,) ground-truth class indices.

    Returns
    -------
    head_acc, mid_acc, tail_acc, overall_acc : floats in [0, 100].
    """
    correct = preds.eq(targets)

    def _acc(indices):
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for i in indices:
            mask |= (targets == i)
        if mask.sum() == 0:
            return 0.0
        return correct[mask].float().mean().item() * 100.0

    head_acc    = _acc(head_idx)
    mid_acc     = _acc(mid_idx)
    tail_acc    = _acc(tail_idx)
    overall_acc = correct.float().mean().item() * 100.0
    return head_acc, mid_acc, tail_acc, overall_acc


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class BalancedSoftmaxCriterion(torch.nn.Module):
    """Drop-in replacement for nn.CrossEntropyLoss using Balanced Softmax.

    Tracks per-epoch average training loss; call ``get_avg_loss()`` after
    each epoch and ``reset_epoch()`` before the next.

    Parameters
    ----------
    class_counts : (C,) pre-computed per-class sample counts (float32 tensor).
    log_interval : emit an INFO log every this many forward calls (default 50).
    """

    def __init__(self,
                 class_counts: torch.Tensor,
                 log_interval: int = 50) -> None:
        super().__init__()
        self.register_buffer("class_counts", class_counts)
        self.log_interval = log_interval
        self._batch_idx   = 0
        self._epoch       = 0
        self._loss_sum    = 0.0
        self._loss_count  = 0

        # Warn at construction time if zeros are present.
        if (class_counts == 0).any():
            logger.warning(
                "[BALANCED_SOFTMAX] class_counts passed to BalancedSoftmaxCriterion "
                "contains zero entries. See balanced_softmax_loss for details."
            )

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch."""
        self._epoch     = epoch
        self._batch_idx = 0

    def reset_epoch(self) -> None:
        """Reset per-epoch loss accumulator. Call after logging each epoch."""
        self._loss_sum   = 0.0
        self._loss_count = 0

    def get_avg_loss(self) -> float:
        """Return the mean training loss accumulated since last reset_epoch()."""
        if self._loss_count == 0:
            return float("nan")
        return self._loss_sum / self._loss_count

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        loss = balanced_softmax_loss(logits, targets, self.class_counts)

        self._loss_sum   += loss.item()
        self._loss_count += 1

        if self._batch_idx % self.log_interval == 0:
            logger.info(
                "[BALANCED_SOFTMAX] epoch=%d batch=%d loss=%.4f",
                self._epoch, self._batch_idx, loss.item(),
            )

        self._batch_idx += 1
        return loss


# ---------------------------------------------------------------------------
# Per-epoch CSV logger
# ---------------------------------------------------------------------------

class BalancedSoftmaxCSVLogger:
    """Writes per-epoch training curves to a CSV file.

    CSV columns: epoch, avg_loss, head_acc, mid_acc, tail_acc, overall_acc

    Parameters
    ----------
    log_dir : directory to write the CSV file into (default "logs").
    """

    def __init__(self, log_dir: str = "logs") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(
            log_dir, f"balanced_softmax_curve_{_TIMESTAMP}.csv"
        )
        with open(self._path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "avg_loss", "head_acc", "mid_acc", "tail_acc", "overall_acc"]
            )
        logger.info("[BALANCED_SOFTMAX] CSV logger initialised at %s", self._path)

    def log_epoch(self,
                  epoch: int,
                  avg_loss: float,
                  head_acc: float,
                  mid_acc: float,
                  tail_acc: float,
                  overall_acc: float) -> None:
        """Append one row to the CSV.  Call once at the end of each epoch."""
        with open(self._path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{head_acc:.2f}",
                f"{mid_acc:.2f}",
                f"{tail_acc:.2f}",
                f"{overall_acc:.2f}",
            ])
        logger.info(
            "[BALANCED_SOFTMAX] epoch=%d avg_loss=%.4f "
            "head=%.2f mid=%.2f tail=%.2f overall=%.2f",
            epoch, avg_loss, head_acc, mid_acc, tail_acc, overall_acc,
        )

    @property
    def path(self) -> str:
        return self._path


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_class_counts(cls_num_list) -> torch.Tensor:
    """Convert dataset_train.get_cls_num_list() output to a float32 tensor.

    Parameters
    ----------
    cls_num_list : list[int]

    Returns
    -------
    class_counts : (C,) float32 tensor on CPU.
    """
    return torch.tensor(cls_num_list, dtype=torch.float32)
