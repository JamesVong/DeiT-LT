"""
Technique 3: Class-Aware Label Smoothing for Long-Tailed Recognition.

Applies per-class label smoothing to the CLS token cross-entropy loss ONLY.
The DIST token distillation loss is left unchanged (hard/soft targets from
the teacher are not smoothed).

Smoothing direction
-------------------
The original formula in the skeleton is:

    eps_max * (count / max_count)          [original]

This gives HEAD classes (large count) MORE smoothing and TAIL classes LESS.
That is the OPPOSITE of what long-tail regularisation typically intends:
tail classes have few samples and are prone to over-confident memorisation,
so they should receive MORE smoothing.

This implementation therefore uses the INVERTED formula:

    eps_max * (1 - count / max_count)      [inverted — DEFAULT]

so that:
  - tail class (count ≈ 0)    → epsilon ≈ eps_max  (heavy smoothing)
  - head class (count = N_max) → epsilon = 0        (hard labels)

To revert to the original (head-smoothing) behaviour pass
``invert=False`` to ``get_smoothing_vector``.

Usage
-----
Pre-compute smoothing_vector ONCE before training:

    cls_num_list     = dataset_train.get_cls_num_list()
    class_counts     = torch.tensor(cls_num_list, dtype=torch.float32)
    smoothing_vector = get_smoothing_vector(class_counts, eps_max=0.2)

Then pass smoothing_vector to ClassAwareSmoothingCriterion (or
class_aware_smoothing_loss directly).  Do NOT recompute per batch.

The epsilon vector is written to logs/eps_vector_<timestamp>.json at
module import time when ``save_eps_vector`` is called.

CLI arg: ``--eps-max`` (float, default 0.2).
         Do NOT conflict with the existing ``--smoothing`` argument in
         arguments.py (which controls uniform label smoothing for DeiT).
"""

import json
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
# Smoothing vector (pre-compute once)
# ---------------------------------------------------------------------------

def get_smoothing_vector(class_counts: torch.Tensor,
                         eps_max: float = 0.2,
                         invert: bool = True) -> torch.Tensor:
    """Pre-compute per-class smoothing epsilons.  Call once before training.

    Parameters
    ----------
    class_counts : (C,) per-class sample counts (float32 tensor).
    eps_max      : maximum smoothing value, applied to the class with the
                   fewest samples when invert=True (default 0.2).
    invert       : if True (default), use the INVERTED formula so that tail
                   classes receive MORE smoothing:

                       epsilon_c = eps_max * (1 - count_c / max_count)

                   If False, use the original skeleton formula (head classes
                   receive more smoothing):

                       epsilon_c = eps_max * (count_c / max_count)

    Returns
    -------
    smoothing_vector : (C,) float32 tensor of per-class epsilon values.

    Direction summary
    -----------------
    invert=True  (default):
        tail → epsilon ≈ eps_max, head → epsilon ≈ 0
        Rationale: regularise rare classes that may overfit their few samples.
    invert=False (original skeleton):
        head → epsilon ≈ eps_max, tail → epsilon ≈ 0
        Rationale: apply more smoothing where the model is most confident.
    """
    max_count = class_counts.max()
    if invert:
        # Tail classes (small count) get HIGHER smoothing — recommended for LT.
        sv = eps_max * (1.0 - class_counts / max_count)
    else:
        # Head classes (large count) get HIGHER smoothing — original formula.
        sv = eps_max * (class_counts / max_count)
    return sv.float()


# ---------------------------------------------------------------------------
# Core loss function
# ---------------------------------------------------------------------------

def class_aware_smoothing_loss(logits: torch.Tensor,
                               targets: torch.Tensor,
                               smoothing_vector: torch.Tensor) -> torch.Tensor:
    """Class-aware label-smoothed cross-entropy.

    IMPORTANT: Apply to CLS token logits ONLY.
    Do NOT apply smoothing to distillation targets — the DIST token DRW loss
    is left unchanged.

    Parameters
    ----------
    logits           : (B, C) raw CLS token logits.
    targets          : (B,)   integer class labels.
    smoothing_vector : (C,)   pre-computed per-class epsilon values from
                              ``get_smoothing_vector``.  Must be pre-computed
                              once before training — do NOT pass raw
                              class_counts here.

    Returns
    -------
    Scalar class-aware label-smoothed cross-entropy loss.
    """
    epsilon     = smoothing_vector[targets.cpu()].to(logits.device)   # (B,)
    log_probs   = F.log_softmax(logits, dim=-1)                 # (B, C)
    C           = logits.size(-1)

    # Build soft targets: (1 - eps) mass on ground truth, eps/C spread uniformly.
    targets_one_hot = torch.zeros_like(logits).scatter_(
        1, targets.unsqueeze(1), 1.0
    )                                                            # (B, C)
    targets_soft = (
        targets_one_hot * (1.0 - epsilon.unsqueeze(1))
        + epsilon.unsqueeze(1) / C
    )                                                            # (B, C)

    return torch.mean(torch.sum(-targets_soft * log_probs, dim=-1))


# ---------------------------------------------------------------------------
# Epsilon vector persistence
# ---------------------------------------------------------------------------

def save_eps_vector(smoothing_vector: torch.Tensor,
                    eps_max: float,
                    invert: bool,
                    cls_num_list,
                    log_dir: str = "logs") -> str:
    """Write the epsilon vector to logs/eps_vector_<timestamp>.json.

    Call once after pre-computing smoothing_vector, before training starts.

    Parameters
    ----------
    smoothing_vector : (C,) tensor produced by get_smoothing_vector.
    eps_max          : the eps_max used.
    invert           : the invert flag used.
    cls_num_list     : original Python list from dataset_train.get_cls_num_list().
    log_dir          : output directory (default "logs").

    Returns
    -------
    Absolute path to the written JSON file.
    """
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"eps_vector_{_TIMESTAMP}.json")

    payload = {
        "timestamp":        _TIMESTAMP,
        "eps_max":          eps_max,
        "invert":           invert,
        "formula":          ("eps_max * (1 - count/max_count)"
                             if invert else "eps_max * (count/max_count)"),
        "num_classes":      len(smoothing_vector),
        "cls_num_list":     cls_num_list if isinstance(cls_num_list, list)
                            else cls_num_list.tolist(),
        "smoothing_vector": smoothing_vector.tolist(),
        "mean_eps":         smoothing_vector.mean().item(),
        "min_eps":          smoothing_vector.min().item(),
        "max_eps":          smoothing_vector.max().item(),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("[CLASS_SMOOTH] Epsilon vector written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class ClassAwareSmoothingCriterion(torch.nn.Module):
    """Drop-in replacement for nn.CrossEntropyLoss with class-aware smoothing.

    Applies smoothing to the CLS token cross-entropy loss ONLY.
    Emits per-batch log lines at a configurable interval.

    Parameters
    ----------
    smoothing_vector : (C,) pre-computed epsilon values from
                       ``get_smoothing_vector``.
    log_interval     : log every this many forward calls (default 50).
    """

    def __init__(self,
                 smoothing_vector: torch.Tensor,
                 log_interval: int = 50) -> None:
        super().__init__()
        self.register_buffer("smoothing_vector", smoothing_vector)
        self.log_interval = log_interval
        self._batch_idx   = 0
        self._epoch       = 0

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch."""
        self._epoch     = epoch
        self._batch_idx = 0

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        loss = class_aware_smoothing_loss(logits, targets, self.smoothing_vector)

        if self._batch_idx % self.log_interval == 0:
            # mean_eps over the current batch's target classes
            mean_eps = self.smoothing_vector[targets.cpu()].mean().item()
            logger.info(
                "[CLASS_SMOOTH] epoch=%d batch=%d mean_eps=%.4f loss=%.4f",
                self._epoch, self._batch_idx, mean_eps, loss.item(),
            )

        self._batch_idx += 1
        return loss


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_smoothing_vector(cls_num_list,
                           eps_max: float = 0.2,
                           invert: bool = True,
                           log_dir: str = "logs") -> torch.Tensor:
    """One-call helper: tensor conversion + get_smoothing_vector + save.

    Parameters
    ----------
    cls_num_list : list[int] from dataset_train.get_cls_num_list().
    eps_max      : maximum smoothing epsilon (default 0.2, CLI: --eps-max).
    invert       : if True (default), tail classes get more smoothing.
    log_dir      : directory for eps_vector JSON (default "logs").

    Returns
    -------
    smoothing_vector : (C,) float32 tensor on CPU.
    """
    class_counts     = torch.tensor(cls_num_list, dtype=torch.float32)
    smoothing_vector = get_smoothing_vector(class_counts, eps_max=eps_max,
                                            invert=invert)
    save_eps_vector(smoothing_vector, eps_max, invert, cls_num_list, log_dir)
    return smoothing_vector
