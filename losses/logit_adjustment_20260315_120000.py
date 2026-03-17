"""
Technique 1: Logit Adjustment Loss for Long-Tailed Recognition.

Reference: Menon et al., "Long-tail learning via logit adjustment" (ICLR 2021).

Usage
-----
Pre-compute log_prior ONCE before training:

    cls_num_list = dataset_train.get_cls_num_list()
    class_counts = torch.tensor(cls_num_list, dtype=torch.float32)
    priors = class_counts / class_counts.sum()
    log_prior = torch.log(priors + 1e-9)   # shape: (num_classes,)

Then pass log_prior to LogitAdjustmentCriterion (or logit_adjustment_loss directly).
Do NOT recompute log_prior per batch — it is a constant w.r.t. the dataset.
"""

import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level logger (configured by the caller; fallback to root logger)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Timestamp used for all output files produced by this module instance.
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Seeding helper (called from train script; kept here for convenience)
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

def logit_adjustment_loss(logits: torch.Tensor,
                          targets: torch.Tensor,
                          log_prior: torch.Tensor,
                          tau: float = 1.0) -> torch.Tensor:
    """Logit-adjusted cross-entropy loss.

    log_prior must be pre-computed ONCE before training as:
        priors    = class_counts / class_counts.sum()
        log_prior = torch.log(priors + 1e-9)

    Pass the pre-computed log_prior here — do NOT recompute per batch.

    Parameters
    ----------
    logits    : (B, C) raw model output (CLS token).
    targets   : (B,)  integer class labels.
    log_prior : (C,)  pre-computed log class prior, device-agnostic (moved
                       to logits.device inside this function).
    tau       : scaling factor for the prior adjustment (default 1.0).

    Returns
    -------
    Scalar cross-entropy loss after adding tau * log_prior to each logit.
    """
    adjusted_logits = logits + tau * log_prior.to(logits.device)
    return F.cross_entropy(adjusted_logits, targets)


# ---------------------------------------------------------------------------
# nn.Module wrapper (used as base_criterion inside DistillationLoss)
# ---------------------------------------------------------------------------

class LogitAdjustmentCriterion(torch.nn.Module):
    """Drop-in replacement for nn.CrossEntropyLoss with logit adjustment.

    Wraps ``logit_adjustment_loss`` and emits per-batch log lines at a
    configurable interval.

    Parameters
    ----------
    log_prior   : (C,) pre-computed log class prior tensor.
    tau         : prior scaling factor (default 1.0).
    log_interval: emit a log line every this many batches (default 50).
    """

    def __init__(self,
                 log_prior: torch.Tensor,
                 tau: float = 1.0,
                 log_interval: int = 50) -> None:
        super().__init__()
        # Register as a buffer so it moves with .to(device) calls.
        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.log_interval = log_interval
        self._batch_idx = 0
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to update the epoch counter."""
        self._epoch = epoch
        self._batch_idx = 0

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        loss = logit_adjustment_loss(logits, targets, self.log_prior, self.tau)

        if self._batch_idx % self.log_interval == 0:
            logger.info(
                "[LOGIT_ADJ] epoch=%d batch=%d tau=%.2f loss=%.4f",
                self._epoch, self._batch_idx, self.tau, loss.item(),
            )

        self._batch_idx += 1
        return loss


# ---------------------------------------------------------------------------
# JSON summary writer (call once at the end of training)
# ---------------------------------------------------------------------------

def write_summary(seed: int,
                  tau: float,
                  dataset: str,
                  epochs: int,
                  best_val_acc: float,
                  per_class_acc: list,
                  log_dir: str = "logs") -> str:
    """Write a JSON run summary for the logit-adjustment experiment.

    Parameters
    ----------
    seed          : random seed used for the run.
    tau           : logit-adjustment tau value.
    dataset       : dataset name string (e.g. "CIFAR10LT").
    epochs        : total training epochs.
    best_val_acc  : best overall validation accuracy (0-100 float).
    per_class_acc : list of per-class validation accuracies (length C).
    log_dir       : directory to write the JSON file into.

    Returns
    -------
    Absolute path to the written JSON file.
    """
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"logit_adj_{_TIMESTAMP}.json")

    summary = {
        "seed": seed,
        "tau": tau,
        "dataset": dataset,
        "epochs": epochs,
        "best_val_acc": best_val_acc,
        "per_class_acc": per_class_acc,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("[LOGIT_ADJ] Summary written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Convenience factory: build log_prior from a class-count list
# ---------------------------------------------------------------------------

def build_log_prior(cls_num_list) -> torch.Tensor:
    """Convert a Python list of class counts (from dataset_train.get_cls_num_list())
    into the log-prior tensor expected by logit_adjustment_loss.

    Parameters
    ----------
    cls_num_list : list[int]  — output of dataset_train.get_cls_num_list().

    Returns
    -------
    log_prior : (C,) float32 tensor on CPU; move to GPU inside the loss fn.
    """
    class_counts = torch.tensor(cls_num_list, dtype=torch.float32)
    priors = class_counts / class_counts.sum()
    log_prior = torch.log(priors + 1e-9)
    return log_prior
