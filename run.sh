#!/usr/bin/env bash
# =============================================================================
# run.sh — DeiT-LT loss-technique comparison
#
# Scope decision
# --------------
# Full schedule (1200 epochs) is required to compare against paper baselines.
#
# Timing analysis (measured on this machine, 1×GPU, 37 s/epoch):
#   CIFAR-10 LT 1200 epochs ≈ 12.4 h/run
#
#   Full matrix  C10+C100, IF=50+IF=100, 4 techniques + CE × 4  ≈ 6.6 days  ✗
#   C10 only,    IF=50+IF=100,           4 techniques + CE × 2  ≈ 4.1 days  ✗
#   C10 only,    IF=100 only,            4 techniques + CE × 1  ≈ 2.1 days  ✓
#
# Chosen: CIFAR-10 LT, IF=100, full schedule (1200 epochs / DRW at 1100).
# IF=100 is the primary benchmark used in the DeiT-LT paper and the most
# challenging setting.
#
# Breakdown: 4 trains (CE baseline + Techniques 1–3) × 12.4 h
#          + 1 fine-tune (Technique 4, 10 epochs on CE checkpoint) ≈ 10 min
#          = ~2.1 days total.
# Adding IF=50 would double the training runs to ~4.1 days, just over budget.
#
# Resume behaviour
# ----------------
# Re-running run.sh is safe.  For each technique, find_resume_ckpt() checks
# for the rolling checkpoint saved every epoch:
#   <output_dir>/<name_exp>/<name_exp>_checkpoint.pth
# If found, --resume is passed automatically and training continues from where
# it left off.  If the run has already completed (best_checkpoint.pth exists
# and the rolling ckpt epoch == total_epochs-1), the run is skipped entirely.
# Technique 4 (fine-tune, ~10 min) is always re-run because it is too short
# to warrant resume logic — it finishes faster than the check would save.
# =============================================================================
set -euo pipefail

# ── 0. Setup ──────────────────────────────────────────────────────────────────
mkdir -p logs checkpoints

SCRIPT="scripts/train_lt_20260315_120000.py"
FINETUNE="scripts/finetune_classifier_20260315_120000.py"
TOTAL_EPOCHS=1200

# ── helper: find the rolling (latest) checkpoint for a given output_dir ───────
# Returns the path of <name_exp>_checkpoint.pth (updated every epoch),
# which is distinct from *_best_checkpoint.pth and *_epoch_*_checkpoint.pth.
find_resume_ckpt() {
    local outdir="$1"
    # The rolling checkpoint has no epoch number in the name and is NOT "best".
    local ckpt
    ckpt=$(find "$outdir" -maxdepth 2 -name "*_checkpoint.pth" \
               ! -name "*_best_checkpoint.pth" \
               ! -name "*_epoch_*_checkpoint.pth" \
               ! -name "*_DRW_checkpoint.pth" \
               2>/dev/null | sort | tail -1)
    echo "$ckpt"
}

# ── helper: check if a run is already complete (reached TOTAL_EPOCHS) ─────────
is_complete() {
    local outdir="$1"
    local ckpt
    ckpt=$(find_resume_ckpt "$outdir")
    if [ -z "$ckpt" ]; then
        return 1   # no checkpoint at all → not complete
    fi
    # Read the saved epoch from the checkpoint dict with Python.
    local saved_epoch
    saved_epoch=$(python -c "
import torch, sys
c = torch.load('$ckpt', map_location='cpu')
print(c.get('epoch', -1))
" 2>/dev/null)
    # Training saves epoch index (0-based); complete when epoch == TOTAL_EPOCHS-1
    if [ "$saved_epoch" -eq $((TOTAL_EPOCHS - 1)) ] 2>/dev/null; then
        return 0   # complete
    fi
    return 1       # not yet complete
}

# ── helper: run one torchrun job with automatic resume ────────────────────────
run_technique() {
    local label="$1"
    local outdir="$2"
    shift 2
    local extra_flags=("$@")

    echo ""
    echo "=== ${label} ==="

    if is_complete "$outdir"; then
        echo "  Already complete (epoch $((TOTAL_EPOCHS-1)) checkpoint found). Skipping."
        return 0
    fi

    local resume_flag=""
    local ckpt
    ckpt=$(find_resume_ckpt "$outdir")
    if [ -n "$ckpt" ]; then
        echo "  Resuming from: $ckpt"
        resume_flag="--resume $ckpt"
    else
        echo "  Starting fresh."
    fi

    torchrun --nproc_per_node=1 $SCRIPT $COMMON \
        $resume_flag \
        "${extra_flags[@]}"
}

# ── 1. Common flags — CIFAR-10 LT IF=100, full schedule ──────────────────────
#       Mirrors sh/train_c10_if100.sh (full schedule) exactly.
COMMON="--model deit_base_distilled_patch16_224 \
    --batch-size 128 \
    --epochs ${TOTAL_EPOCHS} \
    --drw 1100 \
    --gpu 0 \
    --teacher-path paco_sam_ckpt_cf10_if100.pth.tar \
    --teacher-model resnet32 \
    --teacher-size 32 \
    --distillation-type hard \
    --data-path cifar10 \
    --data-set CIFAR10LT \
    --imb_factor 0.01 \
    --student-transform 0 \
    --teacher-transform 0 \
    --no-mixup-drw \
    --custom_model \
    --accum-iter 4 \
    --save_freq 300 \
    --weighted-distillation \
    --moco-t 0.05 --moco-k 1024 --moco-dim 32 --feat_dim 64 --paco"

# ── 2. CE baseline ────────────────────────────────────────────────────────────
run_technique "[1/5] CE baseline (reproduces original DeiT-LT)" \
    deit_out_ce_if100 \
    --loss ce \
    --experiment "[lt_ce_if100]" \
    --output_dir deit_out_ce_if100

# ── 3. Technique 1: Logit Adjustment ─────────────────────────────────────────
run_technique "[2/5] Technique 1: Logit Adjustment (tau=1.0)" \
    deit_out_logit_adj_if100 \
    --loss logit_adj --tau 1.0 \
    --experiment "[lt_logit_adj_if100]" \
    --output_dir deit_out_logit_adj_if100

# ── 4. Technique 2: Balanced Softmax ─────────────────────────────────────────
run_technique "[3/5] Technique 2: Balanced Softmax" \
    deit_out_balsoftmax_if100 \
    --loss balanced_softmax \
    --experiment "[lt_balsoftmax_if100]" \
    --output_dir deit_out_balsoftmax_if100

# ── 5. Technique 3: Class-Aware Label Smoothing ───────────────────────────────
run_technique "[4/5] Technique 3: Class-Aware Label Smoothing (eps-max=0.2)" \
    deit_out_classsmooth_if100 \
    --loss class_smooth --eps-max 0.2 \
    --experiment "[lt_classsmooth_if100]" \
    --output_dir deit_out_classsmooth_if100

# ── 6. Technique 4: Decoupled Classifier Fine-tuning ─────────────────────────
# Fine-tune is ~10 min — always re-run rather than resume.
echo ""
echo "=== [5/5] Technique 4: Decoupled fine-tuning (stage-2 on CE checkpoint) ==="
STAGE1_CKPT=$(ls deit_out_ce_if100/*/*_best_checkpoint.pth 2>/dev/null | tail -1 || true)
if [ -z "$STAGE1_CKPT" ]; then
    echo "  WARNING: No best_checkpoint.pth under deit_out_ce_if100/. Skipping stage-2."
else
    python $FINETUNE \
        --checkpoint "$STAGE1_CKPT" \
        --finetune-epochs 10
fi

# ── 7. Results comparison table ───────────────────────────────────────────────
python scripts/report.py
