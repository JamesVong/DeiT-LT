#!/usr/bin/env bash
# =============================================================================
# run.sh — DeiT-LT loss-technique comparison
#
# Scope decision
# --------------
# Full schedule (1200 epochs) is required to compare against paper baselines.
#
# Timing analysis (measured on this machine, 1×GPU, 37 s/epoch):
#   CIFAR-100 LT 1200 epochs ≈ 12.4 h/run
#
#   Full matrix  C10+C100, IF=50+IF=100, 4 techniques + CE × 4  ≈ 6.6 days  ✗
#   C100 only,   IF=50+IF=100,           4 techniques + CE × 2  ≈ 4.1 days  ✗
#   C100 only,   IF=100 only,            4 techniques + CE × 1  ≈ 2.1 days  ✓
#
# Chosen: CIFAR-100 LT, IF=100, full schedule (1200 epochs / DRW at 1100).
# IF=100 is the primary benchmark used in the DeiT-LT paper and the most
# challenging setting.
#
# Breakdown: 3 trains (Techniques 1–3) × 12.4 h
#          + 1 fine-tune (Technique 4, 10 epochs on downloaded CE checkpoint) ≈ 10 min
#          = ~1.5 days total.
# CE baseline is skipped — a pre-trained checkpoint is downloaded instead.
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

# ── Download teacher model (CIFAR-100 LT IF=100, PaCo+SAM) ───────────────────
TEACHER_CKPT="cifar100_paco_sam_if100.pth.tar"
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Downloading teacher model → ${TEACHER_CKPT} ..."
    wget -q --show-progress -O "$TEACHER_CKPT" \
        "https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/fc4a80f43013c11729e28426b64ef70b/cifar100_paco_sam_if100.pth.tar"
    echo "  Teacher model downloaded."
else
    echo "  Teacher model already present: ${TEACHER_CKPT}"
fi

# ── Download pre-trained student CE checkpoint (skips CE training run) ────────
# Matches name_exp produced by: model_teacher_epochs_CIFAR100LT_imb100_bs_[exp]
_STUDENT_SUBDIR="deit_out_ce_if100/deit_base_distilled_patch16_224_resnet32_1200_CIFAR100LT_imb100_128_[paco_sam_teacher]"
_STUDENT_CKPT="${_STUDENT_SUBDIR}/deit_base_distilled_patch16_224_resnet32_1200_CIFAR100LT_imb100_128_[paco_sam_teacher]_best_checkpoint.pth"
if [ ! -f "$_STUDENT_CKPT" ]; then
    echo "Downloading student CE checkpoint → ${_STUDENT_CKPT} ..."
    mkdir -p "$_STUDENT_SUBDIR"
    wget -q --show-progress -O "$_STUDENT_CKPT" \
        "https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/cf0bdaca962d76f5d79be436478cfd5d/deit_base_distilled_patch16_224_resnet32_1200_CIFAR100LT_imb100_128_%5Bpaco_sam_teacher%5D_best_checkpoint.pth"
    echo "  Student CE checkpoint downloaded."
else
    echo "  Student CE checkpoint already present."
fi

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

# ── 1. Common flags — CIFAR-100 LT IF=100, full schedule ─────────────────────
COMMON="--model deit_base_distilled_patch16_224 \
    --batch-size 128 \
    --epochs ${TOTAL_EPOCHS} \
    --drw 1100 \
    --gpu 0 \
    --teacher-path ${TEACHER_CKPT} \
    --teacher-model resnet32 \
    --teacher-size 32 \
    --distillation-type hard \
    --data-path cifar100 \
    --data-set CIFAR100LT \
    --imb_factor 0.01 \
    --student-transform 0 \
    --teacher-transform 0 \
    --no-mixup-drw \
    --custom_model \
    --accum-iter 4 \
    --save_freq 300 \
    --weighted-distillation \
    --moco-t 0.05 --moco-k 1024 --moco-dim 32 --feat_dim 64 --paco"

# ── 2. CE baseline — SKIPPED (pre-trained checkpoint downloaded above) ────────
echo ""
echo "=== [1/5] CE baseline — SKIPPED (using downloaded checkpoint) ==="
echo "  Checkpoint: ${_STUDENT_CKPT}"

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
STAGE1_CKPT=$(find deit_out_ce_if100 -maxdepth 2 -name "*_best_checkpoint.pth" 2>/dev/null | sort | tail -1 || true)
if [ -z "$STAGE1_CKPT" ]; then
    echo "  WARNING: No best_checkpoint.pth under deit_out_ce_if100/. Skipping stage-2."
else
    python $FINETUNE \
        --checkpoint "$STAGE1_CKPT" \
        --finetune-epochs 10
fi

# ── 7. Results comparison table ───────────────────────────────────────────────
echo ""
echo "############################################################"
echo "#  RESULTS  —  CIFAR-100 LT  IF=100  (1200 epochs)"
echo "############################################################"
python - <<'PYEOF'
import glob, json, os
import torch

PAPER_BASELINE = {
    "DeiT-LT (paper, CE, CIFAR-100 IF=100)": {"head": 74.0, "mid": 43.5, "tail": 25.2, "overall": 53.8},
}

rows = []

# Read best_checkpoint.pth files for all our trained runs
for d in sorted(glob.glob("deit_out_*_if100")):
    ckpts = sorted(glob.glob(f"{d}/**/*_best_checkpoint.pth", recursive=True))
    if not ckpts:
        continue
    tag = os.path.basename(d).replace("deit_out_", "").replace("_if100", "")
    try:
        c = torch.load(ckpts[-1], map_location="cpu")
        rows.append({
            "tag":     tag,
            "head":    round(c.get("head_acc_avg",  c.get("head_acc_cls",  0.0)), 2),
            "mid":     round(c.get("med_acc_avg",   c.get("med_acc_cls",   0.0)), 2),
            "tail":    round(c.get("tail_acc_avg",  c.get("tail_acc_cls",  0.0)), 2),
            "overall": round(c.get("best_acc_avg",  c.get("best_acc_cls",  0.0)), 2),
        })
    except Exception as e:
        print(f"  [WARN] {ckpts[-1]}: {e}")


# Finetune summaries
for f in sorted(glob.glob("logs/finetune_summary_*.json")):
    try:
        d = json.load(open(f))
        rows.append({
            "tag":     "finetune (T4)",
            "head":    d.get("stage2_head_acc",    "-"),
            "mid":     d.get("stage2_mid_acc",     "-"),
            "tail":    d.get("stage2_tail_acc",    "-"),
            "overall": d.get("stage2_overall_acc", "-"),
        })
    except Exception:
        pass

print(f"\n{'Run':<42} {'Head':>7} {'Mid':>7} {'Tail':>7} {'Overall':>9}")
print("-" * 76)
for name, acc in PAPER_BASELINE.items():
    print(f"  {name:<40} {acc['head']:>7.1f} {acc['mid']:>7.1f} "
          f"{acc['tail']:>7.1f} {acc['overall']:>9.1f}  ← paper")
print()
if not rows:
    print("  (no completed checkpoints yet — run training first)")
else:
    for r in rows:
        h = f"{r['head']:.2f}" if isinstance(r['head'], float) else str(r['head'])
        m = f"{r['mid']:.2f}"  if isinstance(r['mid'],  float) else str(r['mid'])
        t = f"{r['tail']:.2f}" if isinstance(r['tail'], float) else str(r['tail'])
        o = f"{r['overall']:.2f}" if isinstance(r['overall'], float) else str(r['overall'])
        print(f"  {r['tag']:<40} {h:>7} {m:>7} {t:>7} {o:>9}")
print()
PYEOF
