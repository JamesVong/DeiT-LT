#!/usr/bin/env bash
# =============================================================================
# setup.sh — Restore the DeiT-LT Python environment after a RunPod pod restart
#
# Usage (run from any directory):
#   bash /workspace/DeiT-LT/setup.sh
# =============================================================================
set -euo pipefail

echo "=== DeiT-LT environment setup ==="

# ── 1. System packages ────────────────────────────────────────────────────────
which tmux &>/dev/null || apt-get install -y tmux

# ── 2. Claude Code ────────────────────────────────────────────────────────────
which claude &>/dev/null || curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# ── 3. Core ML packages ───────────────────────────────────────────────────────
# Restore torch 2.4.1+cu124 (must be pinned — xformers conflicts otherwise)
pip install --quiet \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 4. Project-specific packages ──────────────────────────────────────────────
# timm 0.8.15.dev0 is required — newer versions moved generate_default_cfgs
# xformers 0.0.27.post2 is compatible with torch 2.4.x (ignore resolver warning)
pip install --quiet \
    "timm==0.8.15.dev0" \
    "xformers==0.0.27.post2" \
    scikit-learn \
    wandb

# ── 5. Verify ─────────────────────────────────────────────────────────────────
echo ""
python -c "
import torch, timm, xformers, sklearn, wandb
import sys; sys.path.insert(0, '/workspace/DeiT-LT')
import os; os.chdir('/workspace/DeiT-LT')
import deit_models
print('torch  :', torch.__version__)
print('timm   :', timm.__version__)
print('xformers:', xformers.__version__)
print('CUDA   :', torch.cuda.is_available())
print('All imports OK')
" 2>&1 | grep -v FutureWarning | grep -v "impl_abstract"

# ── 6. Restore bash history ───────────────────────────────────────────────────
if [ -f /workspace/.bash_history ]; then
    cp /workspace/.bash_history ~/.bash_history
fi
grep -q 'HISTFILE=/workspace' ~/.bashrc 2>/dev/null \
    || printf '\nexport HISTFILE=/workspace/.bash_history\nexport HISTSIZE=10000\n' >> ~/.bashrc

# ── 7. Start tmux session ─────────────────────────────────────────────────────
tmux new-session -d -s main -c /workspace/DeiT-LT 2>/dev/null || true

echo ""
echo "Setup complete. Run:  tmux attach -t main"
