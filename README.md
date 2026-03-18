# DeiT-LT: Distillation strikes back for Vision Transformer training on Long-Tailed datasets

[![Static Badge](https://img.shields.io/badge/CVPR_2024-DeiT_LT-blue)](https://openaccess.thecvf.com/content/CVPR2024/papers/Rangwani_DeiT-LT_Distillation_Strikes_Back_for_Vision_Transformer_Training_on_Long-Tailed_CVPR_2024_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2404.02900-b31b1b.svg)](https://arxiv.org/abs/2404.02900)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deit-lt-distillation-strikes-back-for-vision/long-tail-learning-on-cifar-10-lt-r-50)](https://paperswithcode.com/sota/long-tail-learning-on-cifar-10-lt-r-50?p=deit-lt-distillation-strikes-back-for-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deit-lt-distillation-strikes-back-for-vision/long-tail-learning-on-cifar-10-lt-r-100)](https://paperswithcode.com/sota/long-tail-learning-on-cifar-10-lt-r-100?p=deit-lt-distillation-strikes-back-for-vision)

This repository contains the training, evaluation codes and checkpoints for the paper DeiT-LT: Distillation strikes back for Vision Transformer training on Long-Tailed datasets accepted at CVPR 2024. <br>


![DeiT-LT Teaser](static/DeiT-LT-Teaser.jpeg)
<!-- 
## Effect of distillation in DeiT-LT
![DeiT-LT Analysis](static/analysis.png)

A. We train DeiT-B with teachers trained on in-distribution images (RegNetY-16GF) and
out-of-distribution images (ResNet32). The out-of-distribution distillation leads to diverse experts, which become more diverse with deferred
re-weighting on the distillation token (DRW).

B. We plot the Mean Attention Distance for the patches across the early self attention block
1 (solid) and block 2 (dashed) for baselines, where we find that DeiT-LT leads to highly local and generalizable features.

C. We show the
rank of features for DIST token, where we demonstrate that students trained with SAM are more low-rank in comparison to baselines -->



## Extensions: Loss Technique Comparison (this fork)

This repository extends the original DeiT-LT with four additional long-tail loss techniques and a unified training pipeline for comparison against the CE baseline.

### Added techniques

| # | Technique | Key idea | CLI flag |
|---|-----------|----------|----------|
| 1 | **Logit Adjustment** | Shift logits by `τ · log(prior)` to correct for class-frequency bias | `--loss logit_adj --tau 1.0` |
| 2 | **Balanced Softmax** | Additive log-frequency offset; mathematically equivalent to LogitAdj at τ=1 | `--loss balanced_softmax` |
| 3 | **Class-Aware Label Smoothing** | Per-class ε inversely proportional to class count (tail → more smoothing) | `--loss class_smooth --eps-max 0.2` |
| 4 | **Decoupled Classifier Fine-tuning** | Freeze backbone after stage-1 CE training; re-train head with balanced sampler | `scripts/finetune_classifier_*.py` |

### Experimental results — CIFAR-10 LT, IF=100 (1200 epochs, DRW at epoch 1100)

All metrics are **AVG = (CLS token + DIST token) / 2**, consistent with Table 6 of the DeiT-LT paper.
Head / Mid / Tail splits: Head = 3 majority classes, Mid = 4 middle classes, Tail = 3 minority classes (CIFAR-10 LT ρ=100).

| Run | Overall | Head | Mid | Tail | Δ Overall |
|-----|--------:|-----:|----:|-----:|----------:|
| Paper DeiT-LT (Table 6) | 87.50 | 94.50 | 84.10 | 85.00 | — |
| CE baseline (reproduced) | 87.69 | 94.97 | 83.85 | 85.53 | +0.19 |
| T1: Logit Adjustment (τ=1.0) | 87.89 | 92.47 | 84.10 | 88.37 | +0.39 |
| T2: Balanced Softmax | **88.06** | 91.90 | 84.60 | **88.83** | **+0.56** |
| T3: Class-Aware Label Smoothing | 87.66 | 94.20 | 83.73 | 86.37 | +0.16 |
| T4: Decoupled Classifier Fine-tuning | 87.70 | 92.00 | 83.85 | 88.53 | +0.20 |

Key observations:
- **T2 (Balanced Softmax)** achieves the best overall (+0.56) and tail (+3.83) accuracy.
- **T1 and T2** consistently trade head accuracy (−2.0% and −2.6%) for tail gains (+3.4% and +3.8%), reflecting the frequency-corrective nature of these losses.
- **T3 (Class-Aware Label Smoothing)** improves tail (+0.84) with minimal impact elsewhere, preserving head accuracy near the CE baseline.
- **T4 (Decoupled Fine-tuning)** retrains only the CLS head with a balanced sampler; the DIST head is frozen. The CLS tail improves from 58.4% → 75.6% (+17.2%), and the ensemble (CLS+DIST)/2 reaches 87.70% overall.

To reproduce, run `python scripts/report.py` from the repo root.

### Running the full comparison

```bash
# CIFAR-10 LT, IF=100, full schedule (1200 epochs, DRW at 1100)
# Runs CE baseline + Techniques 1–3 sequentially; Technique 4 fine-tunes from CE checkpoint.
# Safe to Ctrl+C and re-run — resumes automatically from the latest checkpoint.
bash run.sh
```

Results are printed as a comparison table at the end. Individual training logs are written to `logs/`.

### SLURM scripts (HPC / V100)

For running CIFAR-100 LT experiments on a SLURM cluster:

```bash
# 1. Download dataset (run once on a node with internet access)
sbatch sh/slurm_download_cifar100.sh

# 2. Submit each technique (replace <download_jobid> with the job ID from step 1)
sbatch --dependency=afterok:<download_jobid> --export=TECHNIQUE=logit_adj    sh/slurm_train_c100_if100.sh
sbatch --dependency=afterok:<download_jobid> --export=TECHNIQUE=bal_softmax  sh/slurm_train_c100_if100.sh
sbatch --dependency=afterok:<download_jobid> --export=TECHNIQUE=class_smooth sh/slurm_train_c100_if100.sh
```

Each job requests 1× V100, 22 h wall time, and supports automatic resume if resubmitted.

### New files

```
losses/
  logit_adjustment_20260315_120000.py       # Technique 1
  balanced_softmax_20260315_120000.py       # Technique 2
  class_aware_smoothing_20260315_120000.py  # Technique 3
scripts/
  train_lt_20260315_120000.py               # Unified train wrapper (all loss types)
  finetune_classifier_20260315_120000.py    # Technique 4 stage-2 fine-tuning
run.sh                                      # Orchestrates all 5 runs with auto-resume
sh/slurm_download_cifar100.sh              # SLURM: download CIFAR-100
sh/slurm_train_c100_if100.sh               # SLURM: train CIFAR-100 LT IF=100
```

---

## RunPod / SSH Tips

**Always use tmux** so training survives SSH disconnections.

```bash
# Check tmux is available
which tmux

# Start a named session before running anything
tmux new -s train

# Run your job inside the session
bash run.sh

# Detach (leave running): Ctrl+B, then D
# Reattach later:
tmux attach -t train
```

**Output must go to `/workspace`** — the root filesystem (`/`) is only 20G and will fill up fast with checkpoints. The repo should live at `/workspace/DeiT-LT`, with `/root/DeiT-LT` as a symlink:

```bash
# One-time setup (if not already done)
cp -r /root/DeiT-LT /workspace/DeiT-LT
rm -rf /root/DeiT-LT
ln -s /workspace/DeiT-LT /root/DeiT-LT
```

---

## Usage

1. Clone the respository.
```
git clone https://github.com/val-iisc/DeiT-LT.git
```

2. CIFAR-10 and CIFAR-100 datasets are downloaded on its own through the code. However, in case of [Imagenet-LT](http://image-net.org/) and [iNaturalist-2018](https://github.com/visipedia/inat_comp/tree/master/2018), download these datasets from the given links.

3. Create the conda environment and activate it.

```
conda env create -f environment.yml
conda activate deitlt
```

4. Download the teacher models as given in Results table.

5. Run the corresponding training script for any dataset after adding teacher path in the script file for the argument `--teacher-path`. For example:
```
bash sh/train_c10_if100.sh
bash sh/train_imagenetlt.sh
```

6. For evaluation a DeiT-LT checkpoint, run the eval bash script for the corresponding dataset. Ensure that the checkpoint path is provided to the script for the argument `--resume`. For example: 
```
bash sh/eval_c10.sh
bash sh/eval_imagenetlt.sh
```

## Results

<table>
<tr>
<th>Dataset</th>
<th>Imbalance Factor</th>
<th>Overall</th>
<th>Head</th>
<th>Mid</th>
<th>Tail</th>
<th>Teacher path</th>
<th>Student path</th>
</tr>
<tr>
<td rowspan="2">CIFAR 10-LT</td>
<td>100</td>
<td>87.5</td>
<td>94.5</td>
<td>84.1</td>
<td>85.0</td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/3d5f8683cecaf84b2e4130f4f9d1f192/paco_sam_ckpt_cf10_if100.pth.tar">Link</a></td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/574dab8b51e97a8fca2b88d9f42e53c5/deit_base_distilled_patch16_224_resnet32_1200_CIFAR10LT_imb100_128_%5Bpaco_sam_teacher%5D_best_checkpoint.pth">Link</a></td>
</tr>
<tr>
<td>50</td>
<td>89.8</td>
<td>94.9</td>
<td>87.0</td>
<td>88.6</td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/fc15814c0ce158e6987110b256248e18/paco_sam_ckpt_cf10_if50.pth.tar">Link</a></td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/a1ee5cb061bb9e888dc54d84fa183a58/deit_base_distilled_patch16_224_resnet32_1200_CIFAR10LT_imb50_256_%5BIF50_paco_sam%5D_best_checkpoint.pth">Link</a></td>
</tr>
<tr>
<td rowspan="2">CIFAR 100-LT</td>
<td>100</td>
<td>55.6</td>
<td>72.8</td>
<td>55.4</td>
<td>31.4</td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/fc4a80f43013c11729e28426b64ef70b/cifar100_paco_sam_if100.pth.tar">Link</a></td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/cf0bdaca962d76f5d79be436478cfd5d/deit_base_distilled_patch16_224_resnet32_1200_CIFAR100LT_imb100_128_%5Bpaco_sam_teacher%5D_best_checkpoint.pth">Link</a></td>
</tr>
<tr>
<td>50</td>
<td>60.5</td>
<td>74.8</td>
<td>60.3</td>
<td>43.1</td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/f32d7dd157a9b2b27ff60f1900769ce0/cifar100_paco_sam_if50.pth.tar">Link</a></td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3NzA4NTEx/69282bd245c8510b6edda1abb9459682/deit_base_distilled_patch16_224_resnet32_1200_CIFAR100LT_imb50_128_%5BIF50_paco_sam_teacher_cf100%5D_best_checkpoint.pth">Link</a></td>
</tr>
<tr>
<td>ImageNet-LT</td>
<td>-</td>
<td>59.1</td>
<td>66.6</td>
<td>58.3</td>
<td>40.0</td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3Njk4NTk4/9bae08b41c6c4416e21e8250a955cdda/imagenetlt_paco_sam.pth.tar">Link</a></td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3Njk4NTk4/79821d09b6603bdea5a18d957fd38b99/deit_base_distilled_patch16_224_resnet50_1400_IMAGENETLT_128_%5Bnew_paco_sam_teacher_flashv2%5D_best_checkpoint.pth">Link</a></td>
</tr>
<tr>
<td>iNaturalist-2018</td>
<td>-</td>
<td>75.1</td>
<td>70.3</td>
<td>75.2</td>
<td>76.2</td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3Njk4NTk4/9db822ec842c5d6c6ada53ce686fe9a7/inat_paco_sam.pth.tar">Link</a></td>
<td><a href="https://api.wandb.ai/artifactsV2/default/pradipto611/QXJ0aWZhY3Q6Nzk3Njk4NTk4/5d643af66f1ca8f49de3ef6606166103/deit_base_distilled_patch16_224_resnet50_1000_INAT18_128_%5Bpaco_sam_teacher_long_schedule%5D_best_checkpoint.pth">Link</a></td>
</tr>
</table>


## Acknowledgement
This codebase is heavily inspired from [DeiT](https://github.com/facebookresearch/deit) (ICML 2021). The concepts and methodologies adopted from DeiT have been instrumental in enabling us to push the boundaries of our research and development. We extend our sincerest thanks to the developers and contributors of DeiT. 

## License
DeiT-LT is an open-source project released under the MIT license (MIT). The codebase is derived from that of DeiT (ICML 2021), which is released under [Apache 2.0 license](https://github.com/val-iisc/DeiT-LT/blob/main/LICENSE-FACEBOOK).

## BibTex
If you find this code or idea useful, please consider citing our work:
```
@InProceedings{Rangwani_2024_CVPR,
    author    = {Rangwani, Harsh and Mondal, Pradipto and Mishra, Mayank and Asokan, Ashish Ramayee and Babu, R. Venkatesh},
    title     = {DeiT-LT: Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {23396-23406}
}
```
 
 
 
 

