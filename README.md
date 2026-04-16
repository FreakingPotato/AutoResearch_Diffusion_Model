# AutoResearch: DNA Diffusion Model

Autonomous diffusion model research on human DNA sequences (GRCh38/hg38). The goal: train a discrete diffusion model to generate realistic DNA, then evaluate via Genomic Benchmarks linear probe.

## Project Overview

We explored three major phases, each building on lessons from the previous:

| Phase | Backbone | Key Question | Results |
|-------|----------|-------------|---------|
| **v1** | Custom DiT (from scratch) | What architecture works for DNA diffusion? | `BASELINE_PHASE1.md`, `results.tsv` |
| **v2** | NucEL (pretrained ModernBERT) | Does a pretrained DNA LM backbone help? | `results_nucel.tsv` |
| **v3** | NucEL + Linear Attention variants | Can we improve with better attention/optimizer? | `results_phase3.tsv` ← **latest** |

---

## Phase v1: From-Scratch DNA DiT

**Script:** `train.py` | **Results:** `results.tsv` | **Details:** `BASELINE_PHASE1.md`

Custom DiT trained from scratch with 6-mer tokenization. 50-experiment sweep over architecture, learning rate, and scale.

**Key findings:**
- Small models (10-45M params) dominate at 5-min budget
- Learning rate is most critical hyperparameter (lr=5e-4 best)
- Shallow+wide beats deep+narrow
- **Major issue:** Sampling produces MASK/UNK tokens (gen_gc=0.0)
- Best val_loss: 6.19 (10M params, 6L/256d)

---

## Phase v2: NucEL Pretrained Backbone

**Script:** `train_nucel.py` | **Results:** `results_nucel.tsv`

Switched to NucEL (FreakingPotato/NucEL) — a ModernBERT 22L/512d pretrained on DNA with single-nucleotide tokenization. DiffusionBERT-style discrete absorbing-state diffusion.

**Key changes from v1:**
- Single-nucleotide tokenization (27 vocab) instead of 6-mer (4096 vocab)
- Pretrained backbone instead of training from scratch
- Noise steps reduced 1000→128 for faster evaluation
- 20-experiment sweep: optimizer, time embedding, noise schedule, freeze strategy

**Key findings:**
- Pretrained backbone converges much faster (val_loss 0.06-0.08 vs 6.19)
- GB NLL evaluation: 6-11 range (baseline)
- AdaLN time embedding ≈ additive
- Cosine/spindle schedule slightly better than uniform

---

## Phase v3: Linear Attention & Optimizer Variants (Current)

**Script:** `train_phase3.py` | **Results:** `results_phase3.tsv`

11 experiments exploring attention mechanism replacements, optimizers, and combinations. GB evaluation changed from NLL to **linear probe accuracy** (CLS embedding → linear classifier → accuracy).

### Experimental Design

- **Backbone:** NucEL (ModernBERT 22L/512d/16h) with Flash Attention 2
- **Data:** chr21 (train), chr22 (val), seq_len=1024
- **Evaluation:** GB Linear Probe — extract CLS embedding, train linear classifier (3 epochs), report test accuracy on 8 subtasks
- **Training:** BF16 mixed precision, gradient accumulation (32K tokens/step), cosine LR schedule

### Results Summary

| # | Experiment | val_loss | Train Loss | Steps | GB Acc (avg) | Params | Notes |
|---|-----------|----------|------------|-------|-------------|--------|-------|
| 0 | NucEL Frozen Probe | - | - | 0 | 0.57 | 100M | Baseline (no diffusion training) |
| 1 | Diffusion Baseline (AdamW) | 0.00006 | 1.61 | 2249 | 0.55 | 100M | |
| 2 | Muon Optimizer | 0.021 | 1.23 | 2505 | **0.62** | 100M | ✅ SingleDeviceMuon |
| 3 | AdaLN (DiT-style) | 0.00005 | 1.09 | 2238 | **0.64** | 106M | |
| 4 | Performer | 0.013 | **1.00** | 2056 | **0.64** | 123M | Lowest train loss |
| 5 | RetNet | 0.096 | 1.02 | 2254 | 0.58 | 146M | High val_loss |
| 6 | GLA | 0.026 | 1.18 | 546 | 0.50 | 124M | Under-trained (546 steps) |
| 7 | **Mamba-2** | 0.023 | 1.05 | 1010 | **0.65** | 137M | 🏆 Best GB accuracy |
| 8 | Hybrid (GLA+Global) | 0.018 | 1.21 | 1427 | **0.63** | 116M | |
| 9 | Muon + AdaLN | 0.000 | 3.30 | 304 | 0.49 | 106M | ⚠️ Unstable (lr too high) |
| 10 | Muon + GLA | 0.00007 | 1.10 | 1806 | 0.57 | 124M | |

### Key Findings

1. **Mamba-2 achieves best GB accuracy (0.65)** — SSM architecture excels at DNA sequence modeling
2. **Performer and AdaLN close second (0.64)** — Performer has lowest train loss
3. **Muon optimizer improves over AdamW** (0.62 vs 0.55) — momentum orthogonalization helps
4. **Frozen NucEL baseline is 0.57** — diffusion training adds ~0.08 improvement with Mamba-2
5. **GLA under-trained** — only 546 steps due to slow convergence, needs longer budget
6. **Muon+AdaLN failed** — lr=0.02 too high for this combination, needs tuning

### Per-Benchmark Breakdown (Phase v3, selected experiments)

| Benchmark | Frozen | Baseline | Mamba2 | Performer | Hybrid |
|-----------|--------|----------|--------|-----------|--------|
| human_enhancers_cohn | 0.60 | 0.53 | 0.64 | 0.64 | **0.65** |
| demo_coding_vs_intergenomic | 0.76 | 0.67 | **0.78** | 0.75 | **0.81** |
| demo_human_or_worm | 0.57 | 0.66 | **0.68** | 0.65 | **0.72** |
| dummy_mouse_enhancers | 0.50 | 0.50 | **0.71** | 0.71 | **0.70** |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Download and prepare DNA data (chr21, chr22)
uv run prepare.py --chromosomes 21,22

# Run Phase v3 experiment
uv run python train_phase3.py --gpu 0 --exp 7     # Single Mamba2 experiment
uv run python train_phase3.py --gpu 0 --sweep       # Full sweep
```

## File Guide

| File | Description |
|------|-------------|
| `prepare.py` | Data download & tokenization |
| `train.py` | Phase v1: Custom DiT sweep (50 experiments) |
| `train_v2.py` | Phase v2: Intermediate refactoring |
| `train_nucel.py` | Phase v2: NucEL backbone + DiffusionBERT (20 experiments) |
| `train_phase3.py` | **Phase v3: Linear attention + GB linear probe (11 experiments)** |
| `results.tsv` | Phase v1 results |
| `results_v2.tsv` | Phase v2 intermediate results |
| `results_nucel.tsv` | Phase v2 final results (20 experiments) |
| `results_phase3.tsv` | **Phase v3 results (11 experiments)** |
| `BASELINE_PHASE1.md` | Phase v1 detailed analysis |
| `program.md` | Original autonomous research instructions |
| `checkpoints_phase3/` | Phase v3 model checkpoints |

## Hardware

- 2× NVIDIA RTX 3090 (24GB each)
- CUDA 13.0, PyTorch 2.11, Flash Attention 2.8.3
- Python 3.12, managed with `uv`

## Dependencies

Key packages (see `pyproject.toml` for full list):
- `torch>=2.5.0` + `flash-attn>=2.0`
- `transformers>=5.5` (ModernBERT support)
- `flash-linear-attention>=0.4` (GLA, RetNet)
- `mamba-ssm>=2.3` (Mamba-2, compiled from source for CUDA 13)
- `performer-pytorch>=1.1` (Performer attention)
- `muon-optimizer>=0.1` (Muon optimizer)

## References

- MDLM (Sahoo et al., 2024) — Simple and Effective Masked Diffusion Language Models
- DiffusionBERT (Li et al., 2023) — Diffusion models with pretrained LMs
- DiT (Peebles & Xie, 2023) — Scalable Diffusion Models with Transformers
- Mamba-2 (Dao & Gu, 2024) — Structured State Space Models
- Performer (Choromanski et al., 2021) — Rethinking Attention with Performers
- GLA (Yang et al., 2024) — Gated Linear Attention
- Muon (Jordan, 2024) — Momentum Orthogonalized by Newton-Schulz
- NucEL — DNA pretrained ModernBERT (FreakingPotato/NucEL)
