# AutoResearch: DNA Diffusion Model

Autonomous diffusion model research on human DNA sequences, inspired by [AutoResearch](https://github.com/karpathy/autoresearch).

## Concept

Instead of training an LLM on text, we train a **discrete diffusion model** on human DNA sequences (GRCh38/hg38). The model learns to generate realistic DNA by iteratively denoising from pure noise (all-MASK tokens).

## Architecture

- **Backbone**: Diffusion Transformer (DiT) with AdaLN-Zero conditioning
- **Diffusion**: Discrete mask diffusion (absorbing state model, inspired by MDLM)
- **Tokenization**: 6-mer DNA tokens (4^6 = 4096 tokens + 4 special)
- **Size**: ~100M parameters (configurable)
- **Data**: Human genome (GRCh38), chromosome-level splits

## Quick Start

```bash
# Install dependencies
uv sync

# Download and prepare DNA data (~5 min for chr21,22)
uv run prepare.py --chromosomes 21,22

# Train for 5 minutes (baseline)
uv run train.py
```

## Project Structure

```
prepare.py    — Data download (GRCh38), k-mer tokenization, train/val splits (do not modify)
train.py      — Model, diffusion, training loop (modify this for experiments)
program.md    — Autonomous research instructions
pyproject.toml — Dependencies
```

## Hardware

- 2x NVIDIA RTX 3090 (24GB each)
- Dual-GPU via DataParallel/DDP supported

## Metrics

- **val_loss**: Cross-entropy on masked token prediction (lower = better)
- **nt_dist_error**: L1 error between generated and real nucleotide distributions
- **gc_error**: |generated GC% - real GC%|
- **training_seconds**: Wall-clock training time (fixed budget)

## Design Choices

- **Discrete diffusion**: DNA is naturally discrete (4 nucleotides), so we use mask/absorbing diffusion rather than continuous Gaussian diffusion
- **6-mer tokens**: Captures short-range dependencies while keeping vocab size manageable (4096)
- **Confident sampling**: During generation, unmask most-confident positions first (MDLM-style)
- **DiT with AdaLN**: Proven architecture for diffusion models, adapted for discrete tokens

## Autonomous Research

See `program.md` for instructions on running autonomous experiments. The loop:
1. Modify `train.py` (architecture, hyperparams, schedule, etc.)
2. Run training (fixed 5-min budget)
3. Check results
4. Keep improvements, discard regressions
5. Repeat

## References

- DDPM (Ho et al., 2020) — Denoising Diffusion Probabilistic Models
- MDLM (Sahoo et al., 2024) — Simple and Effective Masked Diffusion Language Models
- DiT (Peebles & Xie, 2023) — Scalable Diffusion Models with Transformers
- DiscDiff (2024) — Discrete Diffusion for DNA
- DNADiffusion (2024) — DNA sequence generation with diffusion
