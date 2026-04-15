# DNA Diffusion Model — Autonomous Research Program

This is an experiment to have the LLM autonomously iterate on a DNA diffusion model.

## Setup

1. **Agree on a run tag**: e.g. `apr15`. Branch `autoresearch/<tag>` must not exist.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read files**: `README.md`, `prepare.py` (read-only), `train.py` (you modify this).
4. **Verify data**: Check `~/.cache/dna-diffusion/data/` has train.bin and val.bin. If not, run `uv run prepare.py --chromosomes 21,22`.
5. **Initialize results.tsv**: Create with header row.
6. **Confirm and go**.

## Experimentation

Each experiment runs for a **fixed 5 minutes** (wall clock). Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — model architecture, diffusion schedule, optimizer, hyperparams, batch size.
- Everything is fair game: try continuous diffusion in embedding space, flow matching, different transformer architectures, different tokenization strategies (if you also update prepare.py output format), different noise schedules, etc.

**What you CANNOT do:**
- Modify the evaluation harness in `prepare.py` (the `evaluate_metrics` function is ground truth).
- Add new pip packages without documenting why.

**Goal: Minimize val_loss while keeping nt_dist_error and gc_error low.** The model should generate DNA that looks statistically similar to real human DNA.

## Domain Knowledge

Human DNA characteristics:
- ~41% GC content on average (varies by region)
- A≈T≈29%, C≈G≈21% approximately
- Contains repetitive elements (Alu, LINE, SINE)
- Codon structure in coding regions
- Promoter motifs, enhancers, etc.

Key bio-sequence diffusion papers to draw inspiration from:
- **DiscDiff** (2024): Discrete diffusion for DNA
- **DNADiffusion** (2024): Conditional DNA generation
- **EvoDiff** (2023): Protein sequence diffusion
- **DPLM** (2024): Protein language model + diffusion
- **MDLM** (2024): Masked diffusion for discrete data

## Output Format

After training, the script prints:
```
---
val_loss:           2.345678
nt_dist_error:      0.012345
gc_error:           0.003456
training_seconds:   300.1
total_seconds:      325.9
peak_vram_mb:       12345.6
total_tokens_M:     499.6
num_steps:          953
num_params_M:       100.3
depth:              12
```

## Logging

Log to `results.tsv`:
```
commit	val_loss	memory_gb	status	description
a1b2c3d	2.345678	12.3	keep	baseline with DiT-12
```

## Experiment Ideas

### Architecture
- Try different depths/widths (8, 12, 16 layers; 512, 768, 1024 dim)
- GPT-style causal attention vs bidirectional attention
- Rotary embeddings vs learned positional
- Different MLP ratios (2, 4, 8)

### Diffusion
- Flow matching instead of discrete diffusion
- Continuous diffusion in embedding space (DiffusionLM-style)
- Different noise schedules (cosine, linear, sqrt)
- VQ-VAE + continuous diffusion
- Score entropy (SEDD) approach

### Training
- Learning rate tuning
- Different optimizers (Muon, AdamW variants)
- Batch size experiments
- Data augmentation (reverse complement)
- Curriculum learning (short → long sequences)

### Evaluation
- Reverse complement symmetry
- Motif preservation
- Longer sequence generation

## The Loop

NEVER STOP. Run experiments continuously until interrupted.
Each experiment: modify train.py → commit → run → read results → log → keep or revert.
