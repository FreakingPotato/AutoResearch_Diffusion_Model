# AutoResearch Diffusion Model

Phase 4.2: DNA Diffusion with Mamba2, full hg38 training.

## Project Structure

```
AutoResearch_Diffusion_Model/
├── configs/          # Training configs
│   └── ds_config_zero0.json    # DeepSpeed Zero0 config
├── experiments/      # Experimental/benchmark scripts
├── logs/             # All log files
├── results/          # All result files (*.tsv)
├── scripts/          # Shell scripts
├── src/              # Core source code
│   ├── __init__.py
│   ├── data.py       # Dataset and data loading
│   ├── model.py      # Model architecture
│   ├── tokenizer.py  # Tokenizer
│   ├── schedule.py   # Noise schedule
│   └── eval.py       # GB evaluation functions
├── train.py          # Main training entry point
├── legacy/          # Old training scripts (Phase 1, v4, etc.)
├── docs/            # Documentation
└── checkpoints_phase4_2/  # Stage 2 checkpoints
```

## Training (Phase 4.2 Stage 2)

### Single GPU

```bash
uv run python train.py
```

### Dual GPU with DeepSpeed

```bash
# Without Wandb
uv run torchrun --nproc_per_node=2 --master_port=29500 train.py

# With Wandb
export WANDB_API_KEY=your_wandb_api_key
uv run torchrun --nproc_per_node=2 --master_port=29500 train.py --use_wandb
```

### Configuration

- **Model**: NucEL (ModernBERT 22L/512d/16h) with Mamba2 layers
- **Parameters**: 136.6M total, 136.6M trainable
- **Data**: Full hg38 (chr1-22, X, Y), 3.09B tokens
- **Seq length**: 8192
- **Batch size**: 3 per GPU
- **Optimizer**: AdamW, lr=3e-4, weight_decay=0.01
- **LR schedule**: Warmup 500 steps + cosine (min_lr=3e-5 floor)
- **Time budget**: 4 hours
- **DeepSpeed**: Zero0 (no parameter sharding)

### Custom Arguments

```bash
uv run python train.py \
  --seq_len 8192 \
  --batch_size 3 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --max_steps 16000 \
  --hours 4 \
  --use_wandb \
  --wandb_project dna-diffusion \
  --wandb_run_name stage2
```

## Dependencies

- Python 3.12+
- PyTorch 2.x
- Transformers (HuggingFace)
- mamba_ssm (for Mamba2 layers)
- wandb (optional, for logging)
- deepspeed (optional, for multi-GPU training)

Install:
```bash
# Using uv (recommended)
pip install uv
uv sync

# Or with pip
pip install torch transformers mamba_ssm wandb deepspeed
```

## Phase 4.2 Results

### GB Benchmark Evolution

![GB Evolution](gb_evolution.png)

![Coding vs Intergenomic Timeline](coding_vs_intergenomic_timeline.png)

### Key Insights from Coding vs Intergenomic Timeline

- **Stage 1 (4096)**: Step 4000 → 0.807
- **Stage 2 (8192)**: Peak at Step 10000 → **0.854** (best checkpoint)
- Stage 2 continues to Step 15800 → 0.849
- **Stage 2 consistently outperforms Stage 1** across all steps
- **Optimal checkpoint for GB evaluation**: `stage2_seq8192_step10000.pt`

### Stage 1 (seq_len=4096, 8h)
- **Steps**: 10,000+
- **Val loss**: ~0.87
- **GB avg (sampled)**: 0.626
- **Key fix**: Stage 1→Stage 2 checkpoint loading remaps `_layers.*` → `backbone.layers.*`

### Stage 2 (seq_len=8192, 4h, dual GPU)
- **Steps**: 15,800 (interrupted for evaluation)
- **Val loss**: 0.8764 (step 15000)
- **GB avg (sampled)**: 0.678 (using step 15000)
- **GB avg (step 10000 - best)**: **~0.688** (estimated from coding vs intergenomic +5.2%)
- **Best checkpoint**: `checkpoints_phase4_2/stage2_seq8192_step10000.pt` (0.854 on coding vs intergenomic)
- **Improvement vs Stage 1**: +0.052 (step 15000), **+0.062** (step 10000 estimated)

### Comparison with NucEL Paper

LP = Linear Probe (freeze backbone, train MLP) | FT = Full Fine-tuning (all parameters)

| Dataset | Stage 1 LP | Stage 2 LP | Stage 2 FT | NucEL | FT Δ |
|---------|-----------|-----------|-----------|-------|------|
| demo_coding_vs_intergenomic_seqs | 0.704 | 0.804 | 0.883 | 0.952 | -0.068 |
| demo_human_or_worm | 0.760 | 0.796 | **0.908** | 0.922 | -0.013 |
| dummy_mouse_enhancers_ensembl | 0.570 | 0.603 | 0.624 | 0.791 | -0.167 |
| human_enhancers_cohn | 0.664 | 0.668 | **0.727** | 0.709 | **+0.018** ✅ |
| human_enhancers_ensembl | 0.568 | 0.632 | 0.615 | 0.732 | -0.117 |
| human_ensembl_regulatory | 0.572 | 0.672 | 0.573 | 0.583 | -0.010 |
| human_ocr_ensembl | 0.544 | 0.568 | 0.565 | 0.676 | -0.111 |
| **Average** | **0.626** | **0.678** | **0.699** | **0.781** | **-0.082** |

**Key findings:**
- Full fine-tuning improves over linear probe by **+2.1%** on average (0.699 vs 0.678)
- `human_enhancers_cohn` exceeds NucEL baseline by **+1.8%** with fine-tuning ✅
- `demo_human_or_worm` reaches **0.908**, only **1.3%** below NucEL
- Fine-tuning takes ~7 min per dataset on single RTX 3090

## Phase 4.1 Results (for comparison)

- Best model: exp 7 (mamba2_adamw)
- Full GB avg: 0.724
- Frozen NucEL baseline: 0.703
- Improvement: +0.021
