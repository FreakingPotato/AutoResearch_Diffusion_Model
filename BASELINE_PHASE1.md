# Phase 1 Baseline: From-Scratch DNA Diffusion Model

## Experiment Date
2026-04-15

## Design Philosophy

### Model Architecture
- **Type:** Discrete Masked Diffusion (MDLM-style, absorbing state model)
- **Backbone:** Custom DiT (Diffusion Transformer) with AdaLN time conditioning
- **Tokenization:** 6-mer DNA (vocab_size=4100, 4096 k-mers + 4 special tokens)
- **Context length:** 4096 tokens (= 4096×6 = 24,576 bp)

### Key Design Choices
1. **Absorbing state diffusion:** Forward process replaces tokens with [MASK]; reverse process learns to denoise
2. **Confidence-based unmasking:** At each reverse step, unmask tokens the model is most confident about
3. **AdaLN conditioning:** Timestep injected via adaptive layer norm (DiT-style)
4. **RMSNorm + SiLU** activations throughout
5. **Bidirectional attention** by default (DNA is non-autoregressive)

### Training Setup
- **Data:** Human genome GRCh38, chr21 (train) / chr22 (val)
- **Budget:** 5 minutes per experiment (300s training time)
- **Optimizer:** AdamW (β1=0.9, β2=0.95)
- **Effective batch:** 131,072 tokens/step (via gradient accumulation)
- **Precision:** BF16 mixed precision
- **LR schedule:** Linear warmup (5%) + linear warmdown (40%)
- **Compilation:** torch.compile

### Sweep Design (50 experiments)
1. **Phase 1 (exp 0-9):** Architecture search — depth, width, noise steps, attention type
2. **Phase 2 (exp 10-19):** Learning rate & regularization — lr, dropout, weight decay, batch size
3. **Phase 3 (exp 20-29):** Best architecture refinement — combining top configs
4. **Phase 4 (exp 30-39):** Scale up — larger models (12-16 layers, 768-1024 dim)
5. **Phase 5 (exp 40-49):** Final tuning — refined combinations

## Key Results

### Top 5 by Val Loss
| Rank | Experiment | Params | Val Loss | Config |
|------|-----------|--------|----------|--------|
| 1 | tiny_256 | 10.4M | 6.189 | 6L/256d, lr=1e-4 |
| 2 | lr_5e4 | 44.6M | 6.209 | 8L/512d, lr=5e-4 |
| 3 | scale_8L_512_5e4 | 44.6M | 6.232 | 8L/512d, lr=5e-4 |
| 4 | best_10L_640_drop | 82.5M | 6.238 | 10L/640d, lr=3e-4, drop=0.1 |
| 5 | final_8L_768_5e4 | 95.6M | 6.245 | 8L/768d, lr=5e-4, drop=0.1 |

### Key Findings
1. **Small models dominate at 5min budget** — 10-45M params optimal, >100M can't converge
2. **Learning rate is most critical** — lr=5e-4 > 3e-4 > 1e-4
3. **Shallow+wide > deep+narrow** — 6L/1024d > 16L/384d
4. **Dropout=0.1 helps**, 0.2 hurts
5. **Noise steps (32-256) have minimal impact**
6. **Bidirectional ≈ causal attention** (val_loss 6.27 vs 6.32)
7. **Weight decay 0.01-0.1:** minimal difference

### Known Issues
- **gen_gc = 0.0** for all experiments — sampling produces only MASK/UNK tokens
- 5-minute budget too short for meaningful convergence
- Only chr21/chr22 used — limited genomic diversity

## Files
- `results.tsv` — Full 50-experiment results
- `checkpoints/` — Saved model weights (0xx_name.pt)
- `train.py` — Training script with sweep configs
- `prepare.py` — Data preparation and evaluation metrics
