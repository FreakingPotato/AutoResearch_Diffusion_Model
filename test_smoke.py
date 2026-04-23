"""Quick test: train on 1% data to verify everything works end-to-end."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = "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"

import time
import math
import torch
from pathlib import Path

import wandb

from src import (
    DiffusionConfig,
    build_model,
    FullHg38Dataset,
    build_full_data,
    evaluate_val_loss,
    evaluate_gb_sampled,
)

CKPT_DIR = Path("checkpoints_phase4_2")
STAGE1_CKPT = CKPT_DIR / "stage1_seq4096_best.pt"

# --- Config ---
SEQ_LEN = 4096  # Use 4096 for quick test
BATCH_SIZE = 2
LR = 3e-4
WARMUP_STEPS = 5
MAX_STEPS = 50
DEVICE = torch.device("cuda:0")

# --- Init Wandb ---
wandb.init(
    project="dna-diffusion",
    name="test-1pct-smoke",
    config={
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "max_steps": MAX_STEPS,
        "data_fraction": "1%",
        "model": "NucEL+Mamba2",
    }
)

# --- Build Model ---
print("=== Building model ===")
config = DiffusionConfig(seq_len=SEQ_LEN, attention_type="mamba2", d_state=64, d_conv=4)
model = build_model(config, DEVICE, mask_id=4)

# Load stage 1 checkpoint
if STAGE1_CKPT.exists():
    print(f"Loading {STAGE1_CKPT}...")
    ckpt = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(loaded)
    model.load_state_dict(model_state)
    print(f"Loaded {len(loaded)}/{len(state)} params")
else:
    print("No stage 1 checkpoint, training from scratch")

# --- Data (1%) ---
print("=== Loading 1% data ===")
build_full_data(SEQ_LEN)
full_dataset = FullHg38Dataset("train", SEQ_LEN)
subset_size = max(len(full_dataset) // 100, 100)
subset = torch.utils.data.Subset(full_dataset, range(subset_size))
loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
print(f"Using {subset_size} samples (1% of {len(full_dataset)})")

# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

# --- Training Loop ---
print(f"\n=== Training ({MAX_STEPS} steps) ===")
model.train()
data_iter = iter(loader)
t_start = time.time()

for step in range(MAX_STEPS):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)
    batch = batch.to(DEVICE)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model.compute_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # LR schedule
    if step < WARMUP_STEPS:
        lr = LR * step / WARMUP_STEPS
    else:
        progress = (step - WARMUP_STEPS) / max(MAX_STEPS - WARMUP_STEPS, 1)
        lr = LR * 0.1 + 0.5 * (LR - LR * 0.1) * (1 + math.cos(math.pi * min(progress, 1.0)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.step()

    elapsed = time.time() - t_start
    wandb.log({"train_loss": loss.item(), "lr": lr, "step": step}, step=step)

    if step % 10 == 0:
        print(f"  step {step:03d}/{MAX_STEPS} | loss={loss.item():.4f} | lr={lr:.6f} | elapsed={elapsed:.1f}s")

# --- Final Eval ---
print("\n=== Quick GB Eval ===")
model.eval()
try:
    gb = evaluate_gb_sampled(model, DEVICE, SEQ_LEN)
    avg = sum(v for v in gb.values() if not math.isnan(v)) / max(sum(1 for v in gb.values() if not math.isnan(v)), 1)
    print(f"  GB avg: {avg:.4f}")
    wandb.log({"gb_avg": avg, **{f"gb_{k}": v for k, v in gb.items()}})
except Exception as e:
    print(f"  GB eval error: {e}")
    avg = -1

# --- Save Checkpoint ---
ckpt_path = CKPT_DIR / "test_1pct_final.pt"
torch.save({
    "test": "1pct_smoke",
    "seq_len": SEQ_LEN,
    "step": MAX_STEPS,
    "final_loss": loss.item(),
    "gb_avg": avg,
    "state_dict": model.state_dict(),
}, ckpt_path)
print(f"Checkpoint saved: {ckpt_path}")

total_time = time.time() - t_start
wandb.log({"total_time_s": total_time, "final_loss": loss.item()})
wandb.finish()

print(f"\n=== Test Complete: {MAX_STEPS} steps in {total_time:.1f}s ({total_time/60:.1f}min) ===")
print("Check results at: https://wandb.ai/")
