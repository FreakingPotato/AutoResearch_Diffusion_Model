"""Benchmark different batch sizes for seq_len=8192 on dual GPU."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import time, gc, math
import torch
import torch.nn as nn
from train_phase4_2 import (
    Phase42Config, build_model, get_tokenizer,
    UniformSchedule, make_loader, build_full_data
)

torch.manual_seed(42); torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

device = torch.device("cuda")

# Build model (single GPU first for reference)
print("=== Building model ===")
cfg = Phase42Config(seq_len=8192, attention_type="mamba2", d_state=64, d_conv=4)
model = build_model(cfg, device)
total_p = sum(p.numel() for p in model.parameters())
print(f"Model: {total_p/1e6:.1f}M params")

# Load stage 1 checkpoint
ckpt_path = "checkpoints_phase4_2/stage1_seq4096_best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt["state_dict"]
model_state = model.state_dict()
loaded = {k:v for k,v in state.items() if k in model_state and v.shape == model_state[k].shape}
model_state.update(loaded)
model.load_state_dict(model_state)
print(f"Loaded {len(loaded)}/{len(state)} params from stage 1")

# Data
print("=== Loading data ===")
build_full_data(8192)

# Test configs
configs = [
    # (label, batch_size, use_dp, grad_accum)
    ("GPU0 bs=1, no DP",           1, False, 4),
    ("GPU0 bs=2, no DP",           2, False, 2),
    ("GPU0 bs=4, no DP",           4, False, 1),
    ("DP bs=1 (eff=2)",            1, True,  4),
    ("DP bs=2 (eff=4)",            2, True,  2),
    ("DP bs=4 (eff=8)",            4, True,  1),
]

print(f"\n{'Config':<30} {'GPU0 MB':>8} {'GPU1 MB':>8} {'ms/step':>8} {'tok/step':>10} {'OOM':>5}")
print("-" * 75)

for label, bs, use_dp, ga in configs:
    # Rebuild model for clean state
    model2 = build_model(cfg, device)
    model2_state = model2.state_dict()
    model2_state.update({k:v for k,v in loaded.items() if k in model2_state})
    model2.load_state_dict(model2_state)
    
    if use_dp:
        model2 = nn.DataParallel(model2)
    
    trainable = [p for p in model2.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_peak_memory_stats(1)
    
    try:
        loader = make_loader(bs, "train", 8192, num_workers=0)
        data_iter = iter(loader)
        
        # Warmup
        batch = next(data_iter).to(device)
        if use_dp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model2.module.compute_loss(batch)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model2.compute_loss(batch)
        loss.backward()
        opt.zero_grad(set_to_none=True)
        
        # Benchmark 5 steps with grad_accum
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            for micro in range(ga):
                try: batch = next(data_iter)
                except StopIteration: data_iter = iter(loader); batch = next(data_iter)
                batch = batch.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if use_dp:
                        loss = model2.module.compute_loss(batch) / ga
                    else:
                        loss = model2.compute_loss(batch) / ga
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        torch.cuda.synchronize()
        
        ms = (time.time() - t0) / 5 * 1000
        gpu0_mb = torch.cuda.max_memory_allocated(0) / 1024 / 1024
        gpu1_mb = torch.cuda.max_memory_allocated(1) / 1024 / 1024
        toks = bs * (2 if use_dp else 1) * 8192 * ga
        
        print(f"{label:<30} {gpu0_mb:>8.0f} {gpu1_mb:>8.0f} {ms:>8.0f} {toks:>10,} {'':>5}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gpu0_mb = torch.cuda.max_memory_allocated(0) / 1024 / 1024
            gpu1_mb = torch.cuda.max_memory_allocated(1) / 1024 / 1024
            print(f"{label:<30} {gpu0_mb:>8.0f} {gpu1_mb:>8.0f} {'OOM':>8} {'':>10} {'YES':>5}")
        else:
            print(f"{label:<30} ERROR: {e}")
        torch.cuda.empty_cache()
    
    del model2, opt
    gc.collect()
    torch.cuda.empty_cache()

# Check LR issue
print("\n=== LR Schedule Check ===")
# The original code: print(f'lr={3e-4*lr_s:.2e}')
# With .2e format, 1e-7 prints as "1.00e-07" but 5e-10 prints as "5.00e-10"
# Both would show. But if lr_s is truly 0, it would show "0.00e+00"
# Let's check the cosine schedule logic
est_steps = 38000  # approximate for stage 2
warmup = est_steps // 20  # = 1900
for step in [0, 100, 1000, 1900, 5000, 10000, 20000, 30000, 38000]:
    if step < warmup:
        lr_s = step / warmup
    else:
        progress = (step - warmup) / max(est_steps - warmup, 1)
        lr_s = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    lr = 3e-4 * lr_s
    print(f"  step {step:>6d}: lr_s={lr_s:.6f}, lr={lr:.8f} (printed as {lr:.2e})")
