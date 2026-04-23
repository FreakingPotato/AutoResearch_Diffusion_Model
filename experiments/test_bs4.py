"""Test if bs=4 fits on a single GPU for seq_len=8192."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time, gc
import torch
from train_phase4_2 import (
    Phase42Config, build_model,
    make_loader, build_full_data,
)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

print("=== Building model ===")
cfg = Phase42Config(seq_len=8192, attention_type="mamba2", d_state=64, d_conv=4)
model = build_model(cfg, device)

print("\n=== Loading stage 1 checkpoint ===")
ckpt = torch.load("checkpoints_phase4_2/stage1_seq4096_best.pt", map_location=device, weights_only=False)
state = ckpt["state_dict"]
model_state = model.state_dict()
loaded = {k:v for k,v in state.items() if k in model_state and v.shape == model_state[k].shape}
model_state.update(loaded)
model.load_state_dict(model_state)
print(f"Loaded {len(loaded)} params")

print("\n=== Loading data ===")
build_full_data(8192)

print("\n=== Testing bs=4 ===")
BATCH_SIZE = 4

trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=3e-4)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

try:
    loader = make_loader(BATCH_SIZE, "train", 8192, num_workers=0)
    data_iter = iter(loader)
    
    # Warmup
    batch = next(data_iter).to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model.compute_loss(batch)
    loss.backward()
    opt.zero_grad(set_to_none=True)
    
    # Benchmark 3 steps
    model.train()
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(3):
        opt.zero_grad(set_to_none=True)
        batch = next(data_iter).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024
        print(f"  Step {i+1}: loss={loss.item():.4f}, VRAM={peak:.0f}MB")
    torch.cuda.synchronize()
    
    ms = (time.time() - t0) / 3 * 1000
    final_peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024
    toks = BATCH_SIZE * 8192
    
    print(f"\n✅ bs=4 SUCCESS:")
    print(f"  Peak VRAM: {final_peak:.0f}MB / 24576MB ({final_peak/24576*100:.1f}%)")
    print(f"  {ms:.0f}ms/step, {toks:,} toks/step, {toks*1000/ms/1000:.1f}K toks/sec")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024
        print(f"\n❌ bs=4 OOM at {peak:.0f}MB (need ~{peak - 24576:.0f}MB more)")
    else:
        print(f"\n❌ ERROR: {e}")
