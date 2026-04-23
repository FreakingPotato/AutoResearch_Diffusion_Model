"""Test bs=4 with grad_accum=2 (micro-batch bs=2)."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
from train_phase4_2 import Phase42Config, build_model, make_loader, build_full_data

torch.manual_seed(42); torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

cfg = Phase42Config(seq_len=8192, attention_type="mamba2", d_state=64, d_conv=4)
model = build_model(cfg, device)

ckpt = torch.load("checkpoints_phase4_2/stage1_seq4096_best.pt", map_location=device, weights_only=False)
state = ckpt["state_dict"]
model_state = model.state_dict()
loaded = {k:v for k,v in state.items() if k in model_state and v.shape == model_state[k].shape}
model_state.update(loaded)
model.load_state_dict(model_state)

build_full_data(8192)

print("=== Testing bs=2 with grad_accum=2 (effective bs=4) ===")
BATCH_SIZE = 2  # micro-batch
GRAD_ACCUM = 2  # effective bs = 4

trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=3e-4)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

loader = make_loader(BATCH_SIZE, "train", 8192, num_workers=0)
data_iter = iter(loader)

# Benchmark 5 steps
model.train()
torch.cuda.synchronize(); t0 = time.time()
for i in range(5):
    opt.zero_grad(set_to_none=True)
    for micro in range(GRAD_ACCUM):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(loader); batch = next(data_iter)
        batch = batch.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch) / GRAD_ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
torch.cuda.synchronize()

ms = (time.time() - t0) / 5 * 1000
peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024
effective_bs = BATCH_SIZE * GRAD_ACCUM
toks = effective_bs * 8192

print(f"✅ SUCCESS:")
print(f"  Micro-batch bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, effective_bs={effective_bs}")
print(f"  Peak VRAM: {peak:.0f}MB / 24576MB ({peak/24576*100:.1f}%)")
print(f"  {ms:.0f}ms/step, {toks:,} toks/step, {toks*1000/ms/1000:.1f}K toks/sec")
