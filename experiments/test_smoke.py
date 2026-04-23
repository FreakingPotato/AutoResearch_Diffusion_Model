"""Quick smoke test: model import + 3 training steps."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import torch
from src.model import build_model, DiffusionConfig
from src.data import FullHg38Dataset
from src.schedule import UniformSchedule
from src.tokenizer import get_tokenizer

print("=== Test 1: Model import ===")
_, _, mask_id, _, _ = get_tokenizer()
print(f"  mask_id={mask_id}")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"  device={device}")

cfg = DiffusionConfig(seq_len=512, attention_type="mamba2", d_state=64, d_conv=4)
model = build_model(cfg, device, mask_id=mask_id)
model = model.to(dtype=torch.bfloat16)
total, trainable = model.count_params()
print(f"  Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
print("  OK")

print("\n=== Test 2: Data loading ===")
ds = FullHg38Dataset("train", seq_len=512)
sample = ds[0]
print(f"  Dataset: {len(ds):,} sequences, sample.shape={sample.shape}")
print("  OK")

print("\n=== Test 3: 3 training steps ===")
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

model.train()
for step, batch in enumerate(loader):
    if step >= 3:
        break
    batch = batch.to(device)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model.compute_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"  step {step}: loss={loss.item():.4f}")

print("  OK")
print("\nAll tests PASSED")
