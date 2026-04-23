#!/usr/bin/env bash
# Stage 2: seq_len=8192 on dual GPU with DataParallel
# Waits for Stage 1 to finish, then starts Stage 2

cd ~/AutoResearch_Diffusion_Model

# Find best stage 1 checkpoint
CKPT=$(ls -t checkpoints_phase4_2/stage1_*_best.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "No stage 1 checkpoint found!"
    exit 1
fi

echo "Using checkpoint: $CKPT"
echo "Starting Stage 2 on dual GPU..."

# Use CUDA_VISIBLE_DEVICES=0,1 for both GPUs
CUDA_VISIBLE_DEVICES=0,1 uv run python -c "
import os, sys, torch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from train_phase4_2 import *

device = torch.device('cuda')

# Build model
cfg = Phase42Config(seq_len=8192, attention_type='mamba2', d_state=64, d_conv=4)
model = build_model(cfg, device)

# Load stage 1 checkpoint
ckpt = torch.load('$CKPT', map_location=device, weights_only=False)
state = ckpt['state_dict']
model_state = model.state_dict()
loaded = {k:v for k,v in state.items() if k in model_state and v.shape == model_state[k].shape}
skipped = {k: (v.shape, model_state[k].shape) for k,v in state.items() if k in model_state and v.shape != model_state[k].shape}
model_state.update(loaded)
model.load_state_dict(model_state)
print(f'Loaded {len(loaded)}/{len(state)} params')
if skipped: print(f'Shape mismatch (expected): {list(skipped.keys())[:5]}')

# Wrap with DataParallel for dual GPU
model = torch.nn.DataParallel(model)
print(f'Dual GPU: {torch.cuda.device_count()} GPUs')

# Now run stage 2 training manually
total_p = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total: {total_p/1e6:.1f}M, Trainable: {trainable_p/1e6:.1f}M')

# Data
build_full_data(8192)
train_loader = make_loader(1, 'train', 8192)  # bs=1 per GPU, DataParallel doubles effective bs
val_loader = make_loader(1, 'val', 8192, num_workers=0)

trainable_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))

# Speed test
model.train(); data_iter = iter(train_loader)
torch.cuda.synchronize(); t0 = __import__('time').time()
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    batch = next(data_iter).to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model.module.compute_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    opt.step()
torch.cuda.synchronize()
ms_per_step = (__import__('time').time() - t0) / 3 * 1000
print(f'Speed: {ms_per_step:.0f}ms/step')

budget_s = 4 * 3600  # 4 hours
est_steps = int(budget_s / (ms_per_step / 1000))
print(f'Estimated steps: {est_steps}')

# Training loop
import math, gc, time as _time
data_iter = iter(train_loader)
train_time = 0.0; step = 0; final_loss = float('nan'); smooth_loss = 0.0
warmup = max(50, est_steps // 20)

while True:
    torch.cuda.synchronize(); t0 = _time.time()
    opt.zero_grad(set_to_none=True)
    batch = next(data_iter, None)
    if batch is None: data_iter = iter(train_loader); batch = next(data_iter)
    batch = batch.to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model.module.compute_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0); opt.step()
    
    train_loss = loss.item()
    if not math.isnan(train_loss): final_loss = train_loss
    torch.cuda.synchronize(); dt = _time.time() - t0
    if step > 2: train_time += dt
    
    b = 0.95; smooth_loss = b*smooth_loss + (1-b)*train_loss
    debi = smooth_loss / (1-b**(step+1))
    
    if step < warmup: lr_s = step / warmup
    else:
        progress = (step-warmup) / max(est_steps-warmup, 1)
        lr_s = 0.5*(1+math.cos(math.pi*min(progress,1.0)))
    for pg in opt.param_groups: pg['lr'] = 3e-4 * lr_s
    
    if step % 200 == 0:
        rem = max(0, budget_s - train_time)
        print(f'step {step:05d} | loss={debi:.4f} | lr={3e-4*lr_s:.2e} | dt={dt*1000:.0f}ms | rem={rem/60:.0f}min')
        sys.stdout.flush()
    
    if math.isnan(train_loss): print('NaN'); break
    step += 1
    
    if step % 2000 == 0:
        model.eval()
        vl = 0; vn = 0
        for vi, vb in enumerate(val_loader):
            if vi >= 50: break
            vb = vb.to(device)
            t = torch.randint(1, model.module.noise_steps+1, (vb.shape[0],), device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                vl += model.module.compute_loss(vb, t).item(); vn += 1
        val_loss = vl / max(vn, 1)
        print(f'  ★ Step {step}: val_loss={val_loss:.6f}')
        ckpt_path = f'checkpoints_phase4_2/stage2_seq8192_step{step}.pt'
        torch.save({'stage':2, 'seq_len':8192, 'step':step, 'val_loss':val_loss,
                    'state_dict':{k:v for k,v in model.module.state_dict().items() if not k.startswith('nucel.')}},
                   ckpt_path)
        model.train()
        sys.stdout.flush()
    
    if train_time >= budget_s: print(f'Budget reached ({4}h)'); break

print(f'Stage 2 done: {step} steps in {train_time:.0f}s')
" > logs_v42_stage2.log 2>&1

echo "Stage 2 complete!"
