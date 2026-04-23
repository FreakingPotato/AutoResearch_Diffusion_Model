"""Phase 4.2 Stage 2: Dual GPU with mp.spawn, bs=3 per GPU."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, time, math, gc
import torch
import torch.multiprocessing as mp
from train_phase4_2 import (
    Phase42Config, build_model, make_loader, build_full_data,
    CKPT_DIR, evaluate_val_loss, evaluate_gb_sampled,
)

CKPT_PATH = "checkpoints_phase4_2/stage1_seq4096_best.pt"
SEQ_LEN = 8192
BATCH_SIZE = 3  # per GPU
GRAD_ACCUM = 1  # effective bs = 3 per GPU
LR = 3e-4
MIN_LR = LR * 0.1  # min_lr floor
WARMUP_STEPS = 500
BUDGET_HOURS = 4

def worker(gpu_id, results_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    torch.manual_seed(42 + gpu_id)
    torch.cuda.manual_seed(42 + gpu_id)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    
    print(f"[GPU {gpu_id}] Initializing...")
    
    cfg = Phase42Config(seq_len=SEQ_LEN, attention_type="mamba2", d_state=64, d_conv=4)
    model = build_model(cfg, device)
    
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    loaded = {k:v for k,v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(loaded)
    model.load_state_dict(model_state)
    print(f"[GPU {gpu_id}] Loaded {len(loaded)} params from stage 1")
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    
    loader = make_loader(BATCH_SIZE, "train", SEQ_LEN, num_workers=0)
    data_iter = iter(loader)
    
    # Speed test
    model.train()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        batch = next(data_iter).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    torch.cuda.synchronize()
    ms_per_step = (time.time() - t0) / 3 * 1000
    
    budget_s = BUDGET_HOURS * 3600
    est_steps = int(budget_s / (ms_per_step / 1000))
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"[GPU {gpu_id}] {ms_per_step:.0f}ms/step, ~{est_steps} steps, {peak_mb:.0f}MB VRAM")
    
    # Training loop
    data_iter = iter(loader)
    train_time = 0.0; step = 0; final_loss = float("nan"); smooth_loss = 0.0
    
    while True:
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        batch = next(data_iter).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        
        train_loss = loss.item()
        if not math.isnan(train_loss): final_loss = train_loss
        torch.cuda.synchronize(); dt = time.time() - t0
        if step > 2: train_time += dt
        
        b = 0.95; smooth_loss = b*smooth_loss + (1-b)*train_loss
        debi = smooth_loss / (1-b**(step+1))
        
        # LR schedule with min_lr floor
        if step < WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / max(est_steps - WARMUP_STEPS, 1)
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg in opt.param_groups: pg["lr"] = lr
        
        if gpu_id == 0 and step % 200 == 0:
            rem = max(0, budget_s - train_time)
            print(f"[GPU 0] step {step:05d} | loss={debi:.4f} | lr={lr:.6f} | dt={dt*1000:.0f}ms | rem={rem/60:.0f}min")
            sys.stdout.flush()
        
        if math.isnan(train_loss): 
            print(f"[GPU {gpu_id}] NaN at step {step}!")
            results_dict[gpu_id] = ("NaN", step, train_time)
            return
        
        step += 1
        
        if step % 5000 == 0 and gpu_id == 0:
            val = evaluate_val_loss(model, BATCH_SIZE, device, SEQ_LEN)
            print(f"[GPU 0] ★ Step {step}: val_loss={val:.6f}")
            ckpt_path = f"checkpoints_phase4_2/stage2_seq8192_step{step}.pt"
            torch.save({"stage": 2, "seq_len": SEQ_LEN, "step": step, "val_loss": val,
                        "train_loss": final_loss,
                        "state_dict": {k:v for k,v in model.state_dict().items() if not k.startswith("nucel.")}},
                       ckpt_path)
            sys.stdout.flush()
        
        if train_time >= budget_s:
            print(f"[GPU {gpu_id}] Budget reached ({BUDGET_HOURS}h)")
            break
    
    results_dict[gpu_id] = (final_loss, step, train_time)
    
    if gpu_id == 0:
        # Save final checkpoint
        final_ckpt = CKPT_DIR / f"stage2_seq8192_gpu0_final.pt"
        torch.save({"stage": 2, "seq_len": SEQ_LEN, "step": step,
                    "train_loss": final_loss, "train_time": train_time,
                    "state_dict": {k:v for k,v in model.state_dict().items() if not k.startswith("nucel.")}},
                   final_ckpt)
        # Quick GB eval
        gb = evaluate_gb_sampled(model, device, SEQ_LEN)
        avg = sum(v for v in gb.values() if not math.isnan(v)) / max(sum(1 for v in gb.values() if not math.isnan(v)), 1)
        print(f"[GPU 0] GB avg: {avg:.4f}")
    
    print(f"[GPU {gpu_id}] Done: {step} steps in {train_time:.0f}s ({train_time/3600:.1f}h)")

if __name__ == "__main__":
    print("=== Phase 4.2 Stage 2: Dual GPU Training ===")
    print(f"bs={BATCH_SIZE} per GPU, budget={BUDGET_HOURS}h, lr={LR:.6f}\n")
    
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    results = manager.dict()
    
    processes = []
    for gpu_id in [0, 1]:
        p = mp.Process(target=worker, args=(gpu_id, results))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("\n=== Summary ===")
    for gpu_id in sorted(results.keys()):
        final_loss, step, train_time = results[gpu_id]
        print(f"GPU {gpu_id}: {step} steps, {train_time/3600:.1f}h, final_loss={final_loss}")
