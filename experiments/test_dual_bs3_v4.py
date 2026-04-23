"""Dual GPU test with num_workers=1."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import torch.multiprocessing as mp
from train_phase4_2 import Phase42Config, build_model, make_loader, build_full_data

def worker(gpu_id, return_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    torch.manual_seed(42 + gpu_id)
    torch.cuda.manual_seed(42 + gpu_id)
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
    
    BATCH_SIZE = 3
    loader = make_loader(BATCH_SIZE, "train", 8192, num_workers=1)
    data_iter = iter(loader)
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4)
    
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(5):
        batch = next(data_iter).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch)
        loss.backward()
        opt.zero_grad(set_to_none=True)
    
    # Benchmark 20 steps
    model.train()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        batch = next(data_iter).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    torch.cuda.synchronize()
    
    ms = (time.time() - t0) / 20 * 1000
    peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024
    toks = BATCH_SIZE * 8192
    
    print(f"[GPU {gpu_id}] bs={BATCH_SIZE}, VRAM={peak:.0f}MB, {ms:.0f}ms/step, {toks:,} toks/step")
    return_dict[gpu_id] = (peak, ms, toks)

if __name__ == "__main__":
    print("=== Dual GPU: bs=3 per GPU, num_workers=1 ===")
    
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    
    processes = []
    for gpu_id in [0, 1]:
        p = mp.Process(target=worker, args=(gpu_id, return_dict))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print(f"\n=== Summary ===")
    if len(return_dict) == 2:
        results = {k: return_dict[k] for k in sorted(return_dict.keys())}
        avg_ms = sum(results[g][1] for g in results) / 2
        total_toks_per_step = sum(results[g][2] for g in results)
        total_toks_per_sec = total_toks_per_step * 1000 / avg_ms
        print(f"GPU 0: {results[0][0]:.0f}MB, {results[0][1]:.0f}ms/step ({results[0][2]*1000/results[0][1]/1000:.1f}K t/s)")
        print(f"GPU 1: {results[1][0]:.0f}MB, {results[1][1]:.0f}ms/step ({results[1][2]*1000/results[1][1]/1000:.1f}K t/s)")
        print(f"Total: {total_toks_per_sec/1000:.1f}K toks/sec")
        print(f"Speedup: {total_toks_per_sec / (24576*1000/852):.2f}x vs single GPU")
    else:
        print(f"Only {len(return_dict)} GPU(s) completed")
