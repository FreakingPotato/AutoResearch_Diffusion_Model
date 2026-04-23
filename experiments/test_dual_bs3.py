"""Test dual GPU with bs=3 per GPU."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import torch.multiprocessing as mp
from train_phase4_2 import Phase42Config, build_model, make_loader, build_full_data

def worker(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import importlib, train_phase4_2
    importlib.reload(train_phase4_2)
    
    torch.manual_seed(42 + gpu_id)
    torch.cuda.manual_seed(42 + gpu_id)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    
    cfg = Phase42Config(seq_len=8192, attention_type="mamba2", d_state=64, d_conv=4)
    model = train_phase4_2.build_model(cfg, device)
    
    ckpt = torch.load("checkpoints_phase4_2/stage1_seq4096_best.pt", map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    loaded = {k:v for k,v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(loaded)
    model.load_state_dict(model_state)
    
    BATCH_SIZE = 3
    GRAD_ACCUM = 1
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4)
    
    loader = train_phase4_2.make_loader(BATCH_SIZE, "train", 8192, num_workers=0)
    data_iter = iter(loader)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10):
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
    
    ms = (time.time() - t0) / 10 * 1000
    peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024
    toks = BATCH_SIZE * GRAD_ACCUM * 8192
    
    print(f"[GPU {gpu_id}] bs={BATCH_SIZE}, ga={GRAD_ACCUM}, VRAM={peak:.0f}MB, {ms:.0f}ms/step, {toks:,} toks/step")
    return peak, ms

if __name__ == "__main__":
    print("=== Dual GPU Benchmark: bs=3 per GPU ===")
    with mp.get_context('spawn').Pool(2) as pool:
        results = pool.map(worker, [0, 1])
    
    avg_ms = sum(r[1] for r in results) / 2
    total_toks_per_step = 2 * 3 * 8192  # 2 GPUs × bs=3
    total_toks_per_sec = total_toks_per_step * 1000 / avg_ms
    
    print(f"\n=== Summary ===")
    print(f"Avg: {avg_ms:.0f}ms/step")
    print(f"Total throughput: {total_toks_per_sec/1000:.1f}K toks/sec")
    print(f"vs single GPU bs=2: {total_toks_per_sec / (16384*1000/1141):.2f}x speedup")
    print(f"vs single GPU bs=3: {total_toks_per_sec / (24576*1000/852):.2f}x speedup")
