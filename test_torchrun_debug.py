"""Debug torchrun dual GPU: find exactly where CPU tensors appear."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    print(f"[Rank {local_rank}] device={device}, world_size={world_size}")
    
    # Step 1: Build model
    from src import DiffusionConfig, build_model
    config = DiffusionConfig(seq_len=4096, attention_type="mamba2", d_state=64, d_conv=4)
    model = build_model(config, device, mask_id=4)
    model = model.to(device=device, dtype=torch.bfloat16)
    
    # Check all params
    bad = [(n, p.device) for n, p in model.named_parameters() if p.device.type != "cuda"]
    if bad:
        print(f"[Rank {local_rank}] BAD PARAMS after build: {bad[:5]}")
    else:
        print(f"[Rank {local_rank}] All params on {next(model.parameters()).device} ✅")
    
    # Step 2: Load checkpoint
    ckpt_path = "checkpoints_phase4_2/stage1_seq4096_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["state_dict"]
        model_state = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        
        # Check if loaded tensors are on correct device
        bad_ckpt = [(k, v.device) for k, v in loaded.items() if v.device.type != "cuda"]
        if bad_ckpt:
            print(f"[Rank {local_rank}] BAD CKPT tensors: {bad_ckpt[:3]}")
        
        model_state.update(loaded)
        model.load_state_dict(model_state)
        model = model.to(device=device, dtype=torch.bfloat16)
    
    bad2 = [(n, p.device) for n, p in model.named_parameters() if p.device.type != "cuda"]
    if bad2:
        print(f"[Rank {local_rank}] BAD PARAMS after ckpt: {bad2[:5]}")
    else:
        print(f"[Rank {local_rank}] All params on {next(model.parameters()).device} after ckpt ✅")
    
    # Step 3: Data
    from src import FullHg38Dataset, build_full_data
    build_full_data(4096)
    ds = FullHg38Dataset("train", 4096)
    
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=local_rank)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, sampler=sampler, num_workers=0, drop_last=True)
    else:
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    
    batch = next(iter(loader)).to(device)
    print(f"[Rank {local_rank}] Batch device: {batch.device}")
    
    # Step 4: Forward pass
    model.train()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model.compute_loss(batch)
    print(f"[Rank {local_rank}] Loss: {loss.item():.4f} ✅")
    
    # Step 5: Backward pass
    loss.backward()
    print(f"[Rank {local_rank}] Backward pass ✅")
    
    if world_size > 1:
        dist.destroy_process_group()
    
    print(f"[Rank {local_rank}] === ALL TESTS PASSED ===")

if __name__ == "__main__":
    main()
