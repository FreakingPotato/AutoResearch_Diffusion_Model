"""Phase 4.2 Stage 2: Training with DeepSpeed Zero0 + Wandb (simplified).

Usage:
    # Single GPU
    uv run python train.py

    # Dual GPU with DeepSpeed
    uv run torchrun --nproc_per_node=2 --master_port=29500 train.py

    # Dual GPU with DeepSpeed + Wandb
    WANDB_API_KEY=xxx uv run torchrun --nproc_per_node=2 train.py --use_wandb
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import math
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src import (
    DiffusionConfig,
    build_model,
    FullHg38Dataset,
    build_full_data,
    evaluate_val_loss,
    evaluate_gb_sampled,
)

# Paths
PROJECT_DIR = Path(__file__).parent
CKPT_DIR = PROJECT_DIR / "checkpoints_phase4_2"
STAGE1_CKPT = CKPT_DIR / "stage1_seq4096_best.pt"
DS_CONFIG = PROJECT_DIR / "configs" / "ds_config_zero0.json"

# Default config
DEFAULT_SEQ_LEN = 8192
DEFAULT_BATCH_SIZE = 3
DEFAULT_LR = 3e-4
DEFAULT_WARMUP_STEPS = 500
DEFAULT_MAX_STEPS = 16000
DEFAULT_HOURS = 4


def get_cosine_lr_with_min_lr(step, num_training_steps, warmup_steps, min_lr_ratio=0.1):
    """Cosine LR schedule with minimum LR floor."""
    if step < warmup_steps:
        return DEFAULT_LR * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
    return DEFAULT_LR * max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(progress))))


def load_stage1_checkpoint(model, device, ckpt_path=STAGE1_CKPT):
    """Load stage 1 checkpoint into stage 2 model."""
    if not ckpt_path.exists():
        print(f"Warning: Stage 1 checkpoint not found at {ckpt_path}")
        return

    print(f"Loading stage 1 checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    model_state = model.state_dict()

    # Load matching parameters
    loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(loaded)
    model.load_state_dict(model_state)
    print(f"Loaded {len(loaded)}/{len(state)} params from stage 1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dna-diffusion")
    parser.add_argument("--wandb_run_name", type=str, default="stage2")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Wandb initialization (only on main process)
    if args.use_wandb and local_rank == 0:
        import wandb
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if not wandb_api_key:
            print("Error: WANDB_API_KEY environment variable not set")
            print("Run: export WANDB_API_KEY=your_key && uv run torchrun --nproc_per_node=2 train.py --use_wandb")
            return

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
                "model": "NucEL+Mamba2",
                "world_size": world_size,
            }
        )

    if local_rank == 0:
        print("\n=== Starting Training ===")
        print(f"  Seq len: {args.seq_len}")
        print(f"  Batch size: {args.batch_size} per GPU")
        print(f"  LR: {args.lr}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Use Wandb: {args.use_wandb}")
        print(f"  World size: {world_size}")

    # Build model
    config = DiffusionConfig(
        seq_len=args.seq_len,
        attention_type="mamba2",
        d_state=64,
        d_conv=4,
        dropout=0.0,
    )
    model = build_model(config, device, mask_id=4)
    # Ensure model is on correct device and dtype
    model = model.to(device=device, dtype=torch.bfloat16)

    # Load stage 1 checkpoint
    load_stage1_checkpoint(model, device)
    # Ensure model is on correct device after checkpoint loading
    model = model.to(device=device, dtype=torch.bfloat16)

    # Prepare data
    build_full_data(args.seq_len)
    dataset = FullHg38Dataset("train", args.seq_len)
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    else:
        sampler = None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=2, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Training loop
    model.train()
    train_time = 0.0
    step = 0
    final_loss = float("nan")
    smooth_loss = 0.0

    data_iter = iter(loader)

    while step < args.max_steps:
        torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        # Get batch
        try:
            batch = next(data_iter).to(device)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step // max(len(dataset), 1))
            data_iter = iter(loader)
            batch = next(data_iter).to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Get LR
        lr = get_cosine_lr_with_min_lr(step, args.max_steps, args.warmup_steps, min_lr_ratio=0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 2:
            train_time += dt

        # Exponential moving average
        train_loss = loss.item()
        if not math.isnan(train_loss):
            final_loss = train_loss
        b = 0.95
        smooth_loss = b * smooth_loss + (1 - b) * train_loss
        debi = smooth_loss / (1 - b**(step + 1))

        # Print progress
        if local_rank == 0 and step % 200 == 0:
            rem = max(0, args.hours * 3600 - train_time)
            print(f"step {step:05d} | loss={debi:.4f} | lr={lr:.6f} | dt={dt*1000:.0f}ms | rem={rem/60:.0f}min")

        # Save checkpoint
        if step % 5000 == 0 and step > 0:
            if local_rank == 0:
                val_loss = evaluate_val_loss(model, args.batch_size, device, args.seq_len)
                print(f"  ★ Step {step}: val_loss={val_loss:.6f}")

                if args.use_wandb:
                    import wandb
                    wandb.log({"train_loss": debi, "val_loss": val_loss, "lr": lr}, step=step)

                ckpt_path = CKPT_DIR / f"stage2_seq8192_step{step}.pt"
                torch.save({
                    "stage": 2,
                    "seq_len": args.seq_len,
                    "step": step,
                    "val_loss": val_loss,
                    "train_loss": final_loss,
                    "state_dict": model.state_dict(),
                }, ckpt_path)
                print(f"  Checkpoint saved to {ckpt_path}")

        step += 1

    # Final evaluation
    if local_rank == 0:
        print("\n=== Final Evaluation ===")
        gb_results = evaluate_gb_sampled(model, device, args.seq_len)
        avg_acc = np.mean([v for v in gb_results.values() if not np.isnan(v)])

        results = {
            "train_loss": final_loss,
            "val_loss": evaluate_val_loss(model, args.batch_size, device, args.seq_len),
            "gb_avg": avg_acc,
            "gb_results": gb_results,
        }

        print(f"  GB avg: {avg_acc:.4f}")

        if args.use_wandb:
            import wandb
            wandb.log(results)
            wandb.finish()

        # Save final checkpoint
        final_ckpt_path = CKPT_DIR / f"stage2_final_trainer.pt"
        torch.save({
            "stage": 2,
            "seq_len": args.seq_len,
            "step": step,
            **results,
            "state_dict": model.state_dict(),
        }, final_ckpt_path)
        print(f"Final checkpoint saved to {final_ckpt_path}")
        print(f"\n=== Done: {step} steps in {train_time:.0f}s ({train_time/3600:.1f}h) ===")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
