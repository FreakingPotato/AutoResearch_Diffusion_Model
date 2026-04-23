"""
Re-evaluate Phase 4 experiments (GB linear probe).
The GB eval failed in the main run due to a runtime issue.
This script loads each checkpoint and re-runs GB evaluation.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from train_phase4 import (
    EXPERIMENTS, GENOMIC_BENCHMARKS, RESULTS_FILE, CKPT_DIR,
    get_tokenizer, tokenize_nt, NucELDiffusionV4, Phase4Config,
    DNA_ALPHABET, DEFAULT_SEQ_LEN, NUCEL_REPO, NUCEL_HIDDEN_SIZE,
    build_nucel_data, make_loader, UniformSchedule,
)
from train_phase4 import tokenize_sequences, load_gb_dataset, LinearProbe
from torch.utils.data import DataLoader

device = torch.device("cuda")

def extract_embs(model, input_ids, batch_size=8):
    model.eval()
    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model.nucel(input_ids=batch)
            hidden = out.last_hidden_state
            all_embs.append(hidden[:, 0].float().cpu())  # CLS
        gc.collect(); torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)

def train_probe(embs, labels, epochs=5, lr=1e-3):
    n_classes = len(set(labels.tolist()))
    probe = nn.Linear(embs.shape[1], n_classes)  # Simple linear, no nn.Module wrapper
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Move everything to CPU for probe training (avoids GPU memory issues)
    embs_d = embs.detach()
    labels_d = labels
    ds = torch.utils.data.TensorDataset(embs_d, labels_d)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = criterion(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe

def eval_probe(probe, embs, labels):
    probe.eval()
    with torch.no_grad():
        preds = probe(embs).argmax(dim=-1)
        return (preds == labels).float().mean().item()

def load_model_for_exp(exp_cfg):
    """Build model matching experiment config."""
    cfg = Phase4Config(
        seq_len=exp_cfg.get("seq_len", DEFAULT_SEQ_LEN),
        noise_steps=exp_cfg.get("noise_steps", 128),
        time_embed=exp_cfg.get("time_embed", "additive"),
        attention_type=exp_cfg.get("attention_type", "mamba2"),
        d_state=exp_cfg.get("d_state", 64),
        d_conv=exp_cfg.get("d_conv", 4),
        hybrid_every=exp_cfg.get("hybrid_every", 3),
        dropout=exp_cfg.get("dropout", 0.0),
    )
    
    from transformers import AutoModel
    _, _, mask_id, _, _ = get_tokenizer()
    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation='flash_attention_2').to(device)
    schedule = UniformSchedule(cfg.noise_steps, mask_id)
    model = NucELDiffusionV4(cfg, backbone, mask_id, schedule).to(device)
    
    # Load checkpoint (non-backbone weights only)
    exp_idx = EXPERIMENTS.index(exp_cfg)
    ckpts = sorted(CKPT_DIR.glob(f"{exp_idx:03d}_*.pt"))
    if ckpts:
        ckpt = torch.load(ckpts[0], map_location=device, weights_only=False)
        state = ckpt["state_dict"]
        model_state = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(loaded)
        model.load_state_dict(model_state)
        print(f"  Loaded {len(loaded)}/{len(state)} params from checkpoint")
    else:
        print("  WARNING: No checkpoint found, using random weights")
    
    # Apply freeze if needed
    freeze_n = exp_cfg.get("freeze_layers", 0)
    if freeze_n > 0 and model._layers:
        for i in range(min(freeze_n, len(model._layers))):
            for p in model._layers[i].parameters(): p.requires_grad = False
    
    return model

def eval_one_exp(exp_idx, sample_frac=0.20):
    exp_cfg = EXPERIMENTS[exp_idx]
    print(f"\n{'='*60}")
    print(f"EXP {exp_idx}: {exp_cfg['id']}")
    print(f"{'='*60}")
    
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    seq_len = exp_cfg.get("seq_len", DEFAULT_SEQ_LEN)
    
    results = {}
    for bname in GENOMIC_BENCHMARKS:
        try:
            data = load_gb_dataset(bname, sample_frac)
            if len(data) < 20:
                print(f"  {bname}: too few ({len(data)})")
                results[bname] = float("nan")
                continue
            
            seqs, labels = zip(*data)
            labels = torch.tensor(labels, dtype=torch.long)
            n = len(seqs)
            np.random.seed(42)
            perm = np.random.permutation(n)
            split = n // 2
            train_seqs = [seqs[i] for i in perm[:split]]
            test_seqs = [seqs[i] for i in perm[split:]]
            train_labels = labels[list(perm[:split])]
            test_labels = labels[list(perm[split:])]
            
            # Load model fresh for each benchmark to manage memory
            model = load_model_for_exp(exp_cfg)
            
            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, seq_len)
            train_embs = extract_embs(model, train_ids, batch_size=4)
            
            test_ids = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, seq_len)
            test_embs = extract_embs(model, test_ids, batch_size=4)
            
            del model; gc.collect(); torch.cuda.empty_cache()
            
            # Train probe on CPU
            probe = train_probe(train_embs, train_labels, epochs=5)
            acc = eval_probe(probe, test_embs, test_labels)
            results[bname] = acc
            
            print(f"  {bname}: acc={acc:.4f} ({len(train_seqs)} train, {len(test_seqs)} test)")
            
            del probe, train_embs, test_embs
            gc.collect(); torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  {bname}: FAILED ({e})")
            import traceback; traceback.print_exc()
            results[bname] = float("nan")
            gc.collect(); torch.cuda.empty_cache()
    
    return results

def update_results(exp_idx, gb_results):
    """Update the results TSV with GB accuracies."""
    if not RESULTS_FILE.exists():
        print("No results file found!")
        return
    
    lines = []
    with open(RESULTS_FILE) as f:
        lines = f.readlines()
    
    header = lines[0].strip().split("\t")
    gb_cols = {b: header.index(f"gb_acc_{b}") for b in GENOMIC_BENCHMARKS}
    
    for i in range(1, len(lines)):
        parts = lines[i].strip().split("\t")
        if len(parts) > 1 and parts[1] == str(exp_idx):
            for bname, col_idx in gb_cols.items():
                val = gb_results.get(bname, float("nan"))
                parts[col_idx] = "nan" if math.isnan(val) else f"{val:.6f}"
            lines[i] = "\t".join(parts) + "\n"
            break
    
    with open(RESULTS_FILE, "w") as f:
        f.writelines(lines)
    print(f"Updated results for exp {exp_idx}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, help="Single experiment to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all completed experiments")
    parser.add_argument("--sample-frac", type=float, default=0.20, help="GB sample fraction")
    args = parser.parse_args()
    
    if args.exp is not None:
        gb = eval_one_exp(args.exp, args.sample_frac)
        update_results(args.exp, gb)
    elif args.all:
        # Find completed experiments
        completed = []
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) > 1:
                        try: completed.append(int(parts[1]))
                        except: pass
        print(f"Found {len(completed)} completed experiments: {completed}")
        for idx in completed:
            gb = eval_one_exp(idx, args.sample_frac)
            update_results(idx, gb)
    else:
        print("Specify --exp N or --all")

if __name__ == "__main__":
    main()
