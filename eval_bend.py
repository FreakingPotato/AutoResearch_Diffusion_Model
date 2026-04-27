"""Evaluate on BEND tasks (histone_modification, chromatin_accessibility, cpg_methylation).

All three tasks have 512bp sequences — well within our 8192 context window.
Uses our NucEL-Mamba2 model with Linear Probe and Full Fine-tuning.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import time
import gc
from pathlib import Path
from functools import lru_cache

from src import build_model, DiffusionConfig
from src.tokenizer import get_tokenizer, tokenize_nt
from train import load_stage1_checkpoint

device = torch.device('cuda:0')

# --- Data loading ---
HG38_DIR = Path("/home/stark/.cache/dna-diffusion/raw")
BEND_DIR = Path("/home/stark/BEND/data")

@lru_cache(maxsize=24)
def load_chr_fa(chrom):
    """Load a chromosome fasta file and return the sequence."""
    # Strip 'chr' prefix if needed
    chrom_num = chrom.replace('chr', '')
    fa_path = HG38_DIR / f"hg38.{chrom_num}.fa"
    if not fa_path.exists():
        return None
    with open(fa_path) as f:
        lines = f.readlines()
    # Skip header line
    seq = ''.join(line.strip() for line in lines[1:])
    return seq.upper()

def get_sequence(chrom, start, end, strand='+'):
    """Get DNA sequence from our per-chromosome fasta files."""
    seq = load_chr_fa(chrom)
    if seq is None:
        return None
    subseq = seq[start:end]
    if strand == '-':
        comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        subseq = ''.join(comp.get(b, 'N') for b in reversed(subseq))
    return subseq

# --- Model loading ---
def load_stage2_model():
    config = DiffusionConfig(seq_len=512, attention_type="mamba2", d_state=64, d_conv=4, dropout=0.0)
    model = build_model(config, device)
    load_stage1_checkpoint(model, device, Path("checkpoints_phase4_2/stage1_seq4096_best.pt"))
    ckpt = torch.load("checkpoints_phase4_2/stage2_final_trainer.pt", map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    ms = model.state_dict()
    for k, v in state.items():
        if k in ms and v.shape == ms[k].shape:
            ms[k] = v
    model.load_state_dict(ms)
    return model

# --- BEND task data loading ---
def load_bend_task(task_name):
    """Load BEND task data. Returns list of (sequence, labels, split)."""
    bed_path = BEND_DIR / task_name / f"{task_name}.bed"
    df = pd.read_csv(bed_path, sep='\t')
    
    samples = []
    for _, row in df.iterrows():
        chrom = str(row.iloc[0])
        start = int(row.iloc[1])
        end = int(row.iloc[2])
        split = row.iloc[-1]
        
        # Get strand
        if 'strand' in df.columns:
            strand = row['strand']
        else:
            strand = '+'
        
        seq = get_sequence(chrom, start, end, strand)
        if seq is None or len(seq) != (end - start):
            continue
        
        # Get labels
        if 'label' in df.columns:
            label_str = str(row['label'])
            if label_str == 'nan' or label_str == '':
                labels = []
            else:
                try:
                    labels = list(map(int, label_str.split(',')))
                except ValueError:
                    labels = []
        else:
            labels = []
        
        samples.append({
            'sequence': seq,
            'labels': labels,
            'split': split,
            'chrom': chrom,
            'start': start,
            'end': end,
        })
    
    return samples

# --- Linear Probe evaluation ---
def linear_probe_eval(model, samples, task_name, n_classes):
    """Evaluate with linear probe (freeze backbone)."""
    _, nt_to_id, _, _, pad_id = get_tokenizer()
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    
    # Split data
    train_data = [s for s in samples if s['split'] == 'train']
    test_data = [s for s in samples if s['split'] == 'test']
    if not test_data:
        test_data = [s for s in samples if s['split'] == 'valid']
    
    random.seed(42)
    # Limit size for efficiency
    max_train = min(len(train_data), 5000)
    max_test = min(len(test_data), 1000)
    train_data = random.sample(train_data, max_train)
    test_data = random.sample(test_data, max_test)
    
    SL = 512
    BS = 16
    
    def get_embeddings(data_list):
        all_embs = []
        for s in data_list:
            ids = tokenize_nt(s['sequence'][:SL], nt_to_id, 1)
            ids = (ids[:SL] + [pad_id] * max(0, SL - len(ids)))[:SL]
            all_embs.append(ids)
        
        input_ids = torch.tensor(all_embs, dtype=torch.long)
        embs = []
        with torch.no_grad():
            for i in range(0, len(input_ids), BS):
                batch = input_ids[i:i+BS].to(device)
                out = model.backbone(input_ids=batch)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                embs.append(h.mean(dim=1).float().cpu())
        return torch.cat(embs, 0)
    
    train_embs = get_embeddings(train_data)
    test_embs = get_embeddings(test_data)
    
    # For binned tasks, use the labels directly (multi-hot averaged or first label)
    # BEND uses multi-label classification
    train_labels = []
    for s in train_data:
        if s['labels']:
            # Multi-hot encode
            label_vec = np.zeros(n_classes, dtype=np.float32)
            for l in s['labels']:
                if l < n_classes:
                    label_vec[l] = 1.0
            train_labels.append(label_vec)
        else:
            train_labels.append(np.zeros(n_classes, dtype=np.float32))
    train_labels = torch.tensor(np.array(train_labels))
    
    test_labels = []
    for s in test_data:
        if s['labels']:
            label_vec = np.zeros(n_classes, dtype=np.float32)
            for l in s['labels']:
                if l < n_classes:
                    label_vec[l] = 1.0
            test_labels.append(label_vec)
        else:
            test_labels.append(np.zeros(n_classes, dtype=np.float32))
    test_labels = torch.tensor(np.array(test_labels))
    
    # Multi-label linear probe
    probe = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_classes)
    ).float().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(10):
        probe.train()
        for i in range(0, len(train_embs), 128):
            bx = train_embs[i:i+128].to(device)
            by = train_labels[i:i+128].to(device)
            opt.zero_grad()
            criterion(probe(bx), by).backward()
            opt.step()
    
    # Evaluate with AUPRC (BEND's metric)
    probe.eval()
    with torch.no_grad():
        logits = probe(test_embs.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = test_labels.numpy()
    
    # Compute AUROC per class (BEND metric) + AUPRC
    from sklearn.metrics import roc_auc_score, average_precision_score
    aurocs = []
    auprcs = []
    for c in range(n_classes):
        if targets[:, c].sum() > 0 and targets[:, c].sum() < len(targets):
            try:
                auroc = roc_auc_score(targets[:, c], probs[:, c])
                aurocs.append(auroc)
                auprc = average_precision_score(targets[:, c], probs[:, c])
                auprcs.append(auprc)
            except:
                pass
    
    mean_auroc = np.mean(aurocs) if aurocs else 0.0
    mean_auprc = np.mean(auprcs) if auprcs else 0.0
    
    del train_embs, test_embs, probe
    gc.collect()
    torch.cuda.empty_cache()
    
    return mean_auroc, mean_auprc

# --- Main ---
if __name__ == "__main__":
    tasks = {
        'histone_modification': {'n_classes': 18, 'seq_len': 512},
        'cpg_methylation': {'n_classes': 7, 'seq_len': 512},
        'chromatin_accessibility': {'n_classes': 125, 'seq_len': 512},
    }
    
    # BEND paper baselines (from Table 4/5/6)
    # One-hot CNN baseline AUPRC values
    bend_baselines = {
        'histone_modification': {'onehot_auprc': None, 'nt_ms_auprc': None},
        'chromatin_accessibility': {'onehot_auprc': None, 'nt_ms_auprc': None},
        'cpg_methylation': {'onehot_auprc': None, 'nt_ms_auprc': None},
    }
    
    print("Loading model...")
    model = load_stage2_model()
    
    print(f"\n{'Task':<35} {'AUROC':>8} {'AUPRC':>8}")
    print("-" * 53)
    
    results = {}
    for task_name, task_info in tasks.items():
        print(f"\nLoading {task_name}...", flush=True)
        samples = load_bend_task(task_name)
        n_train = sum(1 for s in samples if s['split'] == 'train')
        n_test = sum(1 for s in samples if s['split'] == 'test')
        n_valid = sum(1 for s in samples if s['split'] == 'valid')
        print(f"  Samples: {len(samples)} total ({n_train} train, {n_test} test, {n_valid} valid)")
        
        if len(samples) == 0:
            print(f"  Skipping (no data)")
            continue
        
        auroc, auprc = linear_probe_eval(model, samples, task_name, task_info['n_classes'])
        results[task_name] = {'auroc': auroc, 'auprc': auprc}
        print(f"{task_name:<35} {auroc:>8.4f} {auprc:>8.4f}")
    
    print("=" * 53)
    
    # Save
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    np.save(results_dir / "bend_results.npy", results)
    print(f"\nSaved to {results_dir / 'bend_results.npy'}")
