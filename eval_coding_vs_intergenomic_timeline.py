"""Evaluate coding vs intergenomic across all checkpoints for timeline plot."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import random
import gc
from pathlib import Path

from src import build_model, DiffusionConfig
from src.eval import load_gb_dataset
from src.tokenizer import get_tokenizer, tokenize_nt
from train import load_stage1_checkpoint

# Extract step from filename
def extract_step(ckpt_path):
    name = ckpt_path.stem
    if "step" in name:
        try:
            return int(name.split("step")[1].split("_")[0])
        except:
            pass
    if "best" in name:
        return 10000  # Assume best is around 10k steps
    if "final" in name:
        return 16000  # Assume final is 16k steps
    return 0

# Define checkpoints
ckpt_dir = Path("checkpoints_phase4_2")

stage1_ckpts = [
    ("stage1_seq4096_best.pt", "Stage 1 (4096)"),
]

stage2_ckpts = [
    ("stage2_seq8192_step5000.pt", 5000),
    ("stage2_seq8192_step10000.pt", 10000),
    ("stage2_seq8192_step15000.pt", 15000),
    ("stage2_final_trainer.pt", 15800),  # Approximate
]

device = torch.device('cuda:0')
config = DiffusionConfig(seq_len=512, attention_type="mamba2", d_state=64, d_conv=4, dropout=0.0)

# Load dataset once
dataset_name = "demo_coding_vs_intergenomic_seqs"
print(f"Loading {dataset_name} dataset...")
data = load_gb_dataset(dataset_name)
random.seed(42)
sample_size = min(int(len(data) * 1.0), 2000)  # Full dataset
data = random.sample(data, sample_size)
seqs, labels = zip(*data)

_, nt_to_id, _, _, pad_id = get_tokenizer()
SL = 512
ids_list = []
for seq in seqs:
    ids = tokenize_nt(seq, nt_to_id, 1)
    ids_list.append((ids[:SL] + [pad_id] * max(0, SL - len(ids)))[:SL])
input_ids = torch.tensor(ids_list, dtype=torch.long)
labels_t = torch.tensor(labels, dtype=torch.long)
print(f"Dataset loaded: {len(input_ids)} samples")

# Results storage
results = []

# Evaluate Stage 1 checkpoints
for ckpt_name, label in stage1_ckpts:
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_name} ({label})")
    print('='*60)

    model = build_model(config, device)
    ckpt_path = ckpt_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    ms = model.state_dict()
    for k, v in state.items():
        if k in ms and v.shape == ms[k].shape:
            ms[k] = v
    model.load_state_dict(ms)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    step = ckpt.get('step', extract_step(ckpt_path))
    print(f"Loaded checkpoint (step ~{step})")

    # Extract embeddings
    all_embs = []
    BS = 16
    with torch.no_grad():
        for i in range(0, len(input_ids), BS):
            batch = input_ids[i:i+BS].to(device)
            out = model.backbone(input_ids=batch)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            all_embs.append(h.mean(dim=1).float().cpu())
    embeddings = torch.cat(all_embs, 0)

    # Linear probe
    n_train = len(embeddings) // 2
    n_classes = labels_t.max().item() + 1
    probe = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_classes)
    ).float().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-4)

    best_acc = 0
    for epoch in range(15):
        for i in range(0, n_train, 128):
            bx = embeddings[:n_train][i:i+128].to(device)
            by = labels_t[:n_train][i:i+128].to(device)
            opt.zero_grad()
            nn.CrossEntropyLoss()(probe(bx), by).backward()
            opt.step()

        # Val
        probe.eval()
        with torch.no_grad():
            logits = probe(embeddings[n_train:].to(device))
            acc = (logits.argmax(1) == labels_t[n_train:].to(device)).float().mean().item()
        if acc > best_acc:
            best_acc = acc
        probe.train()

    results.append({
        'stage': 1,
        'step': step,
        'accuracy': best_acc,
        'label': label,
    })
    print(f"Best accuracy: {best_acc:.4f}")

    del model, embeddings, probe
    gc.collect()
    torch.cuda.empty_cache()

# Evaluate Stage 2 checkpoints
for ckpt_name, step in stage2_ckpts:
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_name} (Stage 2)")
    print('='*60)

    model = build_model(config, device)

    # Load Stage 1 first
    load_stage1_checkpoint(model, device, ckpt_dir / "stage1_seq4096_best.pt")

    # Load Stage 2
    ckpt_path = ckpt_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    ms = model.state_dict()
    for k, v in state.items():
        if k in ms and v.shape == ms[k].shape:
            ms[k] = v
    model.load_state_dict(ms)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded checkpoint (step ~{step})")

    # Extract embeddings
    all_embs = []
    BS = 16
    with torch.no_grad():
        for i in range(0, len(input_ids), BS):
            batch = input_ids[i:i+BS].to(device)
            out = model.backbone(input_ids=batch)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            all_embs.append(h.mean(dim=1).float().cpu())
    embeddings = torch.cat(all_embs, 0)

    # Linear probe
    n_train = len(embeddings) // 2
    n_classes = labels_t.max().item() + 1
    probe = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_classes)
    ).float().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-4)

    best_acc = 0
    for epoch in range(15):
        for i in range(0, n_train, 128):
            bx = embeddings[:n_train][i:i+128].to(device)
            by = labels_t[:n_train][i:i+128].to(device)
            opt.zero_grad()
            nn.CrossEntropyLoss()(probe(bx), by).backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            logits = probe(embeddings[n_train:].to(device))
            acc = (logits.argmax(1) == labels_t[n_train:].to(device)).float().mean().item()
        if acc > best_acc:
            best_acc = acc
        probe.train()

    results.append({
        'stage': 2,
        'step': step,
        'accuracy': best_acc,
        'label': 'Stage 2 (8192)',
    })
    print(f"Best accuracy: {best_acc:.4f}")

    del model, embeddings, probe
    gc.collect()
    torch.cuda.empty_cache()

# Print summary
print(f"\n{'='*60}")
print("Summary: Coding vs Intergenomic Accuracy by Step")
print('='*60)
print(f"{'Stage':<10} {'Step':>8} {'Accuracy':>10}")
print('-'*28)
for r in sorted(results, key=lambda x: (x['stage'], x['step'])):
    print(f"{r['label']:<10} {r['step']:>8} {r['accuracy']:>10.4f}")

# Save results
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
np.save(results_dir / "coding_vs_intergenomic_timeline.npy", results)
print(f"\nResults saved to {results_dir / 'coding_vs_intergenomic_timeline.npy'}")

# Generate plot
import matplotlib.pyplot as plt

stage1_results = [r for r in results if r['stage'] == 1]
stage2_results = [r for r in results if r['stage'] == 2]

fig, ax = plt.subplots(figsize=(12, 6))

# Plot Stage 1
if stage1_results:
    s1_steps = sorted([r['step'] for r in stage1_results])
    s1_accs = [next(r['accuracy'] for r in stage1_results if r['step'] == s) for s in s1_steps]
    ax.plot(s1_steps, s1_accs, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='Stage 1 (seq_len=4096)')

# Plot Stage 2
if stage2_results:
    s2_steps = sorted([r['step'] for r in stage2_results])
    s2_accs = [next(r['accuracy'] for r in stage2_results if r['step'] == s) for s in s2_steps]
    ax.plot(s2_steps, s2_accs, 's-', color='#4ECDC4', linewidth=2, markersize=8, label='Stage 2 (seq_len=8192)')

ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Coding vs Intergenomic: Accuracy Evolution Across Checkpoints', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.65, 0.90)

# Add NucEL baseline
nucel_baseline = 0.9516
ax.axhline(y=nucel_baseline, color='#45B7D1', linestyle='--', linewidth=2, alpha=0.7, label=f'NucEL Baseline: {nucel_baseline:.3f}')

plt.tight_layout()
plt.savefig('coding_vs_intergenomic_timeline.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to coding_vs_intergenomic_timeline.png")
plt.close()
