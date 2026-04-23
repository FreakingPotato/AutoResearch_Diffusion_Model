"""Genomic benchmark evaluation functions."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from .tokenizer import get_tokenizer, tokenize_nt

# Genomic benchmarks
GENOMIC_BENCHMARKS = [
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "dummy_mouse_enhancers_ensembl",
]

DEFAULT_SEQ_LEN = 8192


def load_gb_dataset(benchmark_name):
    """Load genomic benchmark dataset."""
    base = Path.home() / '.genomic_benchmarks' / benchmark_name
    if not base.exists():
        return []
    results = []
    for split_name in ['train', 'test']:
        split_path = base / split_name
        if not split_path.exists():
            continue
        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
        if len(class_dirs) < 2:
            continue
        for label_idx, class_dir in enumerate(class_dirs):
            for f in sorted(class_dir.glob('*.txt')):
                seq = f.read_text().strip()
                if len(seq) >= 10:
                    results.append((seq, label_idx))
    return results


def tokenize_sequences(seqs, nt_to_id, unk_id, pad_id, seq_len):
    """Tokenize sequences to fixed length."""
    ids_list = []
    for seq in seqs:
        ids = tokenize_nt(seq, nt_to_id, unk_id)
        if len(ids) >= seq_len:
            ids_list.append(ids[:seq_len])
        else:
            ids_list.append(ids + [pad_id] * (seq_len - len(ids)))
    return torch.tensor(ids_list, dtype=torch.long)


@torch.no_grad()
def extract_embs(backbone_or_model, input_ids, device, batch_size=8):
    """Extract embeddings from model."""
    backbone_or_model.eval()
    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        if hasattr(backbone_or_model, 'get_embeddings'):
            emb = backbone_or_model.get_embeddings(batch, pool="mean")
        else:
            out = backbone_or_model(input_ids=batch)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            emb = hidden.mean(dim=1)
        all_embs.append(emb.float().cpu())
    return torch.cat(all_embs, dim=0)


@torch.enable_grad()
def train_probe(X_train, y_train, device, hidden_dim=512):
    """Train linear probe on embeddings."""
    n_classes = y_train.max().item() + 1
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 256
    for epoch in range(10):
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            opt.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            opt.step()
    return model


def eval_probe(probe, X_test, y_test, device):
    """Evaluate probe accuracy."""
    probe.eval()
    with torch.no_grad():
        correct = 0
        batch_size = 256
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size].to(device)
            logits = probe(batch_x)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
    return correct / len(X_test)


@torch.no_grad()
def evaluate_gb_sampled(model, device, seq_len=DEFAULT_SEQ_LEN):
    """Evaluate genomic benchmarks (20% sample for speed)."""
    _, nt_to_id, _, _, pad_id = get_tokenizer()

    results = {}
    for bench_name in GENOMIC_BENCHMARKS:
        data = load_gb_dataset(bench_name)
        if len(data) == 0:
            results[bench_name] = float('nan')
            continue

        import random
        random.seed(42)
        data = random.sample(data, int(len(data) * 0.2))

        seqs, labels = zip(*data)
        input_ids = tokenize_sequences(seqs, nt_to_id, 1, pad_id, seq_len)

        # Use backbone for embedding extraction
        embeddings = extract_embs(model.backbone, input_ids, device)

        labels_t = torch.tensor(labels, dtype=torch.long)
        n_train = len(embeddings) // 2
        X_train, y_train = embeddings[:n_train], labels_t[:n_train]
        X_test, y_test = embeddings[n_train:], labels_t[n_train:]

        probe = train_probe(X_train, y_train, device)
        acc = eval_probe(probe, X_test, y_test, device)
        results[bench_name] = acc
        print(f"    {bench_name}: {acc:.4f}")

    return results


@torch.no_grad()
def evaluate_val_loss(model, batch_size, device, seq_len=DEFAULT_SEQ_LEN):
    """Evaluate validation loss."""
    from .data import make_loader
    val_loader = make_loader(batch_size, "val", seq_len, num_workers=0)

    model.eval()
    total_loss = 0
    total_batches = 0

    schedule = None
    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch, schedule=schedule)
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)
