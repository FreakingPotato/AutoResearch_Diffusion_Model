"""
train_phase4.py — Phase 4: Mamba-2 × Muon Deep Optimization

14 experiments with dynamic dual-GPU scheduling.

Usage:
    # Dynamic scheduler (recommended): auto-distributes across both GPUs
    uv run python train_phase4.py --schedule

    # Manual single-experiment
    uv run python train_phase4.py --gpu 0 --exp 0

    # Manual sweep on one GPU
    uv run python train_phase4.py --gpu 0 --sweep --start 0 --end 7

    # v4.1: Full GB evaluation (no sampling) on completed experiments
    uv run python train_phase4.py --full-gb-eval
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import argparse
import traceback
import json
import subprocess
import sys
import threading
import queue
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCEL_REPO        = "FreakingPotato/NucEL"
NUCEL_HIDDEN_SIZE = 512
NUCEL_VOCAB_SIZE  = 27

DEFAULT_SEQ_LEN   = 1024
TARGET_TOKENS_PER_STEP = 32_768

DNA_ALPHABET      = "ACGT"

CACHE_DIR      = Path.home() / ".cache" / "dna-diffusion"
RAW_DIR        = CACHE_DIR / "raw"
NUCEL_DATA_DIR = CACHE_DIR / "nucel_data"
NUCEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DIR    = Path(__file__).parent
RESULTS_FILE   = PROJECT_DIR / "results_phase4.tsv"
CKPT_DIR       = PROJECT_DIR / "checkpoints_phase4"
CKPT_DIR.mkdir(exist_ok=True)

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

# ---------------------------------------------------------------------------
# NucEL tokenizer (singleton)
# ---------------------------------------------------------------------------

_TOK = _NT_TO_ID = _MASK_ID = _UNK_ID = _PAD_ID = None

def get_tokenizer():
    global _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID
    if _TOK is not None:
        return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID
    from huggingface_hub import hf_hub_download
    print(f"Loading NucEL vocab from {NUCEL_REPO} ...")
    vocab_path = hf_hub_download(NUCEL_REPO, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    class _Tok: pass
    tok = _Tok()
    tok.vocab = vocab; tok.vocab_size = len(vocab)
    tok.mask_token_id = vocab.get("\u2282", 4)
    tok.pad_token_id = vocab.get("[PAD]", 0)
    tok.unk_token_id = vocab.get("[UNK]", 1)
    def _encode(text, add_special_tokens=False):
        ids = []
        for c in text.upper():
            ids.append(vocab.get(c, vocab.get("[UNK]", 1)))
        return ids
    tok.encode = _encode
    _TOK = tok; _UNK_ID = tok.unk_token_id
    _MASK_ID = tok.mask_token_id; _PAD_ID = tok.pad_token_id
    nt_to_id = {}
    for nt in DNA_ALPHABET:
        ids = tok.encode(nt)
        nt_to_id[nt] = ids[0] if ids else _UNK_ID
        nt_to_id[nt.lower()] = nt_to_id[nt]
    for nt in "NnRrYyWwSsKkMmBbDdHhVv":
        nt_to_id[nt] = _UNK_ID
    _NT_TO_ID = nt_to_id
    print(f"  vocab_size={tok.vocab_size}, mask_id={_MASK_ID}")
    return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID

def tokenize_nt(seq, nt_to_id, unk_id):
    return [nt_to_id.get(c, unk_id) for c in seq]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_nucel_data(seq_len=DEFAULT_SEQ_LEN):
    train_path = NUCEL_DATA_DIR / f"train_{seq_len}.bin"
    val_path   = NUCEL_DATA_DIR / f"val_{seq_len}.bin"
    if train_path.exists() and val_path.exists():
        return
    _, nt_to_id, _, unk_id, _ = get_tokenizer()
    for split, chrom in [("train", "21"), ("val", "22")]:
        fa_path = RAW_DIR / f"hg38.{chrom}.fa"
        if not fa_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fa_path}")
        print(f"Tokenising chr{chrom} → {split} (seq_len={seq_len}) ...")
        all_ids, cur = [], []
        with open(fa_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if cur: all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))
                    cur = []
                else: cur.append(line)
        if cur: all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))
        n_seqs = len(all_ids) // seq_len
        arr = np.array(all_ids[:n_seqs*seq_len], dtype=np.uint16)
        arr.tofile(str(NUCEL_DATA_DIR / f"{split}_{seq_len}.bin"))
        print(f"  → {n_seqs:,} sequences saved")

class NucELDataset(Dataset):
    def __init__(self, split="train", seq_len=DEFAULT_SEQ_LEN):
        self.seq_len = seq_len
        path = NUCEL_DATA_DIR / f"{split}_{seq_len}.bin"
        if not path.exists(): build_nucel_data(seq_len)
        self.data = np.memmap(str(path), dtype=np.uint16, mode="r")
        self.n_seq = len(self.data) // seq_len
    def __len__(self): return self.n_seq
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return torch.from_numpy(self.data[s:s+self.seq_len].copy()).long()

def make_loader(batch_size, split="train", seq_len=DEFAULT_SEQ_LEN, num_workers=4):
    ds = NucELDataset(split, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"),
                      num_workers=num_workers, pin_memory=True, drop_last=True)

# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class UniformSchedule:
    def __init__(self, noise_steps=128, mask_id=1):
        self.noise_steps = noise_steps; self.mask_id = mask_id
    def sample_t(self, B, device):
        return torch.randint(1, self.noise_steps+1, (B,), device=device)
    def forward_process(self, x, t):
        prob = t.float() / self.noise_steps
        mask = torch.rand_like(x, dtype=torch.float) < prob.unsqueeze(1)
        xn = x.clone(); xn[mask] = self.mask_id
        return xn, mask

# ---------------------------------------------------------------------------
# Attention wrappers (reused from phase 3)
# ---------------------------------------------------------------------------

class PerformerAttentionWrapper(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        from performer_pytorch import SelfAttention
        self.attn = SelfAttention(dim=hidden_size, heads=num_heads, causal=False)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        out = self.attn(hidden_states)
        return self.norm(residual + out)

class RetNetAttentionWrapper(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        from fla.layers import MultiScaleRetention
        self.attn = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        result = self.attn(hidden_states)
        return self.norm(residual + result[0])

class GLAAttentionWrapper(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        from fla.layers import GatedLinearAttention
        self.attn = GatedLinearAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        result = self.attn(hidden_states)
        return self.norm(residual + result[0])

class Mamba2Wrapper(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, d_state=64, d_conv=4):
        super().__init__()
        from mamba_ssm import Mamba2
        self.attn = Mamba2(d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=2)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        out = self.attn(hidden_states)
        return self.norm(residual + out)

class IdentityAttentionWrapper(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, hidden_states, **kwargs):
        return self.norm(hidden_states)

def get_attention_wrapper(attn_type, hidden_size=512, num_heads=8, **kwargs):
    wrappers = {
        "standard": IdentityAttentionWrapper,
        "performer": PerformerAttentionWrapper,
        "retnet": RetNetAttentionWrapper,
        "gla": GLAAttentionWrapper,
        "mamba2": Mamba2Wrapper,
    }
    if attn_type not in wrappers:
        raise ValueError(f"Unknown attention: {attn_type}")
    return wrappers[attn_type](hidden_size, num_heads, **kwargs)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class Phase4Config:
    seq_len: int = DEFAULT_SEQ_LEN
    noise_steps: int = 128
    time_embed: str = "additive"
    attention_type: str = "mamba2"
    d_state: int = 64
    d_conv: int = 4
    hybrid_every: int = 3
    dropout: float = 0.0

class NucELDiffusionV4(nn.Module):
    def __init__(self, cfg, backbone, mask_id, schedule):
        super().__init__()
        self.cfg = cfg; self.nucel = backbone; self.mask_id = mask_id
        self.schedule = schedule; self.noise_steps = cfg.noise_steps
        self.hidden_size = NUCEL_HIDDEN_SIZE; self.vocab_size = NUCEL_VOCAB_SIZE
        self.te_type = cfg.time_embed

        self.time_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size*4), nn.SiLU(),
            nn.Linear(self.hidden_size*4, self.hidden_size),
        )

        self._layers = self._find_layers()
        n = len(self._layers) if self._layers else 0
        print(f"  Backbone layers: {n}")

        # Replace attention
        if cfg.attention_type != "standard" and self._layers is not None:
            self._replace_attention(cfg.attention_type, cfg.hybrid_every)

        # Time injection
        if cfg.time_embed == "additive":
            self.time_inject = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                for _ in range(max(n,1))
            ])
            for m in self.time_inject: nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        elif cfg.time_embed == "adaln":
            self.adaln_proj = nn.Linear(self.hidden_size, 2*self.hidden_size*max(n,1), bias=True)
            nn.init.zeros_(self.adaln_proj.weight); nn.init.zeros_(self.adaln_proj.bias)

        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        self.dropout_layer = nn.Dropout(cfg.dropout)

    def _find_layers(self):
        for path in ["encoder.layers","layers","model.layers","encoder.layer","transformer.h"]:
            obj = self.nucel; found = True
            for attr in path.split("."):
                if hasattr(obj, attr): obj = getattr(obj, attr)
                else: found = False; break
            if found and isinstance(obj, nn.ModuleList) and len(obj) > 2: return obj
        for _, m in self.nucel.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 5: return m
        return None

    def _replace_attention(self, attn_type, hybrid_every=3):
        if not self._layers: return
        for i, layer in enumerate(self._layers):
            is_hybrid = attn_type.startswith("hybrid_")
            actual_type = attn_type.replace("hybrid_", "") if is_hybrid else attn_type
            if is_hybrid and (i % hybrid_every == hybrid_every - 1): continue  # keep standard
            attn_mod = None
            for name, mod in layer.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    attn_mod = (name, mod); break
            if not attn_mod: continue
            try:
                if actual_type == "mamba2":
                    replacement = Mamba2Wrapper(self.hidden_size, 8,
                                                d_state=self.cfg.d_state, d_conv=self.cfg.d_conv)
                else:
                    replacement = get_attention_wrapper(actual_type, self.hidden_size, 8)
                parts = attn_mod[0].split('.')
                parent = layer
                for p in parts[:-1]: parent = getattr(parent, p)
                setattr(parent, parts[-1], replacement)
                print(f"  Layer {i}: replaced {attn_mod[0]} with {actual_type}")
            except Exception as e:
                print(f"  Layer {i}: failed ({e})")

    def _sinusoidal(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000)*torch.arange(half, device=t.device, dtype=torch.float32)/max(half-1,1))
        emb = t.float().unsqueeze(1)*freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        if dim % 2 == 1: emb = F.pad(emb, (0,1))
        return emb

    def forward(self, input_ids, t):
        t_hidden = self.time_proj(self._sinusoidal(t, self.hidden_size))
        handles = []
        if self._layers is not None:
            if self.te_type == "additive" and hasattr(self, 'time_inject'):
                for i, layer in enumerate(self._layers):
                    if i >= len(self.time_inject): break
                    delta = self.time_inject[i](t_hidden)
                    def _hook(mod, inp, out, d=delta):
                        h = out[0] if isinstance(out, tuple) else out
                        return (h+d.unsqueeze(1),)+out[1:] if isinstance(out, tuple) else h+d.unsqueeze(1)
                    handles.append(layer.register_forward_hook(_hook))
            elif self.te_type == "adaln" and hasattr(self, 'adaln_proj'):
                n = len(self._layers); raw = self.adaln_proj(t_hidden).view(-1,n,2,self.hidden_size)
                for i, layer in enumerate(self._layers):
                    if i >= n: break
                    s, sh = raw[:,i,0,:], raw[:,i,1,:]
                    def _hook(mod, inp, out, s=s, sh=sh):
                        h = out[0] if isinstance(out, tuple) else out
                        return (h*(1+s.unsqueeze(1))+sh.unsqueeze(1),)+out[1:] if isinstance(out, tuple) else h*(1+s.unsqueeze(1))+sh.unsqueeze(1)
                    handles.append(layer.register_forward_hook(_hook))
        try:
            out = self.nucel(input_ids=input_ids)
        finally:
            for h in handles: h.remove()
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden = self.dropout_layer(hidden)
        return self.lm_head(hidden)

    def compute_loss(self, x_clean, t=None):
        if t is None: t = self.schedule.sample_t(x_clean.shape[0], x_clean.device)
        x_noisy, mask = self.schedule.forward_process(x_clean, t)
        if mask.sum() == 0: return torch.tensor(0.0, device=x_clean.device, requires_grad=True)
        logits = self.forward(x_noisy, t)
        return F.cross_entropy(logits[mask], x_clean[mask])

    @torch.no_grad()
    def get_embeddings(self, input_ids, pool="cls"):
        out = self.nucel(input_ids=input_ids)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden[:, 0] if pool == "cls" else hidden.mean(dim=1)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    from transformers import AutoModel
    print(f"Loading NucEL ({NUCEL_REPO}) flash_attention_2 ...")
    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation='flash_attention_2').to(device)
    _, _, mask_id, _, _ = get_tokenizer()
    schedule = UniformSchedule(cfg.noise_steps, mask_id)
    return NucELDiffusionV4(cfg, backbone, mask_id, schedule).to(device)

# ---------------------------------------------------------------------------
# GB evaluation
# ---------------------------------------------------------------------------

def load_gb_dataset(benchmark_name, sample_frac=0.20, max_seqs=2000):
    base = Path.home() / '.genomic_benchmarks' / benchmark_name
    if not base.exists(): return []
    results = []
    for split_name in ['train', 'test']:
        split_path = base / split_name
        if not split_path.exists(): continue
        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
        if len(class_dirs) < 2: continue
        for label_idx, class_dir in enumerate(class_dirs):
            files = sorted(class_dir.glob('*.txt'))
            n = min(max(1, int(len(files)*sample_frac)), max_seqs//(2*len(class_dirs)))
            np.random.seed(42)
            chosen = np.random.choice(len(files), min(n, len(files)), replace=False)
            for fi in chosen:
                seq = files[fi].read_text().strip()
                if len(seq) >= 10: results.append((seq, label_idx))
    return results

def load_gb_dataset_full(benchmark_name):
    """Load FULL GB dataset (no sampling)."""
    base = Path.home() / '.genomic_benchmarks' / benchmark_name
    if not base.exists(): return []
    results = []
    for split_name in ['train', 'test']:
        split_path = base / split_name
        if not split_path.exists(): continue
        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
        if len(class_dirs) < 2: continue
        for label_idx, class_dir in enumerate(class_dirs):
            for f in sorted(class_dir.glob('*.txt')):
                seq = f.read_text().strip()
                if len(seq) >= 10: results.append((seq, label_idx))
    return results

def tokenize_sequences(seqs, nt_to_id, unk_id, pad_id, seq_len):
    ids_list = []
    for seq in seqs:
        ids = tokenize_nt(seq, nt_to_id, unk_id)
        ids = ids[:seq_len] if len(ids) >= seq_len else ids + [pad_id]*(seq_len-len(ids))
        ids_list.append(ids)
    return torch.tensor(ids_list, dtype=torch.long)

@torch.no_grad()
def extract_embeddings(model, input_ids, device, batch_size=16, pool="cls"):
    model.eval()
    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            embs = model.get_embeddings(batch, pool=pool)
        all_embs.append(embs.float().cpu())
    return torch.cat(all_embs, dim=0)

class LinearProbe(nn.Module):
    def __init__(self, dim, n_classes=2):
        super().__init__(); self.fc = nn.Linear(dim, n_classes)
    def forward(self, x): return self.fc(x)

def train_linear_probe(embs, labels, epochs=5, lr=1e-3):
    """Train linear probe on CPU to avoid GPU memory conflicts."""
    embs = embs.detach().float().cpu()
    labels = labels.cpu()
    n_classes = len(set(labels.tolist()))
    probe = nn.Linear(embs.shape[1], n_classes)  # plain nn.Linear, no wrapper
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(embs, labels)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = criterion(probe(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe

def eval_probe(probe, embs, labels):
    probe.eval()
    with torch.no_grad():
        preds = probe(embs.float().cpu()).argmax(dim=-1)
        return (preds == labels.cpu()).float().mean().item()

@torch.no_grad()
def evaluate_gb(model, device, seq_len=DEFAULT_SEQ_LEN, sample_frac=0.20, max_seqs=2000):
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    results = {}
    for bname in GENOMIC_BENCHMARKS:
        try:
            data = load_gb_dataset(bname, sample_frac, max_seqs)
            if len(data) < 20:
                print(f"    {bname}: too few ({len(data)})"); results[bname] = float("nan"); continue
            seqs, labels = zip(*data); labels = torch.tensor(labels, dtype=torch.long)
            n = len(seqs); np.random.seed(42)
            perm = np.random.permutation(n); split = n//2
            train_seqs = [seqs[i] for i in perm[:split]]
            test_seqs = [seqs[i] for i in perm[split:]]
            train_labels = labels[list(perm[:split])]; test_labels = labels[list(perm[split:])]
            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, seq_len)
            test_ids = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, seq_len)
            model.eval()
            train_embs = extract_embeddings(model, train_ids, device).to(device)
            test_embs = extract_embeddings(model, test_ids, device).to(device)
            probe = train_linear_probe(train_embs, train_labels)
            acc = eval_probe(probe, test_embs, test_labels)
            results[bname] = acc
            print(f"    {bname}: acc={acc:.4f} ({len(train_seqs)} train, {len(test_seqs)} test)")
            del probe, train_embs, test_embs; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {bname}: failed ({e})"); traceback.print_exc()
            results[bname] = float("nan")
    model.train()
    return results

@torch.no_grad()
def evaluate_gb_full(model, device, seq_len=DEFAULT_SEQ_LEN):
    """Full GB eval — NO sampling, uses entire dataset."""
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    results = {}
    for bname in GENOMIC_BENCHMARKS:
        try:
            data = load_gb_dataset_full(bname)
            if len(data) < 100:
                print(f"    {bname}: too few ({len(data)})"); results[bname] = float("nan"); continue
            seqs, labels = zip(*data); labels = torch.tensor(labels, dtype=torch.long)
            # Use official train/test split (first half from train folder, second from test)
            train_data = [(s,l) for s,l in data if len(s) > 0]
            n = len(train_data)
            # Split 80/20
            np.random.seed(42); perm = np.random.permutation(n)
            sp = int(n * 0.8)
            train_seqs = [train_data[i][0] for i in perm[:sp]]
            test_seqs = [train_data[i][0] for i in perm[sp:]]
            train_labels = labels[list(perm[:sp])]; test_labels = labels[list(perm[sp:])]
            print(f"    {bname}: {len(train_seqs)} train, {len(test_seqs)} test")
            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, seq_len)
            test_ids = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, seq_len)
            model.eval()
            train_embs = extract_embeddings(model, train_ids, device, batch_size=8).to(device)
            test_embs = extract_embeddings(model, test_ids, device, batch_size=8).to(device)
            probe = train_linear_probe(train_embs, train_labels, epochs=5, lr=1e-3)
            acc = eval_probe(probe, test_embs, test_labels)
            results[bname] = acc
            print(f"    {bname}: FULL acc={acc:.4f}")
            del probe, train_embs, test_embs; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {bname}: failed ({e})"); traceback.print_exc()
            results[bname] = float("nan")
    return results

@torch.no_grad()
def evaluate_val_loss(model, batch_size, device, seq_len=DEFAULT_SEQ_LEN):
    model.eval()
    loader = make_loader(min(batch_size, 4), "val", seq_len, num_workers=0)
    total, n_ok = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= 50: break
        batch = batch.to(device)
        t = torch.randint(1, model.noise_steps+1, (batch.shape[0],), device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(batch, t)
        if not loss.isnan(): total += loss.item(); n_ok += 1
    model.train()
    return total / max(n_ok, 1)

# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(name, params, lr, wd):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    elif name == "muon":
        from muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam
        params_list = list(params)
        params_2d = [p for p in params_list if p.ndim >= 2]
        params_1d = [p for p in params_list if p.ndim < 2]
        if params_2d and params_1d:
            return SingleDeviceMuonWithAuxAdam([
                dict(params=params_2d, lr=lr, momentum=0.95, weight_decay=wd, use_muon=True),
                dict(params=params_1d, lr=lr*0.5, betas=(0.9,0.95), eps=1e-10, weight_decay=wd, use_muon=False),
            ])
        elif params_2d:
            return SingleDeviceMuon(params_2d, lr=lr, momentum=0.95, weight_decay=wd)
        else:
            return torch.optim.AdamW(params_list, lr=lr, weight_decay=wd, betas=(0.9,0.95))
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9,0.95))

# ---------------------------------------------------------------------------
# Experiment definitions (14 experiments)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # A组: Mamba-2 + Muon lr sweep
    {"id": "mamba2_muon_lr002",  "optimizer": "muon", "lr": 0.02,  "time_embed": "additive", "attention_type": "mamba2", "d_state": 64, "d_conv": 4},
    {"id": "mamba2_muon_lr01",   "optimizer": "muon", "lr": 0.01,  "time_embed": "additive", "attention_type": "mamba2", "d_state": 64, "d_conv": 4},
    {"id": "mamba2_muon_lr005",  "optimizer": "muon", "lr": 0.005, "time_embed": "additive", "attention_type": "mamba2", "d_state": 64, "d_conv": 4},
    {"id": "mamba2_muon_lr03",   "optimizer": "muon", "lr": 0.03,  "time_embed": "additive", "attention_type": "mamba2", "d_state": 64, "d_conv": 4},
    {"id": "mamba2_muon_state128","optimizer":"muon",  "lr": 0.01,  "time_embed": "additive", "attention_type": "mamba2", "d_state":128, "d_conv": 4},
    {"id": "mamba2_muon_conv8",  "optimizer": "muon",  "lr": 0.01,  "time_embed": "additive", "attention_type": "mamba2", "d_state": 64, "d_conv": 8},

    # B组: Time embedding variants
    {"id": "mamba2_muon_adaln",       "optimizer": "muon",  "lr": 0.01,  "time_embed": "adaln", "attention_type": "mamba2", "d_state": 64, "d_conv": 4},
    {"id": "mamba2_adamw",            "optimizer": "adamw", "lr": 3e-4,  "time_embed": "additive","attention_type": "mamba2", "d_state": 64, "d_conv": 4},
    {"id": "mamba2_muon_adaln_lr005", "optimizer": "muon",  "lr": 0.005, "time_embed": "adaln", "attention_type": "mamba2", "d_state": 64, "d_conv": 4},

    # C组: Hybrid + training
    {"id": "hybrid_mamba2_muon",  "optimizer": "muon",  "lr": 0.01,  "time_embed": "additive", "attention_type": "hybrid_mamba2", "d_state": 64, "d_conv": 4, "hybrid_every": 3},
    {"id": "mamba2_muon_long",    "optimizer": "muon",  "lr": 0.01,  "time_embed": "additive", "attention_type": "mamba2", "d_state": 64, "d_conv": 4, "time_multiplier": 2.0},
    {"id": "mamba2_muon_freeze10","optimizer": "muon",  "lr": 0.01,  "time_embed": "additive", "attention_type": "mamba2", "d_state": 64, "d_conv": 4, "freeze_layers": 10},

    # D组: Other attention + Muon
    {"id": "retnet_muon",     "optimizer": "muon", "lr": 0.01,  "time_embed": "additive", "attention_type": "retnet"},
    {"id": "performer_muon",  "optimizer": "muon", "lr": 0.01,  "time_embed": "additive", "attention_type": "performer"},
]

assert len(EXPERIMENTS) == 14

# Defaults for experiments
for e in EXPERIMENTS:
    e.setdefault("noise_steps", 128)
    e.setdefault("dropout", 0.0)
    e.setdefault("seq_len", DEFAULT_SEQ_LEN)
    e.setdefault("batch_size", 8)
    e.setdefault("weight_decay", 0.01)
    e.setdefault("d_state", 64)
    e.setdefault("d_conv", 4)
    e.setdefault("hybrid_every", 3)
    e.setdefault("time_multiplier", 1.0)
    e.setdefault("freeze_layers", 0)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def init_results():
    if RESULTS_FILE.exists(): return
    gb_cols = [f"gb_acc_{b}" for b in GENOMIC_BENCHMARKS]
    header = ["exp_id","exp_idx","timestamp","optimizer","lr","time_embed","attention_type",
              "d_state","d_conv","total_params_M","trainable_params_M","train_steps",
              "training_seconds","total_seconds","final_train_loss","val_loss","peak_vram_mb"] + gb_cols
    with open(RESULTS_FILE, "w") as f: f.write("\t".join(header)+"\n")

def log_result(idx, cfg, info, gb):
    def fmt(v):
        if isinstance(v, float): return "nan" if math.isnan(v) else f"{v:.6f}"
        return str(v)
    row = [cfg["id"], str(idx), datetime.now().isoformat(), cfg["optimizer"],
           str(cfg["lr"]), cfg["time_embed"], cfg["attention_type"],
           str(cfg.get("d_state",64)), str(cfg.get("d_conv",4)),
           f"{info.get('total_M',0):.1f}", f"{info.get('trainable_M',0):.1f}",
           str(info.get("steps",0)), f"{info.get('train_secs',0):.1f}",
           f"{info.get('total_secs',0):.1f}", fmt(info.get("final_loss",float("nan"))),
           fmt(info.get("val_loss",float("nan"))), f"{info.get('peak_vram_mb',0):.0f}"
    ] + [fmt(gb.get(b,float("nan"))) for b in GENOMIC_BENCHMARKS]
    with open(RESULTS_FILE, "a") as f: f.write("\t".join(row)+"\n")
    print(f"Logged → {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp_idx, exp_cfg, gpu_id=None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] ", end="")

    print(f"\n{'='*70}\nEXPERIMENT {exp_idx}: {exp_cfg['id']}\n{'='*70}")
    for k, v in exp_cfg.items():
        if k != "id": print(f"  {k}: {v}")
    sys.stdout.flush()

    t_wall_start = time.time()
    torch.manual_seed(42+exp_idx); torch.cuda.manual_seed(42+exp_idx)
    torch.set_float32_matmul_precision("high"); torch.cuda.empty_cache()
    device = torch.device("cuda")

    seq_len = exp_cfg.get("seq_len", DEFAULT_SEQ_LEN)
    build_nucel_data(seq_len)

    cfg = Phase4Config(
        seq_len=seq_len, noise_steps=exp_cfg.get("noise_steps", 128),
        time_embed=exp_cfg.get("time_embed", "additive"),
        attention_type=exp_cfg.get("attention_type", "mamba2"),
        d_state=exp_cfg.get("d_state", 64), d_conv=exp_cfg.get("d_conv", 4),
        hybrid_every=exp_cfg.get("hybrid_every", 3),
        dropout=exp_cfg.get("dropout", 0.0),
    )
    model = build_model(cfg, device)

    # Freeze layers
    freeze_n = exp_cfg.get("freeze_layers", 0)
    if freeze_n > 0 and model._layers:
        layers = model._layers
        for i in range(min(freeze_n, len(layers))):
            for p in layers[i].parameters(): p.requires_grad = False
        for attr in ["embeddings","embed_tokens"]:
            if hasattr(model.nucel, attr):
                for p in getattr(model.nucel, attr).parameters(): p.requires_grad = False
        print(f"  Froze bottom {freeze_n} layers")

    total_p, trainable_p = model.count_params()
    print(f"  Total: {total_p/1e6:.1f}M, Trainable: {trainable_p/1e6:.1f}M")
    sys.stdout.flush()

    # Optimizer
    lr = exp_cfg.get("lr", 0.01); wd = exp_cfg.get("weight_decay", 0.01)
    opt_name = exp_cfg.get("optimizer", "adamw")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(opt_name, trainable_params, lr, wd)

    # Data
    bs = exp_cfg.get("batch_size", 8)
    train_loader = make_loader(bs, "train", seq_len)
    toks_per_step = bs * seq_len
    grad_accum = max(1, TARGET_TOKENS_PER_STEP // toks_per_step)

    # Speed benchmark
    model.train(); data_iter = iter(train_loader)
    torch.cuda.synchronize(); bench_start = time.time()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            try: batch = next(data_iter)
            except StopIteration: break
            batch = batch.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model.compute_loss(batch) / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0); opt.step()
    torch.cuda.synchronize()
    bench_time = time.time() - bench_start
    ms_per_step = bench_time / 5 * 1000

    n_batches = len(train_loader)
    est_steps_per_epoch = n_batches // grad_accum
    est_epoch_time = est_steps_per_epoch * ms_per_step / 1000
    time_mult = exp_cfg.get("time_multiplier", 1.0)
    time_budget = min(max(est_epoch_time * 1.3 + 300, 600), 7200) * time_mult
    print(f"  Speed: {ms_per_step:.0f}ms/step, ~{est_epoch_time:.0f}min/epoch, budget={time_budget/60:.0f}min")
    sys.stdout.flush()

    # Training loop
    data_iter = iter(train_loader); train_time = 0.0; step = 0
    final_loss = float("nan"); smooth_loss = 0.0
    warmup = max(50, est_steps_per_epoch // 20)

    while True:
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            try: batch = next(data_iter)
            except StopIteration: data_iter = iter(train_loader); batch = next(data_iter)
            batch = batch.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model.compute_loss(batch) / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0); opt.step()

        train_loss = loss.item() * grad_accum
        if not math.isnan(train_loss): final_loss = train_loss
        torch.cuda.synchronize(); dt = time.time() - t0
        if step > 2: train_time += dt

        b = 0.95; smooth_loss = b*smooth_loss + (1-b)*train_loss
        debi = smooth_loss / (1-b**(step+1))

        # LR schedule
        if step < warmup: lr_s = step / warmup
        else:
            progress = (step-warmup) / max(est_steps_per_epoch-warmup, 1)
            lr_s = 0.5*(1+math.cos(math.pi*min(progress,1.0)))
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_s

        if step % 100 == 0:
            rem = max(0, time_budget - train_time)
            print(f"  step {step:05d} | loss={debi:.4f} | lr={lr*lr_s:.2e} | dt={dt*1000:.0f}ms | rem={rem:.0f}s")
            sys.stdout.flush()

        if math.isnan(train_loss): print("  NaN — stopping"); break
        step += 1
        if train_time >= time_budget: print(f"  Budget reached ({time_budget/60:.0f}min)"); break

    print(f"  Training done: {step} steps in {train_time:.0f}s")
    sys.stdout.flush()

    del opt; gc.collect(); torch.cuda.empty_cache()

    # Validation
    print("── Validation ──"); sys.stdout.flush()
    try: val_loss = evaluate_val_loss(model, bs, device, seq_len)
    except Exception as e: print(f"  Val failed: {e}"); val_loss = float("nan")

    # GB
    print("── GB Linear Probe ──"); sys.stdout.flush()
    try: gb = evaluate_gb(model, device, seq_len)
    except Exception as e: print(f"  GB failed: {e}"); traceback.print_exc()
    gb = {b: float("nan") for b in GENOMIC_BENCHMARKS}

    # Checkpoint
    ckpt = CKPT_DIR / f"{exp_idx:03d}_{exp_cfg['id']}.pt"
    try:
        torch.save({"exp_idx":exp_idx, "config":exp_cfg,
                    "state_dict":{k:v for k,v in model.state_dict().items() if not k.startswith("nucel.")},
                    "val_loss":val_loss, "gb":gb, "final_loss":final_loss, "step":step}, ckpt)
    except: pass

    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    info = {"total_M":total_p/1e6, "trainable_M":trainable_p/1e6, "steps":step,
            "train_secs":train_time, "total_secs":time.time()-t_wall_start,
            "final_loss":final_loss, "val_loss":val_loss, "peak_vram_mb":peak_vram}
    log_result(exp_idx, exp_cfg, info, gb)

    del model; gc.collect(); torch.cuda.empty_cache()
    return gb

# ---------------------------------------------------------------------------
# Dynamic GPU scheduler
# ---------------------------------------------------------------------------

def run_scheduler():
    """Dynamically assign experiments to whichever GPU is free."""
    import subprocess, threading, queue

    init_results()

    # Track completed experiments from results file
    completed = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split("\t")
                if len(parts) > 1:
                    try: completed.add(int(parts[1]))
                    except: pass

    # Build work queue
    work_queue = queue.Queue()
    for i, cfg in enumerate(EXPERIMENTS):
        if i not in completed:
            work_queue.put(i)

    total = len(EXPERIMENTS); done = len(completed)
    print(f"Scheduler: {total-len(completed)} remaining, {len(completed)} already done")

    gpu_locks = {0: threading.Lock(), 1: threading.Lock()}
    gpu_busy = {0: False, 1: False}
    results_lock = threading.Lock()

    def gpu_worker(gpu_id):
        nonlocal done
        while True:
            try:
                exp_idx = work_queue.get_nowait()
            except queue.Empty:
                break

            cfg = EXPERIMENTS[exp_idx]
            print(f"\n🚀 [GPU {gpu_id}] Starting exp {exp_idx}: {cfg['id']}")
            sys.stdout.flush()

            # Run as subprocess to ensure clean GPU state
            cmd = [
                sys.executable, "train_phase4.py",
                "--gpu", str(gpu_id), "--exp", str(exp_idx)
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(cmd, cwd=str(PROJECT_DIR), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            # Stream output
            for line in proc.stdout:
                print(f"  [GPU{gpu_id}] {line.decode().rstrip()}")
                sys.stdout.flush()

            proc.wait()
            status = "✓" if proc.returncode == 0 else "✗"
            with results_lock: done += 1
            print(f"\n{status} [GPU {gpu_id}] Exp {exp_idx} done ({done}/{total})")
            sys.stdout.flush()

            work_queue.task_done()

    # Launch one thread per GPU
    threads = []
    for gpu_id in [0, 1]:
        t = threading.Thread(target=gpu_worker, args=(gpu_id,))
        t.start(); threads.append(t)

    for t in threads: t.join()
    print(f"\n{'='*70}\nAll {total} experiments complete!")

# ---------------------------------------------------------------------------
# v4.1: Full GB evaluation
# ---------------------------------------------------------------------------

def run_full_gb_eval():
    """v4.1: Run full GB evaluation (no sampling) on best experiment + NucEL baseline."""
    import glob

    # Find best experiment by average GB acc
    best_idx = None; best_acc = -1
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 17: continue
            accs = []
            for v in parts[17:]:
                try: accs.append(float(v))
                except: pass
            avg = sum(accs)/len(accs) if accs else 0
            if avg > best_acc: best_acc = avg; best_idx = int(parts[1])

    if best_idx is None:
        print("No results found!"); return

    print(f"v4.1: Full GB eval for best experiment (idx={best_idx}, avg_acc={best_acc:.4f})")

    # Load checkpoint
    ckpts = sorted(CKPT_DIR.glob(f"{best_idx:03d}_*.pt"))
    if not ckpts:
        print("No checkpoint found!"); return

    device = torch.device("cuda")
    cfg_dict = EXPERIMENTS[best_idx]
    cfg = Phase4Config(
        attention_type=cfg_dict.get("attention_type","mamba2"),
        d_state=cfg_dict.get("d_state",64), d_conv=cfg_dict.get("d_conv",4),
        time_embed=cfg_dict.get("time_embed","additive"),
        hybrid_every=cfg_dict.get("hybrid_every",3),
    )
    model = build_model(cfg, device)

    # Load non-backbone weights
    ckpt = torch.load(ckpts[0], map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(loaded)
    model.load_state_dict(model_state)
    print(f"  Loaded {len(loaded)}/{len(state)} params from checkpoint")

    # Full GB eval
    print("\n── Full GB Evaluation (no sampling) ──")
    gb_full = evaluate_gb_full(model, device)
    print("\n  Results:")
    avg = 0; n = 0
    for b, acc in gb_full.items():
        print(f"    {b}: {acc:.4f}")
        if not math.isnan(acc): avg += acc; n += 1
    print(f"  Average: {avg/max(n,1):.4f}")

    # Save
    full_results = PROJECT_DIR / "results_phase4_full_gb.tsv"
    with open(full_results, "w") as f:
        f.write("benchmark\tsampled_acc\tfull_acc\n")
        for b in GENOMIC_BENCHMARKS:
            f.write(f"{b}\t{ckpt.get('gb',{}).get(b,float('nan'))}\t{gb_full.get(b,float('nan'))}\n")
    print(f"\nSaved → {full_results}")

    # Also eval frozen NucEL baseline
    print("\n── Frozen NucEL Baseline (full GB) ──")
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation='flash_attention_2').to(device)
    backbone.eval()
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()

    nucel_results = {}
    for bname in GENOMIC_BENCHMARKS:
        try:
            data = load_gb_dataset_full(bname)
            if len(data) < 100:
                nucel_results[bname] = float("nan"); continue
            seqs, labels = zip(*data); labels = torch.tensor(labels, dtype=torch.long)
            n = len(data); np.random.seed(42); perm = np.random.permutation(n)
            sp = int(n*0.8)
            train_seqs = [data[i][0] for i in perm[:sp]]
            test_seqs = [data[i][0] for i in perm[sp:]]
            train_labels = labels[list(perm[:sp])]; test_labels = labels[list(perm[sp:])]
            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, DEFAULT_SEQ_LEN)
            test_ids = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, DEFAULT_SEQ_LEN)

            all_embs = []
            for i in range(0, len(train_ids), 8):
                batch = train_ids[i:i+8].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = backbone(input_ids=batch)
                    all_embs.append(out.last_hidden_state[:,0].float().cpu())
            train_embs = torch.cat(all_embs).to(device)

            all_embs = []
            for i in range(0, len(test_ids), 8):
                batch = test_ids[i:i+8].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = backbone(input_ids=batch)
                    all_embs.append(out.last_hidden_state[:,0].float().cpu())
            test_embs = torch.cat(all_embs).to(device)

            probe = train_linear_probe(train_embs, train_labels, epochs=5)
            acc = eval_probe(probe, test_embs, test_labels)
            nucel_results[bname] = acc
            print(f"    {bname}: {acc:.4f} ({len(train_seqs)} train, {len(test_seqs)} test)")
            del probe, train_embs, test_embs; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {bname}: failed ({e})"); nucel_results[bname] = float("nan")

    del backbone; gc.collect(); torch.cuda.empty_cache()

    # Save NucEL baseline
    nucel_file = PROJECT_DIR / "results_nucel_full_gb.tsv"
    with open(nucel_file, "w") as f:
        f.write("benchmark\tfrozen_nucel_acc\tbest_v4_acc\n")
        for b in GENOMIC_BENCHMARKS:
            f.write(f"{b}\t{nucel_results.get(b,float('nan'))}\t{gb_full.get(b,float('nan'))}\n")
    print(f"\nSaved → {nucel_file}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Mamba-2 × Muon Optimization")
    parser.add_argument("--schedule", action="store_true", help="Dynamic dual-GPU scheduler")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--exp", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--full-gb-eval", action="store_true", help="v4.1: Full GB evaluation")
    args = parser.parse_args()

    if args.full_gb_eval:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        run_full_gb_eval()
        return

    if args.schedule:
        run_scheduler()
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU {args.gpu}")
    init_results()

    start = args.start
    end = args.end if args.end is not None else len(EXPERIMENTS)

    if args.exp is not None:
        if args.exp >= len(EXPERIMENTS):
            print(f"Error: exp {args.exp} out of range"); return
        run_experiment(args.exp, EXPERIMENTS[args.exp], args.gpu)
    elif args.sweep:
        exps = list(range(start, min(end, len(EXPERIMENTS))))
        print(f"Running exps {start}–{end-1} ({len(exps)}) on GPU {args.gpu}")
        for i in exps:
            try:
                run_experiment(i, EXPERIMENTS[i], args.gpu)
                print(f"\n✓ Exp {i} done")
            except Exception as e:
                print(f"\n✗ Exp {i} FAILED: {e}"); traceback.print_exc()
                blank = {b:float("nan") for b in GENOMIC_BENCHMARKS}
                blank_i = {"total_M":0,"trainable_M":0,"steps":0,"train_secs":0,
                           "total_secs":0,"final_loss":float("nan"),"val_loss":float("nan"),"peak_vram_mb":0}
                log_result(i, EXPERIMENTS[i], blank_i, blank)
    else:
        run_experiment(0, EXPERIMENTS[0], args.gpu)

if __name__ == "__main__":
    main()
