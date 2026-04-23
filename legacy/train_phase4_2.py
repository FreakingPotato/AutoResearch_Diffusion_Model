"""
train_phase4_2.py — Phase 4.2: Full hg38 training with Mamba2, two-stage seq_len progression.

Stage 1: seq_len=4096, full hg38 (chr1-22,X,Y), Mamba2+AdamW
Stage 2: seq_len=8192, load stage 1 checkpoint, continue training

Usage:
    uv run python train_phase4_2.py --stage 1              # Stage 1 only
    uv run python train_phase4_2.py --stage 2              # Stage 2 (loads stage 1 ckpt)
    uv run python train_phase4_2.py --both                  # Run both stages sequentially
    uv run python train_phase4_2.py --eval-only             # Full GB eval on best checkpoint
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import json
import traceback
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

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
DNA_ALPHABET      = "ACGT"
TARGET_TOKENS_PER_STEP = 32_768
DEFAULT_SEQ_LEN   = 4096

CACHE_DIR      = Path.home() / ".cache" / "dna-diffusion"
RAW_DIR        = CACHE_DIR / "raw"
NUCEL_DATA_DIR = CACHE_DIR / "nucel_data"
NUCEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DIR    = Path(__file__).parent
CKPT_DIR       = PROJECT_DIR / "checkpoints_phase4_2"
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_FILE   = PROJECT_DIR / "results_phase4_2.tsv"

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

# Chromosomes for training and validation
TRAIN_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]  # chr1-22,X,Y
VAL_CHROMS = ["22"]  # Use chr22 for validation (consistent with earlier phases)

# ---------------------------------------------------------------------------
# Tokenizer (singleton)
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
        return [vocab.get(c, vocab.get("[UNK]", 1)) for c in text.upper()]
    tok.encode = _encode
    _TOK = tok; _UNK_ID = tok.unk_token_id
    _MASK_ID = tok.mask_token_id; _PAD_ID = tok.pad_token_id
    nt_to_id = {}
    for nt in DNA_ALPHABET:
        ids = tok.encode(nt); nt_to_id[nt] = ids[0] if ids else _UNK_ID
        nt_to_id[nt.lower()] = nt_to_id[nt]
    for nt in "NnRrYyWwSsKkMmBbDdHhVv":
        nt_to_id[nt] = _UNK_ID
    _NT_TO_ID = nt_to_id
    return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID

def tokenize_nt(seq, nt_to_id, unk_id):
    return [nt_to_id.get(c, unk_id) for c in seq]

# ---------------------------------------------------------------------------
# Data preparation — full hg38
# ---------------------------------------------------------------------------

def download_chrom(chrom):
    """Download a single chromosome FASTA if not cached."""
    fa_path = RAW_DIR / f"hg38.{chrom}.fa"
    if fa_path.exists():
        return fa_path
    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr{chrom}.fa.gz"
    gz_path = RAW_DIR / f"hg38.{chrom}.fa.gz"
    print(f"  Downloading chr{chrom} ...")
    import urllib.request
    urllib.request.urlretrieve(url, str(gz_path))
    print(f"  Decompressing chr{chrom} ...")
    import gzip
    with gzip.open(gz_path, 'rt') as f_in, open(fa_path, 'w') as f_out:
        f_out.write(f_in.read())
    gz_path.unlink()
    print(f"  chr{chrom} ready ({fa_path.stat().st_size / 1e9:.1f} GB)")
    return fa_path

def build_full_data(seq_len, train_chroms=None, val_chroms=None):
    """Build tokenized data for all specified chromosomes."""
    if train_chroms is None: train_chroms = TRAIN_CHROMS
    if val_chroms is None: val_chroms = VAL_CHROMS
    
    train_path = NUCEL_DATA_DIR / f"full_train_{seq_len}.bin"
    val_path = NUCEL_DATA_DIR / f"full_val_{seq_len}.bin"
    
    if train_path.exists() and val_path.exists():
        return
    
    _, nt_to_id, _, unk_id, _ = get_tokenizer()
    
    for split, chroms, out_path in [("train", train_chroms, train_path), ("val", val_chroms, val_path)]:
        print(f"Building {split} data (seq_len={seq_len}, chroms={chroms}) ...")
        all_ids = []
        for chrom in chroms:
            fa_path = download_chrom(chrom)
            print(f"  Tokenising chr{chrom} ...")
            cur = []
            with open(fa_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if cur: all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))
                        cur = []
                    else: cur.append(line)
            if cur: all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))
            print(f"    chr{chrom}: {len(all_ids):,} tokens so far")
        
        n_seqs = len(all_ids) // seq_len
        arr = np.array(all_ids[:n_seqs * seq_len], dtype=np.uint16)
        arr.tofile(str(out_path))
        print(f"  {split}: {n_seqs:,} sequences ({len(all_ids)/1e9:.2f}B tokens) → {out_path}")

class FullHg38Dataset(Dataset):
    def __init__(self, split="train", seq_len=DEFAULT_SEQ_LEN):
        self.seq_len = seq_len
        train_chroms = TRAIN_CHROMS
        val_chroms = VAL_CHROMS
        chroms = train_chroms if split == "train" else val_chroms
        
        path = NUCEL_DATA_DIR / f"full_{split}_{seq_len}.bin"
        if not path.exists():
            build_full_data(seq_len, train_chroms if split == "train" else None,
                          val_chroms if split == "val" else None)
        self.data = np.memmap(str(path), dtype=np.uint16, mode="r")
        self.n_seq = len(self.data) // seq_len
        print(f"  {split}: {self.n_seq:,} sequences (seq_len={seq_len})")
    
    def __len__(self): return self.n_seq
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return torch.from_numpy(self.data[s:s+self.seq_len].copy()).long()

def make_loader(batch_size, split="train", seq_len=DEFAULT_SEQ_LEN, num_workers=4):
    ds = FullHg38Dataset(split, seq_len)
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
# Mamba2 wrapper
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class Phase42Config:
    seq_len: int = DEFAULT_SEQ_LEN
    noise_steps: int = 128
    time_embed: str = "additive"
    attention_type: str = "mamba2"
    d_state: int = 64
    d_conv: int = 4
    dropout: float = 0.0

class NucELDiffusionV42(nn.Module):
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

        # Replace attention with Mamba2
        if cfg.attention_type == "mamba2" and self._layers is not None:
            self._replace_attention(cfg.d_state, cfg.d_conv)

        # Time injection
        if cfg.time_embed == "additive":
            self.time_inject = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                for _ in range(max(n,1))
            ])
            for m in self.time_inject: nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

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

    def _replace_attention(self, d_state=64, d_conv=4):
        if not self._layers: return
        for i, layer in enumerate(self._layers):
            attn_mod = None
            for name, mod in layer.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    attn_mod = (name, mod); break
            if not attn_mod: continue
            try:
                replacement = Mamba2Wrapper(self.hidden_size, 8, d_state=d_state, d_conv=d_conv)
                parts = attn_mod[0].split('.')
                parent = layer
                for p in parts[:-1]: parent = getattr(parent, p)
                setattr(parent, parts[-1], replacement)
                print(f"  Layer {i}: replaced {attn_mod[0]} with mamba2")
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
        if self._layers is not None and hasattr(self, 'time_inject'):
            for i, layer in enumerate(self._layers):
                if i >= len(self.time_inject): break
                delta = self.time_inject[i](t_hidden)
                def _hook(mod, inp, out, d=delta):
                    h = out[0] if isinstance(out, tuple) else out
                    return (h+d.unsqueeze(1),)+out[1:] if isinstance(out, tuple) else h+d.unsqueeze(1)
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
    return NucELDiffusionV42(cfg, backbone, mask_id, schedule).to(device)

# ---------------------------------------------------------------------------
# GB evaluation (reused)
# ---------------------------------------------------------------------------

def load_gb_dataset_full(benchmark_name):
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
def extract_embs(backbone_or_model, input_ids, device, batch_size=8, is_nucel=False):
    backbone_or_model.eval()
    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if is_nucel:
                out = backbone_or_model(input_ids=batch)
            else:
                out = backbone_or_model.nucel(input_ids=batch)
            all_embs.append(out.last_hidden_state[:, 0].float().cpu())
        if i % (batch_size * 20) == 0:
            gc.collect(); torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)

def train_probe(embs, labels, epochs=5, lr=1e-3):
    embs = embs.detach().float().cpu(); labels = labels.cpu()
    probe = nn.Linear(embs.shape[1], len(set(labels.tolist())))
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

def evaluate_gb_full(model, device, seq_len=DEFAULT_SEQ_LEN, is_nucel=False):
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    results = {}
    for bname in GENOMIC_BENCHMARKS:
        try:
            t0 = datetime.now()
            data = load_gb_dataset_full(bname)
            if len(data) < 100:
                results[bname] = float("nan"); continue
            seqs, labels = zip(*data)
            labels = torch.tensor(labels, dtype=torch.long)
            n = len(data); np.random.seed(42); perm = np.random.permutation(n)
            sp = int(n * 0.8)
            train_seqs = [data[i][0] for i in perm[:sp]]
            test_seqs = [data[i][0] for i in perm[sp:]]
            train_labels = labels[list(perm[:sp])]; test_labels = labels[list(perm[sp:])]
            
            # Use seq_len appropriate for GB (max 1024 for efficiency)
            gb_seq_len = min(seq_len, 1024)
            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, gb_seq_len)
            test_ids = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, gb_seq_len)
            
            train_embs = extract_embs(model, train_ids, device, batch_size=8, is_nucel=is_nucel)
            test_embs = extract_embs(model, test_ids, device, batch_size=8, is_nucel=is_nucel)
            probe = train_probe(train_embs, train_labels, epochs=5)
            acc = eval_probe(probe, test_embs, test_labels)
            results[bname] = acc
            dt = (datetime.now() - t0).total_seconds()
            print(f"  {bname}: acc={acc:.4f} ({dt:.0f}s)")
            del probe, train_embs, test_embs; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {bname}: FAILED ({e})"); traceback.print_exc()
            results[bname] = float("nan"); gc.collect(); torch.cuda.empty_cache()
    return results

@torch.no_grad()
def evaluate_val_loss(model, batch_size, device, seq_len):
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
# Results logging
# ---------------------------------------------------------------------------

def init_results():
    if RESULTS_FILE.exists(): return
    gb_cols = [f"gb_acc_{b}" for b in GENOMIC_BENCHMARKS]
    header = ["stage","seq_len","timestamp","total_params_M","trainable_params_M",
              "train_steps","training_seconds","total_seconds","final_train_loss",
              "val_loss","peak_vram_mb","data_size"] + gb_cols
    with open(RESULTS_FILE, "w") as f: f.write("\t".join(header)+"\n")

def log_result(stage, seq_len, info, gb, data_size="full_hg38"):
    def fmt(v):
        if isinstance(v, float): return "nan" if math.isnan(v) else f"{v:.6f}"
        return str(v)
    gb_cols = [fmt(gb.get(b, float("nan"))) for b in GENOMIC_BENCHMARKS]
    row = [str(stage), str(seq_len), datetime.now().isoformat(),
           f"{info.get('total_M',0):.1f}", f"{info.get('trainable_M',0):.1f}",
           str(info.get("steps",0)), f"{info.get('train_secs',0):.1f}",
           f"{info.get('total_secs',0):.1f}", fmt(info.get("final_loss",float("nan"))),
           fmt(info.get("val_loss",float("nan"))), f"{info.get('peak_vram_mb',0):.0f}",
           data_size] + gb_cols
    with open(RESULTS_FILE, "a") as f: f.write("\t".join(row)+"\n")
    print(f"Logged → {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_stage(stage_num, seq_len, time_budget_hours, ckpt_load=None, gpu_id=0):
    """Run a single training stage."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"\n{'='*70}")
    print(f"STAGE {stage_num}: seq_len={seq_len}, budget={time_budget_hours}h, GPU={gpu_id}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    t_wall_start = time.time()
    torch.manual_seed(42 + stage_num); torch.cuda.manual_seed(42 + stage_num)
    torch.set_float32_matmul_precision("high"); torch.cuda.empty_cache()
    device = torch.device("cuda")
    
    # Build data
    print("Preparing data ...")
    build_full_data(seq_len)
    
    # Build model
    cfg = Phase42Config(seq_len=seq_len, attention_type="mamba2", d_state=64, d_conv=4)
    model = build_model(cfg, device)
    
    # Load checkpoint if continuing
    if ckpt_load:
        print(f"Loading checkpoint: {ckpt_load}")
        ckpt = torch.load(ckpt_load, map_location=device, weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", {}))
        model_state = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        skipped = {k: v.shape for k, v in state.items() if k not in model_state or v.shape != model_state[k].shape}
        model_state.update(loaded)
        model.load_state_dict(model_state)
        print(f"  Loaded {len(loaded)}/{len(state)} params")
        if skipped: print(f"  Skipped: {list(skipped.keys())[:10]}")
    
    total_p, trainable_p = model.count_params()
    print(f"  Total: {total_p/1e6:.1f}M, Trainable: {trainable_p/1e6:.1f}M")
    sys.stdout.flush()
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    
    # Data
    # Adjust batch size based on seq_len to fit in 24GB
    if seq_len <= 1024: bs = 8
    elif seq_len <= 2048: bs = 4
    elif seq_len <= 4096: bs = 2
    else: bs = 1  # 8192
    
    train_loader = make_loader(bs, "train", seq_len)
    toks_per_step = bs * seq_len
    grad_accum = max(1, TARGET_TOKENS_PER_STEP // toks_per_step)
    print(f"  batch_size={bs}, grad_accum={grad_accum}, toks/step={bs*seq_len*grad_accum}")
    sys.stdout.flush()
    
    # Speed benchmark
    model.train(); data_iter = iter(train_loader)
    torch.cuda.synchronize(); bench_start = time.time()
    for _ in range(3):
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
    ms_per_step = (time.time() - bench_start) / 3 * 1000
    
    time_budget_s = time_budget_hours * 3600
    est_steps = int(time_budget_s / (ms_per_step / 1000))
    print(f"  Speed: {ms_per_step:.0f}ms/step, ~{est_steps:,} steps in {time_budget_hours}h")
    sys.stdout.flush()
    
    # Training loop
    data_iter = iter(train_loader); train_time = 0.0; step = 0
    final_loss = float("nan"); smooth_loss = 0.0
    warmup = max(50, est_steps // 20)
    best_val = float("inf")
    
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
        
        # Cosine LR
        if step < warmup: lr_s = step / warmup
        else:
            progress = (step-warmup) / max(est_steps-warmup, 1)
            lr_s = 0.5*(1+math.cos(math.pi*min(progress,1.0)))
        for pg in opt.param_groups: pg["lr"] = 3e-4 * lr_s
        
        if step % 200 == 0:
            rem = max(0, time_budget_s - train_time)
            print(f"  step {step:05d} | loss={debi:.4f} | lr={3e-4*lr_s:.2e} | dt={dt*1000:.0f}ms | rem={rem/60:.0f}min")
            sys.stdout.flush()
        
        if math.isnan(train_loss): print("  NaN — stopping"); break
        step += 1
        
        # Checkpoint every 2000 steps
        if step % 2000 == 0:
            val = evaluate_val_loss(model, bs, device, seq_len)
            print(f"  ★ Checkpoint at step {step}: val_loss={val:.6f}")
            if val < best_val:
                best_val = val
                ckpt_path = CKPT_DIR / f"stage{stage_num}_seq{seq_len}_best.pt"
                torch.save({
                    "stage": stage_num, "seq_len": seq_len, "step": step,
                    "val_loss": val, "train_loss": final_loss,
                    "state_dict": {k:v for k,v in model.state_dict().items() if not k.startswith("nucel.")},
                }, ckpt_path)
                print(f"  Saved best checkpoint: {ckpt_path}")
            sys.stdout.flush()
        
        if train_time >= time_budget_s:
            print(f"  Budget reached ({time_budget_hours}h)"); break
    
    print(f"  Training done: {step} steps in {train_time:.0f}s ({train_time/3600:.1f}h)")
    sys.stdout.flush()
    
    # Final checkpoint
    final_ckpt = CKPT_DIR / f"stage{stage_num}_seq{seq_len}_final.pt"
    torch.save({
        "stage": stage_num, "seq_len": seq_len, "step": step,
        "val_loss": best_val, "train_loss": final_loss,
        "state_dict": {k:v for k,v in model.state_dict().items() if not k.startswith("nucel.")},
    }, final_ckpt)
    
    # Validation
    print("── Validation ──"); sys.stdout.flush()
    val_loss = evaluate_val_loss(model, bs, device, seq_len)
    print(f"  Final val_loss: {val_loss:.6f}")
    
    # GB evaluation (sampled for speed)
    print("── GB Linear Probe (20% sample) ──"); sys.stdout.flush()
    gb = evaluate_gb_sampled(model, device, seq_len)
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    info = {"total_M": total_p/1e6, "trainable_M": trainable_p/1e6, "steps": step,
            "train_secs": train_time, "total_secs": time.time()-t_wall_start,
            "final_loss": final_loss, "val_loss": val_loss, "peak_vram_mb": peak_vram}
    log_result(stage_num, seq_len, info, gb)
    
    del opt; gc.collect(); torch.cuda.empty_cache()
    return model, final_ckpt

def evaluate_gb_sampled(model, device, seq_len, sample_frac=0.20, max_seqs=2000):
    """Quick GB eval with sampling."""
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    results = {}
    for bname in GENOMIC_BENCHMARKS:
        try:
            data = load_gb_dataset(bname, sample_frac, max_seqs)
            if len(data) < 20: results[bname] = float("nan"); continue
            seqs, labels = zip(*data); labels = torch.tensor(labels, dtype=torch.long)
            n = len(seqs); np.random.seed(42); perm = np.random.permutation(n)
            sp = n // 2
            train_seqs = [seqs[i] for i in perm[:sp]]; test_seqs = [seqs[i] for i in perm[sp:]]
            train_labels = labels[list(perm[:sp])]; test_labels = labels[list(perm[sp:])]
            gb_seq_len = min(seq_len, 1024)
            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, gb_seq_len)
            test_ids = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, gb_seq_len)
            model.eval()
            train_embs = extract_embs(model, train_ids, device, batch_size=8)
            test_embs = extract_embs(model, test_ids, device, batch_size=8)
            probe = train_probe(train_embs, train_labels)
            acc = eval_probe(probe, test_embs, test_labels)
            results[bname] = acc
            print(f"    {bname}: {acc:.4f}")
            del probe, train_embs, test_embs; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {bname}: failed ({e})"); results[bname] = float("nan")
    model.train()
    return results

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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4.2: Full hg38 Mamba2 Training")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Which stage to run")
    parser.add_argument("--both", action="store_true", help="Run both stages")
    parser.add_argument("--eval-only", action="store_true", help="Full GB eval on best checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--stage1-hours", type=float, default=8.0, help="Stage 1 training hours")
    parser.add_argument("--stage2-hours", type=float, default=4.0, help="Stage 2 training hours")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    init_results()
    
    if args.eval_only:
        # Find best checkpoint
        best_ckpt = None; best_val = float("inf")
        for ckpt in sorted(CKPT_DIR.glob("*.pt")):
            try:
                data = torch.load(ckpt, map_location="cpu", weights_only=False)
                if data.get("val_loss", float("inf")) < best_val:
                    best_val = data["val_loss"]; best_ckpt = ckpt
            except: pass
        if not best_ckpt:
            print("No checkpoints found!"); return
        print(f"Evaluating best checkpoint: {best_ckpt} (val_loss={best_val:.6f})")
        
        cfg = Phase42Config(seq_len=4096, attention_type="mamba2")
        device = torch.device("cuda")
        model = build_model(cfg, device)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        state = ckpt["state_dict"]
        model_state = model.state_dict()
        loaded = {k:v for k,v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(loaded); model.load_state_dict(model_state)
        print(f"Loaded {len(loaded)}/{len(state)} params")
        
        print("\n── Full GB Evaluation ──")
        gb = evaluate_gb_full(model, device, seq_len=4096)
        avg = sum(v for v in gb.values() if not math.isnan(v))
        n = sum(1 for v in gb.values() if not math.isnan(v))
        print(f"\nAverage full GB acc: {avg/max(n,1):.4f}")
        return
    
    if args.stage == 1 or args.both:
        model, ckpt = run_stage(1, seq_len=4096, time_budget_hours=args.stage1_hours, gpu_id=args.gpu)
        if args.both:
            del model; gc.collect(); torch.cuda.empty_cache()
            # Load best stage 1 checkpoint for stage 2
            best_s1 = sorted(CKPT_DIR.glob("stage1_*_best.pt"))
            ckpt_load = best_s1[-1] if best_s1 else ckpt
            run_stage(2, seq_len=8192, time_budget_hours=args.stage2_hours, ckpt_load=str(ckpt_load), gpu_id=args.gpu)
    
    elif args.stage == 2:
        best_s1 = sorted(CKPT_DIR.glob("stage1_*_best.pt"))
        if not best_s1:
            print("No stage 1 checkpoint found!"); return
        run_stage(2, seq_len=8192, time_budget_hours=args.stage2_hours, ckpt_load=str(best_s1[-1]), gpu_id=args.gpu)

if __name__ == "__main__":
    main()
