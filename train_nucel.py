"""
train_nucel.py — Phase 2: DiffusionBERT-style DNA diffusion with NucEL backbone.

Architecture:
  - NucEL (FreakingPotato/NucEL): ModernBERT, 22L/512d/16h, vocab=27 single-nucleotide
  - Discrete absorbing-state diffusion (MDLM-style)
  - DiffusionBERT innovations: spindle schedule, per-layer time injection
  - Three time embedding modes: additive, adaln, token
  - Single-nucleotide tokenization (4096bp context)
  - Genomic Benchmark NLL evaluation (8 subtasks, 5% sample)
  - 20-experiment sweep

Usage:
    uv run train_nucel.py                           # baseline experiment
    uv run train_nucel.py --sweep                   # all 20 experiments
    uv run train_nucel.py --gpu 1 --start 10 --end 20 --sweep
    uv run train_nucel.py --exp 5                   # specific experiment
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
import urllib.request
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
NUCEL_N_LAYERS    = 22
NUCEL_VOCAB_SIZE  = 27
NUCEL_MAX_LEN     = 8192

DEFAULT_SEQ_LEN   = 4096
TIME_BUDGET       = 900          # 15 minutes per experiment

DNA_ALPHABET      = "ACGT"

CACHE_DIR      = Path.home() / ".cache" / "dna-diffusion"
RAW_DIR        = CACHE_DIR / "raw"
NUCEL_DATA_DIR = CACHE_DIR / "nucel_data"
NUCEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DIR    = Path(__file__).parent
RESULTS_FILE   = PROJECT_DIR / "results_nucel.tsv"
CKPT_DIR       = PROJECT_DIR / "checkpoints_nucel"
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
# NucEL tokenizer — singleton, loaded once
# ---------------------------------------------------------------------------

_TOK       = None   # AutoTokenizer instance
_NT_TO_ID  = None   # {"A": id, "C": id, ...}
_MASK_ID   = None
_UNK_ID    = None
_PAD_ID    = None


def get_tokenizer():
    global _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID
    if _TOK is not None:
        return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID

    from huggingface_hub import hf_hub_download
    import json
    print(f"Loading NucEL vocab from {NUCEL_REPO} ...")
    vocab_path = hf_hub_download(NUCEL_REPO, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    id_to_tok = {v: k for k, v in vocab.items()}

    # Build a minimal tokenizer-like object
    class _Tok:
        pass
    tok = _Tok()
    tok.vocab = vocab
    tok.vocab_size = len(vocab)
    tok.mask_token_id = vocab.get("\u2282", 4)  # ⊂ mask token
    tok.pad_token_id = vocab.get("[PAD]", 0)
    tok.unk_token_id = vocab.get("[UNK]", 1)

    def _encode(text, add_special_tokens=False):
        ids = []
        if add_special_tokens:
            ids.append(vocab.get("[CLS]", 2))
        for c in text.upper():
            ids.append(vocab.get(c, vocab.get("[UNK]", 1)))
        if add_special_tokens:
            ids.append(vocab.get("[SEP]", 3))
        return ids
    tok.encode = _encode
    _TOK = tok

    unk_id = tok.unk_token_id if tok.unk_token_id is not None else 0
    _UNK_ID = unk_id

    # Discover mask token
    if tok.mask_token_id is not None:
        _MASK_ID = tok.mask_token_id
    elif "[MASK]" in tok.vocab:
        _MASK_ID = tok.vocab["[MASK]"]
    else:
        # Fall back: encode '[MASK]'
        ids = tok.encode("[MASK]", add_special_tokens=False)
        _MASK_ID = ids[0] if ids else 1

    _PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0

    # Build fast nt→id map
    nt_to_id = {}
    for nt in DNA_ALPHABET:
        ids = tok.encode(nt, add_special_tokens=False)
        nt_to_id[nt] = ids[0] if ids else unk_id
        nt_to_id[nt.lower()] = nt_to_id[nt]
    # Ambiguous bases → unk
    for nt in "NnRrYyWwSsKkMmBbDdHhVv":
        nt_to_id[nt] = unk_id
    _NT_TO_ID = nt_to_id

    print(f"  vocab_size={tok.vocab_size}, mask_id={_MASK_ID}, unk_id={_UNK_ID}")
    print(f"  NT mapping: { {k: v for k, v in nt_to_id.items() if k in DNA_ALPHABET} }")
    return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID


def tokenize_nt(seq, nt_to_id, unk_id):
    """Fast single-nucleotide tokenization of a DNA string."""
    return [nt_to_id.get(c, unk_id) for c in seq]


# ---------------------------------------------------------------------------
# Data preparation (single-nucleotide, reuses Phase-1 FASTA files)
# ---------------------------------------------------------------------------

def build_nucel_data(seq_len=DEFAULT_SEQ_LEN):
    """Tokenise chr21 (train) and chr22 (val) FASTA into per-nucleotide tokens."""
    train_path = NUCEL_DATA_DIR / f"train_{seq_len}.bin"
    val_path   = NUCEL_DATA_DIR / f"val_{seq_len}.bin"
    if train_path.exists() and val_path.exists():
        return  # already done

    _, nt_to_id, _, unk_id, _ = get_tokenizer()

    for split, chrom in [("train", "21"), ("val", "22")]:
        fa_path = RAW_DIR / f"hg38.{chrom}.fa"
        if not fa_path.exists():
            raise FileNotFoundError(
                f"FASTA not found: {fa_path}\n"
                f"Run first: uv run prepare.py --chromosomes 21,22"
            )

        print(f"Tokenising chr{chrom} → {split} (seq_len={seq_len}) ...")
        all_ids = []
        cur = []
        with open(fa_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if cur:
                        all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))
                    cur = []
                else:
                    cur.append(line)
        if cur:
            all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))

        n_seqs = len(all_ids) // seq_len
        arr    = np.array(all_ids[: n_seqs * seq_len], dtype=np.uint16)
        out    = NUCEL_DATA_DIR / f"{split}_{seq_len}.bin"
        arr.tofile(str(out))
        print(f"  → {n_seqs:,} sequences  ({len(arr):,} tokens)  saved to {out}")

    print("NucEL data ready.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NucELDataset(Dataset):
    def __init__(self, split="train", seq_len=DEFAULT_SEQ_LEN):
        self.seq_len = seq_len
        path = NUCEL_DATA_DIR / f"{split}_{seq_len}.bin"
        if not path.exists():
            build_nucel_data(seq_len)
        self.data  = np.memmap(str(path), dtype=np.uint16, mode="r")
        self.n_seq = len(self.data) // seq_len
        assert self.n_seq > 0, f"Empty dataset at {path}"

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        s = idx * self.seq_len
        return torch.from_numpy(self.data[s: s + self.seq_len].copy()).long()


def make_loader(batch_size, split="train", seq_len=DEFAULT_SEQ_LEN, num_workers=4):
    ds = NucELDataset(split, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, pin_memory=True, drop_last=True)


# ---------------------------------------------------------------------------
# Token frequency (for spindle schedule)
# ---------------------------------------------------------------------------

def compute_token_freqs(seq_len=DEFAULT_SEQ_LEN, vocab_size=NUCEL_VOCAB_SIZE):
    cache = NUCEL_DATA_DIR / f"freqs_{seq_len}.npy"
    if cache.exists():
        return np.load(str(cache))
    path = NUCEL_DATA_DIR / f"train_{seq_len}.bin"
    if not path.exists():
        return np.ones(vocab_size, dtype=np.float32) / vocab_size
    data   = np.memmap(str(path), dtype=np.uint16, mode="r")
    counts = np.bincount(data, minlength=vocab_size).astype(np.float64)
    freqs  = (counts / counts.sum()).astype(np.float32)
    np.save(str(cache), freqs)
    return freqs


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

class UniformSchedule:
    name = "uniform"

    def __init__(self, noise_steps=128, mask_id=1):
        self.noise_steps  = noise_steps
        self.mask_id      = mask_id

    def sample_t(self, B, device):
        return torch.randint(1, self.noise_steps + 1, (B,), device=device)

    def forward_process(self, x, t):
        prob = t.float() / self.noise_steps                        # (B,)
        mask = torch.rand_like(x, dtype=torch.float) < prob.unsqueeze(1)
        xn   = x.clone()
        xn[mask] = self.mask_id
        return xn, mask


class CosineSchedule(UniformSchedule):
    name = "cosine"

    def forward_process(self, x, t):
        alpha = torch.cos(math.pi / 2 * t.float() / self.noise_steps) ** 2
        prob  = 1.0 - alpha
        mask  = torch.rand_like(x, dtype=torch.float) < prob.unsqueeze(1)
        xn    = x.clone()
        xn[mask] = self.mask_id
        return xn, mask


class SpindleSchedule(UniformSchedule):
    """
    DiffusionBERT spindle schedule: more-frequent tokens are masked earlier.
    Per-token mask probability = base_prob * token_weight / avg_weight,
    where token_weight ∈ [0.5, 1.5] (higher = more frequent = mask sooner).
    """
    name = "spindle"

    def __init__(self, noise_steps=128, mask_id=1, token_freqs=None,
                 vocab_size=NUCEL_VOCAB_SIZE):
        super().__init__(noise_steps, mask_id)
        if token_freqs is None:
            token_freqs = np.ones(vocab_size, dtype=np.float32) / vocab_size
        f      = np.asarray(token_freqs, dtype=np.float32)
        lo, hi = f.min(), f.max()
        w      = 0.5 + (f - lo) / (hi - lo + 1e-8) if hi > lo else np.ones_like(f)
        self._w_np = w.astype(np.float32)
        self._w    = None   # lazy GPU tensor

    def _weights(self, device):
        if self._w is None or self._w.device != device:
            self._w = torch.tensor(self._w_np, device=device)
        return self._w

    def forward_process(self, x, t):
        base_prob = t.float() / self.noise_steps          # (B,)
        w         = self._weights(x.device)[x]            # (B, L)
        avg_w     = w.mean(dim=1, keepdim=True).clamp(min=1e-8)
        prob      = (base_prob.unsqueeze(1) * w / avg_w).clamp(0, 1)
        mask      = torch.rand_like(prob) < prob
        xn        = x.clone()
        xn[mask]  = self.mask_id
        return xn, mask


def make_schedule(name, noise_steps, mask_id, token_freqs=None):
    if name == "uniform":
        return UniformSchedule(noise_steps, mask_id)
    if name == "cosine":
        return CosineSchedule(noise_steps, mask_id)
    if name == "spindle":
        return SpindleSchedule(noise_steps, mask_id, token_freqs)
    raise ValueError(f"Unknown schedule: {name}")


# ---------------------------------------------------------------------------
# NucEL Diffusion Model
# ---------------------------------------------------------------------------

@dataclass
class NucELConfig:
    seq_len:       int   = DEFAULT_SEQ_LEN
    noise_steps:   int   = 128
    time_embed:    str   = "additive"   # "additive" | "adaln" | "token"
    schedule:      str   = "uniform"    # "uniform" | "cosine" | "spindle"
    dropout:       float = 0.0
    freeze_layers: int   = 0


class NucELDiffusion(nn.Module):
    """
    DiffusionBERT using NucEL (ModernBERT) as backbone.
    Time embedding injected into every transformer layer via forward hooks.
    """

    def __init__(self, cfg: NucELConfig, backbone: nn.Module,
                 mask_id: int, schedule):
        super().__init__()
        self.cfg         = cfg
        self.nucel       = backbone
        self.mask_id     = mask_id
        self.schedule    = schedule
        self.noise_steps = cfg.noise_steps
        self.hidden_size = NUCEL_HIDDEN_SIZE
        self.vocab_size  = NUCEL_VOCAB_SIZE
        self.te_type     = cfg.time_embed

        # Sinusoidal → projected time embedding
        self.time_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        # Discover transformer layers in backbone
        self._layers = self._find_layers()
        n = len(self._layers) if self._layers else 0
        print(f"  Backbone transformer layers found: {n}")

        if cfg.time_embed == "additive":
            # Per-layer residual injection, init to zero (no-op at start)
            self.time_inject = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                for _ in range(max(n, 1))
            ])
            for m in self.time_inject:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        elif cfg.time_embed == "adaln":
            # Per-layer scale + shift, init to zero
            self.adaln_proj = nn.Linear(
                self.hidden_size, 2 * self.hidden_size * max(n, 1), bias=True
            )
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)

        elif cfg.time_embed == "token":
            # No per-layer hooks; add time embedding to hidden state post-backbone
            pass  # uses time_proj output directly

        # Output LM head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

        self.dropout_layer = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # Utility: find transformer layer list inside NucEL
    # ------------------------------------------------------------------

    def _find_layers(self):
        for path in ["encoder.layers", "layers", "model.layers",
                     "encoder.layer", "transformer.h"]:
            obj   = self.nucel
            found = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            if found and isinstance(obj, nn.ModuleList) and len(obj) > 2:
                return obj
        # Fallback: find first large ModuleList
        for _, m in self.nucel.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 5:
                return m
        return None

    # ------------------------------------------------------------------
    # Sinusoidal time embedding
    # ------------------------------------------------------------------

    def _sinusoidal(self, t, dim):
        half  = dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    # ------------------------------------------------------------------
    # Forward with hook-based time injection
    # ------------------------------------------------------------------

    def forward(self, input_ids, t):
        """
        input_ids : (B, L)  noisy token IDs
        t         : (B,)    diffusion timesteps
        returns     (B, L, vocab_size) logits
        """
        t_sin    = self._sinusoidal(t, self.hidden_size)
        t_hidden = self.time_proj(t_sin)                   # (B, hidden_size)

        handles = []

        if self._layers is not None and self.te_type != "token":
            if self.te_type == "additive":
                for i, layer in enumerate(self._layers):
                    if i >= len(self.time_inject):
                        break
                    delta = self.time_inject[i](t_hidden)  # (B, H)

                    def _hook_add(mod, inp, out, d=delta):
                        h = out[0] if isinstance(out, tuple) else out
                        h = h + d.unsqueeze(1)
                        return (h,) + out[1:] if isinstance(out, tuple) else h

                    handles.append(layer.register_forward_hook(_hook_add))

            elif self.te_type == "adaln":
                n   = len(self._layers)
                raw = self.adaln_proj(t_hidden)            # (B, 2*H*n)
                raw = raw.view(-1, n, 2, self.hidden_size) # (B, n, 2, H)

                for i, layer in enumerate(self._layers):
                    if i >= n:
                        break
                    scale = raw[:, i, 0, :]  # (B, H)
                    shift = raw[:, i, 1, :]  # (B, H)

                    def _hook_ada(mod, inp, out, s=scale, sh=shift):
                        h = out[0] if isinstance(out, tuple) else out
                        h = h * (1 + s.unsqueeze(1)) + sh.unsqueeze(1)
                        return (h,) + out[1:] if isinstance(out, tuple) else h

                    handles.append(layer.register_forward_hook(_hook_ada))

        try:
            out = self.nucel(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        # "token" mode: add time embedding broadcast over sequence
        if self.te_type == "token":
            hidden = hidden + t_hidden.unsqueeze(1)

        hidden = self.dropout_layer(hidden)
        return self.lm_head(hidden)

    # ------------------------------------------------------------------
    # Loss & sampling
    # ------------------------------------------------------------------

    def compute_loss(self, x_clean, t=None):
        if t is None:
            t = self.schedule.sample_t(x_clean.shape[0], x_clean.device)
        x_noisy, mask = self.schedule.forward_process(x_clean, t)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_clean.device, requires_grad=True)
        logits = self.forward(x_noisy, t)
        return F.cross_entropy(logits[mask], x_clean[mask])

    @torch.no_grad()
    def sample(self, n, seq_len, device, steps=None):
        steps = steps or min(self.noise_steps, 32)
        x     = torch.full((n, seq_len), self.mask_id, dtype=torch.long, device=device)

        for tv in reversed(range(1, steps + 1)):
            t      = torch.full((n,), tv, dtype=torch.long, device=device)
            logits = self.forward(x, t)
            masked = (x == self.mask_id)

            if tv > 1:
                p_now  = tv / steps
                p_next = (tv - 1) / steps
                frac   = 1.0 - p_next / max(p_now, 1e-8)
                probs, pred = F.softmax(logits, dim=-1).max(dim=-1)
                probs[~masked] = float("inf")
                n_masked   = masked.sum(dim=1)
                n_unmask   = (n_masked.float() * frac).long().clamp(min=1)
                for b in range(n):
                    if n_masked[b] == 0:
                        continue
                    idx  = masked[b].nonzero(as_tuple=True)[0]
                    k    = min(n_unmask[b].item(), len(idx))
                    _, top = probs[b, idx].topk(k)
                    x[b, idx[top]] = pred[b, idx[top]]
            else:
                x[masked] = logits.argmax(dim=-1)[masked]

        return x

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: NucELConfig, device):
    from transformers import AutoModel

    print(f"Loading NucEL backbone ({NUCEL_REPO}) ...")
    backbone = AutoModel.from_pretrained(
        NUCEL_REPO,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Freeze bottom layers
    if cfg.freeze_layers > 0:
        layers = None
        for path in ["encoder.layers", "layers", "model.layers", "encoder.layer"]:
            obj   = backbone
            found = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            if found and isinstance(obj, nn.ModuleList):
                layers = obj
                break

        if layers is not None:
            n_freeze = min(cfg.freeze_layers, len(layers))
            for i in range(n_freeze):
                for p in layers[i].parameters():
                    p.requires_grad = False
            print(f"  Froze bottom {n_freeze} transformer layers")

        # Freeze embeddings too
        for attr in ["embeddings", "embed_tokens"]:
            if hasattr(backbone, attr):
                for p in getattr(backbone, attr).parameters():
                    p.requires_grad = False

    _, _, mask_id, _, _ = get_tokenizer()

    token_freqs = compute_token_freqs(cfg.seq_len) if cfg.schedule == "spindle" else None
    schedule    = make_schedule(cfg.schedule, cfg.noise_steps, mask_id, token_freqs)

    model = NucELDiffusion(cfg, backbone, mask_id, schedule).to(device)
    return model


# ---------------------------------------------------------------------------
# Genomic Benchmark evaluation
# ---------------------------------------------------------------------------

# UCSC Genome Browser REST API
# Genome: hg38 (human reference genome)
# Example: https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr16;start=72728821;end=72729321
USCS_API_URL = "https://api.genome.ucsc.edu/getData/sequence"
USCS_TIMEOUT_SEC = 10

# Cache for fetched sequences
GB_SEQ_CACHE_DIR = Path.home() / ".cache" / "gb_sequences"
GB_SEQ_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_seq_ucsc(chrom, start, end):
    """Fetch DNA sequence from UCSC REST API."""
    url = f"{USCS_API_URL}?genome=hg38;chrom={chrom};start={start};end={end}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=USCS_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read())
        return data.get("dna", "").upper()
    except Exception as e:
        return f"Error: {e}"


def _load_gb_sequences(benchmark_name, sample_frac=0.05, max_seqs=400, timeout_sec=30):
    """Load DNA sequences from a Genomic Benchmark subtask using UCSC API."""
    cache_file = GB_SEQ_CACHE_DIR / f"{benchmark_name}.json"
    if cache_file.exists():
        try:
            seqs_json = cache_file.read_text()
            seqs = json.loads(seqs_json)
            if seqs:
                return seqs
        except Exception:
            pass  # Cache corrupted or empty, re-fetch

    # --- Fetch sequences from UCSC API ---
    print(f"  Fetching {benchmark_name} sequences via UCSC API...", flush=True)
    result_seqs = []
    try:
        from datasets import load_dataset
        ds = load_dataset("katielink/genomic-benchmarks", benchmark_name)
    except Exception as e:
        print(f"    HF load failed: {e}", flush=True)
        return []

    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        sp = ds[split_name]
        n = min(max(1, int(len(sp) * sample_frac)), max_seqs // 2)
        indices = np.random.choice(len(sp), min(n, len(sp)), replace=False)

        for idx in indices:
            row = sp[int(idx)]
            region = row.get("region", "")
            start = row.get("start", 0)
            end = row.get("end", 0)

            if region and end > start:
                chrom = region
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"

                fetched_seq = fetch_seq_ucsc(chrom, start, end)
                if fetched_seq.startswith("Error:") or len(fetched_seq) < 10:
                    pass  # skip failed/short sequences
                else:
                    result_seqs.append(fetched_seq)
            time.sleep(0.05)  # Rate limit

    if result_seqs:
        try:
            cache_file.write_text(json.dumps(result_seqs))
            print(f"    Cached {len(result_seqs)} sequences to {cache_file}")
        except Exception as e:
            print(f"    Failed to write cache: {e}")
    else:
        print(f"    No sequences fetched for {benchmark_name}")

    return result_seqs



@torch.no_grad()
def evaluate_genomic_benchmarks(model, device, seq_len=DEFAULT_SEQ_LEN):
    """Compute average NLL on 5% sample of each Genomic Benchmark subtask."""
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    model.eval()
    results = {}

    for bname in GENOMIC_BENCHMARKS:
        try:
            seqs = _load_gb_sequences(bname)
            if not seqs:
                results[bname] = float("nan")
                continue

            total, n_ok = 0.0, 0
            for i in range(0, len(seqs), 4):
                batch_seqs = seqs[i: i + 4]
                ids_list   = []
                for s in batch_seqs:
                    ids = tokenize_nt(s, nt_to_id, unk_id)
                    if len(ids) >= seq_len:
                        ids = ids[:seq_len]
                    else:
                        ids = ids + [pad_id] * (seq_len - len(ids))
                    ids_list.append(ids)

                x = torch.tensor(ids_list, dtype=torch.long, device=device)
                t = torch.randint(1, model.noise_steps + 1, (len(ids_list),), device=device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model.compute_loss(x, t)

                if not loss.isnan() and not loss.isinf():
                    total += loss.item()
                    n_ok  += 1

            results[bname] = total / max(n_ok, 1)
        except Exception as e:
            print(f"    GB {bname}: failed ({e})")
            results[bname] = float("nan")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val(model, batch_size, device, seq_len=DEFAULT_SEQ_LEN):
    model.eval()
    loader = make_loader(min(batch_size, 4), "val", seq_len, num_workers=0)
    _, nt_to_id, _, _, _ = get_tokenizer()
    id_to_nt = {v: k for k, v in nt_to_id.items() if k in DNA_ALPHABET}
    nt_order = list("ACGT")

    total_loss, n_ok = 0.0, 0
    real_counts = np.zeros(4)
    gen_counts  = np.zeros(4)

    for i, batch in enumerate(loader):
        if i >= 50:
            break
        batch = batch.to(device)
        t     = torch.randint(1, model.noise_steps + 1, (batch.shape[0],), device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.compute_loss(batch, t)
        if not loss.isnan():
            total_loss += loss.item()
            n_ok       += 1
        for seq in batch.cpu().numpy():
            for tid in seq:
                nt = id_to_nt.get(int(tid))
                if nt in nt_order:
                    real_counts[nt_order.index(nt)] += 1

    avg_loss = total_loss / max(n_ok, 1)

    # Generate a few samples
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        samples = model.sample(min(batch_size, 4), seq_len, device)
    for seq in samples.cpu().numpy():
        for tid in seq:
            nt = id_to_nt.get(int(tid))
            if nt in nt_order:
                gen_counts[nt_order.index(nt)] += 1

    real_dist = real_counts / max(real_counts.sum(), 1)
    gen_dist  = gen_counts  / max(gen_counts.sum(),  1)
    nt_err    = float(np.abs(real_dist - gen_dist).mean())
    real_gc   = float((real_counts[1] + real_counts[2]) / max(real_counts.sum(), 1))
    gen_gc    = float((gen_counts[1]  + gen_counts[2])  / max(gen_counts.sum(),  1))

    model.train()
    return {
        "val_loss":     avg_loss,
        "nt_dist_error": nt_err,
        "real_gc":      real_gc,
        "gen_gc":       gen_gc,
        "gc_error":     abs(real_gc - gen_gc),
    }


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(name, params, lr, wd):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    if name == "adamw_fused":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True)
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, scale_parameter=False,
                             relative_step=False, warmup_init=False, weight_decay=wd)
        except Exception:
            pass
    # fallback
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))


# ---------------------------------------------------------------------------
# Experiment sweep (20 experiments)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # ── Group A: Baseline & learning rate (0–4) ──────────────────────────
    {"id": "baseline",           "lr": 1e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "lr_3e4",             "lr": 3e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "lr_5e4",             "lr": 5e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "adafactor_1e4",      "lr": 1e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adafactor",   "noise_steps": 128, "weight_decay": 0.01},

    {"id": "fused_adamw_3e4",    "lr": 3e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw_fused", "noise_steps": 128, "weight_decay": 0.01},

    # ── Group B: Time embedding mode (5–7) ───────────────────────────────
    {"id": "adaln_1e4",          "lr": 1e-4, "time_embed": "adaln",    "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "adaln_3e4",          "lr": 3e-4, "time_embed": "adaln",    "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "token_time_1e4",     "lr": 1e-4, "time_embed": "token",    "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    # ── Group C: Noise schedule (8–10) ───────────────────────────────────
    {"id": "cosine_1e4",         "lr": 1e-4, "time_embed": "additive", "schedule": "cosine",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "spindle_1e4",        "lr": 1e-4, "time_embed": "additive", "schedule": "spindle",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "spindle_3e4",        "lr": 3e-4, "time_embed": "additive", "schedule": "spindle",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    # ── Group D: Freeze strategy (11–13) ─────────────────────────────────
    {"id": "freeze_6_1e4",       "lr": 1e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 6,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "freeze_11_3e4",      "lr": 3e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 11, "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "freeze6_drop01_3e4", "lr": 3e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 6,  "dropout": 0.1, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    # ── Group E: Sequence length / attention window (14–16) ──────────────
    {"id": "seq_512",            "lr": 3e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len":  512, "batch_size": 8,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "seq_1024",           "lr": 3e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 1024, "batch_size": 4,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "seq_2048",           "lr": 3e-4, "time_embed": "additive", "schedule": "uniform",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 2048, "batch_size": 4,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    # ── Group F: Combined best configs (17–19) ────────────────────────────
    {"id": "best_adaln_spindle", "lr": 3e-4, "time_embed": "adaln",    "schedule": "spindle",
     "freeze_layers": 0,  "dropout": 0.1, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},

    {"id": "best_freeze_cosine", "lr": 3e-4, "time_embed": "adaln",    "schedule": "cosine",
     "freeze_layers": 6,  "dropout": 0.1, "seq_len": 2048, "batch_size": 4,
     "optimizer": "adamw_fused", "noise_steps": 128, "weight_decay": 0.01},

    {"id": "best_5e4_spindle",   "lr": 5e-4, "time_embed": "additive", "schedule": "spindle",
     "freeze_layers": 0,  "dropout": 0.0, "seq_len": 4096, "batch_size": 2,
     "optimizer": "adamw",       "noise_steps": 128, "weight_decay": 0.01},
]

assert len(EXPERIMENTS) == 20, f"Expected 20 experiments, got {len(EXPERIMENTS)}"


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def init_results():
    if RESULTS_FILE.exists():
        return
    gb_cols = [f"gb_{b}" for b in GENOMIC_BENCHMARKS]
    header  = [
        "exp_id", "exp_idx", "timestamp",
        "optimizer", "lr", "time_embed", "schedule",
        "freeze_layers", "dropout", "seq_len", "batch_size",
        "noise_steps", "weight_decay",
        "total_params_M", "trainable_params_M",
        "train_steps", "training_seconds", "total_seconds",
        "final_train_loss", "val_loss", "nt_dist_error",
        "gc_error", "real_gc", "gen_gc",
        "peak_vram_mb", "total_tokens_M",
    ] + gb_cols
    with open(RESULTS_FILE, "w") as f:
        f.write("\t".join(header) + "\n")
    print(f"Created {RESULTS_FILE}")


def log_result(exp_idx, cfg, metrics, info, gb):
    def fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.6f}"
        return str(v)

    row = [
        cfg.get("id", f"exp_{exp_idx}"),
        str(exp_idx),
        datetime.now().isoformat(),
        cfg.get("optimizer", ""),
        str(cfg.get("lr", "")),
        cfg.get("time_embed", ""),
        cfg.get("schedule", ""),
        str(cfg.get("freeze_layers", 0)),
        str(cfg.get("dropout", 0.0)),
        str(cfg.get("seq_len", DEFAULT_SEQ_LEN)),
        str(cfg.get("batch_size", 2)),
        str(cfg.get("noise_steps", 128)),
        str(cfg.get("weight_decay", 0.01)),
        f"{info.get('total_M', 0):.1f}",
        f"{info.get('trainable_M', 0):.1f}",
        str(info.get("steps", 0)),
        f"{info.get('train_secs', 0):.1f}",
        f"{info.get('total_secs', 0):.1f}",
        fmt(info.get("final_loss", float("nan"))),
        fmt(metrics.get("val_loss",      float("nan"))),
        fmt(metrics.get("nt_dist_error", float("nan"))),
        fmt(metrics.get("gc_error",      float("nan"))),
        f"{metrics.get('real_gc', 0):.4f}",
        f"{metrics.get('gen_gc',  0):.4f}",
        f"{info.get('peak_vram_mb', 0):.0f}",
        f"{info.get('tokens_M', 0):.1f}",
    ] + [fmt(gb.get(b, float("nan"))) for b in GENOMIC_BENCHMARKS]

    with open(RESULTS_FILE, "a") as f:
        f.write("\t".join(row) + "\n")
    print(f"Logged → {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp_idx, exp_cfg):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_idx}: {exp_cfg.get('id', 'unnamed')}")
    print(f"{'='*70}")
    for k, v in exp_cfg.items():
        if k != "id":
            print(f"  {k}: {v}")
    print()

    t_wall_start = time.time()
    torch.manual_seed(42 + exp_idx)
    torch.cuda.manual_seed(42 + exp_idx)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    seq_len = exp_cfg.get("seq_len", DEFAULT_SEQ_LEN)

    # Ensure data prepared
    build_nucel_data(seq_len)

    # Build model
    cfg = NucELConfig(
        seq_len       = seq_len,
        noise_steps   = exp_cfg.get("noise_steps", 128),
        time_embed    = exp_cfg.get("time_embed", "additive"),
        schedule      = exp_cfg.get("schedule", "uniform"),
        dropout       = exp_cfg.get("dropout", 0.0),
        freeze_layers = exp_cfg.get("freeze_layers", 0),
    )
    model = build_model(cfg, device)

    total_p, trainable_p = model.count_params()
    print(f"Total params:     {total_p:,}  ({total_p/1e6:.1f}M)")
    print(f"Trainable params: {trainable_p:,}  ({trainable_p/1e6:.1f}M)")

    # Optimizer
    lr       = exp_cfg.get("lr", 1e-4)
    wd       = exp_cfg.get("weight_decay", 0.01)
    opt_name = exp_cfg.get("optimizer", "adamw")
    opt      = make_optimizer(opt_name, [p for p in model.parameters() if p.requires_grad], lr, wd)

    # Data
    bs             = exp_cfg.get("batch_size", 2)
    train_loader   = make_loader(bs, "train", seq_len)
    toks_per_step  = bs * seq_len
    total_toks     = 2 ** 15   # 32K effective tokens per optimizer step
    grad_accum     = max(1, total_toks // toks_per_step)
    print(f"batch={bs}, grad_accum={grad_accum}, eff_tokens/step={toks_per_step*grad_accum:,}")

    # Training loop
    model.train()
    torch.cuda.synchronize()
    train_time  = 0.0
    step        = 0
    smooth_loss = 0.0
    final_loss  = float("nan")
    data_iter   = iter(train_loader)

    warmup_r   = 0.05
    warmdown_r = 0.4

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        opt.zero_grad(set_to_none=True)

        for micro in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch     = next(data_iter)

            batch = batch.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.compute_loss(batch) / grad_accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        train_loss = loss.item() * grad_accum
        if not math.isnan(train_loss):
            final_loss = train_loss

        torch.cuda.synchronize()
        dt = time.time() - t0

        if step > 5:
            train_time += dt

        b    = 0.95
        smooth_loss = b * smooth_loss + (1 - b) * train_loss
        debi = smooth_loss / (1 - b ** (step + 1))

        prog = min(train_time / TIME_BUDGET, 1.0)

        if prog < warmup_r:
            lr_s = prog / warmup_r
        elif prog < 1.0 - warmdown_r:
            lr_s = 1.0
        else:
            lr_s = max((1.0 - prog) / warmdown_r, 0.0)

        for pg in opt.param_groups:
            pg["lr"] = lr * lr_s

        rem = max(0, TIME_BUDGET - train_time)
        print(
            f"\rstep {step:05d} ({100*prog:.1f}%) | loss={debi:.4f} | "
            f"lr={lr*lr_s:.2e} | dt={dt*1000:.0f}ms | rem={rem:.0f}s    ",
            end="", flush=True,
        )

        if math.isnan(train_loss):
            print("\nNaN loss — stopping.")
            break

        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()
        elif (step + 1) % 2000 == 0:
            gc.collect()

        step += 1
        if step > 5 and train_time >= TIME_BUDGET:
            break

    print()

    del opt
    gc.enable(); gc.collect(); torch.cuda.empty_cache()
    model.eval()

    # Validation
    try:
        metrics = evaluate_val(model, bs, device, seq_len)
        print("── Val metrics ──")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v:.6f}")
    except Exception as e:
        print(f"Val eval failed: {e}")
        traceback.print_exc()
        metrics = {k: float("nan") for k in
                   ["val_loss", "nt_dist_error", "gc_error", "real_gc", "gen_gc"]}

    # Genomic Benchmarks
    print("── Genomic Benchmarks ──")
    try:
        gb = evaluate_genomic_benchmarks(model, device, seq_len)
        for bname, v in gb.items():
            tag = f"{v:.4f}" if not math.isnan(v) else "N/A"
            print(f"  {bname:42s}: {tag}")
    except Exception as e:
        print(f"GB eval failed: {e}")
        gb = {b: float("nan") for b in GENOMIC_BENCHMARKS}

    # Checkpoint (save only new params, not NucEL backbone)
    ckpt = CKPT_DIR / f"{exp_idx:03d}_{exp_cfg.get('id','exp')}.pt"
    try:
        torch.save({
            "exp_idx":    exp_idx,
            "config":     exp_cfg,
            "state_dict": {k: v for k, v in model.state_dict().items()
                           if not k.startswith("nucel.")},
            "metrics":    metrics,
            "gb_metrics": gb,
            "final_loss": final_loss,
            "step":       step,
        }, ckpt)
        print(f"Checkpoint → {ckpt}")
    except Exception as e:
        print(f"Checkpoint save failed: {e}")

    total_tokens_seen = step * toks_per_step * grad_accum
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    t_wall_end = time.time()

    info = {
        "total_M":      total_p / 1e6,
        "trainable_M":  trainable_p / 1e6,
        "steps":        step,
        "train_secs":   train_time,
        "total_secs":   t_wall_end - t_wall_start,
        "final_loss":   final_loss,
        "peak_vram_mb": peak_vram,
        "tokens_M":     total_tokens_seen / 1e6,
    }
    print("── Training info ──")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")

    log_result(exp_idx, exp_cfg, metrics, info, gb)

    del model
    gc.collect(); torch.cuda.empty_cache()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: DiffusionBERT + NucEL DNA diffusion")
    parser.add_argument("--sweep",  action="store_true", help="Run experiment sweep")
    parser.add_argument("--exp",    type=int,            help="Run single experiment by index")
    parser.add_argument("--gpu",    type=int, default=0, help="CUDA device index")
    parser.add_argument("--start",  type=int, default=0, help="Sweep start index (inclusive)")
    parser.add_argument("--end",    type=int, default=None, help="Sweep end index (exclusive)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU {args.gpu}")

    init_results()

    start = args.start
    end   = args.end if args.end is not None else len(EXPERIMENTS)

    if args.exp is not None:
        if args.exp >= len(EXPERIMENTS):
            print(f"Error: experiment {args.exp} out of range (max {len(EXPERIMENTS)-1})")
            return
        run_experiment(args.exp, EXPERIMENTS[args.exp])

    elif args.sweep:
        exps = list(range(start, min(end, len(EXPERIMENTS))))
        print(f"Running experiments {start}–{end-1}  ({len(exps)} total)  on GPU {args.gpu}")
        for i in exps:
            cfg = EXPERIMENTS[i]
            try:
                run_experiment(i, cfg)
                print(f"\n✓ Experiment {i} ({cfg['id']}) done")
            except Exception as e:
                print(f"\n✗ Experiment {i} ({cfg['id']}) FAILED: {e}")
                traceback.print_exc()
                blank_m  = {k: float("nan") for k in
                            ["val_loss", "nt_dist_error", "gc_error", "real_gc", "gen_gc"]}
                blank_gb = {b: float("nan") for b in GENOMIC_BENCHMARKS}
                blank_i  = {k: 0.0 for k in
                            ["total_M", "trainable_M", "steps", "train_secs",
                             "total_secs", "final_loss", "peak_vram_mb", "tokens_M"]}
                blank_i["final_loss"] = float("nan")
                log_result(i, cfg, blank_m, blank_i, blank_gb)
            print()

        print(f"\n{'='*70}")
        print(f"Sweep {start}–{end-1} complete!")

    else:
        run_experiment(0, EXPERIMENTS[0])


if __name__ == "__main__":
    main()
