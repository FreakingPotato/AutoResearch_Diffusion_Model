"""
train_phase3.py — Phase 3: NucEL DNA Diffusion with Linear Attention & GB Linear Probe.

Experiments:
  0: nucel_frozen_probe    — NucEL frozen encoder + linear probe (embedding baseline)
  1: diffusion_baseline    — Current framework: additive embed, uniform schedule, AdamW
  2: muon                  — Muon optimizer replacing AdamW
  3: adaln_dit             — DiT-style AdaLN (scale+shift per layer)
  4: linear_attn_performer — Performer (random feature linear attention)
  5: linear_attn_retnet    — RetNet (data-independent decay)
  6: linear_attn_gla       — GLA (data-dependent gated linear attention)
  7: linear_attn_mamba2    — Mamba-2 (SSM, via flash-linear-attention)
  8: hybrid_attn           — Hybrid: linear attn + global attention alternating
  9: muon_adaln            — Muon + AdaLN
  10: muon_linear          — Muon + best linear attention

Usage:
    uv run train_phase3.py --gpu 0 --sweep --start 0 --end 6   # GPU 0: exps 0-5
    uv run train_phase3.py --gpu 1 --sweep --start 6 --end 11  # GPU 1: exps 6-10
    uv run train_phase3.py --exp 3                              # single experiment
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
from dataclasses import dataclass, field
from typing import Optional, List

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

DEFAULT_SEQ_LEN   = 1024
TARGET_TOKENS_PER_STEP = 32_768  # 32K effective tokens per optimizer step

DNA_ALPHABET      = "ACGT"

CACHE_DIR      = Path.home() / ".cache" / "dna-diffusion"
RAW_DIR        = CACHE_DIR / "raw"
NUCEL_DATA_DIR = CACHE_DIR / "nucel_data"
NUCEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DIR    = Path(__file__).parent
RESULTS_FILE   = PROJECT_DIR / "results_phase3.tsv"
CKPT_DIR       = PROJECT_DIR / "checkpoints_phase3"
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

_TOK       = None
_NT_TO_ID  = None
_MASK_ID   = None
_UNK_ID    = None
_PAD_ID    = None


def get_tokenizer():
    global _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID
    if _TOK is not None:
        return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID

    from huggingface_hub import hf_hub_download
    print(f"Loading NucEL vocab from {NUCEL_REPO} ...")
    vocab_path = hf_hub_download(NUCEL_REPO, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)

    class _Tok:
        pass
    tok = _Tok()
    tok.vocab = vocab
    tok.vocab_size = len(vocab)
    tok.mask_token_id = vocab.get("\u2282", 4)  # ⊂
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
    _MASK_ID = tok.mask_token_id if tok.mask_token_id is not None else vocab.get("\u2282", 4)
    _PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0

    nt_to_id = {}
    for nt in DNA_ALPHABET:
        ids = tok.encode(nt, add_special_tokens=False)
        nt_to_id[nt] = ids[0] if ids else unk_id
        nt_to_id[nt.lower()] = nt_to_id[nt]
    for nt in "NnRrYyWwSsKkMmBbDdHhVv":
        nt_to_id[nt] = unk_id
    _NT_TO_ID = nt_to_id

    print(f"  vocab_size={tok.vocab_size}, mask_id={_MASK_ID}, unk_id={_UNK_ID}")
    return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID


def tokenize_nt(seq, nt_to_id, unk_id):
    return [nt_to_id.get(c, unk_id) for c in seq]


# ---------------------------------------------------------------------------
# Data preparation
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
        print(f"  → {n_seqs:,} sequences saved to {out}")

    print("NucEL data ready.")


class NucELDataset(Dataset):
    def __init__(self, split="train", seq_len=DEFAULT_SEQ_LEN):
        self.seq_len = seq_len
        path = NUCEL_DATA_DIR / f"{split}_{seq_len}.bin"
        if not path.exists():
            build_nucel_data(seq_len)
        self.data  = np.memmap(str(path), dtype=np.uint16, mode="r")
        self.n_seq = len(self.data) // seq_len
        assert self.n_seq > 0

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
# Noise schedules (same as train_nucel.py)
# ---------------------------------------------------------------------------

class UniformSchedule:
    name = "uniform"
    def __init__(self, noise_steps=128, mask_id=1):
        self.noise_steps = noise_steps
        self.mask_id = mask_id

    def sample_t(self, B, device):
        return torch.randint(1, self.noise_steps + 1, (B,), device=device)

    def forward_process(self, x, t):
        prob = t.float() / self.noise_steps
        mask = torch.rand_like(x, dtype=torch.float) < prob.unsqueeze(1)
        xn   = x.clone()
        xn[mask] = self.mask_id
        return xn, mask


def make_schedule(name, noise_steps, mask_id):
    return UniformSchedule(noise_steps, mask_id)


# ---------------------------------------------------------------------------
# Linear attention replacements
# ---------------------------------------------------------------------------

class PerformerAttentionWrapper(nn.Module):
    """Wraps performer_pytorch SelfAttention to replace standard attention."""
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        from performer_pytorch import SelfAttention
        # Performer's SelfAttention includes qkv projection internally
        self.attn = SelfAttention(dim=hidden_size, heads=num_heads, causal=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, **kwargs):
        # hidden_states: (B, L, H)
        residual = hidden_states
        out = self.attn(hidden_states)
        return self.norm(residual + out)


class RetNetAttentionWrapper(nn.Module):
    """Wraps fla MultiScaleRetention for chunk-wise (parallel) mode."""
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        from fla.layers import MultiScaleRetention
        self.attn = MultiScaleRetention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        result = self.attn(hidden_states)
        out = result[0]  # (hidden_states, attn_weights, cache)
        return self.norm(residual + out)


class GLAAttentionWrapper(nn.Module):
    """Wraps fla GatedLinearAttention."""
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        from fla.layers import GatedLinearAttention
        self.attn = GatedLinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        result = self.attn(hidden_states)
        out = result[0]  # returns (hidden_states, attn_weights, cache)
        return self.norm(residual + out)


class Mamba2Wrapper(nn.Module):
    """Wraps native mamba_ssm Mamba2 as attention replacement."""
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        from mamba_ssm import Mamba2
        self.attn = Mamba2(d_model=hidden_size, d_state=64, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        out = self.attn(hidden_states)  # returns Tensor directly
        return self.norm(residual + out)


class IdentityAttentionWrapper(nn.Module):
    """Standard attention — no replacement (for baseline)."""
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.norm(hidden_states)


def get_attention_wrapper(attn_type, hidden_size=512, num_heads=8):
    if attn_type == "standard":
        return IdentityAttentionWrapper(hidden_size, num_heads)
    elif attn_type == "performer":
        return PerformerAttentionWrapper(hidden_size, num_heads)
    elif attn_type == "retnet":
        return RetNetAttentionWrapper(hidden_size, num_heads)
    elif attn_type == "gla":
        return GLAAttentionWrapper(hidden_size, num_heads)
    elif attn_type == "mamba2":
        return Mamba2Wrapper(hidden_size, num_heads)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")


# ---------------------------------------------------------------------------
# NucEL Diffusion Model (Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class Phase3Config:
    seq_len:       int   = DEFAULT_SEQ_LEN
    noise_steps:   int   = 128
    time_embed:    str   = "additive"    # "additive" | "adaln"
    attention_type: str  = "standard"    # "standard" | "performer" | "retnet" | "gla" | "mamba2" | "hybrid_gla"
    hybrid_every:  int   = 3             # For hybrid: global attn every N layers
    dropout:       float = 0.0


class NucELDiffusionV3(nn.Module):
    """
    Phase 3 DiffusionBERT with pluggable attention mechanisms.
    Strategy: Load NucEL backbone, then replace its self-attention layers
    with the chosen linear attention variant.
    """

    def __init__(self, cfg: Phase3Config, backbone: nn.Module, mask_id: int, schedule):
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

        # Find transformer layers
        self._layers = self._find_layers()
        n = len(self._layers) if self._layers else 0
        print(f"  Backbone transformer layers found: {n}")

        # Replace attention in backbone layers
        if cfg.attention_type != "standard" and self._layers is not None:
            self._replace_attention(cfg.attention_type, cfg.hybrid_every)

        # Time injection
        if cfg.time_embed == "additive":
            self.time_inject = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                for _ in range(max(n, 1))
            ])
            for m in self.time_inject:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        elif cfg.time_embed == "adaln":
            self.adaln_proj = nn.Linear(
                self.hidden_size, 2 * self.hidden_size * max(n, 1), bias=True
            )
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)

        # LM head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        self.dropout_layer = nn.Dropout(cfg.dropout)

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
        for _, m in self.nucel.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 5:
                return m
        return None

    def _replace_attention(self, attn_type, hybrid_every=3):
        """Replace self-attention in each transformer layer with linear attention."""
        if self._layers is None:
            return

        for i, layer in enumerate(self._layers):
            # For hybrid: every Nth layer keeps standard attention
            if attn_type.startswith("hybrid_") and (i % hybrid_every == hybrid_every - 1):
                continue  # keep standard attention for this layer

            actual_type = attn_type
            if attn_type.startswith("hybrid_"):
                actual_type = attn_type.replace("hybrid_", "")

            # Find and replace the attention submodule
            attn_mod = None
            for name, mod in layer.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    attn_mod = (name, mod)
                    break

            if attn_mod is None:
                print(f"  Layer {i}: no attention module found, skipping")
                continue

            attn_name, attn_module = attn_mod
            # Create replacement
            try:
                replacement = get_attention_wrapper(
                    actual_type, self.hidden_size, num_heads=8
                )
                # Navigate to parent and replace
                parts = attn_name.split('.')
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], replacement)
                print(f"  Layer {i}: replaced {attn_name} with {actual_type}")
            except Exception as e:
                print(f"  Layer {i}: failed to replace attention ({e})")

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

    def forward(self, input_ids, t):
        t_sin    = self._sinusoidal(t, self.hidden_size)
        t_hidden = self.time_proj(t_sin)

        handles = []
        if self._layers is not None:
            if self.te_type == "additive" and hasattr(self, 'time_inject'):
                for i, layer in enumerate(self._layers):
                    if i >= len(self.time_inject):
                        break
                    delta = self.time_inject[i](t_hidden)

                    def _hook_add(mod, inp, out, d=delta):
                        h = out[0] if isinstance(out, tuple) else out
                        h = h + d.unsqueeze(1)
                        return (h,) + out[1:] if isinstance(out, tuple) else h

                    handles.append(layer.register_forward_hook(_hook_add))

            elif self.te_type == "adaln" and hasattr(self, 'adaln_proj'):
                n   = len(self._layers)
                raw = self.adaln_proj(t_hidden)
                raw = raw.view(-1, n, 2, self.hidden_size)

                for i, layer in enumerate(self._layers):
                    if i >= n:
                        break
                    scale = raw[:, i, 0, :]
                    shift = raw[:, i, 1, :]

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
        hidden = self.dropout_layer(hidden)
        return self.lm_head(hidden)

    def compute_loss(self, x_clean, t=None):
        if t is None:
            t = self.schedule.sample_t(x_clean.shape[0], x_clean.device)
        x_noisy, mask = self.schedule.forward_process(x_clean, t)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_clean.device, requires_grad=True)
        logits = self.forward(x_noisy, t)
        return F.cross_entropy(logits[mask], x_clean[mask])

    @torch.no_grad()
    def get_embeddings(self, input_ids, pool="mean"):
        """Extract embeddings (mean pooling over sequence)."""
        out = self.nucel(input_ids=input_ids)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if pool == "mean":
            return hidden.mean(dim=1)
        return hidden[:, 0]  # CLS-like

    @torch.no_grad()
    def sample(self, n, seq_len, device, steps=None):
        steps = steps or min(self.noise_steps, 64)
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

def build_model(cfg: Phase3Config, device):
    from transformers import AutoModel
    print(f"Loading NucEL backbone ({NUCEL_REPO}) ...")
    backbone = AutoModel.from_pretrained(
        NUCEL_REPO,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    _, _, mask_id, _, _ = get_tokenizer()
    schedule = make_schedule("uniform", cfg.noise_steps, mask_id)
    model = NucELDiffusionV3(cfg, backbone, mask_id, schedule).to(device)
    return model


# ---------------------------------------------------------------------------
# GB Linear Probe Evaluation
# ---------------------------------------------------------------------------

USCS_API_URL = "https://api.genome.ucsc.edu/getData/sequence"
GB_SEQ_CACHE_DIR = Path.home() / ".cache" / "gb_sequences"
GB_SEQ_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_seq_ucsc(chrom, start, end):
    url = f"{USCS_API_URL}?genome=hg38;chrom={chrom};start={start};end={end}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("dna", "").upper()
    except Exception:
        return ""


def load_gb_dataset(benchmark_name, sample_frac=0.05, max_seqs=400):
    """Load GB dataset from local genomic-benchmarks cache.
    Returns list of (sequence_string, label_int) using official train/test split."""
    base = Path.home() / '.genomic_benchmarks' / benchmark_name

    # Try local cache first (has sequences + labels in folder structure)
    if base.exists():
        results = []
        for split_name in ['train', 'test']:
            split_path = base / split_name
            if not split_path.exists():
                continue
            class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
            if len(class_dirs) < 2:
                continue

            for label_idx, class_dir in enumerate(class_dirs):
                files = sorted(class_dir.glob('*.txt'))
                n = min(max(1, int(len(files) * sample_frac)), max_seqs // (2 * len(class_dirs)))
                np.random.seed(42)
                chosen = np.random.choice(len(files), min(n, len(files)), replace=False)
                for fi in chosen:
                    seq = files[fi].read_text().strip()
                    if len(seq) >= 10:
                        results.append((seq, label_idx))
        return results

    # Fallback: HF datasets (slow, no proper labels)
    print(f"    Local cache not found for {benchmark_name}, skipping")
    return []


def tokenize_sequences(seqs, nt_to_id, unk_id, pad_id, seq_len):
    """Tokenize and pad/truncate sequences."""
    ids_list = []
    for seq in seqs:
        ids = tokenize_nt(seq, nt_to_id, unk_id)
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [pad_id] * (seq_len - len(ids))
        ids_list.append(ids)
    return torch.tensor(ids_list, dtype=torch.long)


@torch.no_grad()
def extract_embeddings(model, input_ids, device, batch_size=16):
    """Extract mean-pooled embeddings in batches."""
    model.eval()
    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            embs = model.get_embeddings(batch, pool="mean")
        all_embs.append(embs.float().cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def extract_frozen_nucel_embeddings(input_ids, device, batch_size=16, backbone=None):
    """Extract embeddings from frozen NucEL (no diffusion wrapper)."""
    load_backbone = backbone is None
    if load_backbone:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(
            NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
    backbone.eval()

    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = backbone(input_ids=batch)
            hidden = out.last_hidden_state
            embs = hidden.mean(dim=1)
        all_embs.append(embs.float().cpu())

    if load_backbone:
        del backbone
        gc.collect(); torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(embeddings, labels, num_epochs=3, lr=1e-3):
    """Train a simple linear classifier on embeddings."""
    device = embeddings.device
    n_classes = len(set(labels.tolist()))
    probe = LinearProbe(embeddings.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    labels_t = labels.to(device)
    dataset = torch.utils.data.TensorDataset(embeddings, labels_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    probe.train()
    for epoch in range(num_epochs):
        for xb, yb in loader:
            logits = probe(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return probe


def evaluate_probe(probe, embeddings, labels):
    """Return accuracy."""
    probe.eval()
    with torch.no_grad():
        logits = probe(embeddings.to(next(probe.parameters()).device))
        preds = logits.argmax(dim=-1)
        acc = (preds == labels.to(preds.device)).float().mean().item()
    return acc


def evaluate_gb_linear_probe(model, device, seq_len=DEFAULT_SEQ_LEN, frozen_nucel=False):
    """GB evaluation via linear probe → accuracy per subtask."""
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    results = {}

    # Load frozen backbone once for all benchmarks
    frozen_backbone = None
    if frozen_nucel:
        from transformers import AutoModel
        frozen_backbone = AutoModel.from_pretrained(
            NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        frozen_backbone.eval()

    for bname in GENOMIC_BENCHMARKS:
        try:
            data = load_gb_dataset(bname)
            if len(data) < 10:
                print(f"    GB {bname}: too few samples ({len(data)})")
                results[bname] = float("nan")
                continue

            seqs, labels = zip(*data)
            labels = torch.tensor(labels, dtype=torch.long)

            n = len(seqs)
            np.random.seed(42)
            perm = np.random.permutation(n)
            split = n // 2
            train_idx, test_idx = perm[:split], perm[split:]

            train_seqs = [seqs[i] for i in train_idx]
            test_seqs  = [seqs[i] for i in test_idx]
            train_labels = labels[list(train_idx)]
            test_labels  = labels[list(test_idx)]

            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, seq_len)
            test_ids  = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, seq_len)

            if frozen_nucel:
                train_embs = extract_frozen_nucel_embeddings(train_ids, device, backbone=frozen_backbone)
                test_embs  = extract_frozen_nucel_embeddings(test_ids, device, backbone=frozen_backbone)
            else:
                model.eval()
                train_embs = extract_embeddings(model, train_ids, device)
                test_embs  = extract_embeddings(model, test_ids, device)

            train_embs = train_embs.to(device)
            probe = train_linear_probe(train_embs, train_labels)

            test_embs = test_embs.to(device)
            acc = evaluate_probe(probe, test_embs, test_labels)
            results[bname] = acc
            print(f"    GB {bname}: acc={acc:.4f} ({len(train_seqs)} train, {len(test_seqs)} test)")

            del probe, train_embs, test_embs
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"    GB {bname}: failed ({e})")
            traceback.print_exc()
            results[bname] = float("nan")

    if frozen_backbone is not None:
        del frozen_backbone
        gc.collect(); torch.cuda.empty_cache()
    if not frozen_nucel:
        model.train()
    return results


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val_loss(model, batch_size, device, seq_len=DEFAULT_SEQ_LEN):
    model.eval()
    loader = make_loader(min(batch_size, 4), "val", seq_len, num_workers=0)
    total_loss, n_ok = 0.0, 0

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

    model.train()
    return total_loss / max(n_ok, 1)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(name, params, lr, wd):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    elif name == "muon":
        from muon import Muon
        return Muon(params, lr=lr, momentum=0.95)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # 0: Frozen NucEL baseline (no training)
    {"id": "nucel_frozen_probe", "optimizer": "none", "lr": 0,
     "time_embed": "additive", "attention_type": "standard",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.0, "frozen_probe": True},

    # 1: Diffusion baseline
    {"id": "diffusion_baseline", "optimizer": "adamw", "lr": 3e-4,
     "time_embed": "additive", "attention_type": "standard",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 2: Muon optimizer
    {"id": "muon", "optimizer": "muon", "lr": 0.02,
     "time_embed": "additive", "attention_type": "standard",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 3: AdaLN (DiT-style)
    {"id": "adaln_dit", "optimizer": "adamw", "lr": 3e-4,
     "time_embed": "adaln", "attention_type": "standard",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 4: Performer
    {"id": "linear_attn_performer", "optimizer": "adamw", "lr": 3e-4,
     "time_embed": "additive", "attention_type": "performer",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 5: RetNet
    {"id": "linear_attn_retnet", "optimizer": "adamw", "lr": 3e-4,
     "time_embed": "additive", "attention_type": "retnet",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 6: GLA
    {"id": "linear_attn_gla", "optimizer": "adamw", "lr": 3e-4,
     "time_embed": "additive", "attention_type": "gla",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 7: Mamba-2
    {"id": "linear_attn_mamba2", "optimizer": "adamw", "lr": 3e-4,
     "time_embed": "additive", "attention_type": "mamba2",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 8: Hybrid (GLA + global attention every 3 layers)
    {"id": "hybrid_attn", "optimizer": "adamw", "lr": 3e-4,
     "time_embed": "additive", "attention_type": "hybrid_gla",
     "hybrid_every": 3,
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 9: Muon + AdaLN
    {"id": "muon_adaln", "optimizer": "muon", "lr": 0.02,
     "time_embed": "adaln", "attention_type": "standard",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},

    # 10: Muon + best linear attention (defaults to GLA)
    {"id": "muon_linear", "optimizer": "muon", "lr": 0.02,
     "time_embed": "additive", "attention_type": "gla",
     "noise_steps": 128, "dropout": 0.0, "seq_len": DEFAULT_SEQ_LEN,
     "batch_size": 8, "weight_decay": 0.01},
]

assert len(EXPERIMENTS) == 11, f"Expected 11 experiments, got {len(EXPERIMENTS)}"


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def init_results():
    if RESULTS_FILE.exists():
        return
    gb_cols = [f"gb_acc_{b}" for b in GENOMIC_BENCHMARKS]
    header  = [
        "exp_id", "exp_idx", "timestamp",
        "optimizer", "lr", "time_embed", "attention_type",
        "total_params_M", "trainable_params_M",
        "train_steps", "training_seconds", "total_seconds",
        "final_train_loss", "val_loss",
        "peak_vram_mb",
    ] + gb_cols
    with open(RESULTS_FILE, "w") as f:
        f.write("\t".join(header) + "\n")


def log_result(exp_idx, cfg, info, gb):
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
        cfg.get("attention_type", ""),
        f"{info.get('total_M', 0):.1f}",
        f"{info.get('trainable_M', 0):.1f}",
        str(info.get("steps", 0)),
        f"{info.get('train_secs', 0):.1f}",
        f"{info.get('total_secs', 0):.1f}",
        fmt(info.get("final_loss", float("nan"))),
        fmt(info.get("val_loss", float("nan"))),
        f"{info.get('peak_vram_mb', 0):.0f}",
    ] + [fmt(gb.get(b, float("nan"))) for b in GENOMIC_BENCHMARKS]

    with open(RESULTS_FILE, "a") as f:
        f.write("\t".join(row) + "\n")
    print(f"Logged → {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def estimate_time_per_epoch(seq_len, batch_size, device):
    """Quick benchmark: run 10 steps to estimate time per epoch."""
    print("  Benchmarking training speed (10 steps)...")
    loader = make_loader(batch_size, "train", seq_len, num_workers=0)
    data_iter = iter(loader)
    batch = next(data_iter).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    # We'll return the benchmark results from the caller
    return None


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
    build_nucel_data(seq_len)

    # ── Special case: frozen probe (no training) ──
    if exp_cfg.get("frozen_probe", False):
        print("Running frozen NucEL linear probe (no training)...")
        gb = evaluate_gb_linear_probe(None, device, seq_len, frozen_nucel=True)
        info = {
            "total_M": 0, "trainable_M": 0, "steps": 0,
            "train_secs": 0, "total_secs": time.time() - t_wall_start,
            "final_loss": float("nan"), "val_loss": float("nan"),
            "peak_vram_mb": 0,
        }
        log_result(exp_idx, exp_cfg, info, gb)
        return gb

    # ── Build model ──
    cfg = Phase3Config(
        seq_len        = seq_len,
        noise_steps    = exp_cfg.get("noise_steps", 128),
        time_embed     = exp_cfg.get("time_embed", "additive"),
        attention_type = exp_cfg.get("attention_type", "standard"),
        hybrid_every   = exp_cfg.get("hybrid_every", 3),
        dropout        = exp_cfg.get("dropout", 0.0),
    )
    model = build_model(cfg, device)

    total_p, trainable_p = model.count_params()
    print(f"Total params:     {total_p:,}  ({total_p/1e6:.1f}M)")
    print(f"Trainable params: {trainable_p:,}  ({trainable_p/1e6:.1f}M)")

    # ── Optimizer ──
    lr       = exp_cfg.get("lr", 3e-4)
    wd       = exp_cfg.get("weight_decay", 0.01)
    opt_name = exp_cfg.get("optimizer", "adamw")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(opt_name, trainable_params, lr, wd)

    # ── Data ──
    bs             = exp_cfg.get("batch_size", 8)
    train_loader   = make_loader(bs, "train", seq_len)
    toks_per_step  = bs * seq_len
    grad_accum     = max(1, TARGET_TOKENS_PER_STEP // toks_per_step)
    print(f"batch={bs}, grad_accum={grad_accum}, eff_tokens/step={toks_per_step*grad_accum:,}")

    # ── Estimate time for 1 epoch ──
    n_batches = len(train_loader)
    est_steps_per_epoch = n_batches // grad_accum

    # Quick speed benchmark: 5 steps
    model.train()
    data_iter = iter(train_loader)
    torch.cuda.synchronize()
    bench_start = time.time()
    for _ in range(min(5, n_batches)):
        opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            batch = batch.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.compute_loss(batch) / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
    torch.cuda.synchronize()
    bench_time = time.time() - bench_start
    ms_per_step = bench_time / 5 * 1000
    est_epoch_time = est_steps_per_epoch * ms_per_step / 1000
    print(f"  Speed benchmark: {ms_per_step:.0f}ms/step, ~{est_epoch_time/60:.1f}min/epoch")

    # Time budget: enough for 1 epoch + evaluation, max 90 min
    time_budget = min(max(est_epoch_time * 1.3 + 300, 600), 5400)
    print(f"  Time budget: {time_budget/60:.1f} min")

    # ── Training loop ──
    data_iter = iter(train_loader)
    train_time  = 0.0
    step        = 0
    final_loss  = float("nan")
    smooth_loss = 0.0

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

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

        train_loss = loss.item() * grad_accum
        if not math.isnan(train_loss):
            final_loss = train_loss

        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 2:
            train_time += dt

        b = 0.95
        smooth_loss = b * smooth_loss + (1 - b) * train_loss
        debi = smooth_loss / (1 - b ** (step + 1))

        # Learning rate schedule (cosine)
        if est_steps_per_epoch > 0:
            warmup = max(50, est_steps_per_epoch // 20)
            if step < warmup:
                lr_s = step / warmup
            else:
                progress = (step - warmup) / max(est_steps_per_epoch - warmup, 1)
                lr_s = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = lr * lr_s

        rem = max(0, time_budget - train_time)
        if step % 50 == 0:
            print(
                f"  step {step:05d} | loss={debi:.4f} | "
                f"lr={lr * lr_s:.2e} | dt={dt*1000:.0f}ms | rem={rem:.0f}s"
            )

        if math.isnan(train_loss):
            print("  NaN loss — stopping.")
            break

        step += 1
        if train_time >= time_budget:
            print(f"  Time budget reached ({time_budget/60:.1f} min)")
            break

    print(f"  Training done: {step} steps in {train_time:.0f}s")

    del opt
    gc.collect(); torch.cuda.empty_cache()

    # ── Validation ──
    print("── Validation ──")
    try:
        val_loss = evaluate_val_loss(model, bs, device, seq_len)
        print(f"  val_loss: {val_loss:.4f}")
    except Exception as e:
        print(f"  Val eval failed: {e}")
        val_loss = float("nan")

    # ── GB Linear Probe ──
    print("── GB Linear Probe ──")
    try:
        gb = evaluate_gb_linear_probe(model, device, seq_len)
    except Exception as e:
        print(f"  GB eval failed: {e}")
        traceback.print_exc()
        gb = {b: float("nan") for b in GENOMIC_BENCHMARKS}

    # ── Checkpoint ──
    ckpt = CKPT_DIR / f"{exp_idx:03d}_{exp_cfg.get('id','exp')}.pt"
    try:
        torch.save({
            "exp_idx":    exp_idx,
            "config":     exp_cfg,
            "state_dict": {k: v for k, v in model.state_dict().items()
                           if not k.startswith("nucel.")},
            "val_loss":   val_loss,
            "gb_metrics": gb,
            "final_loss": final_loss,
            "step":       step,
        }, ckpt)
        print(f"  Checkpoint → {ckpt}")
    except Exception as e:
        print(f"  Checkpoint save failed: {e}")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    t_wall_end = time.time()

    info = {
        "total_M":      total_p / 1e6,
        "trainable_M":  trainable_p / 1e6,
        "steps":        step,
        "train_secs":   train_time,
        "total_secs":   t_wall_end - t_wall_start,
        "final_loss":   final_loss,
        "val_loss":     val_loss,
        "peak_vram_mb": peak_vram,
    }
    log_result(exp_idx, exp_cfg, info, gb)

    del model
    gc.collect(); torch.cuda.empty_cache()
    return gb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Linear Attention + GB Linear Probe")
    parser.add_argument("--sweep",  action="store_true")
    parser.add_argument("--exp",    type=int)
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--end",    type=int, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU {args.gpu}")

    init_results()

    start = args.start
    end   = args.end if args.end is not None else len(EXPERIMENTS)

    if args.exp is not None:
        if args.exp >= len(EXPERIMENTS):
            print(f"Error: experiment {args.exp} out of range")
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
                blank_gb = {b: float("nan") for b in GENOMIC_BENCHMARKS}
                blank_i  = {k: 0.0 for k in
                            ["total_M", "trainable_M", "steps", "train_secs",
                             "total_secs", "final_loss", "val_loss", "peak_vram_mb"]}
                blank_i["final_loss"] = float("nan")
                blank_i["val_loss"] = float("nan")
                log_result(i, cfg, blank_i, blank_gb)
            print()

        print(f"\n{'='*70}")
        print(f"Sweep {start}–{end-1} complete!")
    else:
        run_experiment(0, EXPERIMENTS[0])


if __name__ == "__main__":
    main()
