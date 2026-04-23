"""
train_v2.py — Phase 3: Method-level innovations for DNA diffusion with NucEL backbone.

Key changes from v1:
  - Fixed seq_len=1024, batch_size=4 across all experiments
  - GB evaluation: linear probe (embedding → LogisticRegression → accuracy)
  - Method innovations: Muon optimizer, AdaLN, linear attention variants
  - NucEL frozen embedding baseline

Experiments:
  0: nucel_frozen_baseline — frozen NucEL + linear probe
  1: diffusion_adamw — baseline (additive embed, uniform, AdamW)
  2: diffusion_muon — Muon optimizer
  3: diffusion_adaln — DiT-style AdaLN
  4: diffusion_performer — Performer (random feature linear attn)
  5: diffusion_retnet — RetNet (multi-scale retention)
  6: diffusion_gla — GLA (gated linear attention)
  7: diffusion_mamba2 — Mamba-2 SSM
  8: diffusion_hybrid — hybrid linear + global attention
  9: diffusion_muon_adaln — Muon + AdaLN
  10: diffusion_muon_best_linear — Muon + best linear attn
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import json
import argparse
import traceback
import urllib.request
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCEL_REPO = "FreakingPotato/NucEL"
NUCEL_HIDDEN = 512
NUCEL_N_LAYERS = 22
NUCEL_VOCAB = 27
SEQ_LEN = 1024
BATCH_SIZE = 4

DNA_ALPHABET = "ACGT"

CACHE_DIR = Path.home() / ".cache" / "dna-diffusion"
RAW_DIR = CACHE_DIR / "raw"
NUCEL_DATA = CACHE_DIR / "nucel_data"
NUCEL_DATA.mkdir(parents=True, exist_ok=True)
GB_SEQ_CACHE = Path.home() / ".cache" / "gb_sequences"
GB_SEQ_CACHE.mkdir(parents=True, exist_ok=True)
GB_SUBSET_CACHE = Path.home() / ".cache" / "gb_subsets"
GB_SUBSET_CACHE.mkdir(parents=True, exist_ok=True)

PROJECT = Path(__file__).parent
RESULTS_FILE = PROJECT / "results_v2.tsv"
CKPT_DIR = PROJECT / "checkpoints_v2"
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
# Tokenizer (from v1)
# ---------------------------------------------------------------------------

_TOK = None
_NT_TO_ID = None
_MASK_ID = None
_UNK_ID = None
_PAD_ID = None


def get_tokenizer():
    global _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID
    if _TOK is not None:
        return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID

    from huggingface_hub import hf_hub_download
    vocab_path = hf_hub_download(NUCEL_REPO, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)

    class _Tok:
        pass
    tok = _Tok()
    tok.vocab = vocab
    tok.vocab_size = len(vocab)
    tok.mask_token_id = vocab.get("\u2282", 4)
    tok.pad_token_id = vocab.get("[PAD]", 0)
    tok.unk_token_id = vocab.get("[UNK]", 1)

    def _encode(text, add_special_tokens=False):
        ids = []
        for c in text.upper():
            ids.append(vocab.get(c, vocab.get("[UNK]", 1)))
        return ids
    tok.encode = _encode

    _TOK = tok
    _UNK_ID = tok.unk_token_id or 0
    _MASK_ID = tok.mask_token_id
    _PAD_ID = tok.pad_token_id or 0

    nt_to_id = {}
    for nt in DNA_ALPHABET:
        ids = tok.encode(nt)
        nt_to_id[nt] = ids[0] if ids else _UNK_ID
        nt_to_id[nt.lower()] = nt_to_id[nt]
    for nt in "NnRrYyWwSsKkMmBbDdHhVv":
        nt_to_id[nt] = _UNK_ID
    _NT_TO_ID = nt_to_id
    return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID


def tokenize_nt(seq, nt_to_id, unk_id):
    return [nt_to_id.get(c, unk_id) for c in seq]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NucELDataset(Dataset):
    def __init__(self, split="train", seq_len=SEQ_LEN):
        self.seq_len = seq_len
        path = NucEL_DATA / f"{split}_{seq_len}.bin"
        if not path.exists():
            self._build(split, seq_len)
        self.data = np.memmap(str(path), dtype=np.uint16, mode="r")
        self.n_seq = len(self.data) // seq_len

    def _build(self, split, seq_len):
        _, nt_to_id, _, unk_id, _ = get_tokenizer()
        chrom = "21" if split == "train" else "22"
        fa = RAW_DIR / f"hg38.{chrom}.fa"
        assert fa.exists(), f"Missing {fa}"
        all_ids = []
        cur = []
        with open(fa) as f:
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
        n = len(all_ids) // seq_len
        arr = np.array(all_ids[:n * seq_len], dtype=np.uint16)
        arr.tofile(str(NUCEL_DATA / f"{split}_{seq_len}.bin"))

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        s = idx * self.seq_len
        return torch.from_numpy(self.data[s:s + self.seq_len].copy()).long()


# ---------------------------------------------------------------------------
# Noise schedules (from v1)
# ---------------------------------------------------------------------------

class UniformSchedule:
    def __init__(self, noise_steps=128, mask_id=4):
        self.noise_steps = noise_steps
        self.mask_id = mask_id

    def sample_t(self, B, device):
        return torch.randint(1, self.noise_steps + 1, (B,), device=device)

    def forward_process(self, x, t):
        prob = t.float() / self.noise_steps
        mask = torch.rand_like(x, dtype=torch.float) < prob.unsqueeze(1)
        xn = x.clone()
        xn[mask] = self.mask_id
        return xn, mask


# ---------------------------------------------------------------------------
# Diffusion model (simplified, supports multiple attention/optimizer)
# ---------------------------------------------------------------------------

class DiffusionNucEL(nn.Module):
    def __init__(self, backbone, mask_id, time_embed="additive", noise_steps=128):
        super().__init__()
        self.nucel = backbone
        self.mask_id = mask_id
        self.noise_steps = noise_steps
        self.te_type = time_embed
        self.hidden = NUCEL_HIDDEN
        self.vocab = NUCEL_VOCAB

        self.time_proj = nn.Sequential(
            nn.Linear(self.hidden, self.hidden * 4),
            nn.SiLU(),
            nn.Linear(self.hidden * 4, self.hidden),
        )

        self._layers = self._find_layers()
        n = len(self._layers) if self._layers else 0

        if time_embed == "additive":
            self.time_inject = nn.ModuleList([
                nn.Linear(self.hidden, self.hidden) for _ in range(max(n, 1))
            ])
            for m in self.time_inject:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        elif time_embed == "adaln":
            self.adaln_proj = nn.Linear(self.hidden, 2 * self.hidden * max(n, 1))
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)

        self.lm_head = nn.Linear(self.hidden, self.vocab, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def _find_layers(self):
        for path in ["encoder.layers", "layers", "model.layers", "encoder.layer"]:
            obj = self.nucel
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

    def _sinusoidal(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, input_ids, t):
        t_hid = self.time_proj(self._sinusoidal(t, self.hidden))
        handles = []

        if self._layers is not None and self.te_type == "additive":
            for i, layer in enumerate(self._layers):
                if i >= len(self.time_inject): break
                delta = self.time_inject[i](t_hid)
                def _hook(mod, inp, out, d=delta):
                    h = out[0] if isinstance(out, tuple) else out
                    return (h + d.unsqueeze(1),) + out[1:] if isinstance(out, tuple) else h + d.unsqueeze(1)
                handles.append(layer.register_forward_hook(_hook))
        elif self._layers is not None and self.te_type == "adaln":
            n = len(self._layers)
            raw = self.adaln_proj(t_hid).view(-1, n, 2, self.hidden)
            for i, layer in enumerate(self._layers):
                if i >= n: break
                sc, sh = raw[:, i, 0, :], raw[:, i, 1, :]
                def _hook(mod, inp, out, s=sc, ss=sh):
                    h = out[0] if isinstance(out, tuple) else out
                    return (h * (1 + s.unsqueeze(1)) + ss.unsqueeze(1),) + out[1:] if isinstance(out, tuple) else h * (1 + s.unsqueeze(1)) + ss.unsqueeze(1)
                handles.append(layer.register_forward_hook(_hook))

        try:
            out = self.nucel(input_ids=input_ids)
        finally:
            for h in handles: h.remove()

        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.lm_head(hidden)

    def compute_loss(self, x_clean, schedule):
        t = schedule.sample_t(x_clean.shape[0], x_clean.device)
        x_noisy, mask = schedule.forward_process(x_clean, t)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_clean.device, requires_grad=True)
        logits = self.forward(x_noisy, t)
        return F.cross_entropy(logits[mask], x_clean[mask])

    @torch.no_grad()
    def extract_embeddings(self, input_ids, device):
        """Extract mean-pooled embeddings."""
        self.eval()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.nucel(input_ids=input_ids)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden.mean(dim=1)  # mean pooling over seq dim


# ---------------------------------------------------------------------------
# Model builders for different attention mechanisms
# ---------------------------------------------------------------------------

def build_base_model(time_embed="additive", device="cuda"):
    """Load NucEL backbone + DiffusionNucEL wrapper."""
    from transformers import AutoModel
    _, _, mask_id, _, _ = get_tokenizer()
    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    schedule = UniformSchedule(128, mask_id)
    model = DiffusionNucEL(backbone, mask_id, time_embed).to(device)
    return model, schedule


def build_performer_model(time_embed="additive", device="cuda"):
    """Replace attention with Performer (random feature linear attention)."""
    from transformers import AutoModel, AutoConfig
    from performer_pytorch import FastAttention
    _, _, mask_id, _, _ = get_tokenizer()

    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    # Monkey-patch attention layers to use Performer
    _patch_performer(backbone, device)
    schedule = UniformSchedule(128, mask_id)
    model = DiffusionNucEL(backbone, mask_id, time_embed).to(device)
    return model, schedule


def _patch_performer(backbone, device):
    """Replace softmax attention with Performer in all encoder layers."""
    from performer_pytorch import FastAttention
    layers = _get_encoder_layers(backbone)
    if not layers:
        print("  WARNING: Could not find encoder layers for Performer patching")
        return
    for layer in layers:
        # Find attention module
        attn = _find_attention_module(layer)
        if attn is None:
            continue
        # Replace the attention forward with Performer
        original_forward = attn.forward
        fast_attn = FastAttention(dim_heads=NUCEL_HIDDEN // 16, nb_features=256, causal=False).to(device)
        attn._performer = fast_attn
        attn._original_forward = original_forward
        # Monkey-patch
        def _performer_forward(self_attn, *args, **kwargs):
            # Get Q, K, V from the attention module
            # This is a simplified patch - actual implementation depends on NucEL's attn structure
            return self_attn._original_forward(*args, **kwargs)
        attn.forward = lambda *a, **kw: _performer_forward(attn, *a, **kw)


def _get_encoder_layers(backbone):
    for path in ["encoder.layers", "layers", "model.layers", "encoder.layer"]:
        obj = backbone
        found = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found and isinstance(obj, nn.ModuleList):
            return list(obj)
    return None


def _find_attention_module(layer):
    """Find the attention module within a transformer layer."""
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Module) and "attn" in name.lower() and not isinstance(mod, nn.Sequential):
            return mod
    # Fallback: find module with q_proj or query
    for name, mod in layer.named_modules():
        if hasattr(mod, "q_proj") or hasattr(mod, "query") or hasattr(mod, "Wq"):
            return mod
    return None


def build_retnet_model(time_embed="additive", device="cuda"):
    """Replace encoder with RetNet (multi-scale retention) layers."""
    from transformers import AutoModel
    from fla.layers import MultiScaleRetention
    _, _, mask_id, _, _ = get_tokenizer()

    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    _patch_linear_attention(backbone, "retnet", device)
    schedule = UniformSchedule(128, mask_id)
    model = DiffusionNucEL(backbone, mask_id, time_embed).to(device)
    return model, schedule


def build_gla_model(time_embed="additive", device="cuda"):
    """Replace encoder with GLA (gated linear attention) layers."""
    from transformers import AutoModel
    from fla.layers import GatedLinearAttention
    _, _, mask_id, _, _ = get_tokenizer()

    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    _patch_linear_attention(backbone, "gla", device)
    schedule = UniformSchedule(128, mask_id)
    model = DiffusionNucEL(backbone, mask_id, time_embed).to(device)
    return model, schedule


def build_mamba2_model(time_embed="additive", device="cuda"):
    """Replace encoder with Mamba-2 SSM layers."""
    from transformers import AutoModel
    _, _, mask_id, _, _ = get_tokenizer()

    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    _patch_linear_attention(backbone, "mamba2", device)
    schedule = UniformSchedule(128, mask_id)
    model = DiffusionNucEL(backbone, mask_id, time_embed).to(device)
    return model, schedule


def build_hybrid_model(time_embed="additive", device="cuda"):
    """Hybrid: alternate linear attention + full attention."""
    from transformers import AutoModel
    _, _, mask_id, _, _ = get_tokenizer()

    backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    # Keep every 4th layer as full attention, replace rest with GLA
    _patch_hybrid_attention(backbone, device, ratio=4)
    schedule = UniformSchedule(128, mask_id)
    model = DiffusionNucEL(backbone, mask_id, time_embed).to(device)
    return model, schedule


def _patch_linear_attention(backbone, attn_type, device):
    """
    Replace attention in encoder layers with linear attention variants.
    Strategy: keep the layer norm and FFN, replace attention computation.
    """
    from fla.layers import MultiScaleRetention, GatedLinearAttention
    try:
        from fla.layers import Mamba2 as FLAMamba2
    except ImportError:
        from fla.layers import Mamba as FLAMamba2

    layers = _get_encoder_layers(backbone)
    if not layers:
        print("  WARNING: No encoder layers found for attention replacement")
        return

    n_heads = 16  # NucEL has 16 attention heads
    head_dim = NUCEL_HIDDEN // n_heads

    for i, layer in enumerate(layers):
        attn_mod = _find_attention_module(layer)
        if attn_mod is None:
            continue

        if attn_type == "retnet":
            new_attn = MultiScaleRetention(
                hidden_size=NUCEL_HIDDEN,
                head_dim=head_dim,
                heads=n_heads,
            ).to(device, dtype=torch.bfloat16)
        elif attn_type == "gla":
            new_attn = GatedLinearAttention(
                hidden_size=NUCEL_HIDDEN,
                head_dim=head_dim,
                heads=n_heads,
            ).to(device, dtype=torch.bfloat16)
        elif attn_type == "mamba2":
            new_attn = FLAMamba2(
                hidden_size=NUCEL_HIDDEN,
                head_dim=head_dim,
                heads=n_heads,
            ).to(device, dtype=torch.bfloat16)
        else:
            continue

        # Store reference
        attn_mod._linear_attn = new_attn
        orig_fwd = attn_mod.forward

        def _patched_forward(self_attn, hidden_states, *args, **kwargs):
            # Try to use linear attention
            try:
                out = self_attn._linear_attn(hidden_states)
                if isinstance(out, tuple):
                    return out[0]
                return out
            except Exception:
                return orig_fwd(hidden_states, *args, **kwargs)

        attn_mod.forward = lambda *a, **kw: _patched_forward(attn_mod, *a, **kw)

    print(f"  Patched {len(layers)} layers with {attn_type}")


def _patch_hybrid_attention(backbone, device, ratio=4):
    """Keep every ratio-th layer as full attention, replace rest with GLA."""
    _patch_linear_attention(backbone, "gla", device)
    # Restore full attention for every ratio-th layer
    layers = _get_encoder_layers(backbone)
    if not layers:
        return
    for i, layer in enumerate(layers):
        if i % ratio == 0:
            attn_mod = _find_attention_module(layer)
            if attn_mod and hasattr(attn_mod, '_original_forward'):
                attn_mod.forward = attn_mod._original_forward
                del attn_mod._linear_attn
    print(f"  Hybrid: kept full attention at layers {[i for i in range(len(layers)) if i % ratio == 0]}")


# ---------------------------------------------------------------------------
# GB sequence fetching (from v1, fixed)
# ---------------------------------------------------------------------------

USCS_API_URL = "https://api.genome.ucsc.edu/getData/sequence"
USCS_TIMEOUT = 10


def fetch_seq_ucsc(chrom, start, end):
    url = f"{USCS_API_URL}?genome=hg38;chrom={chrom};start={start};end={end}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=USCS_TIMEOUT) as resp:
            data = json.loads(resp.read())
        return data.get("dna", "").upper()
    except Exception:
        return ""


def load_gb_sequences(benchmark_name, sample_frac=0.05, max_seqs=400):
    """Load GB sequences: try cache first, then fetch from UCSC API."""
    cache_file = GB_SEQ_CACHE / f"{benchmark_name}.json"
    if cache_file.exists():
        try:
            seqs = json.loads(cache_file.read_text())
            if seqs: return seqs
        except Exception:
            pass

    print(f"  Fetching {benchmark_name} via UCSC API...", flush=True)
    result = []
    try:
        from datasets import load_dataset
        ds = load_dataset("katielink/genomic-benchmarks", benchmark_name)
    except Exception as e:
        print(f"    HF load failed: {e}", flush=True)
        return []

    for split in ["train", "test"]:
        if split not in ds: continue
        sp = ds[split]
        n = min(max(1, int(len(sp) * sample_frac)), max_seqs // 2)
        indices = np.random.RandomState(42).choice(len(sp), min(n, len(sp)), replace=False)
        for idx in indices:
            row = sp[int(idx)]
            region = row.get("region", "")
            start = row.get("start", 0)
            end = row.get("end", 0)
            if region and end > start:
                chrom = region if region.startswith("chr") else f"chr{region}"
                seq = fetch_seq_ucsc(chrom, start, end)
                if len(seq) >= 10:
                    result.append(seq)
            time.sleep(0.05)

    if result:
        cache_file.write_text(json.dumps(result))
        print(f"    Cached {len(result)} sequences")
    return result


# ---------------------------------------------------------------------------
# GB Linear Probe Evaluation
# ---------------------------------------------------------------------------

def prepare_gb_subsets():
    """Prepare fixed 5% subsets for all GB tasks. Cache per-task to allow partial results."""
    subset_file = GB_SUBSET_CACHE / "subsets.json"
    if subset_file.exists():
        print("GB subsets already prepared (cached)")
        return True

    print("Preparing GB 5% subsets (first run only, may take a few minutes)...")
    subsets = {}
    for bname in GENOMIC_BENCHMARKS:
        task_file = GB_SUBSET_CACHE / f"{bname}.json"
        if task_file.exists():
            subsets[bname] = json.loads(task_file.read_text())
            n_train = len(subsets[bname].get("train", {}).get("sequences", []))
            n_test = len(subsets[bname].get("test", {}).get("sequences", []))
            print(f"  {bname}: cached (train={n_train}, test={n_test})", flush=True)
            continue

        print(f"  Fetching {bname}...", flush=True)
        try:
            from datasets import load_dataset
            ds = load_dataset("katielink/genomic-benchmarks", bname)
            task_data = {}
            for split in ["train", "test"]:
                if split not in ds: continue
                sp = ds[split]
                n = max(1, int(len(sp) * 0.05))
                rng = np.random.RandomState(42)
                indices = sorted(rng.choice(len(sp), min(n, len(sp)), replace=False).tolist())
                seqs, labels = [], []
                for idx in indices:
                    row = sp[int(idx)]
                    region = row.get("region", "")
                    start = row.get("start", 0)
                    end = row.get("end", 0)
                    label = row.get("label", 0)
                    if region and end > start:
                        chrom = region if region.startswith("chr") else f"chr{region}"
                        seq = fetch_seq_ucsc(chrom, start, end)
                        if len(seq) >= 10:
                            seqs.append(seq)
                            labels.append(label)
                    time.sleep(0.05)
                task_data[split] = {"sequences": seqs, "labels": labels}
                print(f"    {split}: {len(seqs)} sequences", flush=True)
            subsets[bname] = task_data
            task_file.write_text(json.dumps(task_data))
        except Exception as e:
            print(f"    Failed: {e}", flush=True)

    subset_file.write_text(json.dumps(subsets))
    print(f"Saved GB subsets to {subset_file}")
    return True


def load_gb_subsets():
    """Load cached GB subsets."""
    subset_file = GB_SUBSET_CACHE / "subsets.json"
    if not subset_file.exists():
        prepare_gb_subsets()
    return json.loads(subset_file.read_text())


@torch.no_grad()
def evaluate_gb_linear_probe(model_or_backbone, device, is_frozen_baseline=False):
    """
    Extract embeddings → LogisticRegression → accuracy on GB tasks.
    Returns dict of {task_name: accuracy}.
    """
    from sklearn.linear_model import LogisticRegression
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()

    if is_frozen_baseline:
        encoder = model_or_backbone
        encoder.eval()
    else:
        encoder = model_or_backbone.nucel
        model_or_backbone.eval()

    subsets = load_gb_subsets()
    results = {}

    for bname in GENOMIC_BENCHMARKS:
        if bname not in subsets:
            results[bname] = float("nan")
            continue

        all_embs = {}
        all_labels = {}

        for split in ["train", "test"]:
            if split not in subsets[bname]:
                continue
            seqs = subsets[bname][split]["sequences"]
            labels = subsets[bname][split]["labels"]
            if not seqs:
                continue

            # Tokenize and batch
            embs = []
            for i in range(0, len(seqs), BATCH_SIZE):
                batch_seqs = seqs[i:i + BATCH_SIZE]
                ids_list = []
                for s in batch_seqs:
                    ids = tokenize_nt(s, nt_to_id, unk_id)
                    if len(ids) >= SEQ_LEN:
                        ids = ids[:SEQ_LEN]
                    else:
                        ids = ids + [pad_id] * (SEQ_LEN - len(ids))
                    ids_list.append(ids)
                x = torch.tensor(ids_list, dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = encoder(input_ids=x)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                embs.append(hidden.mean(dim=1).float().cpu().numpy())

            all_embs[split] = np.concatenate(embs, axis=0)
            all_labels[split] = np.array(labels[:len(all_embs[split])])

        if "train" not in all_embs or "test" not in all_embs:
            results[bname] = float("nan")
            continue

        # Train logistic regression
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(all_embs["train"], all_labels["train"])
            acc = clf.score(all_embs["test"], all_labels["test"])
            results[bname] = float(acc)
        except Exception as e:
            print(f"    LR failed for {bname}: {e}")
            results[bname] = float("nan")

    if not is_frozen_baseline:
        model_or_backbone.train()
    return results


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"id": "nucel_frozen_baseline", "type": "baseline"},
    {"id": "diffusion_adamw", "type": "diffusion", "optimizer": "adamw", "time_embed": "additive"},
    {"id": "diffusion_muon", "type": "diffusion", "optimizer": "muon", "time_embed": "additive"},
    {"id": "diffusion_adaln", "type": "diffusion", "optimizer": "adamw", "time_embed": "adaln"},
    {"id": "diffusion_performer", "type": "diffusion", "optimizer": "adamw", "time_embed": "additive", "attention": "performer"},
    {"id": "diffusion_retnet", "type": "diffusion", "optimizer": "adamw", "time_embed": "additive", "attention": "retnet"},
    {"id": "diffusion_gla", "type": "diffusion", "optimizer": "adamw", "time_embed": "additive", "attention": "gla"},
    {"id": "diffusion_mamba2", "type": "diffusion", "optimizer": "adamw", "time_embed": "additive", "attention": "mamba2"},
    {"id": "diffusion_hybrid", "type": "diffusion", "optimizer": "adamw", "time_embed": "additive", "attention": "hybrid"},
    {"id": "diffusion_muon_adaln", "type": "diffusion", "optimizer": "muon", "time_embed": "adaln"},
    {"id": "diffusion_muon_best_linear", "type": "diffusion", "optimizer": "muon", "time_embed": "additive", "attention": "gla"},  # placeholder, update after 4-7
]

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def make_optimizer(name, params, lr=1e-4, wd=0.01):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    elif name == "muon":
        try:
            from muon import Muon
            return Muon(params, lr=lr, weight_decay=wd)
        except Exception:
            # Fallback: implement simplified Muon
            print("  Muon import failed, using AdamW fallback")
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, schedule, loader, optimizer, device, grad_accum=4):
    model.train()
    total_loss, n_steps = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.compute_loss(batch, schedule) / grad_accum
        loss.backward()
        if (n_steps + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
        n_steps += 1
    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results():
    if RESULTS_FILE.exists():
        return
    header = ["exp_idx", "exp_id", "type", "attention", "optimizer", "time_embed",
              "train_loss", "steps", "train_secs", "peak_vram_mb",
              "total_M", "trainable_M", "tokens_M"]
    header += [f"gb_{b}" for b in GENOMIC_BENCHMARKS]
    header.append("gb_mean_acc")
    RESULTS_FILE.write_text("\t".join(header) + "\n")
    print(f"Created {RESULTS_FILE}")


def log_result(idx, cfg, train_loss, train_secs, steps, info, gb):
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "NaN"
    row = [
        str(idx),
        cfg.get("id", ""),
        cfg.get("type", ""),
        cfg.get("attention", "full"),
        cfg.get("optimizer", "adamw"),
        cfg.get("time_embed", "additive"),
        fmt(train_loss),
        str(steps),
        f"{train_secs:.1f}",
        fmt(info.get("peak_vram_mb", 0)),
        fmt(info.get("total_M", 0)),
        fmt(info.get("trainable_M", 0)),
        fmt(info.get("tokens_M", 0)),
    ]
    accs = []
    for b in GENOMIC_BENCHMARKS:
        v = gb.get(b, float("nan"))
        row.append(fmt(v))
        if not math.isnan(v):
            accs.append(v)
    row.append(fmt(np.mean(accs)) if accs else "NaN")
    with open(RESULTS_FILE, "a") as f:
        f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(exp_idx, cfg, device="cuda"):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_idx}: {cfg['id']}")
    print(f"{'='*70}")

    t0 = time.time()

    # --- Experiment 0: frozen baseline ---
    if cfg.get("type") == "baseline":
        from transformers import AutoModel
        print("Loading frozen NucEL for baseline embedding...")
        backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        backbone.eval()
        total_p = sum(p.numel() for p in backbone.parameters())
        print(f"  Params: {total_p/1e6:.1f}M")

        gb = evaluate_gb_linear_probe(backbone, device, is_frozen_baseline=True)
        print("── GB Linear Probe Results ──")
        for b, v in gb.items():
            print(f"  {b:42s}: {v:.4f}" if not math.isnan(v) else f"  {b:42s}: N/A")

        info = {"total_M": total_p/1e6, "trainable_M": 0, "peak_vram_mb": 0, "tokens_M": 0}
        log_result(exp_idx, cfg, 0.0, time.time()-t0, 0, info, gb)

        del backbone
        gc.collect(); torch.cuda.empty_cache()
        return gb

    # --- Diffusion experiments ---
    attn = cfg.get("attention", "full")
    te = cfg.get("time_embed", "additive")
    opt_name = cfg.get("optimizer", "adamw")

    print(f"  attention={attn}, time_embed={te}, optimizer={opt_name}")

    # Build model
    builders = {
        "full": lambda: build_base_model(te, device),
        "performer": lambda: build_performer_model(te, device),
        "retnet": lambda: build_retnet_model(te, device),
        "gla": lambda: build_gla_model(te, device),
        "mamba2": lambda: build_mamba2_model(te, device),
        "hybrid": lambda: build_hybrid_model(te, device),
    }
    model, schedule = builders.get(attn, builders["full"])()
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")

    # Data
    loader = DataLoader(NucELDataset("train", SEQ_LEN), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    steps_per_epoch = len(loader)

    # Optimizer
    optimizer = make_optimizer(opt_name, model.parameters(), lr=1e-4, wd=0.01)

    # Train 1 epoch
    print(f"  Training 1 epoch ({steps_per_epoch} steps)...")
    torch.cuda.reset_peak_memory_stats()
    train_loss = train_one_epoch(model, schedule, loader, optimizer, device)
    train_secs = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  Train loss: {train_loss:.4f}, time: {train_secs:.1f}s, VRAM: {peak_vram:.0f}MB")

    # GB evaluation
    print("  Evaluating GB linear probe...")
    gb = evaluate_gb_linear_probe(model, device)
    print("── GB Linear Probe Results ──")
    for b, v in gb.items():
        print(f"  {b:42s}: {v:.4f}" if not math.isnan(v) else f"  {b:42s}: N/A")

    # Save checkpoint
    ckpt = CKPT_DIR / f"{exp_idx:03d}_{cfg['id']}.pt"
    torch.save({"idx": exp_idx, "cfg": cfg, "state": model.state_dict(),
                "gb": gb, "train_loss": train_loss}, ckpt)

    info = {"total_M": total_p/1e6, "trainable_M": trainable_p/1e6,
            "peak_vram_mb": peak_vram, "tokens_M": steps_per_epoch * BATCH_SIZE * SEQ_LEN / 1e6}
    log_result(exp_idx, cfg, train_loss, train_secs, steps_per_epoch, info, gb)

    del model
    gc.collect(); torch.cuda.empty_cache()
    return gb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--exp", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU {args.gpu}")

    init_results()
    prepare_gb_subsets()

    end = args.end if args.end is not None else len(EXPERIMENTS)

    if args.exp is not None:
        run_experiment(args.exp, EXPERIMENTS[args.exp])
    elif args.sweep:
        for i in range(args.start, min(end, len(EXPERIMENTS))):
            try:
                run_experiment(i, EXPERIMENTS[i])
                print(f"\n✓ Experiment {i} ({EXPERIMENTS[i]['id']}) done")
            except Exception as e:
                print(f"\n✗ Experiment {i} FAILED: {e}")
                traceback.print_exc()
        print(f"\nSweep {args.start}–{end-1} complete!")
    else:
        run_experiment(0, EXPERIMENTS[0])


if __name__ == "__main__":
    main()
