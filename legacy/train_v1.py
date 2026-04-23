"""
DNA Diffusion Model — Discrete Masked Diffusion (MDLM-style) for human DNA.
Autonomous research: modify this file to iterate.

Usage:
    uv run train.py                    # single run with current config
    uv run train.py --sweep            # run experiment sweep
    uv run train.py --exp 15           # run specific experiment by index
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import time
import csv
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, KMER_SIZE, SPECIAL_TOKENS,
    NUM_SPECIAL, make_dataloader, evaluate_metrics, DNADataset,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
RESULTS_FILE = PROJECT_DIR / "results.tsv"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Diffusion Schedule
# ---------------------------------------------------------------------------

class DiscreteDiffusionSchedule:
    def __init__(self, noise_steps=128):
        self.noise_steps = noise_steps

    def sample_t(self, batch_size, device="cuda"):
        return torch.randint(1, self.noise_steps + 1, (batch_size,), device=device)

    def get_mask_prob(self, t):
        return t.float() / self.noise_steps

    def forward_process(self, x, t):
        mask_prob = self.get_mask_prob(t)
        mask_token = SPECIAL_TOKENS["<MASK>"]
        B, L = x.shape
        rand = torch.rand(B, L, device=x.device)
        mask = rand < mask_prob.unsqueeze(1)
        noisy_x = x.clone()
        noisy_x[mask] = mask_token
        return noisy_x, mask


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 8
    n_head: int = 12
    n_embd: int = 512
    noise_steps: int = 128
    mlp_ratio: int = 4
    bidirectional: bool = True  # DNA is not autoregressive!
    dropout: float = 0.0


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0

        self.c_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_fc = nn.Linear(config.n_embd, config.mlp_ratio * config.n_embd, bias=False)
        self.c_proj2 = nn.Linear(config.mlp_ratio * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6 * config.n_embd, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, bidirectional=True):
        mod = self.adaLN(c).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mod.chunk(6, dim=-1)

        x_norm = norm(x) * (1 + scale_msa) + shift_msa
        B, L, C = x_norm.shape
        qkv = self.c_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        is_causal = not bidirectional
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().view(B, L, C)
        attn = self.c_proj(attn)
        attn = self.dropout(attn)

        x = x + gate_msa * attn

        x_norm2 = norm(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.c_fc(x_norm2)
        mlp_out = F.gelu(mlp_out)
        mlp_out = self.c_proj2(mlp_out)
        mlp_out = self.dropout(mlp_out)
        x = x + gate_mlp * mlp_out

        return x


class DNA_diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.noise_steps = config.noise_steps
        self.bidirectional = config.bidirectional

        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.time_embed = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_layer)])
        self.output_norm = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, config.sequence_len, config.n_embd) * 0.02)
        self.schedule = DiscreteDiffusionSchedule(config.noise_steps)

    @torch.no_grad()
    def init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        self.apply(_init)
        nn.init.zeros_(self.output_head.weight)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)

    def timestep_embedding(self, t, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x_noisy, t):
        B, L = x_noisy.shape
        tok_emb = self.token_embed(x_noisy)
        pos = self.pos_embed[:, :L, :]
        x = tok_emb + pos
        t_emb = self.timestep_embedding(t, self.config.n_embd)
        c = self.time_embed(t_emb)
        for block in self.blocks:
            x = block(x, c, bidirectional=self.bidirectional)
        x = self.output_norm(x)
        logits = self.output_head(x)
        return logits

    def compute_loss(self, x_clean, t=None):
        if t is None:
            t = self.schedule.sample_t(x_clean.shape[0], device=x_clean.device)
        x_noisy, mask = self.schedule.forward_process(x_clean, t)
        logits = self.forward(x_noisy, t)
        mask_token_id = SPECIAL_TOKENS["<MASK>"]
        loss_mask = (x_noisy == mask_token_id)
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=x_clean.device, requires_grad=True)
        loss = F.cross_entropy(logits[loss_mask], x_clean[loss_mask])
        return loss

    @torch.no_grad()
    def sample(self, n_samples, seq_len, device="cuda", steps=None):
        if steps is None:
            steps = self.noise_steps
        mask_token = SPECIAL_TOKENS["<MASK>"]
        x = torch.full((n_samples, seq_len), mask_token, dtype=torch.long, device=device)

        for t_val in reversed(range(1, steps + 1)):
            t = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
            logits = self.forward(x, t)
            is_masked = (x == mask_token)

            if t_val > 1:
                prob_now = t_val / steps
                prob_next = (t_val - 1) / steps
                frac_to_unmask = 1.0 - (prob_next / max(prob_now, 1e-8))

                probs = F.softmax(logits, dim=-1)
                max_probs, predicted = probs.max(dim=-1)
                max_probs[~is_masked] = float('inf')

                n_masked = is_masked.sum(dim=1)
                n_to_unmask = (n_masked.float() * frac_to_unmask).long().clamp(min=1)

                for b in range(n_samples):
                    if n_masked[b] == 0:
                        continue
                    masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                    confidences = max_probs[b, masked_indices]
                    k = min(n_to_unmask[b].item(), len(masked_indices))
                    if k > 0:
                        _, top_k_idx = confidences.topk(k)
                        unmask_indices = masked_indices[top_k_idx]
                        x[b, unmask_indices] = predicted[b, unmask_indices]
            else:
                predicted = logits.argmax(dim=-1)
                x[is_masked] = predicted[is_masked]

        return x

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # --- Phase 1: Baseline & architecture search (exp 0-9) ---
    {"id": "baseline_128steps",  "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "baseline_256steps",  "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 256, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "baseline_64steps",   "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 64,  "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "baseline_32steps",   "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 32,  "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "causal_attn",        "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": False, "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "deeper_12L",         "n_layer": 12, "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "wider_768",          "n_layer": 8,  "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "narrow_384",         "n_layer": 8,  "n_embd": 384, "n_head": 6,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 16, "dropout": 0.0, "weight_decay": 0.01},
    {"id": "tiny_256",           "n_layer": 6,  "n_embd": 256, "n_head": 4,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 16, "dropout": 0.0, "weight_decay": 0.01},
    {"id": "big_1024",           "n_layer": 10, "n_embd": 1024,"n_head": 16, "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 2,  "dropout": 0.0, "weight_decay": 0.01},

    # --- Phase 2: Learning rate & optim (exp 10-19) ---
    {"id": "lr_3e5",             "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 3e-5, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "lr_3e4",             "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "lr_5e4",             "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 5e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "lr_1e4_wd0",        "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.0},
    {"id": "lr_1e4_wd0.1",      "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.1},
    {"id": "lr_1e4_drop0.1",    "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "lr_1e4_drop0.2",    "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.2, "weight_decay": 0.01},
    {"id": "lr_cosine_3e4",     "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01, "warmup": 0.1, "warmdown": 0.3},
    {"id": "batch16",            "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 16, "dropout": 0.0, "weight_decay": 0.01},
    {"id": "batch4_accum",       "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.0, "weight_decay": 0.01},

    # --- Phase 3: Best arch refinement (exp 20-29) ---
    {"id": "best_10L_640",      "n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "best_10L_640_lr3e4","n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "best_10L_640_drop", "n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "best_10L_640_64s",  "n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 64,  "lr": 3e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "best_10L_640_256s", "n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 256, "lr": 3e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "best_14L_512",      "n_layer": 14, "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "best_6L_768",       "n_layer": 6,  "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "best_8L_768",       "n_layer": 8,  "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "best_8L_768_64s",   "n_layer": 8,  "n_embd": 768, "n_head": 12, "noise_steps": 64,  "lr": 3e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "best_8L_768_256s",  "n_layer": 8,  "n_embd": 768, "n_head": 12, "noise_steps": 256, "lr": 3e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.1, "weight_decay": 0.01},

    # --- Phase 4: Scale up (exp 30-39) ---
    {"id": "scale_12L_768",     "n_layer": 12, "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "scale_12L_768_d0",  "n_layer": 12, "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "scale_12L_768_1e4", "n_layer": 12, "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "scale_12L_768_64s", "n_layer": 12, "n_embd": 768, "n_head": 12, "noise_steps": 64,  "lr": 1e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "scale_16L_512",     "n_layer": 16, "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "scale_10L_1024",    "n_layer": 10, "n_embd": 1024,"n_head": 16, "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 2,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "scale_10L_1024_d1", "n_layer": 10, "n_embd": 1024,"n_head": 16, "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 2,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "scale_8L_768_wd0",  "n_layer": 8,  "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 3e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.0, "weight_decay": 0.0},
    {"id": "scale_8L_512_5e4",  "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 5e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},
    {"id": "scale_8L_512_1e4",  "n_layer": 8,  "n_embd": 512, "n_head": 8,  "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.0, "weight_decay": 0.01},

    # --- Phase 5: Final tuning (exp 40-49) ---
    {"id": "final_best_v1",     "n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 128, "lr": 2e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.05, "weight_decay": 0.05},
    {"id": "final_best_v2",     "n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 64,  "lr": 2e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.05, "weight_decay": 0.05},
    {"id": "final_best_v3",     "n_layer": 12, "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 2e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.05, "weight_decay": 0.05},
    {"id": "final_best_v4",     "n_layer": 12, "n_embd": 768, "n_head": 12, "noise_steps": 64,  "lr": 2e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.05, "weight_decay": 0.05},
    {"id": "final_wide_shallow","n_layer": 6,  "n_embd": 1024,"n_head": 16, "noise_steps": 128, "lr": 2e-4, "bidirectional": True,  "batch_size": 2,  "dropout": 0.05, "weight_decay": 0.05},
    {"id": "final_deep_narrow", "n_layer": 16, "n_embd": 384, "n_head": 6,  "noise_steps": 128, "lr": 2e-4, "bidirectional": True,  "batch_size": 16, "dropout": 0.05, "weight_decay": 0.05},
    {"id": "final_10L_768_d02", "n_layer": 10, "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 2e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.2, "weight_decay": 0.05},
    {"id": "final_8L_768_5e4",  "n_layer": 8,  "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 5e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "final_8L_768_1e4_d", "n_layer": 8, "n_embd": 768, "n_head": 12, "noise_steps": 128, "lr": 1e-4, "bidirectional": True,  "batch_size": 4,  "dropout": 0.1, "weight_decay": 0.01},
    {"id": "final_champion",    "n_layer": 10, "n_embd": 640, "n_head": 10, "noise_steps": 128, "lr": 2e-4, "bidirectional": True,  "batch_size": 8,  "dropout": 0.1, "weight_decay": 0.01},
]


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def init_results_file():
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_FILE.exists():
        header = [
            "exp_id", "exp_idx", "timestamp",
            "n_layer", "n_embd", "n_head", "noise_steps", "lr", "bidirectional",
            "batch_size", "dropout", "weight_decay",
            "num_params_M", "train_steps", "train_seconds", "total_seconds",
            "final_train_loss", "val_loss", "nt_dist_error", "gc_error",
            "real_gc", "gen_gc", "peak_vram_mb", "total_tokens_M",
        ]
        with open(RESULTS_FILE, "w") as f:
            f.write("\t".join(header) + "\n")
        print(f"Created {RESULTS_FILE}")


def log_result(exp_idx, config_dict, metrics, train_info):
    """Append experiment result to results.tsv."""
    row = [
        config_dict.get("id", f"exp_{exp_idx}"),
        str(exp_idx),
        datetime.now().isoformat(),
        str(config_dict.get("n_layer", "")),
        str(config_dict.get("n_embd", "")),
        str(config_dict.get("n_head", "")),
        str(config_dict.get("noise_steps", "")),
        str(config_dict.get("lr", "")),
        str(config_dict.get("bidirectional", "")),
        str(config_dict.get("batch_size", "")),
        str(config_dict.get("dropout", "")),
        str(config_dict.get("weight_decay", "")),
        f"{train_info.get('num_params_M', 0):.1f}",
        str(train_info.get("num_steps", 0)),
        f"{train_info.get('training_seconds', 0):.1f}",
        f"{train_info.get('total_seconds', 0):.1f}",
        f"{train_info.get('final_loss', 0):.4f}",
        f"{metrics.get('val_loss', 0):.6f}",
        f"{metrics.get('nt_dist_error', 0):.6f}",
        f"{metrics.get('gc_error', 0):.6f}",
        f"{metrics.get('real_gc', 0):.4f}",
        f"{metrics.get('gen_gc', 0):.4f}",
        f"{train_info.get('peak_vram_mb', 0):.0f}",
        f"{train_info.get('total_tokens_M', 0):.1f}",
    ]
    with open(RESULTS_FILE, "a") as f:
        f.write("\t".join(row) + "\n")
    print(f"Result logged to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(exp_idx, exp_config):
    """Run a single experiment with given config."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_idx}: {exp_config.get('id', 'unnamed')}")
    print(f"{'='*70}")
    for k, v in exp_config.items():
        if k != "id":
            print(f"  {k}: {v}")
    print()

    t_start = time.time()
    torch.manual_seed(42 + exp_idx)
    torch.cuda.manual_seed(42 + exp_idx)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()

    device = torch.device("cuda")

    # Build model
    config = DiffusionConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        n_layer=exp_config.get("n_layer", 8),
        n_head=exp_config.get("n_head", 8),
        n_embd=exp_config.get("n_embd", 512),
        noise_steps=exp_config.get("noise_steps", 128),
        bidirectional=exp_config.get("bidirectional", True),
        dropout=exp_config.get("dropout", 0.0),
    )
    model = DNA_diffusion(config)
    model.init_weights()
    model = model.to(device)

    param_info = model.count_parameters()
    print(f"Model parameters: {param_info['total']:,} ({param_info['total']/1e6:.1f}M)")

    # Optimizer
    lr = exp_config.get("lr", 1e-4)
    weight_decay = exp_config.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )

    # Compile
    model = torch.compile(model, dynamic=False)

    # Data
    batch_size = exp_config.get("batch_size", 8)
    train_loader = make_dataloader(batch_size, "train")
    tokens_per_step = batch_size * MAX_SEQ_LEN
    total_batch_size = 2**17  # 131K tokens per optimizer step
    grad_accum_steps = max(1, total_batch_size // tokens_per_step)
    print(f"Batch size: {batch_size}, Grad accum: {grad_accum_steps}, Tokens/step: {tokens_per_step * grad_accum_steps:,}")

    # LR schedule params
    warmup_ratio = exp_config.get("warmup", 0.05)
    warmdown_ratio = exp_config.get("warmdown", 0.4)

    # Training loop
    model.train()
    torch.cuda.synchronize()
    t_start_training = time.time()
    total_training_time = 0
    step = 0
    smooth_loss = 0
    final_loss = 0

    data_iter = iter(train_loader)

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = batch.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.compute_loss(batch)
                loss = loss / grad_accum_steps
            loss.backward()

        optimizer.step()
        train_loss = loss.item() * grad_accum_steps
        final_loss = train_loss

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        pct = 100 * progress

        # LR schedule
        if progress < warmup_ratio:
            lr_scale = progress / warmup_ratio
        elif progress < 1.0 - warmdown_ratio:
            lr_scale = 1.0
        else:
            cooldown = (1.0 - progress) / warmdown_ratio
            lr_scale = max(cooldown, 0.0)

        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        remaining = max(0, TIME_BUDGET - total_training_time)
        print(f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | lr: {lr*lr_scale:.2e} | dt: {dt*1000:.0f}ms | rem: {remaining:.0f}s    ",
              end="", flush=True)

        if math.isnan(train_loss):
            print("\nNaN loss!")
            break

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()

    # --- Evaluation ---
    # Cleanup before eval to free memory
    del optimizer
    gc.enable()
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()
    raw_model = model
    if hasattr(raw_model, '_orig_mod'):
        raw_model = raw_model._orig_mod

    # Temporarily reduce noise_steps for fast sampling
    original_noise_steps = raw_model.noise_steps
    eval_sample_steps = min(original_noise_steps, 32)  # Very fast eval
    raw_model.noise_steps = eval_sample_steps

    # Use small batch for evaluation to avoid OOM
    eval_batch = min(batch_size, 4)

    try:
        metrics = evaluate_metrics(raw_model, eval_batch, device)
        print("---")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v:.6f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        metrics = {"val_loss": float('nan'), "nt_dist_error": float('nan'),
                   "gc_error": float('nan'), "real_gc": 0, "gen_gc": 0}

    # Restore
    raw_model.noise_steps = original_noise_steps

    # Save checkpoint
    ckpt_name = f"{exp_idx:03d}_{exp_config.get('id', 'exp')}.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    try:
        torch.save({
            "exp_idx": exp_idx,
            "config": exp_config,
            "model_state_dict": raw_model.state_dict(),
            "metrics": metrics,
            "final_loss": final_loss,
            "step": step,
        }, ckpt_path)
        print(f"Checkpoint: {ckpt_path}")
    except Exception as e:
        print(f"Checkpoint save failed: {e}")

    total_tokens = step * tokens_per_step * grad_accum_steps
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    t_end = time.time()

    train_info = {
        "num_params_M": param_info['total'] / 1e6,
        "num_steps": step,
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "final_loss": final_loss,
        "peak_vram_mb": peak_vram_mb,
        "total_tokens_M": total_tokens / 1e6,
    }

    for k, v in train_info.items():
        print(f"{k:20s}: {v}")

    # Log result
    log_result(exp_idx, exp_config, metrics, train_info)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run all experiments")
    parser.add_argument("--exp", type=int, help="Run specific experiment by index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--start", type=int, default=0, help="Start experiment index")
    parser.add_argument("--end", type=int, default=None, help="End experiment index (exclusive)")
    args = parser.parse_args()

    init_results_file()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU {args.gpu}")

    # Determine experiment range
    start = args.start
    end = args.end if args.end is not None else len(EXPERIMENTS)

    if args.exp is not None:
        # Run single experiment
        if args.exp >= len(EXPERIMENTS):
            print(f"Error: experiment {args.exp} not found (max {len(EXPERIMENTS)-1})")
            return
        run_experiment(args.exp, EXPERIMENTS[args.exp])
    elif args.sweep:
        # Run experiments in range
        exps = range(start, min(end, len(EXPERIMENTS)))
        print(f"Running experiments {start}-{end-1} ({len(list(exps))} experiments) on GPU {args.gpu}...")
        for i in exps:
            exp = EXPERIMENTS[i]
            try:
                run_experiment(i, exp)
                print(f"\n✓ Experiment {i} complete")
            except Exception as e:
                print(f"\n✗ Experiment {i} failed: {e}")
                # Log failure
                metrics = {"val_loss": float('nan'), "nt_dist_error": float('nan'),
                           "gc_error": float('nan'), "real_gc": 0, "gen_gc": 0}
                log_result(i, exp, metrics, {"num_params_M": 0, "num_steps": 0,
                          "training_seconds": 0, "total_seconds": 0,
                          "final_loss": float('nan'), "peak_vream_mb": 0, "total_tokens_M": 0})
            print()
        print(f"\n{'='*70}")
        print(f"All experiments in range {start}-{end-1} complete!")
    else:
        # Default: run first experiment
        run_experiment(0, EXPERIMENTS[0])


if __name__ == "__main__":
    main()
