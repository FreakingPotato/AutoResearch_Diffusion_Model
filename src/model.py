"""Model architecture: NucEL backbone with Mamba2 + Diffusion head."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel

from .schedule import UniformSchedule
from .tokenizer import get_tokenizer


# Constants
NUCEL_REPO = "FreakingPotato/NucEL"
NUCEL_HIDDEN_SIZE = 512
NUCEL_VOCAB_SIZE = 27
DEFAULT_NOISE_STEPS = 128
DEFAULT_SEQ_LEN = 4096


# Mamba2 wrapper
class Mamba2Wrapper(nn.Module):
    def __init__(self, hidden_size=512, d_state=64, d_conv=4):
        super().__init__()
        from mamba_ssm import Mamba2
        self.attn = Mamba2(d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=2)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        out = self.attn(hidden_states)
        return self.norm(residual + out)


@dataclass
class DiffusionConfig:
    seq_len: int = DEFAULT_SEQ_LEN
    noise_steps: int = DEFAULT_NOISE_STEPS
    time_embed: str = "additive"
    attention_type: str = "mamba2"
    d_state: int = 64
    d_conv: int = 4
    dropout: float = 0.0


class NucELDiffusion(nn.Module):
    """NucEL backbone with diffusion head and Mamba2 layers."""
    def __init__(self, config, mask_id=4):
        super().__init__()
        self.config = config
        self.mask_id = mask_id
        self.hidden_size = NUCEL_HIDDEN_SIZE
        self.vocab_size = NUCEL_VOCAB_SIZE
        self.noise_steps = config.noise_steps

        # Load NucEL backbone
        print(f"Loading NucEL ({NUCEL_REPO}) flash_attention_2 ...")
        self.backbone = AutoModel.from_pretrained(
            NUCEL_REPO,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
        )

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size*4),
            nn.SiLU(),
            nn.Linear(self.hidden_size*4, self.hidden_size),
        )

        # Find and replace attention layers
        self.layers = self._find_layers()
        n_layers = len(self.layers) if self.layers else 0
        print(f"  Backbone layers: {n_layers}")

        if config.attention_type == "mamba2" and self.layers is not None:
            self._replace_attention(config.d_state, config.d_conv)

        # Time injection (additive)
        if config.time_embed == "additive" and self.layers is not None:
            self.time_inject = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                for _ in range(max(n_layers, 1))
            ])
            for m in self.time_inject:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # Output head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        self.dropout = nn.Dropout(config.dropout)

    def _find_layers(self):
        """Find the transformer layer list in the backbone."""
        for path in ["encoder.layers", "layers", "model.layers", "encoder.layer"]:
            obj = self.backbone
            found = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            if found and isinstance(obj, nn.ModuleList) and len(obj) > 2:
                return obj
        return None

    def _replace_attention(self, d_state, d_conv):
        """Replace attention layers with Mamba2."""
        if not self.layers:
            return
        for i, layer in enumerate(self.layers):
            attn_name = None
            attn_mod = None
            for name, mod in layer.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    attn_name = name
                    attn_mod = mod
                    break
            if not attn_mod:
                continue
            try:
                replacement = Mamba2Wrapper(self.hidden_size, d_state=d_state, d_conv=d_conv)
                parts = attn_name.split('.')
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], replacement)
                print(f"  Layer {i}: replaced {attn_name} with mamba2")
            except Exception as e:
                print(f"  Layer {i}: failed ({e})")

    def _sinusoidal(self, t, dim):
        """Sinusoidal time embedding."""
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half-1, 1))
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, input_ids, t):
        """Forward pass."""
        param_dtype = next(self.time_proj.parameters()).dtype
        t_hidden = self.time_proj(self._sinusoidal(t, self.hidden_size).to(dtype=param_dtype))

        handles = []
        if self.layers is not None and hasattr(self, 'time_inject'):
            for i, layer in enumerate(self.layers):
                if i >= len(self.time_inject):
                    break
                delta = self.time_inject[i](t_hidden)

                def _hook(mod, inp, out, d=delta):
                    h = out[0] if isinstance(out, tuple) else out
                    return (h + d.unsqueeze(1),) + out[1:] if isinstance(out, tuple) else h + d.unsqueeze(1)

                handles.append(layer.register_forward_hook(_hook))

        try:
            out = self.backbone(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden = self.dropout(hidden)
        return self.lm_head(hidden)

    def compute_loss(self, x_clean, t=None, schedule=None):
        """Compute diffusion loss."""
        if schedule is None:
            schedule = UniformSchedule(self.noise_steps, self.mask_id)
        if t is None:
            t = schedule.sample_t(x_clean.shape[0], x_clean.device)
        x_noisy, mask = schedule.forward_process(x_clean, t)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_clean.device, requires_grad=True)
        logits = self.forward(x_noisy, t)
        return F.cross_entropy(logits[mask], x_clean[mask])

    @torch.no_grad()
    def get_embeddings(self, input_ids, pool="cls"):
        """Extract embeddings for downstream tasks."""
        out = self.backbone(input_ids=input_ids)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        emb = hidden[:, 0] if pool == "cls" else hidden.mean(dim=1)
        return emb.float()  # Return float32 for downstream tasks

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(config, device, mask_id=4):
    """Build model and load to device."""
    model = NucELDiffusion(config, mask_id=mask_id)
    total, trainable = model.count_params()
    print(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    return model.to(device)
