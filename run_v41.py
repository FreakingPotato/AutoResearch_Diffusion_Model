"""
v4.1: Full GB evaluation (no sampling) for best experiment + NucEL baseline.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc, sys, math, torch, torch.nn as nn, numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
from train_phase4 import *

def extract_embs(backbone, input_ids, batch_size=8):
    backbone.eval()
    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = backbone(input_ids=batch)
            all_embs.append(out.last_hidden_state[:, 0].float().cpu())
        if i % (batch_size * 20) == 0:
            gc.collect(); torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)

def extract_embs_model(model, input_ids, batch_size=8):
    model.eval()
    all_embs = []
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model.nucel(input_ids=batch)
            all_embs.append(out.last_hidden_state[:, 0].float().cpu())
        if i % (batch_size * 20) == 0:
            gc.collect(); torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)

def train_probe(embs, labels, epochs=5, lr=1e-3):
    embs = embs.detach().float().cpu()
    labels = labels.cpu()
    n_classes = len(set(labels.tolist()))
    probe = nn.Linear(embs.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(embs, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
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

def full_eval(name, backbone_or_model, is_nucel=False):
    _, nt_to_id, _, unk_id, pad_id = get_tokenizer()
    results = {}
    for bname in GENOMIC_BENCHMARKS:
        try:
            t0 = datetime.now()
            data = load_gb_dataset_full(bname)
            if len(data) < 100:
                print(f"  {bname}: too few ({len(data)})")
                results[bname] = float("nan"); continue
            
            seqs, labels = zip(*data)
            labels = torch.tensor(labels, dtype=torch.long)
            n = len(data)
            np.random.seed(42); perm = np.random.permutation(n)
            sp = int(n * 0.8)
            train_seqs = [data[i][0] for i in perm[:sp]]
            test_seqs = [data[i][0] for i in perm[sp:]]
            train_labels = labels[list(perm[:sp])]
            test_labels = labels[list(perm[sp:])]
            
            print(f"  {bname}: {len(train_seqs)} train, {len(test_seqs)} test ... ", end="", flush=True)
            
            train_ids = tokenize_sequences(train_seqs, nt_to_id, unk_id, pad_id, DEFAULT_SEQ_LEN)
            test_ids = tokenize_sequences(test_seqs, nt_to_id, unk_id, pad_id, DEFAULT_SEQ_LEN)
            
            if is_nucel:
                train_embs = extract_embs(backbone_or_model, train_ids, batch_size=8)
                test_embs = extract_embs(backbone_or_model, test_ids, batch_size=8)
            else:
                train_embs = extract_embs_model(backbone_or_model, train_ids, batch_size=8)
                test_embs = extract_embs_model(backbone_or_model, test_ids, batch_size=8)
            
            probe = train_probe(train_embs, train_labels, epochs=5)
            acc = eval_probe(probe, test_embs, test_labels)
            results[bname] = acc
            
            dt = (datetime.now() - t0).total_seconds()
            print(f"acc={acc:.4f} ({dt:.0f}s)")
            
            del probe, train_embs, test_embs; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED ({e})")
            import traceback; traceback.print_exc()
            results[bname] = float("nan")
            gc.collect(); torch.cuda.empty_cache()
    return results

device = torch.device("cuda")

# 1. Find best experiment
print("=" * 60)
print("Finding best experiment from Phase 4 results ...")
best_idx = None; best_acc = -1
with open(RESULTS_FILE) as f:
    lines = f.readlines()[1:]
for line in lines:
    parts = line.strip().split("\t")
    if len(parts) < 17: continue
    accs = []
    for v in parts[17:]:
        try:
            fv = float(v)
            if not math.isnan(fv): accs.append(fv)
        except: pass
    avg = sum(accs) / len(accs) if accs else 0
    if avg > best_acc:
        best_acc = avg; best_idx = int(parts[1])

print(f"Best: exp {best_idx} (avg GB acc={best_acc:.4f})")

# 2. Load best model
print(f"\nLoading model for exp {best_idx} ...")
exp_cfg = EXPERIMENTS[best_idx]
cfg = Phase4Config(
    attention_type=exp_cfg.get("attention_type", "mamba2"),
    d_state=exp_cfg.get("d_state", 64), d_conv=exp_cfg.get("d_conv", 4),
    time_embed=exp_cfg.get("time_embed", "additive"),
    hybrid_every=exp_cfg.get("hybrid_every", 3),
)
model = build_model(cfg, device)

ckpts = sorted(CKPT_DIR.glob(f"{best_idx:03d}_*.pt"))
if ckpts:
    ckpt = torch.load(ckpts[0], map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(loaded)
    model.load_state_dict(model_state)
    print(f"Loaded {len(loaded)}/{len(state)} params")

# 3. Full GB eval for best model
print(f"\n{'='*60}")
print(f"FULL GB EVAL: exp {best_idx} ({exp_cfg['id']})")
print(f"{'='*60}")
gb_best = full_eval(f"exp{best_idx}", model, is_nucel=False)

avg = 0; n = 0
for b, acc in gb_best.items():
    if not math.isnan(acc): avg += acc; n += 1
print(f"\nAverage full GB acc: {avg / max(n, 1):.4f}")

del model; gc.collect(); torch.cuda.empty_cache()

# 4. NucEL frozen baseline
print(f"\n{'='*60}")
print("FROZEN NucEL BASELINE (full GB)")
print(f"{'='*60}")
from transformers import AutoModel
backbone = AutoModel.from_pretrained(NUCEL_REPO, trust_remote_code=True,
    dtype=torch.bfloat16, attn_implementation='flash_attention_2').to(device)
gb_nucel = full_eval("nucel_frozen", backbone, is_nucel=True)

avg_n = 0; nn = 0
for b, acc in gb_nucel.items():
    if not math.isnan(acc): avg_n += acc; nn += 1
print(f"\nAverage frozen NucEL GB acc: {avg_n / max(nn, 1):.4f}")

del backbone; gc.collect(); torch.cuda.empty_cache()

# 5. Save results
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")
out_file = PROJECT_DIR / "results_v41_full_gb.tsv"
with open(out_file, "w") as f:
    f.write("benchmark\tfrozen_nucel_acc\tbest_v4_sampled\tbest_v4_full\n")
    for b in GENOMIC_BENCHMARKS:
        # Get sampled acc from results
        sampled = float("nan")
        with open(RESULTS_FILE) as rf:
            rlines = rf.readlines()[1:]
            for rl in rlines:
                rp = rl.strip().split("\t")
                if len(rp) > 17 and rp[1] == str(best_idx):
                    idx = list(GENOMIC_BENCHMARKS).index(b) + 17
                    try: sampled = float(rp[idx])
                    except: pass
        f.write(f"{b}\t{gb_nucel.get(b, float('nan'))}\t{sampled}\t{gb_best.get(b, float('nan'))}\n")

print(f"Saved → {out_file}")
print(f"\nBest v4 (exp {best_idx}): sampled={best_acc:.4f} → full={avg/max(n,1):.4f}")
print(f"Frozen NucEL baseline: {avg_n/max(nn,1):.4f}")
print(f"Improvement over baseline: {avg/max(n,1) - avg_n/max(nn,1):.4f}")
