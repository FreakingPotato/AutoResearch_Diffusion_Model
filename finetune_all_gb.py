"""Full-parameter fine-tuning on all 8 GB benchmarks."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import random
import time
import gc
from pathlib import Path

from src import build_model, DiffusionConfig
from src.eval import load_gb_dataset
from src.tokenizer import get_tokenizer, tokenize_nt
from train import load_stage1_checkpoint

BENCHMARKS = [
    "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "dummy_mouse_enhancers_ensembl",
    "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
    "human_nontata_promoters", "human_ocr_ensembl",
]

nucel_results = {
    "demo_coding_vs_intergenomic_seqs": 0.9516, "demo_human_or_worm": 0.9216,
    "dummy_mouse_enhancers_ensembl": 0.7910, "human_enhancers_cohn": 0.7090,
    "human_enhancers_ensembl": 0.7320, "human_ensembl_regulatory": 0.5830,
    "human_nontata_promoters": 0.8836, "human_ocr_ensembl": 0.6760,
}

device = torch.device('cuda:0')

class DiffusionClassifier(nn.Module):
    def __init__(self, backbone, hidden_size=512, n_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, n_classes)
        )

    def forward(self, input_ids):
        out = self.backbone(input_ids=input_ids)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = h.mean(dim=1)
        return self.classifier(h)

def load_base_model():
    config = DiffusionConfig(seq_len=512, attention_type="mamba2", d_state=64, d_conv=4, dropout=0.1)
    model = build_model(config, device)
    load_stage1_checkpoint(model, device, Path("checkpoints_phase4_2/stage1_seq4096_best.pt"))
    ckpt = torch.load("checkpoints_phase4_2/stage2_final_trainer.pt", map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    ms = model.state_dict()
    for k, v in state.items():
        if k in ms and v.shape == ms[k].shape:
            ms[k] = v
    model.load_state_dict(ms)
    return model

_, nt_to_id, _, _, pad_id = get_tokenizer()
SL = 512
BS = 16

print(f"\n{'Dataset':<42} {'Finetune':>7} {'NucEL':>7} {'Delta':>7} {'Time':>6}")
print("-" * 72, flush=True)

results = {}
total_start = time.time()

for i, bench_name in enumerate(BENCHMARKS):
    bench_start = time.time()
    print(f"[{i+1}/8] {bench_name}...", flush=True)

    # Load fresh model each time
    model = load_base_model()
    model = model.to(device, dtype=torch.bfloat16)

    # Load dataset
    data = load_gb_dataset(bench_name)
    if not data:
        print(f"  Skipping (no data)")
        del model; gc.collect(); torch.cuda.empty_cache()
        continue

    random.seed(42)
    data = random.sample(data, min(len(data), 3000))
    seqs, labels = zip(*data)

    ids_list = []
    for seq in seqs:
        ids = tokenize_nt(seq, nt_to_id, 1)
        ids_list.append((ids[:SL] + [pad_id] * max(0, SL - len(ids)))[:SL])
    input_ids = torch.tensor(ids_list, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    n_classes = labels_t.max().item() + 1

    # 80/20 split
    n_train = int(len(input_ids) * 0.8)
    train_ids, val_ids = input_ids[:n_train], input_ids[n_train:]
    train_labels, val_labels = labels_t[:n_train], labels_t[n_train:]

    classifier = DiffusionClassifier(model.backbone, hidden_size=512, n_classes=n_classes)
    classifier = classifier.to(device)
    for param in classifier.classifier.parameters():
        param.data = param.data.to(torch.bfloat16)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(10):
        classifier.train()
        perm = torch.randperm(len(train_ids))
        for j in range(0, len(train_ids), BS):
            idx = perm[j:j+BS]
            bx, by = train_ids[idx].to(device), train_labels[idx].to(device)
            optimizer.zero_grad()
            logits = classifier(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Val
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for j in range(0, len(val_ids), BS):
                bx, by = val_ids[j:j+BS].to(device), val_labels[j:j+BS].to(device)
                correct += (classifier(bx).argmax(1) == by).sum().item()
        val_acc = correct / len(val_ids)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    # Final eval with best model
    if best_state:
        classifier.load_state_dict(best_state)
    classifier.eval()
    correct = 0
    with torch.no_grad():
        for j in range(0, len(val_ids), BS):
            bx, by = val_ids[j:j+BS].to(device), val_labels[j:j+BS].to(device)
            correct += (classifier(bx).argmax(1) == by).sum().item()
    final_acc = correct / len(val_ids)
    results[bench_name] = final_acc

    elapsed = time.time() - bench_start
    n = nucel_results.get(bench_name, float('nan'))
    print(f"{bench_name:<42} {final_acc:>7.4f} {n:>7.4f} {final_acc-n:>+7.4f} {elapsed/60:>5.1f}m", flush=True)

    del model, classifier; gc.collect(); torch.cuda.empty_cache()

total_time = time.time() - total_start
print("=" * 72)
avg = np.mean(list(results.values()))
navg = np.mean(list(nucel_results.values()))
print(f"{'Average':<42} {avg:>7.4f} {navg:>7.4f} {avg-navg:>+7.4f} {total_time/60:>5.1f}m")

# Save
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
np.save(results_dir / "finetune_all_gb.npy", results)
print(f"\nSaved to {results_dir / 'finetune_all_gb.npy'}")
