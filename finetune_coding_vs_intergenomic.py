"""Full-parameter fine-tuning on Coding vs Intergenomic."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import random
import time
from pathlib import Path

from src import build_model, DiffusionConfig
from src.eval import load_gb_dataset
from src.tokenizer import get_tokenizer, tokenize_nt
from train import load_stage1_checkpoint

device = torch.device('cuda:0')

# Load dataset
dataset_name = "demo_coding_vs_intergenomic_seqs"
print(f"Loading {dataset_name}...")
data = load_gb_dataset(dataset_name)
random.seed(42)
sample_size = min(len(data), 3000)
data = random.sample(data, sample_size)
seqs, labels = zip(*data)

_, nt_to_id, _, _, pad_id = get_tokenizer()
SL = 512
ids_list = []
for seq in seqs:
    ids = tokenize_nt(seq, nt_to_id, 1)
    ids_list.append((ids[:SL] + [pad_id] * max(0, SL - len(ids)))[:SL])
input_ids = torch.tensor(ids_list, dtype=torch.long)
labels_t = torch.tensor(labels, dtype=torch.long)
n_classes = labels_t.max().item() + 1

# Train/val split
n_train = int(len(input_ids) * 0.8)
n_val = len(input_ids) - n_train
train_ids, val_ids = input_ids[:n_train], input_ids[n_train:]
train_labels, val_labels = labels_t[:n_train], labels_t[n_train:]

print(f"Dataset: {len(train_ids)} train, {len(val_ids)} val, {n_classes} classes")

# Load model
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
model = model.to(device, dtype=torch.bfloat16)  # bf16 for flash attention
print("Model loaded (full parameters trainable)")

# Add classification head
class DiffusionClassifier(nn.Module):
    def __init__(self, backbone, hidden_size=512, n_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )

    def forward(self, input_ids):
        out = self.backbone(input_ids=input_ids)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = h.mean(dim=1)
        return self.classifier(h)

classifier = DiffusionClassifier(model.backbone, hidden_size=512, n_classes=n_classes)
classifier = classifier.to(device)
# Set classifier to bfloat16 to match backbone output
for param in classifier.classifier.parameters():
    param.data = param.data.to(torch.bfloat16)

total_params = sum(p.numel() for p in classifier.parameters())
trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print(f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

# Training setup
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
BS = 16
n_train_batches = (len(train_ids) + BS - 1) // BS

print(f"\n{'='*60}")
print("Starting full-parameter fine-tuning...")
print('='*60)
print(f"Batch size: {BS}")
print(f"Train batches: {n_train_batches}")
print(f"Learning rate: 1e-5")
print(f"{'='*60}\n")

start_time = time.time()
best_val_acc = 0
best_model_state = None

for epoch in range(10):
    epoch_start = time.time()
    classifier.train()
    train_loss = 0
    train_correct = 0
    perm = torch.randperm(len(train_ids))

    for i in range(0, len(train_ids), BS):
        idx = perm[i:i+BS]
        bx, by = train_ids[idx].to(device), train_labels[idx].to(device)

        optimizer.zero_grad()
        logits = classifier(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(bx)
        train_correct += (logits.argmax(1) == by).sum().item()

    # Validation
    classifier.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for i in range(0, len(val_ids), BS):
            bx, by = val_ids[i:i+BS].to(device), val_labels[i:i+BS].to(device)
            logits = classifier(bx)
            loss = criterion(logits, by)
            val_loss += loss.item() * len(bx)
            val_correct += (logits.argmax(1) == by).sum().item()

    train_loss /= len(train_ids)
    val_loss /= len(val_ids)
    train_acc = train_correct / len(train_ids)
    val_acc = val_correct / len(val_ids)
    epoch_time = time.time() - epoch_start

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    print(f"Epoch {epoch+1}/10 | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
          f"time={epoch_time:.1f}s | best_val={best_val_acc:.4f}")

total_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"Fine-tuning completed in {total_time/60:.1f} minutes")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print('='*60)

if best_model_state:
    classifier.load_state_dict(best_model_state)
classifier.eval()
with torch.no_grad():
    all_logits = []
    for i in range(0, len(val_ids), BS):
        bx = val_ids[i:i+BS].to(device)
        all_logits.append(classifier(bx))
    logits = torch.cat(all_logits, 0)
    final_acc = (logits.argmax(1) == val_labels.to(device)).float().mean().item()

print(f"Final best model accuracy: {final_acc:.4f}")
print(f"\nComparison:")
print(f"  Linear Probe (Stage 2 step10000): 0.854")
print(f"  Full Fine-tune: {final_acc:.4f}")
print(f"  NucEL Baseline: 0.9516")

# Save
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
np.save(results_dir / "finetune_coding_vs_intergenomic.npy", {
    'accuracy': final_acc,
    'best_val_acc': best_val_acc,
    'total_time_minutes': total_time / 60,
    'n_train': len(train_ids),
    'n_val': len(val_ids),
})
print(f"\nResults saved to {results_dir / 'finetune_coding_vs_intergenomic.npy'}")
