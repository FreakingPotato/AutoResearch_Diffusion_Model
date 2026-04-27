"""Generate GB metrics evolution plot including fine-tuning results."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")

# Stage 1 results (linear probe)
stage1 = dict(np.load(results_dir / "stage1_gb_results.npy", allow_pickle=True).item())

# Stage 2 results (linear probe)
stage2 = {
    "demo_coding_vs_intergenomic_seqs": 0.8040, "demo_human_or_worm": 0.7960,
    "dummy_mouse_enhancers_ensembl": 0.6033, "human_enhancers_cohn": 0.6680,
    "human_enhancers_ensembl": 0.6320, "human_ensembl_regulatory": 0.6720,
    "human_ocr_ensembl": 0.5680,
}

# Full fine-tuning results
finetune = dict(np.load(results_dir / "finetune_all_gb.npy", allow_pickle=True).item())

# NucEL paper baseline
nucel = {
    "demo_coding_vs_intergenomic_seqs": 0.9516, "demo_human_or_worm": 0.9216,
    "dummy_mouse_enhancers_ensembl": 0.7910, "human_enhancers_cohn": 0.7090,
    "human_enhancers_ensembl": 0.7320, "human_ensembl_regulatory": 0.5830,
    "human_nontata_promoters": 0.8836, "human_ocr_ensembl": 0.6760,
}

dataset_labels = [
    "coding\nvs\nintergenomic", "human\nvs\nworm", "mouse\nenhancers",
    "enhancers\ncohn", "enhancers\nensembl", "regulatory", "OCR",
]
dataset_keys = [
    "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "dummy_mouse_enhancers_ensembl",
    "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
    "human_ocr_ensembl",
]

x = np.arange(len(dataset_labels))
width = 0.20

s1 = [stage1.get(k, np.nan) for k in dataset_keys]
s2 = [stage2.get(k, np.nan) for k in dataset_keys]
ft = [finetune.get(k, np.nan) for k in dataset_keys]
nc = [nucel.get(k, np.nan) for k in dataset_keys]

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(x - 1.5*width, s1, width, label=f'Stage 1 LP (avg={np.nanmean(s1):.3f})', color='#FF6B6B', alpha=0.85)
ax.bar(x - 0.5*width, s2, width, label=f'Stage 2 LP (avg={np.nanmean(s2):.3f})', color='#4ECDC4', alpha=0.85)
ax.bar(x + 0.5*width, ft, width, label=f'Stage 2 FT (avg={np.nanmean(ft):.3f})', color='#FFD93D', alpha=0.85)
ax.bar(x + 1.5*width, nc, width, label=f'NucEL Paper (avg={np.nanmean(nc):.3f})', color='#45B7D1', alpha=0.85)

# Average lines
for vals, color in [(s1, '#FF6B6B'), (s2, '#4ECDC4'), (ft, '#FFD93D'), (nc, '#45B7D1')]:
    ax.axhline(y=np.nanmean(vals), color=color, linestyle='--', alpha=0.4, linewidth=1)

ax.set_xlabel('Genomic Benchmark Dataset', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('GB Benchmark: Linear Probe vs Full Fine-tuning vs NucEL', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels, fontsize=9)
ax.legend(loc='lower left', fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('gb_evolution.png', dpi=150, bbox_inches='tight')
print("Plot saved to gb_evolution.png")
plt.close()
