"""Generate GB metrics evolution plot for README."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path("results")

stage1_results = dict(np.load(results_dir / "stage1_gb_results.npy", allow_pickle=True).item())

# Stage 2 results (from benchmark evaluation)
stage2_results = {
    "demo_coding_vs_intergenomic_seqs": 0.8040,
    "demo_human_or_worm": 0.7960,
    "dummy_mouse_enhancers_ensembl": 0.6033,
    "human_enhancers_cohn": 0.6680,
    "human_enhancers_ensembl": 0.6320,
    "human_ensembl_regulatory": 0.6720,
    "human_ocr_ensembl": 0.5680,
}

# NucEL paper baseline
nucel_results = {
    "demo_coding_vs_intergenomic_seqs": 0.9516,
    "demo_human_or_worm": 0.9216,
    "dummy_mouse_enhancers_ensembl": 0.7910,
    "human_enhancers_cohn": 0.7090,
    "human_enhancers_ensembl": 0.7320,
    "human_ensembl_regulatory": 0.5830,
    "human_nontata_promoters": 0.8836,
    "human_ocr_ensembl": 0.6760,
}

# Dataset names for plotting (abbreviated for readability)
dataset_labels = [
    "coding\nvs\nintergenomic",
    "human\nvs\nworm",
    "mouse\nenhancers",
    "enhancers\ncohn",
    "enhancers\nensembl",
    "regulatory",
    # "promoters",  # Skip due to loading issues
    "OCR",
]

# Extract values for common datasets
x = np.arange(len(dataset_labels))
stage1_vals = [
    stage1_results.get("demo_coding_vs_intergenomic_seqs", np.nan),
    stage1_results.get("demo_human_or_worm", np.nan),
    stage1_results.get("dummy_mouse_enhancers_ensembl", np.nan),
    stage1_results.get("human_enhancers_cohn", np.nan),
    stage1_results.get("human_enhancers_ensembl", np.nan),
    stage1_results.get("human_ensembl_regulatory", np.nan),
    stage1_results.get("human_ocr_ensembl", np.nan),
]
stage2_vals = [
    stage2_results.get("demo_coding_vs_intergenomic_seqs", np.nan),
    stage2_results.get("demo_human_or_worm", np.nan),
    stage2_results.get("dummy_mouse_enhancers_ensembl", np.nan),
    stage2_results.get("human_enhancers_cohn", np.nan),
    stage2_results.get("human_enhancers_ensembl", np.nan),
    stage2_results.get("human_ensembl_regulatory", np.nan),
    stage2_results.get("human_ocr_ensembl", np.nan),
]
nucel_vals = [
    nucel_results.get("demo_coding_vs_intergenomic_seqs", np.nan),
    nucel_results.get("demo_human_or_worm", np.nan),
    nucel_results.get("dummy_mouse_enhancers_ensembl", np.nan),
    nucel_results.get("human_enhancers_cohn", np.nan),
    nucel_results.get("human_enhancers_ensembl", np.nan),
    nucel_results.get("human_ensembl_regulatory", np.nan),
    nucel_results.get("human_ocr_ensembl", np.nan),
]

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

width = 0.25
ax.bar(x - width, stage1_vals, width, label='Stage 1 (4096 seq)', color='#FF6B6B', alpha=0.8)
ax.bar(x, stage2_vals, width, label='Stage 2 (8192 seq)', color='#4ECDC4', alpha=0.8)
ax.bar(x + width, nucel_vals, width, label='NucEL Paper', color='#45B7D1', alpha=0.8)

# Add average lines
stage1_avg = np.nanmean(stage1_vals)
stage2_avg = np.nanmean(stage2_vals)
nucel_avg = np.nanmean(nucel_vals)
ax.axhline(y=stage1_avg, color='#FF6B6B', linestyle='--', alpha=0.5, label=f'Stage 1 Avg: {stage1_avg:.3f}')
ax.axhline(y=stage2_avg, color='#4ECDC4', linestyle='--', alpha=0.5, label=f'Stage 2 Avg: {stage2_avg:.3f}')
ax.axhline(y=nucel_avg, color='#45B7D1', linestyle='--', alpha=0.5, label=f'NucEL Avg: {nucel_avg:.3f}')

ax.set_xlabel('Genomic Benchmark Dataset', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('GB Benchmark Accuracy Evolution: Stage 1 → Stage 2 vs NucEL Baseline', fontsize=14, fontweight='bold')
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

# Also save a summary table
summary = {
    "stage1_avg": stage1_avg,
    "stage2_avg": stage2_avg,
    "nucel_avg": nucel_avg,
    "stage1_vals": stage1_vals,
    "stage2_vals": stage2_vals,
    "nucel_vals": nucel_vals,
}
np.save(results_dir / "gb_summary.npy", summary)
print(f"Summary saved to {results_dir / 'gb_summary.npy'}")
