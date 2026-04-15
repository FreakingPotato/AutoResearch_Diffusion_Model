"""
Data preparation for DNA diffusion model experiments.
Downloads human genome (GRCh38) subset, tokenizes into k-mers,
creates train/val splits as memory-mapped tensors.

Usage:
    uv run prepare.py                  # full prep
    uv run prepare.py --chromosomes 21,22   # only chr21, chr22 (faster for testing)
"""

import os
import sys
import argparse
import urllib.request
import gzip
import struct
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 4096       # context length in tokens
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288 # tokens for validation eval

# DNA tokenization
VOCAB_SIZE = 4**6 + 4     # 4096 k-mers (k=6) + 4 special tokens
KMER_SIZE = 6
DNA_ALPHABET = "ACGT"
SPECIAL_TOKENS = {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2, "<CLS>": 3}
NUM_SPECIAL = len(SPECIAL_TOKENS)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dna-diffusion")
RAW_DIR = os.path.join(CACHE_DIR, "raw")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# UCSC download URL for GRCh38 chromosomes
BASE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/"

# ---------------------------------------------------------------------------
# K-mer tokenization
# ---------------------------------------------------------------------------

def build_kmer_vocab(k=KMER_SIZE):
    """Build k-mer to token ID mapping."""
    vocab = {}
    idx = NUM_SPECIAL
    from itertools import product
    for combo in product(DNA_ALPHABET, repeat=k):
        vocab["".join(combo)] = idx
        idx += 1
    assert idx == 4**k + NUM_SPECIAL
    return vocab

def tokenize_sequence(seq, vocab, k=KMER_SIZE):
    """Tokenize a DNA sequence into k-mer IDs. Unknown chars → UNK."""
    unk_id = SPECIAL_TOKENS["<UNK>"]
    tokens = []
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        # Skip kmers with N or other ambiguous bases
        if all(c in DNA_ALPHABET for c in kmer):
            tokens.append(vocab.get(kmer, unk_id))
        else:
            tokens.append(unk_id)
    return tokens

def detokenize(tokens, id_to_kmer):
    """Convert token IDs back to DNA sequence."""
    parts = []
    for t in tokens:
        if t in id_to_kmer:
            parts.append(id_to_kmer[t])
        elif t == SPECIAL_TOKENS["<UNK>"]:
            parts.append("N" * KMER_SIZE)
    if not parts:
        return ""
    # Overlap k-mers
    result = list(parts[0])
    for kmer in parts[1:]:
        result.append(kmer[-1])
    return "".join(result)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_chromosome(chrom, force=False):
    """Download a single chromosome FASTA from UCSC."""
    os.makedirs(RAW_DIR, exist_ok=True)
    fa_path = os.path.join(RAW_DIR, f"hg38.{chrom}.fa")
    if os.path.exists(fa_path) and not force:
        print(f"  {chrom}: already exists")
        return fa_path

    gz_url = f"{BASE_URL}chr{chrom}.fa.gz"
    gz_path = fa_path + ".gz"

    print(f"  Downloading chr{chrom}...")
    try:
        urllib.request.urlretrieve(gz_url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(fa_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(gz_path)
        print(f"  Extracted chr{chrom}")
        return fa_path
    except Exception as e:
        print(f"  Failed to download chr{chrom}: {e}")
        if os.path.exists(gz_path):
            os.remove(gz_path)
        return None

# ---------------------------------------------------------------------------
# Parse FASTA
# ---------------------------------------------------------------------------

def parse_fasta_chunks(fa_path, chunk_size=MAX_SEQ_LEN * KMER_SIZE * 2, stride=None):
    """Parse FASTA into overlapping chunks of DNA sequence."""
    if stride is None:
        stride = chunk_size // 2

    sequences = []
    current_seq = []
    current_name = None

    with open(fa_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Process accumulated sequence
                if current_seq:
                    full_seq = "".join(current_seq)
                    for start in range(0, len(full_seq) - chunk_size + 1, stride):
                        chunk = full_seq[start:start + chunk_size]
                        if len(chunk) == chunk_size:
                            sequences.append(chunk)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())

        # Last sequence
        if current_seq:
            full_seq = "".join(current_seq)
            for start in range(0, len(full_seq) - chunk_size + 1, stride):
                chunk = full_seq[start:start + chunk_size]
                if len(chunk) == chunk_size:
                    sequences.append(chunk)

    return sequences

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_data(chromosomes=None):
    """Download and tokenize human genome data."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]
    else:
        chromosomes = [c.strip() for c in chromosomes.split(",")]

    # Download
    print(f"Downloading chromosomes: {chromosomes}")
    fa_paths = {}
    for chrom in chromosomes:
        path = download_chromosome(chrom)
        if path:
            fa_paths[chrom] = path

    if not fa_paths:
        print("ERROR: No chromosomes downloaded. Check network connection.")
        sys.exit(1)

    # Tokenize
    vocab = build_kmer_vocab()
    id_to_kmer = {v: k for k, v in vocab.items()}

    # Save vocab
    vocab_path = os.path.join(DATA_DIR, "vocab.npy")
    np.save(vocab_path, np.array([id_to_kmer.get(i, "") for i in range(VOCAB_SIZE)], dtype=object))
    print(f"Vocab saved: {VOCAB_SIZE} tokens (k={KMER_SIZE})")

    # Split: last chromosome is validation
    val_chrom = chromosomes[-1]
    train_chroms = [c for c in fa_paths if c != val_chrom]
    if val_chrom not in fa_paths:
        # Use last available as val
        val_chrom = list(fa_paths.keys())[-1]
        train_chroms = [c for c in fa_paths if c != val_chrom]

    print(f"Train chromosomes: {train_chroms}")
    print(f"Val chromosome: {val_chrom}")

    for split, chroms in [("train", train_chroms), ("val", [val_chrom])]:
        all_tokens = []
        for chrom in chroms:
            if chrom not in fa_paths:
                continue
            print(f"Processing chr{chrom} for {split}...")
            sequences = parse_fasta_chunks(fa_paths[chrom])
            print(f"  {len(sequences)} chunks extracted")
            for i, seq in enumerate(sequences):
                tokens = tokenize_sequence(seq, vocab)
                # Truncate/pad to MAX_SEQ_LEN
                if len(tokens) >= MAX_SEQ_LEN:
                    all_tokens.extend(tokens[:MAX_SEQ_LEN])
                # else skip short chunks

        tokens_arr = np.array(all_tokens, dtype=np.uint16)
        out_path = os.path.join(DATA_DIR, f"{split}.bin")
        tokens_arr.tofile(out_path)
        n_seqs = len(tokens_arr) // MAX_SEQ_LEN
        print(f"{split}: {n_seqs:,} sequences ({len(tokens_arr):,} tokens) saved to {out_path}")

    print("\nDone! Ready to train.")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader

class DNADataset(Dataset):
    """Memory-mapped DNA token dataset."""

    def __init__(self, split="train"):
        path = os.path.join(DATA_DIR, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found at {path}. Run prepare.py first.")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_seqs = len(self.data) // MAX_SEQ_LEN
        assert self.n_seqs > 0, "No sequences found"

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * MAX_SEQ_LEN
        end = start + MAX_SEQ_LEN
        tokens = torch.from_numpy(self.data[start:end].copy()).long()
        return tokens


def make_dataloader(batch_size, split="train", num_workers=4):
    """Create a DataLoader for DNA data."""
    dataset = DNADataset(split)
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def evaluate_metrics(model, batch_size, device="cuda"):
    """
    Evaluate diffusion model on validation set.
    Returns dict with loss, nucleotide distribution error, GC content error.
    """
    model.eval()
    val_loader = make_dataloader(batch_size, "val", num_workers=0)

    total_loss = 0
    n_batches = 0
    generated_nt_counts = np.zeros(4)  # A, C, G, T
    real_nt_counts = np.zeros(4)

    nt_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    vocab = build_kmer_vocab()
    id_to_kmer = {v: k for k, v in vocab.items()}

    max_eval_batches = min(len(val_loader), EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_eval_batches:
                break
            batch = batch.to(device)
            # Sample random timesteps
            t = torch.randint(0, model.noise_steps, (batch.shape[0],), device=device)
            loss = model.compute_loss(batch, t)
            total_loss += loss.item()
            n_batches += 1

            # Count nucleotides in real data
            for token_seq in batch.cpu().numpy():
                for tid in token_seq:
                    kmer = id_to_kmer.get(tid, "")
                    for c in kmer:
                        if c in nt_map:
                            real_nt_counts[nt_map[c]] += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Generate some samples for evaluation
    n_samples = min(batch_size, 16)
    with torch.no_grad():
        samples = model.sample(n_samples, MAX_SEQ_LEN, device=device)

    # Count nucleotides in generated samples
    for token_seq in samples.cpu().numpy():
        for tid in token_seq:
            kmer = id_to_kmer.get(int(tid), "")
            for c in kmer:
                if c in nt_map:
                    generated_nt_counts[nt_map[c]] += 1

    # Normalize
    real_dist = real_nt_counts / max(real_nt_counts.sum(), 1)
    gen_dist = generated_nt_counts / max(generated_nt_counts.sum(), 1)
    nt_error = float(np.abs(real_dist - gen_dist).mean())

    real_gc = (real_nt_counts[1] + real_nt_counts[2]) / max(real_nt_counts.sum(), 1)
    gen_gc = (generated_nt_counts[1] + generated_nt_counts[2]) / max(generated_nt_counts.sum(), 1)

    model.train()
    return {
        "val_loss": avg_loss,
        "nt_dist_error": nt_error,
        "real_gc": float(real_gc),
        "gen_gc": float(gen_gc),
        "gc_error": float(abs(real_gc - gen_gc)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DNA data for diffusion training")
    parser.add_argument("--chromosomes", type=str, default="21,22",
                        help="Comma-separated chromosome numbers (default: 21,22 for fast testing)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    prepare_data(args.chromosomes)
