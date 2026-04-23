"""Data loading for full hg38 DNA dataset."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import get_tokenizer, tokenize_nt

# Data paths
CACHE_DIR = Path.home() / ".cache" / "dna-diffusion"
RAW_DIR = CACHE_DIR / "raw"
NUCEL_DATA_DIR = CACHE_DIR / "nucel_data"
NUCEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Chromosomes
TRAIN_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
VAL_CHROMS = ["22"]


def download_chrom(chrom):
    """Download a single chromosome FASTA if not cached."""
    fa_path = RAW_DIR / f"hg38.{chrom}.fa"
    if fa_path.exists():
        return fa_path
    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr{chrom}.fa.gz"
    gz_path = RAW_DIR / f"hg38.{chrom}.fa.gz"
    print(f"  Downloading chr{chrom} ...")
    import urllib.request, gzip
    urllib.request.urlretrieve(url, str(gz_path))
    with gzip.open(gz_path, 'rt') as f_in, open(fa_path, 'w') as f_out:
        f_out.write(f_in.read())
    gz_path.unlink()
    print(f"  chr{chrom} ready ({fa_path.stat().st_size / 1e9:.1f} GB)")
    return fa_path


def build_full_data(seq_len, train_chroms=None, val_chroms=None):
    """Build tokenized data for all specified chromosomes."""
    if train_chroms is None: train_chroms = TRAIN_CHROMS
    if val_chroms is None: val_chroms = VAL_CHROMS

    train_path = NUCEL_DATA_DIR / f"full_train_{seq_len}.bin"
    val_path = NUCEL_DATA_DIR / f"full_val_{seq_len}.bin"
    if train_path.exists() and val_path.exists():
        return

    _, nt_to_id, _, unk_id, _ = get_tokenizer()

    for split, chroms, out_path in [("train", train_chroms, train_path), ("val", val_chroms, val_path)]:
        print(f"Building {split} data (seq_len={seq_len}, chroms={chroms}) ...")
        all_ids = []
        for chrom in chroms:
            fa_path = download_chrom(chrom)
            print(f"  Tokenising chr{chrom} ...")
            cur = []
            with open(fa_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if cur: all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))
                        cur = []
                    else: cur.append(line)
            if cur: all_ids.extend(tokenize_nt("".join(cur).upper(), nt_to_id, unk_id))
            print(f"    chr{chrom}: {len(all_ids):,} tokens so far")

        n_seqs = len(all_ids) // seq_len
        arr = np.array(all_ids[:n_seqs * seq_len], dtype=np.uint16)
        arr.tofile(str(out_path))
        print(f"  {split}: {n_seqs:,} sequences ({len(all_ids)/1e9:.2f}B tokens) → {out_path}")


class FullHg38Dataset(Dataset):
    def __init__(self, split="train", seq_len=4096):
        self.seq_len = seq_len
        if split == "train":
            build_full_data(seq_len, train_chroms=TRAIN_CHROMS)
        else:
            build_full_data(seq_len, val_chroms=VAL_CHROMS)
        path = NUCEL_DATA_DIR / f"full_{split}_{seq_len}.bin"
        self.data = np.memmap(str(path), dtype=np.uint16, mode="r")
        self.n_seq = len(self.data) // seq_len
        print(f"  {split}: {self.n_seq:,} sequences (seq_len={seq_len})")

    def __len__(self): return self.n_seq
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return torch.from_numpy(self.data[s:s+self.seq_len].copy()).long()


def make_loader(batch_size, split="train", seq_len=4096, num_workers=4):
    ds = FullHg38Dataset(split, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"),
                      num_workers=num_workers, pin_memory=True, drop_last=True)
