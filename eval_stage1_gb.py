"""Evaluate Stage 1 checkpoint on GB benchmarks (dual GPU)."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import random
import gc
import argparse
from pathlib import Path

from src import build_model, DiffusionConfig
from src.eval import load_gb_dataset
from src.tokenizer import get_tokenizer, tokenize_nt


def run_benchmark(rank, world_size, results_dict):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    config = DiffusionConfig(seq_len=512, attention_type="mamba2", d_state=64, d_conv=4, dropout=0.0)
    model = build_model(config, device)

    # Load Stage 1 checkpoint
    ckpt_path = Path("checkpoints_phase4_2/stage1_seq4096_best.pt")
    print(f"[GPU{rank}] Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    ms = model.state_dict()
    for k, v in state.items():
        if k in ms and v.shape == ms[k].shape:
            ms[k] = v
    model.load_state_dict(ms)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    if rank == 0:
        print(f"[GPU{rank}] Model loaded ({len(state)} params)")

    _, nt_to_id, _, _, pad_id = get_tokenizer()

    BENCHMARKS = [
        "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "dummy_mouse_enhancers_ensembl",
        "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
        "human_nontata_promoters", "human_ocr_ensembl",
    ]

    SL, BS = 512, 8

    for bench_name in BENCHMARKS:
        if rank == 0:
            print(f"\nRunning {bench_name}...", flush=True)

        data = load_gb_dataset(bench_name)
        if not data:
            continue

        random.seed(42)
        sample_size = min(int(len(data) * 0.2), 500)
        data = random.sample(data, sample_size)
        seqs, labels = zip(*data)

        # Tokenize to fixed length
        ids_list = []
        for seq in seqs:
            ids = tokenize_nt(seq, nt_to_id, 1)
            ids_list.append((ids[:SL] + [pad_id] * max(0, SL - len(ids)))[:SL])
        input_ids = torch.tensor(ids_list, dtype=torch.long)

        # Extract embeddings
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(input_ids), BS):
                batch = input_ids[i:i+BS].to(device)
                out = model.backbone(input_ids=batch)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                all_embs.append(h.mean(dim=1).float().cpu())
        embeddings = torch.cat(all_embs, 0)
        del input_ids, all_embs
        gc.collect()
        torch.cuda.empty_cache()

        # Linear probe
        labels_t = torch.tensor(labels, dtype=torch.long)
        n_train = len(embeddings) // 2
        n_classes = labels_t.max().item() + 1

        probe = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_classes)
        ).float().to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=3e-4)

        for epoch in range(10):
            for i in range(0, n_train, 64):
                bx = embeddings[:n_train][i:i+64].to(device)
                by = labels_t[:n_train][i:i+64].to(device)
                opt.zero_grad()
                nn.CrossEntropyLoss()(probe(bx), by).backward()
                opt.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            logits = probe(embeddings[n_train:].to(device))
            acc = (logits.argmax(1) == labels_t[n_train:].to(device)).float().mean().item()

        if rank == 0:
            results_dict[bench_name] = acc
            print(f"  {bench_name}: {acc:.4f}", flush=True)

        del embeddings, probe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    from multiprocessing import Manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()

    manager = Manager()
    results_dict = manager.dict()

    mp.spawn(run_benchmark, args=(args.world_size, results_dict), nprocs=args.world_size, join=True)

    # Print results
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

    print("\n" + "=" * 63)
    print(f"{'Dataset':<42} {'Stage1':>7} {'NucEL':>7} {'Delta':>7}")
    print("-" * 63)
    for name in results_dict:
        n = nucel_results.get(name, float('nan'))
        print(f"{name:<42} {results_dict[name]:>7.4f} {n:>7.4f} {results_dict[name]-n:>+7.4f}")

    avg = np.mean(list(results_dict.values()))
    navg = np.mean(list(nucel_results.values()))
    print("=" * 63)
    print(f"{'Average':<42} {avg:>7.4f} {navg:>7.4f} {avg-navg:>+7.4f}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    np.save(results_dir / "stage1_gb_results.npy", dict(results_dict))
    print(f"\nResults saved to {results_dir / 'stage1_gb_results.npy'}")
