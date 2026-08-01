#!/usr/bin/env python3
"""
Causality Analyzer vs causal-learn — Head-to-Head Benchmark.

Runs causal-learn's PC, GES, LiNGAM, FCI on the same CSV data exported
by export-benchmark-data.ts. Compares SHD/TPR/FPR against CA's results.

Usage:
  source /tmp/benchmark-venv/bin/activate
  python3 benchmark/run-causal-learn-benchmark.py

Output:
  benchmark-results/cross-language-comparison.json
  benchmark-results/cross-language-comparison.md
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.FCMBased import lingam
from causallearn.utils.GraphUtils import GraphUtils

DATA_DIR = Path(__file__).parent / '..' / 'benchmark-results' / 'data'
OUTPUT_DIR = Path(__file__).parent / '..' / 'benchmark-results'

# ── Helpers ──────────────────────────────────────────────────────────

def load_csv(path: Path) -> np.ndarray:
    """Load CSV data file, skipping header row."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            rows.append([float(v) for v in row])
    return np.array(rows), header

def load_truth(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def compute_shd(pred_edges: set, true_edges: set) -> dict:
    """Compute SHD, TPR, FPR from edge sets."""
    n_correct = len(pred_edges & true_edges)
    n_missing = len(true_edges - pred_edges)
    n_extra = len(pred_edges - true_edges)
    shd = n_missing + n_extra
    tpr = n_correct / len(true_edges) if len(true_edges) > 0 else 1.0
    fpr = n_extra / len(pred_edges) if len(pred_edges) > 0 else 0.0
    precision = n_correct / len(pred_edges) if len(pred_edges) > 0 else 1.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    return {'shd': shd, 'tpr': round(tpr, 3), 'fpr': round(fpr, 3), 'f1': round(f1, 3)}


def pc_edges_to_set(cg, nodes: list) -> set:
    """Convert causal-learn PC graph to edge set (v0.1.4 API: cg.G.graph)."""
    edges = set()
    graph = cg.G.graph
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # -1: i->j, 1: j->i
            if graph[j, i] == -1 and graph[i, j] == 1:
                edges.add(f"{nodes[i]}->{nodes[j]}")
    return edges


def ges_edges_to_set(record: dict, nodes: list) -> set:
    """Convert causal-learn GES graph to edge set."""
    edges = set()
    graph = record['G'].graph
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if graph[j, i] == -1 and graph[i, j] == 1:
                edges.add(f"{nodes[i]}->{nodes[j]}")
    return edges


def fci_edges_to_set(G, nodes: list) -> set:
    """Convert causal-learn FCI graph to edge set (v0.1.4 API: G.G.graph)."""
    edges = set()
    graph = G.G.graph
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if graph[j, i] == -1 and graph[i, j] == 1:
                edges.add(f"{nodes[i]}->{nodes[j]}")
    return edges


# ── Runner ───────────────────────────────────────────────────────────

def run_benchmark(dataset_name: str, data: np.ndarray, nodes: list,
                  true_edges_raw: list, sample_size: int) -> list:
    """Run all causal-learn algorithms on one dataset."""
    true_edges = set(f"{s}->{t}" for s, t in true_edges_raw)
    results = []

    # ── PC ──
    if len(nodes) <= 15:  # PC is O(2^n) CI tests — skip beyond 15 nodes
        try:
            t0 = time.time()
            cg = pc(data, alpha=0.05, stable=True)
            elapsed = (time.time() - t0) * 1000
            pred = pc_edges_to_set(cg, nodes)
            metrics = compute_shd(pred, true_edges)
            metrics['algorithm'] = 'PC (causal-learn)'
            metrics['timeMs'] = int(elapsed)
            results.append(metrics)
        except Exception as e:
            print(f"    PC failed: {e}")

    # ── GES ──
    if len(nodes) <= 30:  # GES scales better but still O(n³f)
        try:
            t0 = time.time()
            record = ges(data)
            elapsed = (time.time() - t0) * 1000
            pred = ges_edges_to_set(record, nodes)
            metrics = compute_shd(pred, true_edges)
            metrics['algorithm'] = 'GES (causal-learn)'
            metrics['timeMs'] = int(elapsed)
            results.append(metrics)
        except Exception as e:
            print(f"    GES failed: {e}")

    # ── FCI ──
    if len(nodes) <= 20:  # FCI is slow on large graphs
        try:
            t0 = time.time()
            G, _ = fci(data, alpha=0.05)
            elapsed = (time.time() - t0) * 1000
            pred = fci_edges_to_set(G, nodes)
            metrics = compute_shd(pred, true_edges)
            metrics['algorithm'] = 'FCI (causal-learn)'
            metrics['timeMs'] = int(elapsed)
            results.append(metrics)
        except Exception as e:
            print(f"    FCI failed: {e}")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Run 'pnpm benchmark:export-data' first.")
        sys.exit(1)

    all_results = []

    # Find all CSV files
    csv_files = sorted(DATA_DIR.glob('*_n*.csv'))
    print(f"Found {len(csv_files)} benchmark data files\n")

    for csv_path in csv_files:
        # Parse filename: {name}_n{samples}.csv
        stem = csv_path.stem
        parts = stem.rsplit('_n', 1)
        if len(parts) != 2:
            continue
        dataset_name = parts[0]
        sample_size = int(parts[1])

        # Load truth
        truth_path = DATA_DIR / f"{dataset_name}_truth.json"
        if not truth_path.exists():
            continue
        truth = load_truth(truth_path)

        print(f"{dataset_name} n={sample_size} ({len(truth['nodes'])} nodes, {len(truth['edges'])} edges)...")

        data, nodes = load_csv(csv_path)

        results = run_benchmark(dataset_name, data, nodes, truth['edges'], sample_size)
        for r in results:
            r['dataset'] = dataset_name
            r['sampleSize'] = sample_size
            r['nodes'] = len(nodes)
            all_results.append(r)

    # Save JSON
    json_path = OUTPUT_DIR / 'cross-language-comparison.json'
    with open(json_path, 'w') as f:
        json.dump({'benchmark': 'causal-learn', 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                   'results': all_results}, f, indent=2)
    print(f"\nJSON: {json_path} ({len(all_results)} results)")

    # Generate Markdown report
    md_path = OUTPUT_DIR / 'cross-language-comparison.md'
    lines = ['# CA vs causal-learn — Head-to-Head Benchmark',
             '', f'> Generated: {time.strftime("%Y-%m-%d %H:%M:%S UTC")}',
             '', '## Results',
             '', '| Dataset | N | Algorithm | SHD | TPR | FPR | F1 | Time (ms) |',
             '|---------|---|-----------|-----|-----|-----|----|----------|']
    for r in all_results:
        lines.append(f"| {r['dataset']} | {r['sampleSize']} | {r['algorithm']} | {r['shd']} | {r['tpr']:.3f} | {r['fpr']:.3f} | {r['f1']:.3f} | {r['timeMs']} |")
    lines.append('')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Markdown: {md_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
