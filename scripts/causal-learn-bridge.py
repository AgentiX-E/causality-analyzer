#!/usr/bin/env python3
"""
Causal-Learn Bridge — External Parity Suite for Causality Analyzer.

Generates benchmark data, runs causal-learn algorithms, outputs JSON metrics.
"""
import json
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.Endpoint import Endpoint


def generate_linear_data(n_nodes, n_samples, edges, seed=42):
    rng = np.random.RandomState(seed)
    data = np.zeros((n_samples, n_nodes))
    adj = np.zeros((n_nodes, n_nodes))
    for s, t in edges:
        adj[s, t] = 1
    for i in range(n_samples):
        for node in range(n_nodes):
            val = 0.0
            for p in range(n_nodes):
                if adj[p, node] > 0:
                    val += 0.7 * data[i, p]
            val += rng.randn()
            data[i, node] = val
    return data


def extract_directed_edges(general_graph):
    """Extract (source, target) directed edges from causal-learn GeneralGraph."""
    edges = []
    for e in general_graph.get_graph_edges():
        n1 = int(e.get_node1().get_name()[1:]) - 1  # 'X1' -> 0
        n2 = int(e.get_node2().get_name()[1:]) - 1
        e1 = e.get_endpoint1()
        e2 = e.get_endpoint2()
        if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
            edges.append((n1, n2))
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            edges.append((n2, n1))
    return edges


def compute_shd(pred_edges, true_edges):
    true_set = set(true_edges)
    pred_set = set(pred_edges)
    correct = len(pred_set & true_set)
    missing = len(true_set) - correct
    extra = len(pred_set) - correct
    shd = missing + extra
    tpr = correct / max(1, len(true_set))
    fpr = extra / max(1, len(pred_set))
    return shd, tpr, fpr


def main():
    benchmarks = [
        ("ASIA", 8, [(0,2),(1,3),(1,4),(2,5),(3,5),(4,7),(5,6),(5,7)], 500),
        ("Butterfly", 4, [(2,0),(2,1),(0,3),(3,1)], 300),
        ("M-Bias", 5, [(0,1),(0,3),(2,3),(2,4)], 300),
        ("Random-5", 5, [(0,1),(1,2),(0,3),(3,4)], 300),
    ]

    output = []
    for name, n_nodes, edges, n_samples in benchmarks:
        data = generate_linear_data(n_nodes, n_samples, edges)
        results = []

        # PC
        try:
            g_pc = pc(data, alpha=0.05, verbose=False, show_progress=False)
            pc_edges = extract_directed_edges(g_pc.G)
            s, t, f = compute_shd(pc_edges, edges)
            results.append({"algorithm": "cl-PC", "shd": s, "tpr": round(t, 3), "fpr": round(f, 3)})
        except Exception as e:
            results.append({"algorithm": "cl-PC", "shd": -1, "tpr": 0, "fpr": 0, "error": str(e)[:100]})

        # GES
        try:
            g_ges = ges(data)
            ges_edges = extract_directed_edges(g_ges['G'])
            s, t, f = compute_shd(ges_edges, edges)
            results.append({"algorithm": "cl-GES", "shd": s, "tpr": round(t, 3), "fpr": round(f, 3)})
        except Exception as e:
            results.append({"algorithm": "cl-GES", "shd": -1, "tpr": 0, "fpr": 0, "error": str(e)[:100]})

        # FCI
        try:
            g_fci, _ = fci(data, verbose=False, show_progress=False)
            fci_edges = extract_directed_edges(g_fci)
            s, t, f = compute_shd(fci_edges, edges)
            results.append({"algorithm": "cl-FCI", "shd": s, "tpr": round(t, 3), "fpr": round(f, 3)})
        except Exception as e:
            results.append({"algorithm": "cl-FCI", "shd": -1, "tpr": 0, "fpr": 0, "error": str(e)[:100]})

        output.append({
            "graph": name, "nodes": n_nodes, "trueEdges": len(edges),
            "edges": [list(e) for e in edges], "results": results,
        })

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
