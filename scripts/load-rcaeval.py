#!/usr/bin/env python3
"""
RCAEval Data Loader Bridge — reads Parquet metrics and converts to JSON
for consumption by the TypeScript RCAgent benchmark runner.

Usage:
  python3 scripts/load-rcaeval.py case_index <dataset_dir>
  python3 scripts/load-rcaeval.py case_metrics <case_dir>
"""
import pandas as pd
import json
import sys
import os


def load_case_index(dataset_dir):
    """Read cases.parquet and return RE1 cases as JSON."""
    cases_path = os.path.join(dataset_dir, 'cases.parquet')
    df = pd.read_parquet(cases_path)
    # Handle ArrowDtype by converting to string
    df['suite'] = df['suite'].astype(str)
    # Filter to RE1 (metric-only) for reliable benchmarking
    df = df[df['suite'] == 'RE1']
    records = df[[
        'case', 'dataset', 'system', 'root_cause_service',
        'fault', 'inject_time'
    ]].to_dict('records')
    print(json.dumps(records))


def load_case_metrics(case_dir):
    """Read metrics.parquet for a single case and return JSON."""
    metric_path = os.path.join(case_dir, 'metrics.parquet')
    if not os.path.exists(metric_path):
        print(json.dumps({'error': 'No metrics.parquet found'}))
        return

    df = pd.read_parquet(metric_path)

    # Read inject time
    inject_time = 0
    inject_path = os.path.join(case_dir, 'inject_time.txt')
    if os.path.exists(inject_path):
        with open(inject_path) as f:
            inject_time = int(f.read().strip())

    # Identify service columns: pattern is {service}_{metric}
    service_cols = {}
    for col in df.columns:
        if col in ('time', 'timestamp', 'Unnamed: 0', 'index'):
            continue
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            svc = parts[0]
            if svc not in service_cols:
                service_cols[svc] = []
            service_cols[svc].append(col)

    # Aggregate per service: prefer latency column, else first numeric
    service_names = []
    service_series = []
    for svc in sorted(service_cols.keys()):
        cols = service_cols[svc]
        best_col = None
        for c in cols:
            if 'latency' in c.lower() or 'lat' in c.lower() or 'duration' in c.lower():
                best_col = c
                break
        if best_col is None:
            best_col = cols[0]
        series = df[best_col].ffill().fillna(0).values.tolist()
        service_names.append(svc)
        service_series.append(series)

    if not service_names:
        print(json.dumps({'error': 'No service columns found'}))
        return

    # Find fault injection index
    fault_idx = 0
    if 'time' in df.columns and inject_time > 0:
        for i, ts in enumerate(df['time'].values):
            if ts >= inject_time:
                fault_idx = i
                break

    n_points = len(service_series[0])
    print(json.dumps({
        'serviceNames': service_names,
        'data': service_series,
        'nTimesteps': n_points,
        'faultIndex': fault_idx,
    }))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'case_index':
        load_case_index(sys.argv[2])
    elif cmd == 'case_metrics':
        load_case_metrics(sys.argv[2])
    else:
        print(json.dumps({'error': f'Unknown command: {cmd}'}))
