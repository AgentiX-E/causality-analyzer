#!/usr/bin/env python3
"""
RCAEval Data Loader Bridge — reads Parquet data and converts to JSON
for consumption by the TypeScript RCAgent benchmark runner.

Commands:
  python3 scripts/load-rcaeval.py case_index <dataset_dir> [suite_filter]
  python3 scripts/load-rcaeval.py case_metrics <case_dir>
  python3 scripts/load-rcaeval.py case_traces <case_dir>
  python3 scripts/load-rcaeval.py case_logs <case_dir>
"""
import pandas as pd
import numpy as np
import json
import sys
import os
import re


def load_case_index(dataset_dir, suite_filter=None):
    """Read cases.parquet and return cases as JSON."""
    cases_path = os.path.join(dataset_dir, 'cases.parquet')
    df = pd.read_parquet(cases_path)
    df['suite'] = df['suite'].astype(str)
    if suite_filter:
        suites = [s.strip() for s in suite_filter.split(',')]
        df = df[df['suite'].isin(suites)]
    records = df[[
        'case', 'dataset', 'suite', 'system', 'root_cause_service',
        'fault', 'inject_time', 'has_logs', 'has_traces'
    ]].to_dict('records')
    # Convert numpy types to Python native
    for r in records:
        for k, v in r.items():
            if isinstance(v, (np.bool_,)):
                r[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                r[k] = int(v)
    print(json.dumps(records))


def load_case_metrics(case_dir):
    """Read metrics.parquet, return per-service aggregated + ALL metric columns."""
    metric_path = os.path.join(case_dir, 'metrics.parquet')
    if not os.path.exists(metric_path):
        print(json.dumps({'error': 'No metrics.parquet found'}))
        return

    df = pd.read_parquet(metric_path)

    inject_time = 0
    inject_path = os.path.join(case_dir, 'inject_time.txt')
    if os.path.exists(inject_path):
        with open(inject_path) as f:
            inject_time = int(f.read().strip())

    # Identify service columns: {service}_{metric_type}
    service_cols = {}
    all_columns = []
    for col in df.columns:
        if col in ('time', 'timestamp', 'Unnamed: 0', 'index'):
            continue
        all_columns.append(col)
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            svc = parts[0]
            if svc not in service_cols:
                service_cols[svc] = []
            service_cols[svc].append(col)

    # Per-service aggregated (one best column per service)
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

    # ALL metric columns for multivariate BOCPD (latency + error only)
    bocpd_cols = []
    bocpd_data = []
    for col in sorted(all_columns):
        cl = col.lower()
        if any(kw in cl for kw in ('latency', 'duration', 'error', '5xx', 'fail', 'exception')):
            series = df[col].ffill().fillna(0).values.tolist()
            bocpd_cols.append(col)
            bocpd_data.append(series)
    if not bocpd_cols:
        bocpd_cols = sorted(all_columns)[:min(10, len(all_columns))]
        bocpd_data = [df[c].ffill().fillna(0).values.tolist() for c in bocpd_cols]

    # Min-max normalize ALL data to [0,1] (BARO convention for IQR scoring)
    for series in bocpd_data:
        mn, mx = min(series), max(series)
        rng = mx - mn if mx != mn else 1
        for k in range(len(series)):
            series[k] = (series[k] - mn) / rng
    for series in service_series:
        mn, mx = min(series), max(series)
        rng = mx - mn if mx != mn else 1
        for k in range(len(series)):
            series[k] = (series[k] - mn) / rng

    # Fault index
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
        'bocpdColumns': bocpd_cols,
        'bocpdData': bocpd_data,
    }))


def load_case_traces(case_dir):
    """Read traces.parquet and extract caller→callee topology edges."""
    trace_path = os.path.join(case_dir, 'traces.parquet')
    if not os.path.exists(trace_path):
        print(json.dumps({'edges': [], 'error': 'No traces.parquet found'}))
        return

    df = pd.read_parquet(trace_path)
    
    # Try to identify service name and parent-child relationship columns
    # Common RCAEval trace schema fields
    svc_col = None
    parent_col = None
    span_col = None
    error_col = None
    
    for col in df.columns:
        cl = col.lower()
        if svc_col is None and ('service' in cl or 'svc' in cl or 'component' in cl):
            svc_col = col
        if parent_col is None and ('parent' in cl) and ('span' in cl or 'id' in cl):
            parent_col = col
        if span_col is None and ('span' in cl and 'id' in cl) and 'parent' not in cl:
            span_col = col
        if error_col is None and ('error' in cl or 'status' in cl or 'fault' in cl):
            error_col = col

    edges = []
    if svc_col and parent_col:
        # Build parent-child relationships
        for _, row in df.head(5000).iterrows():
            parent_svc = None
            child_svc = str(row[svc_col])
            
            # Find parent service
            if span_col:
                parent_span = row[parent_col] if pd.notna(row[parent_col]) else None
                if parent_span:
                    parent_row = df[df[span_col] == parent_span]
                    if len(parent_row) > 0:
                        parent_svc = str(parent_row.iloc[0][svc_col])
            
            if parent_svc and parent_svc != child_svc:
                edges.append({
                    'source': parent_svc,
                    'target': child_svc,
                    'callCount': 1,
                })

    # Aggregate edges
    edge_map = {}
    for e in edges:
        key = f"{e['source']}→{e['target']}"
        if key not in edge_map:
            edge_map[key] = e
        else:
            edge_map[key]['callCount'] += 1

    print(json.dumps({
        'edges': list(edge_map.values()),
        'totalSpans': int(len(df)),
        'uniqueServices': list(set(e['source'] for e in edge_map.values()) | set(e['target'] for e in edge_map.values())),
    }))


def load_case_logs(case_dir):
    """Read logs.parquet and extract error signals."""
    log_path = os.path.join(case_dir, 'logs.parquet')
    if not os.path.exists(log_path):
        print(json.dumps({'errors': [], 'error': 'No logs.parquet found'}))
        return

    df = pd.read_parquet(log_path)
    
    # Identify content and service columns
    content_col = None
    svc_col = None
    for col in df.columns:
        cl = col.lower()
        if content_col is None and ('content' in cl or 'message' in cl or 'body' in cl or 'log' in cl):
            content_col = col
        if svc_col is None and ('service' in cl or 'svc' in cl or 'component' in cl or 'pod' in cl):
            svc_col = col

    if not content_col:
        print(json.dumps({'errors': [], 'message': 'No log content column found'}))
        return

    # Error patterns (same as our TypeScript LogMatcher)
    error_patterns = [
        ('connection_refused', re.compile(r'connection\s+refused', re.I), 'critical'),
        ('connection_timeout', re.compile(r'(connection|dial|connect).*tim(e)?out', re.I), 'critical'),
        ('out_of_memory', re.compile(r'(out\s+of\s+memory|OOM|memory\s+limit)', re.I), 'critical'),
        ('disk_full', re.compile(r'(no\s+space|disk\s+full|ENOSPC)', re.I), 'critical'),
        ('permission_denied', re.compile(r'(permission\s+denied|EACCES|forbidden|401|403)', re.I), 'error'),
        ('null_pointer', re.compile(r'(null\s+pointer|NullPointerException|undefined\s+is\s+not)', re.I), 'error'),
        ('crash_loop', re.compile(r'(crashloop|restarting|backoff|OOMKilled)', re.I), 'critical'),
        ('dns_failure', re.compile(r'(name\s+resolution|DNS|NXDOMAIN|no\s+such\s+host)', re.I), 'critical'),
    ]

    errors_by_service = {}
    for _, row in df.head(5000).iterrows():
        content = str(row[content_col]) if pd.notna(row[content_col]) else ''
        service = str(row[svc_col]) if svc_col and pd.notna(row[svc_col]) else 'unknown'
        
        for err_type, pattern, severity in error_patterns:
            if pattern.search(content):
                if service not in errors_by_service:
                    errors_by_service[service] = {}
                if err_type not in errors_by_service[service]:
                    errors_by_service[service][err_type] = {'count': 0, 'severity': severity}
                errors_by_service[service][err_type]['count'] += 1

    result = []
    for svc, errs in errors_by_service.items():
        total_errors = sum(e['count'] for e in errs.values())
        max_severity = 'warning'
        for e in errs.values():
            if e['severity'] == 'critical':
                max_severity = 'critical'
            elif e['severity'] == 'error' and max_severity != 'critical':
                max_severity = 'error'
        
        result.append({
            'service': svc,
            'errorCount': total_errors,
            'severity': max_severity,
            'errorTypes': list(errs.keys()),
        })

    print(json.dumps({'errors': result, 'totalLogLines': int(len(df))}))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'case_index':
        suite_filter = sys.argv[3] if len(sys.argv) > 3 else None
        load_case_index(sys.argv[2], suite_filter)
    elif cmd == 'case_metrics':
        load_case_metrics(sys.argv[2])
    elif cmd == 'case_traces':
        load_case_traces(sys.argv[2])
    elif cmd == 'case_logs':
        load_case_logs(sys.argv[2])
    else:
        print(json.dumps({'error': f'Unknown command: {cmd}'}))
