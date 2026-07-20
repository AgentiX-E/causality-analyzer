/**
 * Data preprocessing: Standardizer, Discretizer, WindowExtractor, Imputer.
 *
 * All operations return new instances — the original data is never mutated.
 */
import { ColumnarTable, type TableSchema } from '@agentix-e/causality-analyzer-core';

export type StandardizeMethod = 'zscore' | 'minmax' | 'robust';

/** Z-score standardizer: (x - μ) / σ */
export function standardize<S extends TableSchema>(
  table: ColumnarTable<S>,
  method: StandardizeMethod = 'zscore',
): ColumnarTable<S> {
  const result: Record<string, Float64Array> = {};
  for (const name of table.columnNames) {
    const col = table.column(name);
    const n = col.length;
    if (n < 2) { result[name] = col; continue; }

    if (method === 'minmax') {
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < n; i++) { const v = col[i]!; if (v < min) min = v; if (v > max) max = v; }
      const range = max - min || 1;
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = (col[i]! - min) / range;
      result[name] = out;
    } else if (method === 'robust') {
      const sorted = new Float64Array(col).sort();
      const q1 = sorted[Math.floor(n * 0.25)]!;
      const q3 = sorted[Math.floor(n * 0.75)]!;
      const iqr = q3 - q1 || 1;
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = (col[i]! - q1) / iqr;
      result[name] = out;
    } else {
      let sum = 0; for (let i = 0; i < n; i++) sum += col[i]!;
      const mean = sum / n;
      let ss = 0; for (let i = 0; i < n; i++) { const d = col[i]! - mean; ss += d * d; }
      const std = Math.sqrt(ss / n) || 1;
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = (col[i]! - mean) / std;
      result[name] = out;
    }
  }
  return ColumnarTable.fromColumnar(result as Record<keyof S & string, Float64Array>);
}

/** Equal-width discretizer: bin continuous values into integer labels 0..bins-1 */
export function discretize<S extends TableSchema>(
  table: ColumnarTable<S>,
  bins: number = 5,
): ColumnarTable<S> {
  const result: Record<string, Float64Array> = {};
  for (const name of table.columnNames) {
    const col = table.column(name);
    const n = col.length;
    if (n < 2 || bins < 2) { result[name] = col; continue; }
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) { const v = col[i]!; if (v < min) min = v; if (v > max) max = v; }
    const range = max - min || 1;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.min(bins - 1, Math.floor(((col[i]! - min) / range) * bins));
    result[name] = out;
  }
  return ColumnarTable.fromColumnar(result as Record<keyof S & string, Float64Array>);
}

/** Sliding window extractor: yields consecutive windows of given size */
export function* extractWindows<S extends TableSchema>(
  table: ColumnarTable<S>,
  size: number,
  step: number = 1,
): Generator<ColumnarTable<S>> {
  for (let start = 0; start + size <= table.rowCount; start += step) {
    yield table.slice(start, start + size);
  }
}

/** Mean imputer: fills NaN values with column mean */
export function imputeMean<S extends TableSchema>(table: ColumnarTable<S>): ColumnarTable<S> {
  const result: Record<string, Float64Array> = {};
  for (const name of table.columnNames) {
    const col = table.column(name);
    const n = col.length;
    let sum = 0, count = 0;
    for (let i = 0; i < n; i++) { if (!Number.isNaN(col[i]!)) { sum += col[i]!; count++; } }
    const mean = count > 0 ? sum / count : 0;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Number.isNaN(col[i]!) ? mean : col[i]!;
    result[name] = out;
  }
  return ColumnarTable.fromColumnar(result as Record<keyof S & string, Float64Array>);
}
