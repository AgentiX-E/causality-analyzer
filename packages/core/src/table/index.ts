/**
 * ColumnarTable — A type-safe, column-oriented data table.
 *
 * Design principles:
 * 1. Immutable operations (return new instances), supporting chainable API.
 * 2. Type-safe column access via generic schema inference.
 * 3. Zero-copy views where possible (slice/window operate on views).
 * 4. Optimized for numeric time-series data using Float64Array columns.
 *
 * `ColumnarTable` is the universal data primitive of Causality Analyzer.
 * It is the ONLY implementation class permitted in the core package,
 * by explicit exception — it serves the same foundational role as
 * `Array` in the JavaScript runtime.
 *
 * @packageDocumentation
 */

// ── Types ───────────────────────────────────────────────────────────────

/** Schema type: maps column names to their value types */
// deno-lint-ignore no-explicit-any
export type TableSchema = Record<string, any>;

/** Extract column names from a schema type */
export type ColumnNames<S extends TableSchema> = Extract<keyof S, string>;

/** A single row of data (column name → value) */
export type DataRow<S extends TableSchema = TableSchema> = {
  [K in keyof S]: number;
};

/** Standardization methods */
export type StandardizeMethod = 'zscore' | 'minmax' | 'robust';

/** Discretization strategies */
export type DiscretizeStrategy = 'uniform' | 'quantile';

/**
 * ColumnarTable — column-oriented data table optimized for
 * time-series and metric data.
 *
 * @typeParam S - Schema mapping column names to their value types
 */
export class ColumnarTable<S extends TableSchema = TableSchema> {
  private readonly _columns: Map<string, Float64Array>;
  private readonly _colNames: ReadonlyArray<ColumnNames<S>>;
  private readonly _rowCount: number;

  /**
   * Private constructor. Use static factory methods:
   * `fromRows()`, `fromStream()`, `fromColumnar()`.
   */
  private constructor(
    columns: Map<string, Float64Array>,
    colNames: ReadonlyArray<ColumnNames<S>>,
    rowCount: number,
  ) {
    this._columns = columns;
    this._colNames = colNames;
    this._rowCount = rowCount;
  }

  // ── Accessors ──────────────────────────────────────────────────────

  /** Number of rows in the table */
  get rowCount(): number {
    return this._rowCount;
  }

  /** Ordered list of column names */
  get columnNames(): ReadonlyArray<ColumnNames<S>> {
    return this._colNames;
  }

  /**
   * Retrieve a column by name as a Float64Array (zero-copy reference).
   *
   * @throws If the column name does not exist in the schema
   */
  column<K extends ColumnNames<S>>(name: K): Float64Array {
    const col = this._columns.get(name);
    if (!col) {
      throw new Error(`Column "${name}" not found in table`);
    }
    return col;
  }

  /** Check if a column exists in the table */
  hasColumn(name: string): boolean {
    return this._columns.has(name);
  }

  // ── Slice & Window ─────────────────────────────────────────────────

  /**
   * Return a zero-copy slice view of rows [start, end).
   * The returned ColumnarTable shares the underlying Float64Array
   * buffers with the original — no data is copied.
   */
  slice(start: number, end: number): ColumnarTable<S> {
    if (start < 0 || end > this._rowCount || start >= end) {
      throw new Error(
        `Invalid slice range [${start}, ${end}) for table with ${this._rowCount} rows`,
      );
    }
    const sliced = new Map<string, Float64Array>();
    for (const [name, arr] of this._columns) {
      sliced.set(name, arr.subarray(start, end));
    }
    return new ColumnarTable<S>(sliced, this._colNames, end - start);
  }

  // ── Row Iteration ──────────────────────────────────────────────────

  /** Iterate over all rows as plain objects */
  *rows(): IterableIterator<DataRow<S>> {
    for (let i = 0; i < this._rowCount; i++) {
      const row: Record<string, number> = {};
      for (const name of this._colNames) {
        const col = this._columns.get(name);
        if (col) {
          row[name] = col[i]!;
        }
      }
      yield row as DataRow<S>;
    }
  }

  /** Convert the entire table to an array of row objects */
  toRows(): DataRow<S>[] {
    return Array.from(this.rows());
  }

  // ── Serialization ──────────────────────────────────────────────────

  /** Serialize to a JSON-compatible structure */
  toJSON(): Record<string, number[]> {
    const result: Record<string, number[]> = {};
    for (const name of this._colNames) {
      const col = this._columns.get(name);
      if (col) {
        result[name] = Array.from(col);
      }
    }
    return result;
  }

  // ── Static Factories ───────────────────────────────────────────────

  /**
   * Create a ColumnarTable from an array of row objects.
   *
   * @example
   * ```typescript
   * const table = ColumnarTable.fromRows([
   *   { ts: 1000, latency: 42, cpu: 0.5 },
   *   { ts: 1001, latency: 45, cpu: 0.6 },
   * ]);
   * ```
   */
  static fromRows<S extends TableSchema>(rows: DataRow<S>[]): ColumnarTable<S> {
    if (rows.length === 0) {
      return new ColumnarTable<S>(new Map(), [], 0);
    }

    const colNames = Object.keys(rows[0]!) as ColumnNames<S>[];
    const nRows = rows.length;
    const columns = new Map<string, Float64Array>();

    for (const name of colNames) {
      columns.set(name, new Float64Array(nRows));
    }

    for (let i = 0; i < nRows; i++) {
      const row = rows[i]!;
      for (const name of colNames) {
        columns.get(name)![i] = row[name] ?? NaN;
      }
    }

    return new ColumnarTable<S>(columns, colNames, nRows);
  }

  /**
   * Create a ColumnarTable from a map of column name → Float64Array.
   * All arrays must have the same length. Zero-copy — arrays are
   * stored by reference.
   *
   * @throws If arrays have different lengths
   */
  static fromColumnar<S extends TableSchema>(
    columns: Record<string, Float64Array>,
  ): ColumnarTable<S> {
    const colNames = Object.keys(columns) as ColumnNames<S>[];
    if (colNames.length === 0) {
      return new ColumnarTable<S>(new Map(), [], 0);
    }

    const rowCount = columns[colNames[0]!]!.length;
    for (const name of colNames) {
      if (columns[name]!.length !== rowCount) {
        throw new Error(
          `Column "${name}" has length ${columns[name]!.length}, expected ${rowCount}`,
        );
      }
    }

    const map = new Map<string, Float64Array>();
    for (const name of colNames) {
      map.set(name, columns[name]!);
    }

    return new ColumnarTable<S>(map, colNames, rowCount);
  }

  /**
   * Create a ColumnarTable from an iterable of row objects.
   * Collects all rows into memory before constructing the table.
   */
  static fromIterable<S extends TableSchema>(
    rows: Iterable<DataRow<S>>,
  ): ColumnarTable<S> {
    return ColumnarTable.fromRows(Array.from(rows));
  }
}
