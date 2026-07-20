/**
 * Unit tests for ColumnarTable.
 */
import { describe, it, expect } from 'vitest';
import { ColumnarTable } from '../../index.js';

describe('ColumnarTable', () => {
  // ── Construction ─────────────────────────────────────────────────
  describe('fromRows', () => {
    it('should create a table from an array of rows', () => {
      const rows = [
        { ts: 1000, latency: 42, cpu: 0.5 },
        { ts: 1001, latency: 45, cpu: 0.6 },
        { ts: 1002, latency: 40, cpu: 0.4 },
      ];
      const table = ColumnarTable.fromRows(rows);
      expect(table.rowCount).toBe(3);
      expect(table.columnNames).toEqual(['ts', 'latency', 'cpu']);
    });

    it('should handle an empty row array gracefully', () => {
      const table = ColumnarTable.fromRows([]);
      expect(table.rowCount).toBe(0);
      expect(table.columnNames).toEqual([]);
    });

    it('should handle a single row', () => {
      const table = ColumnarTable.fromRows([{ a: 1, b: 2 }]);
      expect(table.rowCount).toBe(1);
      expect(table.column('a')[0]).toBe(1);
      expect(table.column('b')[0]).toBe(2);
    });

    it('should preserve insertion order of columns', () => {
      const rows = [{ z: 1, a: 2, m: 3 }];
      const table = ColumnarTable.fromRows(rows);
      expect(table.columnNames).toEqual(['z', 'a', 'm']);
    });

    it('should handle NaN values in rows', () => {
      const rows = [{ a: NaN, b: 1 }];
      const table = ColumnarTable.fromRows(rows);
      expect(Number.isNaN(table.column('a')[0])).toBe(true);
    });
  });

  describe('fromColumnar', () => {
    it('should create a table from columnar data', () => {
      const table = ColumnarTable.fromColumnar({
        x: new Float64Array([1, 2, 3]),
        y: new Float64Array([4, 5, 6]),
      });
      expect(table.rowCount).toBe(3);
      expect(table.columnNames).toEqual(['x', 'y']);
    });

    it('should throw on mismatched column lengths', () => {
      expect(() =>
        ColumnarTable.fromColumnar({
          x: new Float64Array([1, 2]),
          y: new Float64Array([3, 4, 5]),
        }),
      ).toThrow(/length/);
    });

    it('should handle empty columnar input', () => {
      const table = ColumnarTable.fromColumnar({});
      expect(table.rowCount).toBe(0);
    });

    it('should store columns by reference (zero-copy)', () => {
      const data = new Float64Array([1, 2, 3]);
      const table = ColumnarTable.fromColumnar({ x: data });
      data[0] = 99;
      // The table's column should reflect the mutation (shared buffer)
      expect(table.column('x')[0]).toBe(99);
    });
  });

  describe('fromIterable', () => {
    it('should create from a generator', () => {
      function* gen() {
        yield { a: 1 } as const;
        yield { a: 2 } as const;
      }
      const table = ColumnarTable.fromIterable([{ a: 1 }, { a: 2 }]);
      expect(table.rowCount).toBe(2);
    });
  });

  // ── Column Access ─────────────────────────────────────────────────
  describe('column', () => {
    it('should retrieve a column by name', () => {
      const table = ColumnarTable.fromRows([
        { ts: 1000, val: 10 },
        { ts: 1001, val: 20 },
      ]);
      expect(Array.from(table.column('ts'))).toEqual([1000, 1001]);
      expect(Array.from(table.column('val'))).toEqual([10, 20]);
    });

    it('should throw for non-existent column', () => {
      const table = ColumnarTable.fromRows([{ a: 1 }]);
      expect(() => table.column('nonexistent' as never)).toThrow(
        /not found/,
      );
    });
  });

  describe('hasColumn', () => {
    it('should return true for existing columns', () => {
      const table = ColumnarTable.fromRows([{ a: 1, b: 2 }]);
      expect(table.hasColumn('a')).toBe(true);
      expect(table.hasColumn('b')).toBe(true);
    });

    it('should return false for non-existent columns', () => {
      const table = ColumnarTable.fromRows([{ a: 1 }]);
      expect(table.hasColumn('b')).toBe(false);
    });
  });

  // ── Slice ─────────────────────────────────────────────────────────
  describe('slice', () => {
    it('should return a zero-copy view of the specified range', () => {
      const table = ColumnarTable.fromRows([
        { a: 1 }, { a: 2 }, { a: 3 }, { a: 4 }, { a: 5 },
      ]);
      const sliced = table.slice(1, 4);
      expect(sliced.rowCount).toBe(3);
      expect(Array.from(sliced.column('a'))).toEqual([2, 3, 4]);
    });

    it('should share underlying buffers (zero-copy)', () => {
      const table = ColumnarTable.fromRows([
        { a: 1 }, { a: 2 }, { a: 3 },
      ]);
      const sliced = table.slice(0, 2);
      // Modify the original column — sliced should see the change
      const originalCol = table.column('a');
      originalCol[0] = 99;
      expect(sliced.column('a')[0]).toBe(99);
    });

    it('should throw for invalid slice ranges', () => {
      const table = ColumnarTable.fromRows([{ a: 1 }, { a: 2 }]);
      expect(() => table.slice(-1, 1)).toThrow(/Invalid slice/);
      expect(() => table.slice(0, 3)).toThrow(/Invalid slice/);
      expect(() => table.slice(1, 0)).toThrow(/Invalid slice/);
    });

    it('should slice the full range correctly', () => {
      const table = ColumnarTable.fromRows([
        { a: 1 }, { a: 2 }, { a: 3 },
      ]);
      const sliced = table.slice(0, 3);
      expect(sliced.rowCount).toBe(3);
    });
  });

  // ── Row Iteration ─────────────────────────────────────────────────
  describe('rows', () => {
    it('should iterate all rows', () => {
      const table = ColumnarTable.fromRows([
        { a: 1, b: 10 },
        { a: 2, b: 20 },
      ]);
      const rows = Array.from(table.rows());
      expect(rows).toEqual([
        { a: 1, b: 10 },
        { a: 2, b: 20 },
      ]);
    });

    it('should handle empty tables', () => {
      const table = ColumnarTable.fromRows([]);
      expect(Array.from(table.rows())).toEqual([]);
    });
  });

  describe('toRows', () => {
    it('should convert the whole table to an array', () => {
      const table = ColumnarTable.fromRows([
        { x: 1 }, { x: 2 }, { x: 3 },
      ]);
      expect(table.toRows()).toEqual([{ x: 1 }, { x: 2 }, { x: 3 }]);
    });
  });

  // ── Serialization ─────────────────────────────────────────────────
  describe('toJSON', () => {
    it('should serialize to a plain object of arrays', () => {
      const table = ColumnarTable.fromRows([
        { a: 1, b: 2 },
        { a: 3, b: 4 },
      ]);
      expect(table.toJSON()).toEqual({ a: [1, 3], b: [2, 4] });
    });

    it('should handle empty tables', () => {
      const table = ColumnarTable.fromRows([]);
      expect(table.toJSON()).toEqual({});
    });
  });

  // ── Property Access ───────────────────────────────────────────────
  describe('rowCount and columnNames', () => {
    it('should report correct counts', () => {
      const table = ColumnarTable.fromRows([
        { ts: 1, val: 10 },
        { ts: 2, val: 20 },
        { ts: 3, val: 30 },
      ]);
      expect(table.rowCount).toBe(3);
      expect(table.columnNames).toHaveLength(2);
    });
  });

  // ── Large Data ────────────────────────────────────────────────────
  describe('large data', () => {
    it('should handle 1000 columns x 100K rows', () => {
      const nCols = 100;
      const nRows = 10_000;
      const row: Record<string, number> = {};
      for (let i = 0; i < nCols; i++) {
        row[`col_${i}`] = 1;
      }
      const rows = Array.from({ length: nRows }, () => ({ ...row }));
      const table = ColumnarTable.fromRows(rows);
      expect(table.rowCount).toBe(nRows);
      expect(table.columnNames.length).toBe(nCols);
      expect(table.column(`col_0` as never).length).toBe(nRows);
    });
  });
});

  describe('fromRows with missing values', () => {
    it('should fill missing columns with NaN', () => {
      const rows = [
        { a: 1, b: 2 },
        { a: 3 }, // missing 'b'
      ] as Array<{ a: number; b: number }>;
      const table = ColumnarTable.fromRows(rows);
      expect(table.column('a')[1]).toBe(3);
      expect(Number.isNaN(table.column('b')[1])).toBe(true);
    });
  });
