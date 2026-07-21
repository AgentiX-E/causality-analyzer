import type { IRelationalStore, ColumnarTable, TableSchema, MetricQuery, DetectionResult, ConditionalProbabilityTable, RegressionParams, RCAResult, ResultQuery } from '@agentix-e/causality-analyzer-core';

/** Stub: PG-wire remote relational store (future implementation) */
export class RemoteRelationalStore implements IRelationalStore {
  async readMetrics<S extends TableSchema>(_q: MetricQuery): Promise<ColumnarTable<S>> { throw new Error('Not implemented: install pg'); }
  async writeDetections(_d: DetectionResult[]): Promise<void> {}
  async saveCPT(_g: string, _n: string, _c: ConditionalProbabilityTable): Promise<void> {}
  async loadCPT(): Promise<null> { return null; }
  async saveRegressionModel(): Promise<void> {}
  async loadRegressionModel(): Promise<null> { return null; }
  async saveRCAResult(): Promise<void> {}
  async queryHistoricalResults(_q: ResultQuery): Promise<RCAResult[]> { return []; }
  async beginTransaction(_s: string): Promise<void> {}
  async commitTransaction(_s: string): Promise<void> {}
  async rollbackToCheckpoint(_s: string, _c: string): Promise<void> {}
  async setCheckpoint(_s: string, _n: string): Promise<void> {}
}
