/**
 * Dependency Injection type definitions for Causality Analyzer.
 *
 * The DI container supports constructor-based injection of
 * storage implementations. External backends (embed, remote)
 * are injected at runtime — the pipeline has zero hard dependency
 * on any specific storage implementation.
 *
 * ISP guidance:
 * - Consumers needing only metrics → depend on IMetricStore
 * - Consumers needing only RCA results → depend on IResultStore
 * - The CausalityAnalyzerConfig requires the full IRelationalStore
 *   because the pipeline may use any subset of methods.
 *
 * @packageDocumentation
 */

import type { IRelationalStore, IGraphStore } from '../interfaces/index.js';

/** DI container interface for constructing a Causality Analyzer instance */
export interface CausalityAnalyzerConfig {
  /** Relational store implementation (required) */
  relational: IRelationalStore;
  /** Graph store implementation (required) */
  graph: IGraphStore;
}
