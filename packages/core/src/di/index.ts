/**
 * Dependency Injection type definitions for Causality Analyzer.
 *
 * The DI container supports constructor-based injection of
 * IRelationalStore and IGraphStore implementations.
 * External storage backends (embed, remote) are injected
 * at runtime — the pipeline package has zero hard dependency
 * on any specific storage implementation.
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
