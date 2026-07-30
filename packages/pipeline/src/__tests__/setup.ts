/**
 * Vitest global setup — resets shared mutable state before each test
 * to prevent cross-test contamination.
 *
 * Critical: Fisher Z test has a module-level LRU cache keyed by
 * (i,j,condSet,n). Without reset, Test A's cached p-value leaks
 * into Test B when they share column indices with different data.
 */
import { beforeEach } from 'vitest';
import { _resetFisherZCache as resetCache } from '@agentix-e/causality-analyzer-core';

beforeEach(() => {
  resetCache();
});
