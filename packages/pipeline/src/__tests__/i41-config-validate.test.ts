/**
 * I41: Config validation tests. Covers all Zod schemas with
 * valid/invalid/edge/boundary test cases.
 */
import { describe, it, expect } from 'vitest';
import {
  SRConfigSchema,
  StatsDetectorConfigSchema,
  SPOTConfigSchema,
  VotingDetectorConfigSchema,
  PCConfigSchema,
  GESConfigSchema,
  KCIConfigSchema,
  RHTConfigSchema,
  DAConfigSchema,
  FusionConfigSchema,
  RateLimiterConfigSchema,
  CausalForestConfigSchema,
  EncryptedStoreConfigSchema,
  validateOrThrow,
  validateConfig,
} from '../config/validate.js';
import { ValidationError, ErrorCode } from '@agentix-e/causality-analyzer-core';

// ── Helpers ───────────────────────────────────────────────────────────────

function expectValid<T>(schema: any, data: Partial<T>) {
  const result = validateConfig(schema, data);
  expect(result.valid).toBe(true);
  return result.data;
}

function expectInvalid<T>(schema: any, data: Partial<T>) {
  const result = validateConfig(schema, data);
  expect(result.valid).toBe(false);
  return result.errors!;
}

// ── Detection Schemas ────────────────────────────────────────────────────

describe('SRConfigSchema', () => {
  it('accepts valid config', () => {
    const data = expectValid(SRConfigSchema, { magWindow: 5, threshold: 2.0 });
    expect(data.magWindow).toBe(5);
  });

  it('applies defaults', () => {
    const data = expectValid(SRConfigSchema, {});
    expect(data.magWindow).toBe(3);
    expect(data.scoreWindow).toBe(21);
    expect(data.threshold).toBe(3.0);
    expect(data.minPoints).toBe(32);
  });

  it('rejects negative magWindow', () => {
    expectInvalid(SRConfigSchema, { magWindow: -1 });
  });

  it('rejects zero threshold', () => {
    expectInvalid(SRConfigSchema, { threshold: 0 });
  });

  it('rejects unknown keys (strict)', () => {
    expectInvalid(SRConfigSchema, { unknown: 42 });
  });

  it('rejects threshold > 100', () => {
    expectInvalid(SRConfigSchema, { threshold: 101 });
  });

  it('rejects minPoints > 65536', () => {
    expectInvalid(SRConfigSchema, { minPoints: 100000 });
  });
});

describe('StatsDetectorConfigSchema', () => {
  it('accepts valid methods', () => {
    expectValid(StatsDetectorConfigSchema, { method: 'mad' });
    expectValid(StatsDetectorConfigSchema, { method: 'iqr' });
    expectValid(StatsDetectorConfigSchema, { method: 'zscore' });
  });

  it('rejects invalid method', () => {
    expectInvalid(StatsDetectorConfigSchema, { method: 'unknown' });
  });

  it('rejects insufficient minSamples', () => {
    expectInvalid(StatsDetectorConfigSchema, { minSamples: 1 });
  });

  it('applies defaults', () => {
    const data = expectValid(StatsDetectorConfigSchema, {});
    expect(data.method).toBe('zscore');
    expect(data.minSamples).toBe(10);
  });
});

describe('SPOTConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(SPOTConfigSchema, { q: 0.02, level: 0.01 });
  });

  it('rejects q <= 0', () => {
    expectInvalid(SPOTConfigSchema, { q: 0 });
  });

  it('rejects q >= 0.5', () => {
    expectInvalid(SPOTConfigSchema, { q: 0.5 });
  });

  it('rejects level > 0.1', () => {
    expectInvalid(SPOTConfigSchema, { level: 0.2 });
  });

  it('rejects initSize < 10', () => {
    expectInvalid(SPOTConfigSchema, { initSize: 5 });
  });

  it('applies defaults', () => {
    const data = expectValid(SPOTConfigSchema, {});
    expect(data.q).toBe(0.01);
    expect(data.maxPeaks).toBe(10000);
  });
});

describe('VotingDetectorConfigSchema', () => {
  it('accepts valid strategies', () => {
    expectValid(VotingDetectorConfigSchema, { strategy: 'majority' });
    expectValid(VotingDetectorConfigSchema, { strategy: 'weighted' });
    expectValid(VotingDetectorConfigSchema, { strategy: 'maximum' });
  });

  it('rejects minAgreement < 0.5', () => {
    expectInvalid(VotingDetectorConfigSchema, { minAgreement: 0.3 });
  });

  it('rejects minAgreement > 1', () => {
    expectInvalid(VotingDetectorConfigSchema, { minAgreement: 1.5 });
  });
});

// ── Causal Discovery Schemas ─────────────────────────────────────────────

describe('PCConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(PCConfigSchema, { alpha: 0.01, maxDegree: 5 });
  });

  it('rejects alpha ≤ 0.001', () => {
    expectInvalid(PCConfigSchema, { alpha: 0 });
  });

  it('rejects alpha > 0.5', () => {
    expectInvalid(PCConfigSchema, { alpha: 0.6 });
  });

  it('accepts maxDegree -1 (auto)', () => {
    expectValid(PCConfigSchema, { maxDegree: -1 });
  });

  it('rejects maxDegree > 1000', () => {
    expectInvalid(PCConfigSchema, { maxDegree: 1001 });
  });

  it('defaults alpha to 0.05', () => {
    const data = expectValid(PCConfigSchema, {});
    expect(data.alpha).toBe(0.05);
    expect(data.stable).toBe(true);
  });
});

describe('GESConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(GESConfigSchema, { maxIter: 50 });
  });

  it('rejects maxIter = 0', () => {
    expectInvalid(GESConfigSchema, { maxIter: 0 });
  });

  it('rejects alpha > 0.5', () => {
    expectInvalid(GESConfigSchema, { alpha: 0.51 });
  });
});

describe('KCIConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(KCIConfigSchema, { permutations: 200, bandwidth: 0.3 });
  });

  it('rejects 0 permutations', () => {
    expectInvalid(KCIConfigSchema, { permutations: 0 });
  });

  it('rejects negative bandwidth', () => {
    expectInvalid(KCIConfigSchema, { bandwidth: -1 });
  });

  it('rejects bandwidth > 10', () => {
    expectInvalid(KCIConfigSchema, { bandwidth: 11 });
  });
});

// ── RCA Schemas ──────────────────────────────────────────────────────────

describe('RHTConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(RHTConfigSchema, { tauMax: 3, aggregator: 'mean' });
  });

  it('rejects invalid aggregator', () => {
    expectInvalid(RHTConfigSchema, { aggregator: 'median' });
  });

  it('rejects negative tauMax', () => {
    expectInvalid(RHTConfigSchema, { tauMax: -1 });
  });
});

describe('DAConfigSchema', () => {
  it('accepts valid config', () => {
    const data = expectValid(DAConfigSchema, { bonus: 0.6, malus: 0.4 });
    expect(data.bonus).toBe(0.6);
  });

  it('rejects bonus > 1', () => {
    expectInvalid(DAConfigSchema, { bonus: 1.5 });
  });

  it('rejects negative malus', () => {
    expectInvalid(DAConfigSchema, { malus: -0.1 });
  });

  it('accepts bonus = 0 (min boundary)', () => {
    expectValid(DAConfigSchema, { bonus: 0 });
  });
});

describe('FusionConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(FusionConfigSchema, { strategy: 'nested' });
  });

  it('accepts weighted with weights', () => {
    const data = expectValid(FusionConfigSchema, {
      strategy: 'weighted',
      weights: { metric: 0.6, trace: 0.3, log: 0.1 },
    });
    expect(data.weights!.metric).toBe(0.6);
  });

  it('rejects weights > 1', () => {
    expectInvalid(FusionConfigSchema, { weights: { metric: 1.5, trace: 0.3, log: 0.1 } });
  });

  it('rejects invalid strategy', () => {
    expectInvalid(FusionConfigSchema, { strategy: 'random' });
  });
});

// ── Infrastructure Schemas ───────────────────────────────────────────────

describe('RateLimiterConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(RateLimiterConfigSchema, { maxRequests: 50, window: 500 });
  });

  it('rejects maxRequests = 0', () => {
    expectInvalid(RateLimiterConfigSchema, { maxRequests: 0 });
  });

  it('rejects window = 0', () => {
    expectInvalid(RateLimiterConfigSchema, { window: 0 });
  });

  it('rejects unknown overflow strategy', () => {
    expectInvalid(RateLimiterConfigSchema, { overflow: 'explode' });
  });

  it('defaults to reject overflow', () => {
    const data = expectValid(RateLimiterConfigSchema, {});
    expect(data.overflow).toBe('reject');
  });
});

describe('CausalForestConfigSchema', () => {
  it('accepts valid config', () => {
    expectValid(CausalForestConfigSchema, { numTrees: 50 });
  });

  it('rejects numTrees = 0', () => {
    expectInvalid(CausalForestConfigSchema, { numTrees: 0 });
  });

  it('rejects numTrees > 1000', () => {
    expectInvalid(CausalForestConfigSchema, { numTrees: 1001 });
  });

  it('rejects maxDepth = 0', () => {
    expectInvalid(CausalForestConfigSchema, { maxDepth: 0 });
  });
});

describe('EncryptedStoreConfigSchema', () => {
  it('accepts AES-256-GCM', () => {
    expectValid(EncryptedStoreConfigSchema, { algorithm: 'AES-256-GCM' });
  });

  it('rejects unknown algorithm', () => {
    expectInvalid(EncryptedStoreConfigSchema, { algorithm: 'DES' });
  });
});

// ── validateOrThrow ──────────────────────────────────────────────────────

describe('validateOrThrow', () => {
  it('returns validated data on success', () => {
    const data = validateOrThrow(SRConfigSchema, { threshold: 5.0 }, 'SR');
    expect(data.threshold).toBe(5.0);
  });

  it('throws ValidationError on failure', () => {
    expect(() => validateOrThrow(PCConfigSchema, { alpha: 2 }, 'PC'))
      .toThrow(ValidationError);
  });

  it('throws with descriptive error message', () => {
    try {
      validateOrThrow(PCConfigSchema, { alpha: -5 }, 'PCConfig');
      expect.fail('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ValidationError);
      if (e instanceof ValidationError) {
        expect(e.code).toBe(ErrorCode.INVALID_CONFIG);
        expect(e.field).toBe('PCConfig');
      }
    }
  });

  it('applies defaults via schema', () => {
    const data = validateOrThrow(VotingDetectorConfigSchema, {}, 'Voting');
    expect(data.strategy).toBe('majority');
    expect(data.minAgreement).toBe(0.5);
  });
});

describe('validateConfig (non-throwing)', () => {
  it('returns { valid: true, data } on success', () => {
    const result = validateConfig(SRConfigSchema, { threshold: 2 });
    expect(result.valid).toBe(true);
    expect(result.data).toBeDefined();
  });

  it('returns { valid: false, errors } on failure', () => {
    const result = validateConfig(SRConfigSchema, { threshold: 0 });
    expect(result.valid).toBe(false);
    expect(result.errors!.length).toBeGreaterThan(0);
  });

  it('returns unrecognized keys error for strict schema', () => {
    const result = validateConfig(SRConfigSchema, { bogusField: true });
    expect(result.valid).toBe(false);
    expect(result.errors!.some(e => e.includes('unrecognized') || e.includes('bogus'))).toBe(true);
  });
});
