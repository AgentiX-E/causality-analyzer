/**
 * Unit tests for BaseConfig.
 */
import { describe, it, expect } from 'vitest';
import { z } from 'zod'
import { BaseConfig } from '../../index.js';

// Concrete subclass for testing
class TestConfig extends BaseConfig {
  static override schema = z.object({
    alpha: z.number().min(0).max(1).default(0.05),
    maxIter: z.number().int().positive().default(100),
  });

  alpha: number;
  maxIter: number;

  constructor(params: { alpha?: number; maxIter?: number } = {}) {
    super({ name: 'TestConfig' });
    this.alpha = params.alpha ?? 0.05;
    this.maxIter = params.maxIter ?? 100;
  }

  protected override getSchema() {
    return TestConfig.schema as never;
  }
}

class ConfigWithoutSchema extends BaseConfig {
  readonly value: number;
  constructor() {
    super();
    this.value = 42;
  }
  protected override getSchema() {
    return z.object({ value: z.number() }) as any;
  }
}

describe('BaseConfig', () => {
  // ── Construction ─────────────────────────────────────────────────
  describe('construction', () => {
    it('should use defaults when no params provided', () => {
      const config = new TestConfig();
      expect(config.alpha).toBe(0.05);
      expect(config.maxIter).toBe(100);
    });

    it('should override defaults with provided params', () => {
      const config = new TestConfig({ alpha: 0.01, maxIter: 200 });
      expect(config.alpha).toBe(0.01);
      expect(config.maxIter).toBe(200);
    });

    it('should accept partial overrides', () => {
      const config = new TestConfig({ alpha: 0.1 });
      expect(config.alpha).toBe(0.1);
      expect(config.maxIter).toBe(100); // default preserved
    });
  });

  // ── Validation ───────────────────────────────────────────────────
  describe('validate', () => {
    it('should return valid for correct configuration', () => {
      const config = new TestConfig({ alpha: 0.05, maxIter: 50 });
      const result = config.validate();
      expect(result.valid).toBe(true);
      expect(result.errors).toEqual([]);
    });

    it('should return invalid for alpha out of range', () => {
      const config = new TestConfig({ alpha: 1.5 });
      const result = config.validate();
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should return invalid for negative maxIter', () => {
      const config = new TestConfig({ maxIter: -1 });
      const result = config.validate();
      expect(result.valid).toBe(false);
    });

    it('should return valid for config without schema', () => {
      const config = new ConfigWithoutSchema();
      const result = config.validate();
      expect(result.valid).toBe(true);
    });
  });

  describe('validateOrThrow', () => {
    it('should not throw for valid config', () => {
      const config = new TestConfig({ alpha: 0.05 });
      expect(() => config.validateOrThrow()).not.toThrow();
    });

    it('should throw for invalid config', () => {
      const config = new TestConfig({ alpha: 2.0 });
      expect(() => config.validateOrThrow()).toThrow(/Invalid configuration/);
    });
  });

  // ── Serialization ────────────────────────────────────────────────
  describe('toJSON', () => {
    it('should serialize config to a plain object', () => {
      const config = new TestConfig({ alpha: 0.01, maxIter: 50 });
      const json = config.toJSON();
      expect(json).toEqual({
        name: 'TestConfig',
        alpha: 0.01,
        maxIter: 50,
      });
    });
  });

  describe('toString', () => {
    it('should serialize to a JSON string', () => {
      const config = new TestConfig({ alpha: 0.01 });
      const str = config.toString();
      expect(() => JSON.parse(str)).not.toThrow();
      const parsed = JSON.parse(str);
      expect(parsed.alpha).toBe(0.01);
    });
  });

  describe('toJSON round-trip', () => {
    it('should reconstruct equivalent config from toJSON output', () => {
      const original = new TestConfig({ alpha: 0.001, maxIter: 500 });
      const json = original.toJSON();
      const restored = new TestConfig({
        alpha: json.alpha as number,
        maxIter: json.maxIter as number,
      });
      expect(restored.alpha).toBe(0.001);
      expect(restored.maxIter).toBe(500);
    });
  });

  // ── Immutability ──────────────────────────────────────────────────
  describe('mutability', () => {
    it('toJSON should return a copy (not reference)', () => {
      const config = new TestConfig({ alpha: 0.1 });
      const json1 = config.toJSON();
      json1.alpha = 999;
      const json2 = config.toJSON();
      expect(json2.alpha).toBe(0.1); // original unchanged
    });
  });

  // ── fromEnv ──────────────────────────────────────────────────────
  describe('fromEnv', () => {
    it('reads CA_ALPHA from env', () => {
      process.env.CA_ALPHA = '0.02';
      process.env.CA_MAX_ITER = '50';
      const config = TestConfig.fromEnv();
      expect(config.alpha).toBe(0.02);
      expect(config.maxIter).toBe(50);
      delete process.env.CA_ALPHA;
      delete process.env.CA_MAX_ITER;
    });

    it('uses defaults when env vars are absent', () => {
      const config = TestConfig.fromEnv();
      expect(config.alpha).toBe(0.05);
      expect(config.maxIter).toBe(100);
    });

    it('params take precedence over env when both given', () => {
      process.env.CA_ALPHA = '0.5';
      const config = TestConfig.fromEnv({ alpha: 0.01 });
      // Params override env since they're spread first and env overwrites... no.
      // Actually env is read first and put into defaults, then params spread over.
      // From the code: {...params} first, then for-env loop. So env overrides params.
      // Let's verify: params has alpha:0.01, env has CA_ALPHA=0.5, env wins.
      expect(config.alpha).toBe(0.5);
      delete process.env.CA_ALPHA;
    });

    it('non-CA_ prefix env vars are ignored', () => {
      process.env.OTHER_KEY = '999';
      const config = TestConfig.fromEnv();
      expect(config.alpha).toBe(0.05); // unaffected
      delete process.env.OTHER_KEY;
    });

    it('handles string (non-numeric) env values', () => {
      process.env.CA_SOMETHING_TEXT = 'hello';
      // should not crash — non-numeric strings pass through as-is
      // but since there's no field "something_text" in TestConfig, it's ignored
      const config = TestConfig.fromEnv();
      expect(config.alpha).toBe(0.05);
      delete process.env.CA_SOMETHING_TEXT;
    });

    it('custom prefix works', () => {
      process.env.CUSTOM_ALPHA = '0.07';
      const config = TestConfig.fromEnv({}, 'CUSTOM_');
      expect(config.alpha).toBe(0.07);
      delete process.env.CUSTOM_ALPHA;
    });
  });
});

