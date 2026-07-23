/**
 * Zod-based configuration validation schemas.
 *
 * Every config interface in the pipeline package has a corresponding
 * Zod schema here. Validation is performed at constructor entry points
 * and throws ValidationError (from @agentix-e/causality-analyzer-core)
 * on failure.
 *
 * Design decisions:
 * - All schemas are exported for external use (e.g., API validation)
 * - `.strict()` prevents unknown keys from passing silently
 * - Default values are applied via `.default()` where appropriate
 * - Runtime validation catches bugs that TypeScript cannot
 *
 * @packageDocumentation
 */
import { z } from 'zod';
import type { ZodIssue } from 'zod';
import { ValidationError, ErrorCode } from '@agentix-e/causality-analyzer-core';

// ── Detection Configs ────────────────────────────────────────────────────

export const SRConfigSchema = z.object({
  magWindow: z.number().int().positive().max(999).default(3),
  scoreWindow: z.number().int().positive().max(999).default(21),
  threshold: z.number().positive().max(100).default(3.0),
  minPoints: z.number().int().positive().max(65536).default(32),
}).strict();

export const StatsDetectorConfigSchema = z.object({
  method: z.enum(['zscore', 'mad', 'iqr']).default('zscore'),
  threshold: z.number().positive().max(100).default(3.0),
  minSamples: z.number().int().min(2).max(1000000).default(10),
}).strict();

export const SPOTConfigSchema = z.object({
  q: z.number().min(0.0001).lt(0.5).default(0.01),
  level: z.number().min(0.0001).max(0.1).default(0.05),
  initSize: z.number().int().min(10).max(100000).default(100),
  maxPeaks: z.number().int().min(100).max(100000).default(10000),
}).strict();

export const DSPOTConfigSchema = SPOTConfigSchema.extend({
  // DSPOT adds no new fields but inherits all from SPOT
}).strict();

export const VotingDetectorConfigSchema = z.object({
  strategy: z.enum(['majority', 'weighted', 'maximum']).default('majority'),
  minAgreement: z.number().min(0.5).max(1).default(0.5),
}).strict();

// ── Causal Discovery Configs ─────────────────────────────────────────────

export const PCConfigSchema = z.object({
  alpha: z.number().min(0.001).max(0.5).default(0.05),
  maxDegree: z.number().int().min(-1).max(1000).default(-1),
  stable: z.boolean().default(true),
}).strict();

export const GESConfigSchema = z.object({
  maxIter: z.number().int().min(1).max(1000).default(100),
  alpha: z.number().min(0.001).max(0.5).default(0.05),
}).strict();

export const KCIConfigSchema = z.object({
  permutations: z.number().int().min(1).max(10000).default(100),
  bandwidth: z.number().positive().max(10).default(0.5),
}).strict();

// ── RCA Configs ──────────────────────────────────────────────────────────

export const RHTConfigSchema = z.object({
  tauMax: z.number().int().min(0).max(100).default(5),
  aggregator: z.enum(['max', 'mean', 'sum']).default('max'),
}).strict();

export const DAConfigSchema = z.object({
  bonus: z.number().min(0).max(1).default(0.5),
  malus: z.number().min(0).max(1).default(0.3),
}).strict();

export const FusionConfigSchema = z.object({
  strategy: z.enum(['weighted', 'nested', 'voting']).default('weighted'),
  weights: z.object({
    metric: z.number().min(0).max(1).default(0.5),
    trace: z.number().min(0).max(1).default(0.35),
    log: z.number().min(0).max(1).default(0.15),
  }).optional(),
}).strict();

// ── Infrastructure Configs ───────────────────────────────────────────────

export const RateLimiterConfigSchema = z.object({
  maxRequests: z.number().int().min(1).max(1000000).default(100),
  window: z.number().int().min(1).max(3600000).default(1000),
  overflow: z.enum(['reject', 'queue', 'drop']).default('reject'),
}).strict();

export const EncryptedStoreConfigSchema = z.object({
  algorithm: z.enum(['AES-256-GCM']).default('AES-256-GCM'),
}).strict();

// ── Inference Configs ────────────────────────────────────────────────────

export const CausalForestConfigSchema = z.object({
  numTrees: z.number().int().min(1).max(1000).default(100),
  minSamplesLeaf: z.number().int().min(1).max(100000).default(5),
  maxDepth: z.number().int().min(1).max(100).default(20),
  honesty: z.boolean().default(true),
}).strict();

// ── Validate Helper ──────────────────────────────────────────────────────

/**
 * Validate a config object against a Zod schema, throwing ValidationError on failure.
 *
 * @param schema — Zod schema to validate against
 * @param config — config object to validate
 * @param configName — human-readable name for error messages
 * @throws {ValidationError} if validation fails
 */
export function validateOrThrow<T>(
  schema: z.ZodSchema<T>,
  config: Partial<T> | unknown,
  configName: string,
): T {
  const result = schema.safeParse(config);
  if (!result.success) {
    const errors = result.error.issues.map((i: ZodIssue) => `  - ${i.path.join('.')}: ${i.message}`).join('\n');
    throw new ValidationError(
      ErrorCode.INVALID_CONFIG,
      `Invalid ${configName} configuration:\n${errors}`,
      {
        field: configName,
        received: config,
        context: { issues: result.error.issues },
      },
    );
  }
  return result.data;
}

/**
 * Validate a config object against a Zod schema, returning validation result.
 * Non-throwing version — use validateOrThrow for constructor entry points.
 */
export function validateConfig<T>(
  schema: z.ZodSchema<T>,
  config: Partial<T> | unknown,
): { valid: boolean; data?: T; errors?: string[] } {
  const result = schema.safeParse(config);
  if (!result.success) {
    return {
      valid: false,
      errors: result.error.issues.map((i: ZodIssue) => `${i.path.join('.')}: ${i.message}`),
    };
  }
  return { valid: true, data: result.data };
}
