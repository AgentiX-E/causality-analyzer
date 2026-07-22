/**
 * BaseConfig — configuration system with zod schema validation.
 *
 * All component configurations in Causality Analyzer extend BaseConfig.
 * It provides JSON/YAML serialization, schema validation, and
 * runtime type-safety on top of TypeScript's compile-time checks.
 *
 * @packageDocumentation
 */

import type { ZodType, ZodTypeDef } from 'zod';

/** Result of a configuration validation */
export interface ValidationResult {
  /** Whether the configuration is valid */
  readonly valid: boolean;
  /** Error messages if invalid (empty if valid) */
  readonly errors: ReadonlyArray<string>;
  /** Warnings that don't prevent use but should be reviewed */
  readonly warnings: ReadonlyArray<string>;
}

/** Options for BaseConfig construction */
export interface BaseConfigOptions {
  /** Human-readable name for this configuration (used in error messages) */
  readonly name?: string;
}

/**
 * Abstract base class for all component configurations.
 *
 * Subclasses define their schema as a static `schema` property
 * (a zod schema) and their configuration fields as constructor
 * parameters with defaults.
 *
 * @example
 * ```typescript
 * class PCConfig extends BaseConfig {
 *   static override schema = z.object({
 *     alpha: z.number().min(0).max(1).default(0.01),
 *     maxDegree: z.number().int().positive().default(5),
 *   });
 *
 *   readonly alpha: number;
 *   readonly maxDegree: number;
 *
 *   constructor(params: { alpha?: number; maxDegree?: number } = {}) {
 *     super();
 *     this.alpha = params.alpha ?? 0.01;
 *     this.maxDegree = params.maxDegree ?? 5;
 *   }
 * }
 * ```
 */
export abstract class BaseConfig {
  /** Human-readable name for this configuration */
  readonly name: string;

  constructor(options: BaseConfigOptions = {}) {
    this.name = options.name ?? this.constructor.name;
  }

  /**
   * Validate this configuration against its zod schema.
   * Override `getSchema()` in subclasses to define the schema.
   */
  validate(): ValidationResult {
    const schema = this.getSchema();
    if (!schema) {
      return { valid: true, errors: [], warnings: [] };
    }

    const result = schema.safeParse(this);
    if (result.success) {
      return { valid: true, errors: [], warnings: [] };
    }

    return {
      valid: false,
      errors: result.error.issues.map(
        (issue) => `${issue.path.join('.')}: ${issue.message}`,
      ),
      warnings: [],
    };
  }

  /**
   * Get the zod schema for this configuration.
   * Override in subclasses to enable validation.
   */
  protected abstract getSchema(): ZodType<this, ZodTypeDef, unknown>;

  /**
   * Validate and throw if invalid.
   * @throws Error with validation details if configuration is invalid
   */
  validateOrThrow(): void {
    const result = this.validate();
    if (!result.valid) {
      throw new Error(
        `Invalid configuration for ${this.name}:\n${result.errors.map((e) => `  - ${e}`).join('\n')}`,
      );
    }
  }

  /** Serialize configuration to a plain JSON object */
  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(this)) {
      const value = (this as Record<string, unknown>)[key];
      if (value !== undefined && typeof value !== 'function') {
        result[key] = value;
      }
    }
    return result;
  }

  /** Serialize configuration to a JSON string */
  toString(): string {
    return JSON.stringify(this.toJSON(), null, 2);
  }

}
