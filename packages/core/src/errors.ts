/**
 * Structured error hierarchy for Causality Analyzer.
 *
 * Every thrown error in the library MUST be an instance of CausalityError
 * or one of its subclasses. This enables:
 * - Machine-readable error codes for programmatic handling
 * - Structured context for debugging and observability
 * - Cause chaining for root-cause tracing
 * - Type-safe instanceof checks for error discrimination
 *
 * Design decisions:
 * - No string-matching on error messages — use `code` property
 * - All subclasses accept `cause` for error wrapping
 * - `toJSON()` for structured logging (OpenTelemetry, audit trail)
 * - Immutable after construction
 *
 * @packageDocumentation
 */

// ── Error Codes ──────────────────────────────────────────────────────────

export const ErrorCode = {
  // Store
  CONNECTION_FAILED: 'CONNECTION_FAILED',
  QUERY_FAILED: 'QUERY_FAILED',
  TRANSACTION_FAILED: 'TRANSACTION_FAILED',
  STORE_CLOSED: 'STORE_CLOSED',

  // Validation
  INVALID_CONFIG: 'INVALID_CONFIG',
  INVALID_DATA: 'INVALID_DATA',
  INVALID_GRAPH: 'INVALID_GRAPH',
  SCHEMA_MISMATCH: 'SCHEMA_MISMATCH',
  DIMENSION_MISMATCH: 'DIMENSION_MISMATCH',

  // Not Found
  NODE_NOT_FOUND: 'NODE_NOT_FOUND',
  GRAPH_NOT_FOUND: 'GRAPH_NOT_FOUND',
  MODEL_NOT_FOUND: 'MODEL_NOT_FOUND',
  RESULT_NOT_FOUND: 'RESULT_NOT_FOUND',

  // Algorithm
  MAX_ITERATIONS: 'MAX_ITERATIONS',
  NO_CONVERGENCE: 'NO_CONVERGENCE',
  SINGULAR_MATRIX: 'SINGULAR_MATRIX',
  NUMERICAL_INSTABILITY: 'NUMERICAL_INSTABILITY',

  // Config
  MISSING_REQUIRED: 'MISSING_REQUIRED',
  CONFIG_CONFLICT: 'CONFIG_CONFLICT',
  UNSUPPORTED_OPERATION: 'UNSUPPORTED_OPERATION',

  // General
  INTERNAL: 'INTERNAL',
  NOT_IMPLEMENTED: 'NOT_IMPLEMENTED',
} as const;

export type ErrorCodeType = (typeof ErrorCode)[keyof typeof ErrorCode];

// ── Base Error ───────────────────────────────────────────────────────────

/**
 * Base error class for all Causality Analyzer errors.
 *
 * Errors thrown by any package in the causality-analyzer ecosystem
 * SHOULD extend this class. External code can catch CausalityError
 * to handle all library errors without string-matching.
 *
 * @example
 * ```typescript
 * try { await store.saveGraph(...); }
 * catch (e) {
 *   if (e instanceof StoreError) handleStoreError(e);
 *   else if (e instanceof CausalityError) handleLibraryError(e);
 *   else throw e; // unknown error
 * }
 * ```
 */
export class CausalityError extends Error {
  /** Machine-readable error code */
  readonly code: ErrorCodeType;
  /** Arbitrary structured context for debugging */
  readonly context: Record<string, unknown>;

  constructor(
    code: ErrorCodeType,
    message: string,
    options?: {
      cause?: unknown;
      context?: Record<string, unknown>;
    },
  ) {
    super(message, options?.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = this.constructor.name;
    this.code = code;
    this.context = options?.context ?? {};

    // Fix prototype chain for instanceof
    Object.setPrototypeOf(this, new.target.prototype);
  }

  /** Serialize to structured log format */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      context: this.context,
      cause: this.cause instanceof Error ? this.cause.message : String(this.cause ?? ''),
    };
  }
}

// ── Store Errors ─────────────────────────────────────────────────────────

/**
 * Database/storage operation failure.
 *
 * @example
 * ```typescript
 * throw new StoreError(
 *   ErrorCode.CONNECTION_FAILED,
 *   'Failed to connect to PostgreSQL',
 *   { store: 'PostgreSQL', operation: 'connect', cause: pgError },
 * );
 * ```
 */
export class StoreError extends CausalityError {
  /** Which store failed */
  readonly store: string;
  /** What operation was being performed */
  readonly operation: string;

  constructor(
    code: ErrorCodeType,
    message: string,
    options: {
      store: string;
      operation: string;
      cause?: unknown;
      context?: Record<string, unknown>;
    },
  ) {
    super(code, message, {
      cause: options.cause,
      context: { store: options.store, operation: options.operation, ...options.context },
    });
    this.store = options.store;
    this.operation = options.operation;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ── Validation Errors ────────────────────────────────────────────────────

/**
 * Configuration or data validation failure.
 *
 * @example
 * ```typescript
 * throw new ValidationError(
 *   ErrorCode.INVALID_CONFIG,
 *   'alpha must be between 0 and 1',
 *   { field: 'alpha', expected: '[0, 1]', received: 2.5 },
 * );
 * ```
 */
export class ValidationError extends CausalityError {
  /** Which field failed validation */
  readonly field?: string;
  /** Expected value description */
  readonly expected?: unknown;
  /** Actual value received */
  readonly received?: unknown;

  constructor(
    code: ErrorCodeType,
    message: string,
    options?: {
      field?: string;
      expected?: unknown;
      received?: unknown;
      cause?: unknown;
      context?: Record<string, unknown>;
    },
  ) {
    super(code, message, {
      cause: options?.cause,
      context: {
        field: options?.field,
        expected: options?.expected,
        received: options?.received,
        ...options?.context,
      },
    });
    this.field = options?.field;
    this.expected = options?.expected;
    this.received = options?.received;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ── Config Errors ────────────────────────────────────────────────────────

/**
 * Configuration loading or parsing failure.
 * Distinct from ValidationError: ConfigError means the config is
 * structurally invalid or missing, not that a value is out of range.
 */
export class ConfigError extends CausalityError {
  /** Which config module was being loaded */
  readonly configName?: string;

  constructor(
    code: ErrorCodeType,
    message: string,
    options?: {
      configName?: string;
      cause?: unknown;
      context?: Record<string, unknown>;
    },
  ) {
    super(code, message, {
      cause: options?.cause,
      context: { configName: options?.configName, ...options?.context },
    });
    this.configName = options?.configName;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ── Not Found Errors ─────────────────────────────────────────────────────

/**
 * Resource not found.
 * Used when a lookup operation fails because the target doesn't exist.
 */
export class NotFoundError extends CausalityError {
  /** Type of resource */
  readonly resource: string;
  /** Identifier used in the lookup */
  readonly identifier: string;

  constructor(
    code: ErrorCodeType,
    message: string,
    options: {
      resource: string;
      identifier: string;
      cause?: unknown;
      context?: Record<string, unknown>;
    },
  ) {
    super(code, message, {
      cause: options.cause,
      context: { resource: options.resource, identifier: options.identifier, ...options.context },
    });
    this.resource = options.resource;
    this.identifier = options.identifier;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ── Convergence Errors ────────────────────────────────────────────────────

/**
 * Algorithm convergence failure.
 * Thrown when an iterative algorithm fails to converge within limits.
 */
export class ConvergenceError extends CausalityError {
  /** Which algorithm failed */
  readonly algorithm: string;
  /** Iterations executed before failure */
  readonly iterations?: number;
  /** Tolerance target that was not met */
  readonly tolerance?: number;

  constructor(
    code: ErrorCodeType,
    message: string,
    options: {
      algorithm: string;
      iterations?: number;
      tolerance?: number;
      cause?: unknown;
      context?: Record<string, unknown>;
    },
  ) {
    super(code, message, {
      cause: options.cause,
      context: {
        algorithm: options.algorithm,
        iterations: options.iterations,
        tolerance: options.tolerance,
        ...options.context,
      },
    });
    this.algorithm = options.algorithm;
    this.iterations = options.iterations;
    this.tolerance = options.tolerance;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
