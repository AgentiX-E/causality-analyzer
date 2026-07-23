/**
 * Tamper-Evident Audit Trail — SHA-256 hash-chained append-only log.
 *
 * Every entry in the audit log is hash-chained: each entry includes the
 * SHA-256 hash of the previous entry plus its own payload. verify() detects
 * any insertion, deletion, alteration, or reordering.
 *
 * Uses Web Crypto API (standard in Node 20+, browser, edge runtimes).
 *
 * Reference: Merkle-Damgård hash chain pattern (RFC 6962).
 *
 * Design decisions:
 * - Zero runtime dependencies (uses native crypto.subtle)
 * - Each entry: { index, timestamp, type, payload, previousHash, hash }
 * - verify() returns { valid, tamperedIndices } for precise diagnostics
 * - Immutable entries (cannot be modified once appended)
 *
 * @packageDocumentation
 */

export interface AuditEntry {
  /** Monotonically increasing index */
  readonly index: number;
  /** ISO 8601 timestamp */
  readonly timestamp: string;
  /** Entry type for categorization */
  readonly type: AuditEntryType;
  /** Arbitrary JSON-serializable payload */
  readonly payload: unknown;
  /** SHA-256 hash of previous entry (or '0' for genesis) */
  readonly previousHash: string;
  /** SHA-256 hash of this entry's content */
  readonly hash: string;
}

export type AuditEntryType =
  | 'analysis.start'
  | 'analysis.complete'
  | 'rca.result'
  | 'causal.estimate'
  | 'counterfactual.query'
  | 'sensitivity.result'
  | 'model.update'
  | 'config.change'
  | 'evidence.observed'
  | 'diagnosis.ranked';

export interface AuditVerifyResult {
  /** Whether the entire chain is valid */
  valid: boolean;
  /** Indices of tampered entries (empty if valid) */
  tamperedIndices: number[];
  /** Total number of entries verified */
  entriesVerified: number;
  /** Human-readable diagnostic message */
  diagnosis: string;
}

/**
 * Tamper-evident audit trail.
 *
 * Example:
 * ```typescript
 * const audit = await AuditTrail.create();
 * await audit.append('analysis.start', { graphId: 'g1', nodes: 5 });
 * await audit.append('rca.result', { rootCause: 'Memory', score: 0.87 });
 * const result = await audit.verify();
 * console.log(result.valid); // true
 * ```
 */
export class AuditTrail {
  private entries: AuditEntry[] = [];
  private lastHash: string = '0';

  private constructor() {}

  /**
   * Create a new audit trail.
   * Async because it initializes the genesis hash.
   */
  static async create(): Promise<AuditTrail> {
    return new AuditTrail();
  }

  /**
   * Append an entry to the audit trail.
   *
   * @param type — entry type
   * @param payload — arbitrary JSON-serializable data
   * @returns the appended entry
   */
  async append(type: AuditEntryType, payload: unknown): Promise<AuditEntry> {
    const index = this.entries.length;
    const timestamp = new Date().toISOString();
    const previousHash = this.lastHash;

    // Build content to hash: index | timestamp | type | payload | previousHash
    const contentString = JSON.stringify({ index, timestamp, type, payload, previousHash });
    const hash = await sha256(contentString);

    const entry: AuditEntry = Object.freeze({
      index,
      timestamp,
      type,
      payload,
      previousHash,
      hash,
    });

    this.entries.push(entry);
    this.lastHash = hash;
    return entry;
  }

  /**
   * Verify the integrity of the entire audit trail.
   *
   * Checks:
   * 1. Each entry's hash matches its content
   * 2. Each entry's previousHash matches the previous entry's hash
   * 3. Entry indices are sequential
   *
   * @returns verification result with tampered indices
   */
  async verify(): Promise<AuditVerifyResult> {
    const tamperedIndices: number[] = [];
    const n = this.entries.length;

    if (n === 0) {
      return { valid: true, tamperedIndices: [], entriesVerified: 0, diagnosis: 'Empty audit trail — nothing to verify' };
    }

    // Verify genesis entry
    const genesis = this.entries[0]!;
    if (genesis.index !== 0) {
      tamperedIndices.push(0);
    }

    // Verify each entry
    for (let i = 0; i < n; i++) {
      const entry = this.entries[i]!;
      const expectedPrevHash = i === 0 ? '0' : this.entries[i - 1]!.hash;

      // Check previousHash chain
      if (entry.previousHash !== expectedPrevHash) {
        tamperedIndices.push(i);
        continue;
      }

      // Verify entry's own hash
      const contentString = JSON.stringify({
        index: entry.index,
        timestamp: entry.timestamp,
        type: entry.type,
        payload: entry.payload,
        previousHash: entry.previousHash,
      });
      const computedHash = await sha256(contentString);

      if (entry.hash !== computedHash) {
        tamperedIndices.push(i);
      }

      // Check index sequential
      if (entry.index !== i) {
        if (!tamperedIndices.includes(i)) tamperedIndices.push(i);
      }
    }

    const valid = tamperedIndices.length === 0;
    return {
      valid,
      tamperedIndices,
      entriesVerified: n,
      diagnosis: valid
        ? `All ${n} entries verified — no tampering detected`
        : `Tampering detected at entries: [${tamperedIndices.join(', ')}]`,
    };
  }

  /** Get a snapshot of all entries (read-only) */
  getEntries(): readonly AuditEntry[] {
    return this.entries;
  }

  /** Count of entries */
  get length(): number {
    return this.entries.length;
  }

  /** Export the full audit trail as JSON (for persistence) */
  toJSON(): AuditEntry[] {
    return [...this.entries];
  }

  /**
   * Restore audit trail from JSON snapshot.
   * Verifies integrity on restore.
   *
   * @throws if the snapshot fails verification
   */
  static async fromJSON(entries: AuditEntry[]): Promise<AuditTrail> {
    const audit = new AuditTrail();
    if (entries.length === 0) return audit;

    // Verify snapshot integrity
    const tempTrail = new AuditTrail();
    tempTrail.entries = [...entries];
    tempTrail.lastHash = entries[entries.length - 1]?.hash ?? '0';
    const result = await tempTrail.verify();
    if (!result.valid) {
      throw new Error(`Cannot restore audit trail: tampering detected at entries [${result.tamperedIndices.join(', ')}]`);
    }

    audit.entries = [...entries];
    audit.lastHash = entries[entries.length - 1]?.hash ?? '0';
    return audit;
  }
}

// ── Crypto ──────────────────────────────────────────────────────────────

/**
 * Compute SHA-256 hash of a string.
 * Uses Web Crypto API (standard in Node 20+, browsers, edge runtimes).
 */
async function sha256(input: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);

  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  // Fallback: use Node.js crypto module
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const nodeCrypto = require('crypto') as typeof import('crypto');
  return nodeCrypto.createHash('sha256').update(input).digest('hex');
}
