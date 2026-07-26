/**
 * Encrypted Store — AES-256-GCM encryption wrapper for embed storage.
 *
 * When sensitive causal analysis results (root cause rankings, anomaly
 * patterns) need protection at rest, wrap the embed store with encryption.
 *
 * Uses Web Crypto API (available in Node.js 19+ and browsers).
 * Falls back gracefully if crypto is unavailable.
 *
 * Security features:
 * - AES-256-GCM authenticated encryption (AEAD)
 * - Fresh 12-byte IV per encryption
 * - Optional AAD (Additional Authenticated Data) binding to context
 * - Key length validation (exactly 32 bytes required)
 *
 * @packageDocumentation
 */

import { ValidationError, ErrorCode } from '@agentix-e/causality-analyzer-core';

export interface EncryptedStoreConfig {
  /** AES-256-GCM key (raw bytes, MUST be exactly 32 bytes for AES-256) */
  key: Uint8Array;
  /** Optional Additional Authenticated Data — binds ciphertext to context.
   *  When set, decryption with different AAD will fail (integrity check).
   *  Use for storeId, record type, or version binding. */
  aad?: string;
}

/**
 * AES-256-GCM encrypt/decrypt utilities for storage encryption.
 *
 * All encryption uses GCM mode with authentication tags (AEAD).
 * Each encrypt call generates a fresh 12-byte IV.
 */
export class EncryptedStore {
  private key: CryptoKey | null = null;
  private rawKey: Uint8Array;
  private aad: Uint8Array | undefined;

  constructor(config: EncryptedStoreConfig) {
    // Validate key length: AES-256 requires exactly 32 bytes
    if (config.key.length !== 32) {
      throw new ValidationError(
        ErrorCode.INVALID_CONFIG,
        `AES-256-GCM requires exactly 32-byte key, got ${config.key.length} bytes`,
        { field: 'key', expected: '32 bytes', received: `${config.key.length} bytes` },
      );
    }
    this.rawKey = config.key;
    this.aad = config.aad ? new TextEncoder().encode(config.aad) : undefined;
  }

  /** Initialize the crypto key (async — call before encrypt/decrypt) */
  async init(): Promise<void> {
    try {
      if (typeof crypto === 'undefined' || !crypto.subtle) {
        throw new Error('Web Crypto API (crypto.subtle) not available in this runtime');
      }
      // Web Crypto types vary between Node and browser — suppress TS2769
      // @ts-expect-error TS2769 importKey overload resolution
      this.key = await crypto.subtle.importKey(
        'raw', this.rawKey, 'AES-GCM', false, ['encrypt', 'decrypt'],
      );
    } catch (err) {
      throw new Error(
        `EncryptedStore.init() failed: ${err instanceof Error ? err.message : String(err)}. ` +
        'Check that crypto.subtle is available and the key is valid.',
      );
    }
  }

  /** Encrypt plaintext → base64-encoded ciphertext with embedded IV */
  async encrypt(plaintext: string): Promise<string> {
    if (!this.key) throw new Error('EncryptedStore not initialized — call init() before encrypt()');
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoded = new TextEncoder().encode(plaintext);
    const ciphertext = await crypto.subtle.encrypt(
      // @ts-expect-error TS2769: Web Crypto API AAD type mismatch across Node/browser
      { name: 'AES-GCM', iv, additionalData: this.aad }, this.key, encoded,
    );
    // Prepend IV to ciphertext for storage
    const combined = new Uint8Array(iv.length + ciphertext.byteLength);
    combined.set(iv);
    combined.set(new Uint8Array(ciphertext), iv.length);
    return btoa(String.fromCharCode(...combined));
  }

  /** Decrypt base64-encoded ciphertext → plaintext */
  async decrypt(ciphertext: string): Promise<string> {
    if (!this.key) throw new Error('EncryptedStore not initialized — call init() before decrypt()');
    const combined = Uint8Array.from(atob(ciphertext), c => c.charCodeAt(0));
    const iv = combined.slice(0, 12);
    const data = combined.slice(12);
    // AAD must match — different AAD will fail integrity check
    const decrypted = await crypto.subtle.decrypt(
      // @ts-expect-error TS2769: Web Crypto API AAD type mismatch across Node/browser
      { name: 'AES-GCM', iv, additionalData: this.aad }, this.key, data,
    );
    return new TextDecoder().decode(decrypted);
  }

  /** Check if encryption is available in the current runtime */
  isAvailable(): boolean { return typeof crypto !== 'undefined' && !!crypto.subtle && !!this.key; }
}
