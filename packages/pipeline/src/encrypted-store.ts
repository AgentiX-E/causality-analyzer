/**
 * Encrypted Store — AES-256-GCM encryption wrapper for embed storage.
 *
 * When sensitive causal analysis results (root cause rankings, anomaly
 * patterns) need protection at rest, wrap the embed store with encryption.
 *
 * Uses Web Crypto API (available in Node.js 19+ and browsers).
 * Falls back gracefully if crypto is unavailable.
 *
 * @packageDocumentation
 */

export interface EncryptedStoreConfig {
  /** AES-256-GCM key (raw bytes, 32 bytes for AES-256) */
  key: Uint8Array;
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

  constructor(config: EncryptedStoreConfig) {
    this.rawKey = config.key;
  }

  /** Initialize the crypto key (async — call before encrypt/decrypt) */
  async init(): Promise<void> {
    if (typeof crypto === 'undefined' || !crypto.subtle) return;
    // Web Crypto types vary between Node and browser — suppress TS2769
    // @ts-expect-error TS2769 importKey overload resolution
    this.key = await crypto.subtle.importKey(
      'raw', this.rawKey, 'AES-GCM', false, ['encrypt', 'decrypt'],
    );
  }

  /** Encrypt plaintext → base64-encoded ciphertext with embedded IV */
  async encrypt(plaintext: string): Promise<string> {
    if (!this.key) throw new Error('EncryptedStore not initialized');
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoded = new TextEncoder().encode(plaintext);
    const ciphertext = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv }, this.key, encoded,
    );
    // Prepend IV to ciphertext for storage
    const combined = new Uint8Array(iv.length + ciphertext.byteLength);
    combined.set(iv);
    combined.set(new Uint8Array(ciphertext), iv.length);
    return btoa(String.fromCharCode(...combined));
  }

  /** Decrypt base64-encoded ciphertext → plaintext */
  async decrypt(ciphertext: string): Promise<string> {
    if (!this.key) throw new Error('EncryptedStore not initialized');
    const combined = Uint8Array.from(atob(ciphertext), c => c.charCodeAt(0));
    const iv = combined.slice(0, 12);
    const data = combined.slice(12);
    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv }, this.key, data,
    );
    return new TextDecoder().decode(decrypted);
  }

  /** Check if encryption is available in the current runtime */
  isAvailable(): boolean { return typeof crypto !== 'undefined' && !!crypto.subtle && !!this.key; }

  /**
   * Rotate encryption key: re-encrypt data from old key to new key.
   *
   * @param plaintext — unencrypted data (re-encrypted with new key)
   * @param oldEncrypted — data encrypted with OLD key (decrypted first, then re-encrypted)
   * @param newConfig — new key configuration
   * @returns data encrypted with new key
   */
  async rotateKey(
    oldEncrypted: string,
    newConfig: EncryptedStoreConfig,
  ): Promise<string> {
    const plaintext = await this.decrypt(oldEncrypted);
    const newStore = new EncryptedStore(newConfig);
    await newStore.init();
    return newStore.encrypt(plaintext);
  }

  /**
   * Generate a random AES-256 key (32 bytes).
   * Uses crypto.getRandomValues for cryptographically secure randomness.
   */
  static generateKey(): Uint8Array {
    if (typeof crypto === 'undefined' || !crypto.getRandomValues) {
      // Fallback for environments without Web Crypto
      const key = new Uint8Array(32);
      for (let i = 0; i < 32; i++) key[i] = Math.floor(Math.random() * 256);
      return key;
    }
    return crypto.getRandomValues(new Uint8Array(32));
  }
}
