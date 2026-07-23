import { describe, it, expect } from 'vitest';
import { EncryptedStore } from '../encrypted-store.js';

const testKey = new Uint8Array(32);
for (let i = 0; i < 32; i++) testKey[i] = i + 1;

describe('EncryptedStore', () => {
  it('creates with a 32-byte key', () => {
    const s = new EncryptedStore({ key: testKey });
    expect(s).toBeDefined();
  });

  it('reports availability based on runtime', () => {
    const s = new EncryptedStore({ key: testKey });
    expect(typeof s.isAvailable()).toBe('boolean');
  });

  it('init does not throw', async () => {
    const s = new EncryptedStore({ key: testKey });
    await s.init();
  });

  it('encrypt/decrypt round-trip preserves data', async () => {
    const s = new EncryptedStore({ key: testKey });
    await s.init();
    if (!s.isAvailable()) return; // skip in crypto-unavailable environments

    const plaintext = 'Hello, Causality Analyzer!';
    const encrypted = await s.encrypt(plaintext);
    expect(encrypted).not.toBe(plaintext);
    const decrypted = await s.decrypt(encrypted);
    expect(decrypted).toBe(plaintext);
  });

  it('encrypt produces different outputs for same plaintext (IV randomness)', async () => {
    const s = new EncryptedStore({ key: testKey });
    await s.init();
    if (!s.isAvailable()) return;

    const c1 = await s.encrypt('test');
    const c2 = await s.encrypt('test');
    expect(c1).not.toBe(c2); // different IVs
    expect(await s.decrypt(c1)).toBe('test');
    expect(await s.decrypt(c2)).toBe('test');
  });
});
