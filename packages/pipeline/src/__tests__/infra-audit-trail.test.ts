/**
 * I38: Tamper-Evident Audit Trail tests.
 */
import { describe, it, expect } from 'vitest';
import { AuditTrail, type AuditEntry } from '../audit-trail.js';

describe('AuditTrail', () => {
  it('creates an empty audit trail', async () => {
    const audit = await AuditTrail.create();
    expect(audit.length).toBe(0);
    const result = await audit.verify();
    expect(result.valid).toBe(true);
  });

  it('appends entries and verifies chain', async () => {
    const audit = await AuditTrail.create();

    await audit.append('analysis.start', { graphId: 'test', nodes: 3 });
    await audit.append('rca.result', { rootCause: 'Memory', score: 0.87 });
    await audit.append('analysis.complete', { duration: '150ms' });

    expect(audit.length).toBe(3);

    const result = await audit.verify();
    expect(result.valid).toBe(true);
    expect(result.entriesVerified).toBe(3);
    expect(result.tamperedIndices).toEqual([]);
  });

  it('detects tampered entry (altered payload)', async () => {
    const audit = await AuditTrail.create();
    await audit.append('evidence.observed', { metric: 'cpu', value: 85 });
    await audit.append('diagnosis.ranked', { causes: ['Memory'] });

    // Tamper: modify an entry's payload
    const entry = audit.getEntries()[0]!;
    const tamperedEntry: AuditEntry = {
      ...entry,
      payload: { metric: 'cpu', value: 999 }, // altered!
    };

    // Create new audit with tampered entries
    const entries = audit.getEntries().map((e, i) => i === 0 ? tamperedEntry : e);
    await expect(AuditTrail.fromJSON(entries)).rejects.toThrow(/tampering/);
  });

  it('detects tampered entry (altered hash)', async () => {
    const audit = await AuditTrail.create();
    await audit.append('model.update', { cpt: 'updated' });

    const entries = audit.toJSON();
    const tampered: AuditEntry = { ...entries[0]!, hash: 'deadbeef' };
    await expect(AuditTrail.fromJSON([tampered])).rejects.toThrow(/tampering/);
  });

  it('detects missing entry (truncation)', async () => {
    const audit = await AuditTrail.create();
    await audit.append('a', 1);
    await audit.append('b', 2);
    await audit.append('c', 3);

    // Remove middle entry
    const withoutMiddle = [audit.toJSON()[0]!, audit.toJSON()[2]!];
    await expect(AuditTrail.fromJSON(withoutMiddle)).rejects.toThrow(/tampering/);
  });

  it('detects reordered entries', async () => {
    const audit = await AuditTrail.create();
    await audit.append('first', 'a');
    await audit.append('second', 'b');

    const entries = audit.toJSON();
    await expect(AuditTrail.fromJSON([entries[1]!, entries[0]!])).rejects.toThrow(/tampering/);
  });

  it('serializes and deserializes correctly', async () => {
    const audit = await AuditTrail.create();
    await audit.append('rca.result', { rootCause: 'CPU', score: 0.92 });
    await audit.append('sensitivity.result', { eValue: 2.5, robust: true });

    const json = audit.toJSON();
    const restored = await AuditTrail.fromJSON(json);

    expect(restored.length).toBe(2);
    const result = await restored.verify();
    expect(result.valid).toBe(true);
  });

  it('entries are immutable (frozen)', async () => {
    const audit = await AuditTrail.create();
    const entry = await audit.append('config.change', { key: 'threshold', value: 3.0 });

    expect(Object.isFrozen(entry)).toBe(true);
  });

  it('handles large payload', async () => {
    const audit = await AuditTrail.create();
    const largePayload = { data: new Array(1000).fill('x').join('') };
    await audit.append('analysis.complete', largePayload);

    const result = await audit.verify();
    expect(result.valid).toBe(true);
  });

  it('genesis entry has previousHash "0"', async () => {
    const audit = await AuditTrail.create();
    const entry = await audit.append('analysis.start', {});

    expect(entry.previousHash).toBe('0');
    expect(entry.index).toBe(0);
  });

  it('sequential indices are enforced', async () => {
    const audit = await AuditTrail.create();
    const e0 = await audit.append('a', 1);
    const e1 = await audit.append('b', 2);
    const e2 = await audit.append('c', 3);

    expect(e0.index).toBe(0);
    expect(e1.index).toBe(1);
    expect(e2.index).toBe(2);
  });

  it('verify on empty trail is valid', async () => {
    const audit = await AuditTrail.create();
    const result = await audit.verify();
    expect(result.valid).toBe(true);
    expect(result.entriesVerified).toBe(0);
  });
});
