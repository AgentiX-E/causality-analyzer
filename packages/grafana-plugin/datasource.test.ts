import { describe, it, expect } from 'vitest';
import { CausalityDataSource } from './datasource';

describe('CausalityDataSource', () => {
  it('constructs with instance settings', () => {
    const ds = new CausalityDataSource(
      { url: 'http://localhost:3000', jsonData: {} },
      { datasourceRequest: async () => ({ data: {} }) },
    );
    expect(ds).toBeDefined();
  });

  it('testDatasource returns success for valid engine', async () => {
    const ds = new CausalityDataSource(
      { url: 'http://localhost:3000' },
      {
        datasourceRequest: async () => ({ data: { version: '1.0.0' } }),
      },
    );
    const result = await ds.testDatasource();
    expect(result.status).toBe('success');
  });

  it('testDatasource returns error on failure', async () => {
    const ds = new CausalityDataSource(
      { url: 'http://localhost:3000' },
      {
        datasourceRequest: async () => { throw new Error('connection refused'); },
      },
    );
    const result = await ds.testDatasource();
    expect(result.status).toBe('error');
  });

  it('query returns empty with hidden targets', async () => {
    const ds = new CausalityDataSource(
      { url: 'http://localhost:3000' },
      { datasourceRequest: async () => ({ data: { edges: [] } }) },
    );
    const result = await ds.query({ targets: [{ hide: true }] });
    expect(result.data).toEqual([]);
  });
});
