import type { IGraphStore, CausalGraph, GraphMetadata, GraphVersion } from '@agentix-e/causality-analyzer-core';

/** Stub: Bolt-protocol remote graph store (future implementation) */
export class RemoteGraphStore implements IGraphStore {
  async saveGraph(_g: CausalGraph, _m: GraphMetadata): Promise<string> { throw new Error('Not implemented: install neo4j-driver-lite'); }
  async loadGraph(): Promise<null> { return null; }
  async loadGraphVersion(): Promise<null> { return null; }
  async listGraphVersions(): Promise<GraphVersion[]> { return []; }
  async findSimilarGraphs(): Promise<CausalGraph[]> { return []; }
}
