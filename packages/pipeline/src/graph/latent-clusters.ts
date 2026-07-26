/**
 * TSC — Trek Separation Clusters.
 *
 * Discovers measurement clusters (groups of variables that behave as
 * indicators of a shared latent parent) using rank constraints from
 * the covariance matrix.
 *
 * Algorithm:
 *  1. Build a clustering graph based on tetrad constraints:
 *     Variables i, j share a latent parent if the vanishing tetrad
 *     test ρ_ij·ρ_kl = ρ_ik·ρ_jl holds for triples.
 *  2. Form clusters via connected components in the clustering graph.
 *  3. Return cluster assignments for downstream measurement model building.
 *
 * Reference: Silva et al. (UAI 2006). "Learning the structure of
 *            linear latent variable models."
 *
 * @packageDocumentation */
import { Matrix } from 'ml-matrix';

export interface ClusterResult {
  /** Cluster assignments: variable idx → cluster id */
  assignments: Map<number, number>;
  /** Number of clusters found */
  nClusters: number;
  /** Cluster cardinalities */
  clusterSizes: number[];
}

/**
 * Discover measurement clusters using tetrad rank constraints.
 *
 * @param data — data matrix (n × d)
 * @param alpha — significance threshold for tetrad test (default 0.05)
 */
export function discoverClusters(
  data: Matrix,
  _alpha: number = 0.05,
): ClusterResult {
  const d = data.columns;
  const n = data.rows;

  if (d < 4 || n < 10) {
    return { assignments: new Map(), nClusters: 0, clusterSizes: [] };
  }

  // Compute correlation matrix
  const corr = correlationMatrix(data);

  // Build clustering graph: connect i,j if they share a latent parent
  const adj: boolean[][] = []; for (let _i=0; _i<d; _i++) adj.push(new Array<boolean>(d).fill(false) as boolean[]);

  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      // Tetrad test: check vanishing tetrad with many (k, l) pairs
      let tetradCount = 0;
      let tetradPass = 0;

      for (let k = 0; k < d && tetradCount < 50; k++) {
        if (k === i || k === j) continue;
        for (let l = k + 1; l < d && tetradCount < 50; l++) {
          if (l === i || l === j) continue;
          tetradCount++;

          // Tetrad: ρ_ij·ρ_kl - ρ_ik·ρ_jl ≈ 0 if i,j share latent
          const rho_ij = Math.abs(corr[i][j]);
          const rho_kl = Math.abs(corr[k][l]);
          const rho_ik = Math.abs(corr[i][k]);
          const rho_jl = Math.abs(corr[j][l]);

          const tetrad = Math.abs(rho_ij * rho_kl - rho_ik * rho_jl);

          // Fisher-transformed test for vanishing tetrad
          const z_ij = 0.5 * Math.log((1 + rho_ij) / Math.max(1e-10, 1 - rho_ij));
          const se = 1 / Math.sqrt(n - 3);
          const threshold = 1.96 * se; // 95% confidence

          if (Math.abs(z_ij * z_ij) < 10 && tetrad < threshold) {
            tetradPass++;
          }
        }
      }

      // If most tetrads vanish, i and j share a latent parent
      if (tetradCount >= 3 && tetradPass / tetradCount > 0.6) {
        adj[i][j] = adj[j][i] = true;
      }
    }
  }

  // Connected components = clusters
  const visited = new Set<number>();
  const assignments = new Map<number, number>();
  let clusterId = 0;

  for (let i = 0; i < d; i++) {
    if (visited.has(i)) continue;
    const comp = new Set<number>();
    const stack = [i];
    while (stack.length > 0) {
      const v = stack.pop()!;
      if (comp.has(v)) continue;
      comp.add(v);
      visited.add(v);
      for (let w = 0; w < d; w++) {
        if (!visited.has(w) && adj[v][w]) stack.push(w);
      }
    }
    // Only clusters with ≥ 3 variables are meaningful latent indicators
    if (comp.size >= 3) {
      for (const v of comp) assignments.set(v, clusterId);
      clusterId++;
    }
  }

  const clusterSizes: number[] = [];
  for (let c = 0; c < clusterId; c++) {
    clusterSizes.push([...assignments.values()].filter(v => v === c).length);
  }

  return { assignments, nClusters: clusterId, clusterSizes };
}

function correlationMatrix(data: Matrix): number[][] {
  const d = data.columns;
  const n = data.rows;
  const means: number[] = new Array<number>(d).fill(0) as number[];
  const stds: number[] = new Array<number>(d).fill(0) as number[];

  for (let j = 0; j < d; j++) {
    let sum = 0; for (let i = 0; i < n; i++) sum += data.get(i, j);
    means[j] = sum / n;
    let sq = 0; for (let i = 0; i < n; i++) sq += (data.get(i, j) - means[j]!) ** 2;
    stds[j] = Math.sqrt(sq / n);
  }

  const corr: number[][] = []; for (let _i=0; _i<d; _i++) corr.push(new Array<number>(d).fill(0) as number[]);
  for (let i = 0; i < d; i++) {
    corr[i][i] = 1;
    for (let j = i + 1; j < d; j++) {
      let cov = 0;
      for (let r = 0; r < n; r++) cov += (data.get(r, i) - means[i]!) * (data.get(r, j) - means[j]!);
      cov /= n;
      const denom = stds[i]! * stds[j]!;
      corr[i][j] = corr[j][i] = denom > 0 ? cov / denom : 0;
    }
  }
  return corr;
}
