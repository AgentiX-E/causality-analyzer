/**
 * Kernel Conditional Independence (KCI) test.
 *
 * Based on Zhang, Peters, Janzing & Schölkopf (UAI 2011):
 * "Kernel-based Conditional Independence Test and Application in Causal Discovery"
 *
 * KCI tests X ⟂ Y | Z using kernel methods. Unlike Fisher's Z (linear
 * Gaussian only), KCI can detect nonlinear conditional dependencies.
 *
 * @packageDocumentation
 */
import { Matrix } from 'ml-matrix';

export interface KCIConfig {
  /** RBF kernel width (median heuristic used when nullish) */
  sigma?: number;
  /** Regularization parameter for kernel matrix inversion */
  epsilon?: number;
  /** Number of Monte Carlo permutations for null distribution (0 = Gamma approx) */
  nPermutations?: number;
}

/**
 * KCI independence test.
 *
 * H₀: X ⟂ Y | Z
 *
 * Returns p-value. Small p-value rejects independence.
 *
 * @param data — data matrix (rows × columns)
 * @param xIdx — column index for variable X
 * @param yIdx — column index for variable Y
 * @param zIndices — column indices for conditioning set Z (can be empty)
 * @param config — optional kernel configuration
 */
export function kciTest(
  data: Matrix,
  xIdx: number,
  yIdx: number,
  zIndices: number[],
  config: KCIConfig = {},
): number {
  const n = data.rows;
  const eps = config.epsilon ?? 1e-3;

  // Extract columns
  const x = extractColumn(data, xIdx);
  const y = extractColumn(data, yIdx);
  const z = zIndices.map(i => extractColumn(data, i));

  // Compute kernel width via median heuristic
  const sigma = config.sigma ?? medianHeuristic([x, y, ...z]);

  // Build kernel matrices
  const Kx = rbfKernel(x, sigma);
  const Ky = rbfKernel(y, sigma);
  const Kz = z.length > 0 ? rbfKernelMulti(z, sigma) : null;

  // Center kernel matrices: K̃ = HKH where H = I - (1/n)11ᵀ
  const KxTilde = centerKernel(Kx, n);
  const KyTilde = centerKernel(Ky, n);

  if (Kz && zIndices.length > 0) {
    const KzTilde = centerKernel(Kz, n);

    // Regress out Z: K̃_{X|Z} = K̃_X - K̃_X (K̃_Z + εI)^{-1} K̃_Z
    // Equivalent to computing Rz * K̃_X * Rz where Rz = ε(K̃_Z + εI)^{-1}
    const KzReg = regularizedInverse(KzTilde, eps, n);

    // Cross-covariance after conditioning on Z
    const KxCond = matrixTripleProduct(KxTilde, KzReg, n);
    const KyCond = matrixTripleProduct(KyTilde, KzReg, n);

    // Test statistic: (1/n) Tr(K̃_{X|Z} K̃_{Y|Z})
    const stat = traceProduct(KxCond, KyCond, n) / n;

    // Use permutation test for p-value (more reliable than Gamma approx)
  return kciPermutationPValue(stat, x, y, z, sigma, eps, n, config.nPermutations ?? 100);
  }

  // Unconditional: HSIC test
  const stat = traceProduct(KxTilde, KyTilde, n) / n;
  return kciPermutationPValue(stat, x, y, [], sigma, eps, n, config.nPermutations ?? 100);
}

// ── Kernel functions ──────────────────────────────────────────────────

/** RBF (Gaussian) kernel: k(x_i, x_j) = exp(-||x_i - x_j||² / (2σ²)) */
function rbfKernel(x: Float64Array, sigma: number): Float64Array {
  const n = x.length;
  const K = new Float64Array(n * n);
  const denom = 2 * sigma * sigma;

  for (let i = 0; i < n; i++) {
    K[i * n + i] = 1; // k(x_i, x_i) = 1
    for (let j = i + 1; j < n; j++) {
      const d = x[i]! - x[j]!;
      const val = Math.exp(-(d * d) / denom);
      K[i * n + j] = val;
      K[j * n + i] = val;
    }
  }
  return K;
}

/** RBF kernel for multivariate Z: k(z_i, z_j) = exp(-||z_i - z_j||² / (2σ²)) */
function rbfKernelMulti(zCols: Float64Array[], sigma: number): Float64Array {
  const n = zCols[0]!.length;
  const K = new Float64Array(n * n);
  const denom = 2 * sigma * sigma;

  for (let i = 0; i < n; i++) {
    K[i * n + i] = 1;
    for (let j = i + 1; j < n; j++) {
      let sqDist = 0;
      for (const col of zCols) {
        const d = (col[i] ?? 0) - (col[j] ?? 0);
        sqDist += d * d;
      }
      const val = Math.exp(-sqDist / denom);
      K[i * n + j] = val;
      K[j * n + i] = val;
    }
  }
  return K;
}

// ── Kernel matrix operations ──────────────────────────────────────────

/** Center a kernel matrix: K̃ = HKH where H_ij = δ_ij - 1/n */
function centerKernel(K: Float64Array, n: number): Float64Array {
  const Kc = new Float64Array(n * n);
  // Row and column means
  const rowMeans = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += K[i * n + j]!;
    rowMeans[i] = sum / n;
  }
  const grandMean = rowMeans.reduce((a, b) => a + b, 0) / n;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Kc[i * n + j] = K[i * n + j]! - rowMeans[i]! - rowMeans[j]! + grandMean;
    }
  }
  return Kc;
}

/** Regularized inverse: (K + εI)⁻¹ using Cholesky decomposition */
function regularizedInverse(K: Float64Array, eps: number, n: number): Float64Array {
  // Add εI to K
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A[i * n + j] = K[i * n + j]! + (i === j ? eps : 0);
    }
  }

  // Cholesky decomposition: A = L Lᵀ
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * n + j]!;
      for (let k = 0; k < j; k++) {
        sum -= L[i * n + k]! * L[j * n + k]!;
      }
      if (i === j) {
        L[i * n + i] = Math.sqrt(Math.max(1e-12, sum));
      } else {
        L[i * n + j] = sum / L[j * n + j]!;
      }
    }
  }

  // Forward and back substitution to compute inverse
  // L Lᵀ X = I → solve L Y = I, then Lᵀ X = Y
  const inv = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    // Forward: solve L y = e_col
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = (i === col ? 1 : 0);
      for (let j = 0; j < i; j++) sum -= L[i * n + j]! * y[j]!;
      y[i] = sum / L[i * n + i]!;
    }
    // Back: solve Lᵀ x = y
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i]!;
      for (let j = i + 1; j < n; j++) sum -= L[j * n + i]! * inv[j * n + col]!;
      inv[i * n + col] = sum / L[i * n + i]!;
    }
  }
  return inv;
}

/** Compute R * K * Rᵀ where R = ε(K̃_Z + εI)⁻¹ */
function matrixTripleProduct(K: Float64Array, R: Float64Array, n: number): Float64Array {
  // We compute R * K * Rᵀ
  // First: temp = K * Rᵀ
  const temp = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += K[i * n + k]! * R[j * n + k]!;
      temp[i * n + j] = sum;
    }
  }
  // Then: result = R * temp
  const result = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += R[i * n + k]! * temp[k * n + j]!;
      result[i * n + j] = sum;
    }
  }
  return result;
}

/** Tr(A B) for n×n matrices stored as Float64Array */
function traceProduct(A: Float64Array, B: Float64Array, n: number): number {
  let trace = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      trace += A[i * n + j]! * B[j * n + i]!;
    }
  }
  return trace;
}

// ── Median heuristic for kernel width ──────────────────────────────────

function medianHeuristic(columns: Float64Array[]): number {
  const n = columns[0]!.length;
  // Sample pairwise distances
  const sampleSize = Math.min(500, n * (n - 1) / 2);
  const dists: number[] = [];
  for (let s = 0; s < sampleSize; s++) {
    const i = Math.floor(Math.random() * n);
    const j = Math.floor(Math.random() * n);
    if (i === j) continue;
    let sqDist = 0;
    for (const col of columns) {
      const d = (col[i] ?? 0) - (col[j] ?? 0);
      sqDist += d * d;
    }
    dists.push(Math.sqrt(sqDist));
  }
  dists.sort((a, b) => a - b);
  const median = dists[Math.floor(dists.length / 2)] ?? 1;
  return Math.max(0.1, median);
}

// ── p-value via permutation test ──────────────────────────────────────

function kciPermutationPValue(
  observedStat: number,
  x: Float64Array,
  y: Float64Array,
  z: Float64Array[],
  sigma: number,
  eps: number,
  n: number,
  nPermutations: number,
): number {
  const yCopy = new Float64Array(y);
  const KxTilde = centerKernel(rbfKernel(x, sigma), n);

  // Pre-compute Z-related matrices (don't change during permutation)
  let KzReg: Float64Array | null = null;
  if (z.length > 0) {
    const Kz = rbfKernelMulti(z, sigma);
    const KzTilde = centerKernel(Kz, n);
    KzReg = regularizedInverse(KzTilde, eps, n);
  }

  let countGreater = 0;
  for (let p = 0; p < nPermutations; p++) {
    // Fisher-Yates shuffle of Y
    for (let i = yCopy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [yCopy[i], yCopy[j]] = [yCopy[j]!, yCopy[i]!];
    }

    const KyPerm = rbfKernel(yCopy, sigma);
    const KyPermTilde = centerKernel(KyPerm, n);

    let permStat: number;
    if (KzReg) {
      const KxCond = matrixTripleProduct(KxTilde, KzReg, n);
      const KyCond = matrixTripleProduct(KyPermTilde, KzReg, n);
      permStat = traceProduct(KxCond, KyCond, n) / n;
    } else {
      permStat = traceProduct(KxTilde, KyPermTilde, n) / n;
    }

    if (permStat >= observedStat) countGreater++;
  }

  return (countGreater + 1) / (nPermutations + 1);
}

// ── Helpers ───────────────────────────────────────────────────────────

function extractColumn(data: Matrix, idx: number): Float64Array {
  const n = data.rows;
  const col = new Float64Array(n);
  for (let i = 0; i < n; i++) col[i] = data.get(i, idx);
  return col;
}
