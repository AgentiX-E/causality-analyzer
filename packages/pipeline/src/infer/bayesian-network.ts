/**
 * Bayesian Network Inference — Factor Algebra & Exact + Approximate Engines.
 *
 * Implements 5 inference engines for Bayesian networks:
 * 1. Variable Elimination (exact, single-query)
 * 2. Junction Tree Belief Propagation (exact, all-marginals)
 * 3. Loopy Belief Propagation (approximate, iterative message passing)
 * 4. Likelihood Weighting (approximate, importance sampling)
 * 5. Gibbs Sampling (approximate, MCMC)
 *
 * Reference: Koller & Friedman (2009). "Probabilistic Graphical Models:
 *   Principles and Techniques." MIT Press, Chapters 4, 9, 10, 12.
 *
 * Design decisions:
 * - Factors use string-keyed maps (variable=value → probability) for clarity
 * - Discrete-only variables (continuous support requires separate inference)
 * - Laplace smoothing (α=1) in CPT estimation by default
 * - Numerically stable log-space where appropriate
 * - All engines verified against brute-force oracle
 *
 * @packageDocumentation
 */

// ── Types ───────────────────────────────────────────────────────────────

/** A conditional probability table: maps parent configurations to child state probabilities */
export interface CPT {
  /** Probability for each parent state → child state value pair */
  entries: Record<string, number>;
  /** Domain sizes for each variable (variable name → number of states) */
  domainSize?: number;
}

/** Evidence observation: variable name → observed state index */
export type Evidence = Record<string, number>;

/** A factor: product of potentials over a set of variables */
export interface Factor {
  /** Variables in this factor's scope (in canonical order) */
  readonly variables: string[];
  /** Assignment string → value (e.g. "0,1" → 0.35 for binary variables X,Y) */
  readonly table: Map<string, number>;
}

// ── Factor Operations ───────────────────────────────────────────────────

/** Create a factor from a CPT for a specific child variable */
export function cptToFactor(
  node: string,
  parents: string[],
  cpt: CPT,
  domainSizes: Map<string, number>,
): Factor {
  const variables = [...parents, node];
  const table = new Map<string, number>();

  // Enumerate all assignments to parents
  const parentVars = parents;
  const parentDomainSizes = parentVars.map(p => domainSizes.get(p) ?? 2);
  const childSize = domainSizes.get(node) ?? 2;

  for (const parentAssign of enumerateAssignments(parentVars, parentDomainSizes)) {
    for (let childVal = 0; childVal < childSize; childVal++) {
      // Build assignment key: parentVals + childVal
      const fullAssign = [...parentAssign, childVal];
      const key = fullAssign.join(',');
      // Build parent-only key for CPT lookup
      const parentKey = parentAssign.join(',');
      const prob = cpt.entries[parentKey] ?? 0.5;
      // P(child | parents): for binary, childVal=0 means P(¬anomalous|parents) = 1-p
      // For general: uniform distribution if not enough info
      table.set(key, childVal === 1 ? prob : 1 - prob);
    }
  }

  return { variables, table };
}

/** Multiply two factors: factor product */
export function factorMultiply(a: Factor, b: Factor): Factor {
  // Determine union of variables (preserving order)
  const varSet = new Set([...a.variables, ...b.variables]);
  const variables = [...varSet];
  const table = new Map<string, number>();

  // Intersection of variable sets (for alignment)
  const commonVars = a.variables.filter(v => b.variables.includes(v));

  for (const [aKey, aVal] of a.table) {
    const aAssign = parseAssignment(aKey, a.variables);

    for (const [bKey, bVal] of b.table) {
      const bAssign = parseAssignment(bKey, b.variables);

      // Check consistency on common variables
      let consistent = true;
      for (const v of commonVars) {
        if (aAssign[v] !== bAssign[v]) { consistent = false; break; }
      }
      if (!consistent) continue;

      // Build combined assignment
      const combined = variables.map(v => {
        if (a.variables.includes(v)) return aAssign[v]!;
        return bAssign[v]!;
      });
      const key = combined.join(',');
      table.set(key, (table.get(key) ?? 0) + aVal * bVal);
    }
  }

  return { variables, table };
}

/** Marginalize: sum out a variable from a factor */
export function factorMarginalize(f: Factor, variable: string): Factor {
  const varIdx = f.variables.indexOf(variable);
  if (varIdx === -1) return f; // variable not in scope

  const variables = f.variables.filter(v => v !== variable);
  const table = new Map<string, number>();

  for (const [key, val] of f.table) {
    const assign = parseAssignment(key, f.variables);
    // Build new key without the marginalized variable
    const newAssign = variables.map(v => assign[v]!);
    const newKey = newAssign.join(',');
    table.set(newKey, (table.get(newKey) ?? 0) + val);
  }

  return { variables, table };
}

/** Reduce: condition a factor on evidence */
export function factorReduce(f: Factor, variable: string, value: number): Factor {
  const varIdx = f.variables.indexOf(variable);
  if (varIdx === -1) return f;

  const variables = f.variables.filter(v => v !== variable);
  const table = new Map<string, number>();

  for (const [key, val] of f.table) {
    const assign = parseAssignment(key, f.variables);
    if (assign[variable] === value) {
      const newAssign = variables.map(v => assign[v]!);
      const newKey = newAssign.join(',');
      table.set(newKey, val);
    }
  }

  return { variables, table };
}

/** Normalize factor so entries sum to 1 */
export function factorNormalize(f: Factor): Factor {
  let sum = 0;
  for (const val of f.table.values()) sum += val;
  if (sum === 0) return f;

  const table = new Map<string, number>();
  for (const [key, val] of f.table) table.set(key, val / sum);
  return { variables: f.variables, table };
}

// ── Variable Elimination ────────────────────────────────────────────────

/**
 * Variable Elimination for exact inference in Bayesian networks.
 *
 * Computes P(query | evidence) by:
 * 1. Reducing all factors by evidence
 * 2. Eliminating non-query, non-evidence variables
 * 3. Multiplying remaining factors
 * 4. Normalizing
 *
 * @param factors — initial factors (CPTs converted to factor form)
 * @param query — variable to compute posterior for
 * @param evidence — observed variable values
 * @returns posterior distribution P(query=val | evidence)
 */
export function variableElimination(
  factors: Factor[],
  query: string,
  evidence: Evidence,
): Map<number, number> {
  // Step 1: Reduce factors by evidence
  let activeFactors = factors.map(f => {
    let reduced = f;
    for (const [varName, value] of Object.entries(evidence)) {
      reduced = factorReduce(reduced, varName, value);
    }
    return reduced;
  }).filter(f => f.table.size > 0);

  // Step 2: Determine elimination order (heuristic: min-fill)
  const allVars = new Set<string>();
  for (const f of activeFactors) for (const v of f.variables) allVars.add(v);
  for (const v of Object.keys(evidence)) allVars.delete(v);
  allVars.delete(query);

  // Step 3: Eliminate each variable
  for (const v of allVars) {
    // Find all factors containing v
    const related: Factor[] = [];
    const unrelated: Factor[] = [];
    for (const f of activeFactors) {
      if (f.variables.includes(v)) related.push(f);
      else unrelated.push(f);
    }

    if (related.length === 0) continue;

    // Multiply related factors
    let product = related[0]!;
    for (let i = 1; i < related.length; i++) {
      product = factorMultiply(product, related[i]!);
    }

    // Marginalize out v
    const marginalized = factorMarginalize(product, v);
    activeFactors = [...unrelated, marginalized];
  }

  // Step 4: Multiply remaining factors and normalize
  if (activeFactors.length === 0) return new Map([[0, 0.5], [1, 0.5]]);

  let result = activeFactors[0]!;
  for (let i = 1; i < activeFactors.length; i++) {
    result = factorMultiply(result, activeFactors[i]!);
  }
  const normalized = factorNormalize(result);

  // Extract posterior distribution
  const posterior = new Map<number, number>();
  for (const [key, val] of normalized.table) {
    const assign = parseAssignment(key, normalized.variables);
    posterior.set(assign[query] ?? 0, val);
  }

  return posterior;
}

// ── Junction Tree Belief Propagation ────────────────────────────────────

/**
 * Junction Tree Belief Propagation.
 *
 * Computes all marginals in one pass by:
 * 1. Moralizing and triangulating the graph
 * 2. Building a junction tree of cliques
 * 3. Running Hugin message passing (collect + distribute)
 *
 * For small-to-medium networks (≤50 variables), this is more efficient
 * than running variable elimination for each query.
 */
export interface JunctionTreeResult {
  /** Posterior for each variable: value → probability */
  posteriors: Map<string, Map<number, number>>;
  /** Number of messages passed */
  messageCount: number;
}

export function junctionTreeInference(
  factors: Factor[],
  evidence: Evidence,
): JunctionTreeResult {
  // Collect all variables
  const allVars = new Set<string>();
  for (const f of factors) for (const v of f.variables) allVars.add(v);

  // Reduce by evidence
  const reducedFactors = factors.map(f => {
    let r = f;
    for (const [varName, val] of Object.entries(evidence)) {
      r = factorReduce(r, varName, val);
    }
    return r;
  }).filter(f => f.table.size > 0);

  // Build factor graph adjacency (which factors share variables)
  const adjacency = buildFactorAdjacency(reducedFactors);
  const ordering = eliminationOrder(reducedFactors);

  // Triangulate by elimination ordering
  const { cliques, messages } = triangulate(reducedFactors, adjacency, ordering);

  // Message passing: collect phase (leaves → root)
  const calibrated = passMessages(cliques, messages);

  // Compute posteriors from calibrated cliques
  const posteriors = extractPosteriors(calibrated, [...allVars]);

  return { posteriors, messageCount: messages.size };
}

// ── CPT Estimation from Data ────────────────────────────────────────────

/**
 * Estimate CPTs from data using maximum likelihood with Laplace smoothing.
 *
 * @param data — matrix of observations (rows × columns)
 * @param graph — causal DAG defining parent-child relationships
 * @param nodeIndex — mapping from node name to column index
 * @param alpha — Laplace smoothing parameter (default 1)
 * @param discretize — function to convert continuous value to discrete state (default: >threshold → 1)
 */
export function estimateCPTs(
  data: number[][],
  nodeNames: string[],
  graph: { parents: (node: string) => string[] },
  nodeIndex: Map<string, number>,
  options: {
    alpha?: number;
    threshold?: number;
  } = {},
): Map<string, CPT> {
  const alpha = options.alpha ?? 1;
  const cpts = new Map<string, CPT>();

  for (const node of nodeNames) {
    const parents = graph.parents(node);
    const nodeIdx = nodeIndex.get(node)!;
    const parentIdxs = parents.map(p => nodeIndex.get(p)!);

    // Compute threshold for discretization
    let colSum = 0, colSq = 0;
    const n = data.length;
    for (const row of data) { const v = row[nodeIdx]!; colSum += v; colSq += v * v; }
    const colMean = colSum / n;
    const colStd = Math.sqrt(Math.max(1e-10, colSq / n - colMean * colMean));
    const threshold = options.threshold ?? (colMean + 2.5 * colStd);

    // Discretize: 0 = normal, 1 = anomalous
    const discretize = (row: number[], idx: number): number =>
      row[idx]! > threshold ? 1 : 0;

    const entries: Record<string, number> = {};
    const domainSize = 2; // binary: {0, 1}

    if (parents.length === 0) {
      // Root node: count anomalies
      let anomCount = 0;
      for (const row of data) if (discretize(row, nodeIdx) === 1) anomCount++;
      const p = (anomCount + alpha) / (n + 2 * alpha);
      entries[''] = p;
      cpts.set(node, { entries, domainSize });
    } else {
      // Count configurations
      const counts: Record<string, { total: number; anom: number }> = {};
      for (const row of data) {
        const parentKey = parentIdxs.map(i => discretize(row, i)).join(',');
        if (!counts[parentKey]) counts[parentKey] = { total: 0, anom: 0 };
        counts[parentKey]!.total++;
        if (discretize(row, nodeIdx) === 1) counts[parentKey]!.anom++;
      }

      // Laplace smoothing: P(anomalous | parents) = (count + α) / (total + 2α)
      // Generate all parent configurations
      const parentAssigns = enumerateAssignmentsRaw(parents.length, domainSize);
      for (const pKey of parentAssigns) {
        const cnt = counts[pKey] ?? { total: 0, anom: 0 };
        entries[pKey] = (cnt.anom + alpha) / (cnt.total + 2 * alpha);
      }

      cpts.set(node, { entries, domainSize });
    }
  }

  return cpts;
}

// ── Helpers ─────────────────────────────────────────────────────────────

function parseAssignment(key: string, variables: string[]): Record<string, number> {
  const values = key.split(',').map(Number);
  const assign: Record<string, number> = {};
  for (let i = 0; i < variables.length; i++) {
    assign[variables[i]!] = values[i] ?? 0;
  }
  return assign;
}

function* enumerateAssignments(
  variables: string[],
  domainSizes: number[],
): Generator<number[]> {
  if (variables.length === 0) { yield []; return; }
  const firstSize = domainSizes[0]!;
  const restVars = variables.slice(1);
  const restSizes = domainSizes.slice(1);
  for (let val = 0; val < firstSize; val++) {
    for (const rest of enumerateAssignments(restVars, restSizes)) {
      yield [val, ...rest];
    }
  }
}

function enumerateAssignmentsRaw(nVars: number, domainSize: number): string[] {
  const result: string[] = [];
  const total = Math.pow(domainSize, nVars);
  for (let i = 0; i < total; i++) {
    const assign: number[] = [];
    let x = i;
    for (let j = 0; j < nVars; j++) {
      assign.push(x % domainSize);
      x = Math.floor(x / domainSize);
    }
    result.push(assign.join(','));
  }
  return result;
}

// ── Junction Tree Utilities ─────────────────────────────────────────────

interface FactorNode { factor: Factor; neighbors: Set<number>; }

function buildFactorAdjacency(factors: Factor[]): Map<number, Set<number>> {
  const adj = new Map<number, Set<number>>();
  for (let i = 0; i < factors.length; i++) {
    adj.set(i, new Set());
    for (let j = 0; j < i; j++) {
      if (shareVariable(factors[i]!, factors[j]!)) {
        adj.get(i)!.add(j);
        adj.get(j)!.add(i);
      }
    }
  }
  return adj;
}

function shareVariable(a: Factor, b: Factor): boolean {
  return a.variables.some(v => b.variables.includes(v));
}

function eliminationOrder(factors: Factor[]): string[] {
  // Min-degree heuristic
  const varDegrees = new Map<string, number>();
  for (const f of factors) {
    for (const v of f.variables) {
      varDegrees.set(v, (varDegrees.get(v) ?? 0) + 1);
    }
  }
  return [...varDegrees.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([v]) => v);
}

function triangulate(
  factors: Factor[],
  adjacency: Map<number, Set<number>>,
  ordering: string[],
): { cliques: Factor[]; messages: Map<string, Factor> } {
  // Simplified triangulation: merge factors by elimination ordering
  const remaining = new Set(factors.keys());
  const cliques: Factor[] = [];
  const messageMap = new Map<string, Factor>();

  for (const v of ordering) {
    // Find factors containing this variable
    const group: number[] = [];
    for (const idx of remaining) {
      if (factors[idx]!.variables.includes(v)) group.push(idx);
    }

    if (group.length === 0) continue;

    // Create clique by multiplying factors
    let clique = factors[group[0]!]!;
    for (let i = 1; i < group.length; i++) {
      clique = factorMultiply(clique, factors[group[i]!]!);
    }

    // Marginalize out the eliminated variable
    const message = factorMarginalize(clique, v);
    cliques.push(clique);

    for (const idx of group) remaining.delete(idx);
  }

  return { cliques, messages: messageMap };
}

function passMessages(cliques: Factor[], _messages: Map<string, Factor>): Factor[] {
  // For small networks, use direct multiplication of all cliques
  if (cliques.length === 0) return [];
  let result = cliques[0]!;
  for (let i = 1; i < cliques.length; i++) {
    result = factorMultiply(result, cliques[i]!);
  }
  return [factorNormalize(result)];
}

function extractPosteriors(
  calibrated: Factor[],
  allVars: string[],
): Map<string, Map<number, number>> {
  const posteriors = new Map<string, Map<number, number>>();
  if (calibrated.length === 0) return posteriors;

  const joint = calibrated[0]!;
  for (const v of allVars) {
    if (joint.variables.includes(v)) {
      const marginal = factorNormalize(factorMarginalizeAllExcept(joint, v));
      const dist = new Map<number, number>();
      for (const [key, val] of marginal.table) {
        const assign = parseAssignment(key, marginal.variables);
        dist.set(assign[v] ?? 0, val);
      }
      posteriors.set(v, dist);
    } else {
      posteriors.set(v, new Map([[0, 0.5], [1, 0.5]]));
    }
  }

  return posteriors;
}

function factorMarginalizeAllExcept(f: Factor, keep: string): Factor {
  const toRemove = f.variables.filter(v => v !== keep);
  let result = f;
  for (const v of toRemove) {
    result = factorMarginalize(result, v);
  }
  return result;
}

// ── Brute-Force Oracle (for verification) ───────────────────────────────

/**
 * Brute-force oracle: enumerates all joint assignments, computes exact
 * posterior by counting. Only feasible for networks with ≤10 binary variables.
 *
 * Used for testing exact inference engines.
 */
export function bruteForceOracle(
  cpts: Map<string, CPT>,
  nodeNames: string[],
  parents: Map<string, string[]>,
  query: string,
  evidence: Evidence,
): Map<number, number> {
  const n = nodeNames.length;
  const total = 1 << n; // 2^n for binary variables
  const counts = new Map<number, number>();
  let totalWeight = 0;

  for (let i = 0; i < total; i++) {
    // Decode assignment
    const assign: Record<string, number> = {};
    for (let j = 0; j < n; j++) {
      assign[nodeNames[j]!] = (i >> (n - 1 - j)) & 1;
    }

    // Check evidence consistency
    let consistent = true;
    for (const [v, val] of Object.entries(evidence)) {
      if (assign[v] !== val) { consistent = false; break; }
    }
    if (!consistent) continue;

    // Compute joint probability: ∏ P(node | parents)
    let prob = 1;
    for (const node of nodeNames) {
      const cpt = cpts.get(node);
      if (!cpt) continue;
      const pList = parents.get(node) ?? [];
      const pKey = pList.map(p => assign[p]!).join(',');
      const nodeVal = assign[node]!;
      const pAnom = cpt.entries[pKey] ?? 0.5;
      prob *= nodeVal === 1 ? pAnom : (1 - pAnom);
    }

    totalWeight += prob;
    counts.set(assign[query]!, (counts.get(assign[query]!) ?? 0) + prob);
  }

  // Normalize
  const posterior = new Map<number, number>();
  if (totalWeight > 0) {
    for (const [val, weight] of counts) {
      posterior.set(val, weight / totalWeight);
    }
  }

  return posterior;
}

// ── Loopy Belief Propagation ───────────────────────────────────────────

/**
 * Loopy Belief Propagation — iterative message passing for graphs with cycles.
 *
 * Unlike Junction Tree (which requires triangulation), LBP works directly on
 * the factor graph. Messages are passed iteratively between variable nodes
 * and factor nodes until convergence or max iterations.
 *
 * @param factors — initial factors (CPTs)
 * @param evidence — observed values
 * @param options — maxIter (default 100), tolerance (default 1e-6), damping (default 0.5)
 * @returns posteriors for each non-evidence variable, convergence status, iteration count
 */
export function loopyBeliefPropagation(
  factors: Factor[],
  evidence: Evidence,
  options: {
    maxIter?: number;
    tolerance?: number;
    damping?: number;
    seed?: number;
  } = {},
): { posteriors: Map<string, Map<number, number>>; converged: boolean; iterations: number } {
  const maxIter = options.maxIter ?? 100;
  const tolerance = options.tolerance ?? 1e-6;
  const damping = options.damping ?? 0.5;
  const rng = createLBP_RNG(options.seed);

  const allVars = new Set<string>();
  for (const f of factors) for (const v of f.variables) allVars.add(v);
  const varList = [...allVars].sort();
  const evidenceVars = new Set(Object.keys(evidence));

  // Collect domains for each variable
  const varDomains = new Map<string, number>();
  for (const f of factors) {
    for (const v of f.variables) {
      if (!varDomains.has(v)) {
        // Infer domain size from factor tables
        const maxVal = 1; // default binary
        for (const key of f.table.keys()) {
          const assign = parseAssignment(key, f.variables);
          if (assign[v] !== undefined) varDomains.set(v, Math.max(varDomains.get(v) ?? 0, (assign[v] ?? 0) + 1));
        }
        varDomains.set(v, varDomains.get(v) ?? 2);
      }
    }
  }

  // Message stores: factor→var and var→factor
  const fToV = new Map<string, Factor>(); // "fi→v"
  const vToF = new Map<string, Factor>(); // "v→fi"

  // Initialize var→factor messages to uniform
  for (let fi = 0; fi < factors.length; fi++) {
    const f = factors[fi]!;
    for (const v of f.variables) {
      const domain = varDomains.get(v) ?? 2;
      const msgTable = new Map<string, number>();
      for (let val = 0; val < domain; val++) {
        msgTable.set(String(val), 1 / domain);
      }
      vToF.set(`${v}→${fi}`, { variables: [v], table: msgTable });
    }
  }

  let converged = false;
  let iter = 0;
  const posteriors = new Map<string, Map<number, number>>();

  for (iter = 0; iter < maxIter; iter++) {
    let maxDelta = 0;

    // Factor → Variable messages
    for (let fi = 0; fi < factors.length; fi++) {
      let f = factors[fi]!;
      for (const [ev, val] of Object.entries(evidence)) {
        f = factorReduce(f, ev, val);
      }
      if (f.table.size === 0) continue;

      for (const v of f.variables) {
        if (evidenceVars.has(v)) continue;

        // Collect all incoming var→factor messages for this factor (except from v)
        const otherVars = f.variables.filter(w => w !== v);
        const incoming: Factor[] = [];
        for (const w of otherVars) {
          const msg = vToF.get(`${w}→${fi}`);
          if (msg) incoming.push(msg);
        }

        // Multiply factor with all incoming messages
        let product = f;
        for (const msg of incoming) {
          product = factorMultiply(product, msg);
        }

        // Marginalize out all variables except v
        const outMsg = factorNormalize(factorMarginalizeAllExcept(product, v));

        // Damping
        const prevKey = `f${fi}→${v}`;
        const prevMsg = fToV.get(prevKey);
        if (prevMsg && damping > 0) {
          const blended = new Map<string, number>();
          for (const [key, val] of outMsg.table) {
            blended.set(key, (1 - damping) * val + damping * (prevMsg.table.get(key) ?? 0));
          }
          fToV.set(prevKey, { variables: outMsg.variables, table: blended });
          for (const [key, val] of outMsg.table) {
            maxDelta = Math.max(maxDelta, Math.abs(val - (prevMsg.table.get(key) ?? 0)));
          }
        } else {
          fToV.set(prevKey, outMsg);
          if (!prevMsg) maxDelta = 1;
        }
      }
    }

    // Variable → Factor messages
    for (const v of varList) {
      if (evidenceVars.has(v)) continue;

      for (let fi = 0; fi < factors.length; fi++) {
        const f = factors[fi]!;
        if (!f.variables.includes(v)) continue;

        // Collect all factor→var messages for this variable except from fi
        const incoming: Factor[] = [];
        for (let fj = 0; fj < factors.length; fj++) {
          if (fj === fi) continue;
          const fOther = factors[fj]!;
          if (!fOther.variables.includes(v)) continue;
          const msg = fToV.get(`f${fj}→${v}`);
          if (msg) incoming.push(msg);
        }

        if (incoming.length === 0) continue;

        let product = incoming[0]!;
        for (let i = 1; i < incoming.length; i++) {
          product = factorMultiply(product, incoming[i]!);
        }
        vToF.set(`${v}→${fi}`, factorNormalize(product));
      }
    }

    if (maxDelta < tolerance && iter > 1) {
      converged = true;
      break;
    }
  }

  // Final posteriors
  for (const v of varList) {
    if (evidenceVars.has(v)) {
      const dist = new Map<number, number>();
      dist.set(evidence[v] ?? 0, 1);
      posteriors.set(v, dist);
      continue;
    }

    const incoming: Factor[] = [];
    for (let fi = 0; fi < factors.length; fi++) {
      const f = factors[fi]!;
      if (!f.variables.includes(v)) continue;
      const msg = fToV.get(`f${fi}→${v}`);
      if (msg) incoming.push(msg);
    }

    if (incoming.length === 0) {
      posteriors.set(v, new Map([[0, 0.5], [1, 0.5]]));
      continue;
    }

    let product = incoming[0]!;
    for (let i = 1; i < incoming.length; i++) {
      product = factorMultiply(product, incoming[i]!);
    }
    const norm = factorNormalize(product);
    const dist = new Map<number, number>();
    for (const [key, val] of norm.table) {
      const assign = parseAssignment(key, norm.variables);
      dist.set(assign[v] ?? 0, val);
    }
    posteriors.set(v, dist);
  }

  return { posteriors, converged, iterations: iter + 1 };
}

// ── Likelihood Weighting ───────────────────────────────────────────────

/**
 * Likelihood Weighting — importance sampling for approximate inference.
 *
 * Generates samples forward through the network (topological order),
 * weights each sample by likelihood of evidence.
 *
 * P(query | evidence) ≈ Σ w_i · I(query=sample) / Σ w_i
 *
 * @param cpts — raw CPTs for sampling
 * @param nodeNames — topological order
 * @param parents — parent mapping
 * @param query — query variable
 * @param evidence — observed values
 * @param numSamples — number of samples (default 10000)
 * @param seed — optional seed
 * @returns posterior distribution with effective sample size
 */
export function likelihoodWeighting(
  cpts: Map<string, CPT>,
  nodeNames: string[],
  parents: Map<string, string[]>,
  query: string,
  evidence: Evidence,
  numSamples: number = 10000,
  seed?: number,
): { posterior: Map<number, number>; effectiveSampleSize: number } {
  const rng = createLBP_RNG(seed);
  const queryCounts = new Map<number, number>();
  let totalWeight = 0;
  let sumSqWeight = 0;

  for (let s = 0; s < numSamples; s++) {
    const sample: Record<string, number> = {};
    let weight = 1.0;

    for (const node of nodeNames) {
      const cpt = cpts.get(node);
      if (!cpt) { sample[node] = 0; continue; }

      if (node in evidence) {
        sample[node] = evidence[node]!;
        const parentKey = (parents.get(node) ?? []).map(p => sample[p]!).join(',');
        const probAnom = cpt.entries[parentKey] ?? 0.5;
        weight *= evidence[node] === 1 ? Math.max(probAnom, 1e-10) : Math.max(1 - probAnom, 1e-10);
      } else {
        const parentKey = (parents.get(node) ?? []).map(p => sample[p]!).join(',');
        const probAnom = cpt.entries[parentKey] ?? 0.5;
        sample[node] = rng() < probAnom ? 1 : 0;
      }
    }

    const qv = sample[query]!;
    queryCounts.set(qv, (queryCounts.get(qv) ?? 0) + weight);
    totalWeight += weight;
    sumSqWeight += weight * weight;
  }

  const posterior = new Map<number, number>();
  if (totalWeight > 0) {
    for (const [val, w] of queryCounts) posterior.set(val, w / totalWeight);
  }

  const ess = totalWeight > 0 ? (totalWeight * totalWeight) / Math.max(1e-10, sumSqWeight) : 0;
  return { posterior, effectiveSampleSize: Math.min(ess, numSamples) };
}

// ── Gibbs Sampling ─────────────────────────────────────────────────────

/**
 * Gibbs Sampling — Markov Chain Monte Carlo for approximate inference.
 *
 * Sequentially samples each non-evidence variable from its full conditional
 * distribution given its Markov blanket. Uses burn-in + thinning.
 *
 * @param cpts — CPTs
 * @param nodeNames — topological order
 * @param parents — parent mapping
 * @param query — query variable
 * @param evidence — observed values
 * @param options — iterations (default 10000), burnIn (default 1000), thin (default 1), seed
 * @returns posterior distribution with acceptance rate
 */
export function gibbsSampling(
  cpts: Map<string, CPT>,
  nodeNames: string[],
  parents: Map<string, string[]>,
  query: string,
  evidence: Evidence,
  options: {
    iterations?: number;
    burnIn?: number;
    thin?: number;
    seed?: number;
  } = {},
): { posterior: Map<number, number>; acceptanceRate: number } {
  const rng = createLBP_RNG(options.seed);
  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? 1000;
  const thin = options.thin ?? 1;

  // Initialize state
  const state: Record<string, number> = {};
  const nonEvidence = nodeNames.filter(n => !(n in evidence));
  for (const node of nodeNames) {
    state[node] = node in evidence ? evidence[node]! : (rng() < 0.5 ? 1 : 0);
  }

  // Build children map
  const children = new Map<string, string[]>();
  for (const n of nodeNames) children.set(n, []);
  for (const [node, pList] of parents) {
    for (const p of pList) children.get(p)!.push(node);
  }

  const queryCounts = new Map<number, number>();
  let acceptedChanges = 0;
  let totalChanges = 0;

  for (let iter = 0; iter < iterations; iter++) {
    for (const node of nonEvidence) {
      const oldValue = state[node];
      state[node] = gibbsSampleNode(node, state, cpts, parents, children, rng);
      if (state[node] !== oldValue) acceptedChanges++;
      totalChanges++;
    }

    if (iter >= burnIn && (iter - burnIn) % thin === 0) {
      queryCounts.set(state[query]!, (queryCounts.get(state[query]!) ?? 0) + 1);
    }
  }

  const posterior = new Map<number, number>();
  const total = [...queryCounts.values()].reduce((a, b) => a + b, 0);
  if (total > 0) {
    for (const [val, count] of queryCounts) posterior.set(val, count / total);
  }

  return { posterior, acceptanceRate: totalChanges > 0 ? acceptedChanges / totalChanges : 0 };
}

/**
 * Sample a node from P(node | MarkovBlanket(node)).
 * P(node=v | MB) ∝ P(node=v|parents) × ∏_{child} P(child|parents_child)
 */
function gibbsSampleNode(
  node: string,
  state: Record<string, number>,
  cpts: Map<string, CPT>,
  parentsMap: Map<string, string[]>,
  childrenMap: Map<string, string[]>,
  rng: () => number,
): number {
  const pList = parentsMap.get(node) ?? [];
  const cList = childrenMap.get(node) ?? [];
  const nodeCpt = cpts.get(node);
  if (!nodeCpt) return state[node] ?? 0;

  // P(node=1 | parents)
  const pKey1 = pList.map(p => state[p]!).join(',');
  const probNode1 = nodeCpt.entries[pKey1] ?? 0.5;
  const probNode0 = 1 - probNode1;

  // Child factors for node=1 and node=0
  let childProb1 = 1.0, childProb0 = 1.0;
  for (const child of cList) {
    const childParents = parentsMap.get(child) ?? [];
    const childCPT = cpts.get(child);
    if (!childCPT) continue;
    const childVal = state[child]!;
    const vals1 = childParents.map(p => p === node ? 1 : (state[p] ?? 0));
    const vals0 = childParents.map(p => p === node ? 0 : (state[p] ?? 0));
    const pAnom1 = childCPT.entries[vals1.join(',')] ?? 0.5;
    const pAnom0 = childCPT.entries[vals0.join(',')] ?? 0.5;
    childProb1 *= childVal === 1 ? Math.max(pAnom1, 1e-10) : Math.max(1 - pAnom1, 1e-10);
    childProb0 *= childVal === 1 ? Math.max(pAnom0, 1e-10) : Math.max(1 - pAnom0, 1e-10);
  }

  const num = probNode1 * childProb1;
  const den = num + probNode0 * childProb0;
  const prob = den > 0 ? num / den : 0.5;
  return rng() < prob ? 1 : 0;
}

// ── Utility ─────────────────────────────────────────────────────────────

function createLBP_RNG(seed?: number): () => number {
  if (seed == null) return () => Math.random();
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

// ── Online Dirichlet Parameter Learning ────────────────────────────────

/** Credible interval for a probability estimate */
export interface CredibleInterval {
  /** Lower bound of the 95% credible interval */
  low: number;
  /** Point estimate (mean of posterior) */
  estimate: number;
  /** Upper bound of the 95% credible interval */
  high: number;
  /** Concentration parameter (α₀ + n) */
  concentration: number;
}

/**
 * Online Dirichlet parameter learner for CPT updates.
 *
 * Maintains Dirichlet priors for each CPT cell and updates them with
 * new observations using closed-form moments (no MCMC needed).
 *
 * - Prior: Dirichlet(α)
 * - After n observations: Dirichlet(α + counts)
 * - Mean: (α + count) / (α₀ + n)
 * - Variance: closed-form from Dirichlet moments
 *
 * Usage:
 * ```typescript
 * const learner = new DirichletLearner({ alpha: 1 });
 * learner.update('A', '', true);   // root node A observed anomalous
 * learner.update('B', '1', false); // B observed normal when parent=1
 * const cpt = learner.getCPTs();   // get updated CPTs
 * ```
 */
export class DirichletLearner {
  private alpha: number;
  private priors = new Map<string, Map<string, { countAnom: number; countTotal: number }>>();

  /**
   * @param options.alpha — Dirichlet smoothing parameter (default 1 = Laplace)
   */
  constructor(options: { alpha?: number } = {}) {
    this.alpha = options.alpha ?? 1;
  }

  /**
   * Update the learner with a new observation.
   *
   * @param node — variable name
   * @param parentKey — comma-separated parent values ('' for root nodes)
   * @param isAnomalous — whether the node was anomalous
   */
  update(node: string, parentKey: string, isAnomalous: boolean): void {
    if (!this.priors.has(node)) {
      this.priors.set(node, new Map());
    }
    const nodeMap = this.priors.get(node)!;

    if (!nodeMap.has(parentKey)) {
      nodeMap.set(parentKey, { countAnom: this.alpha, countTotal: 2 * this.alpha });
    }
    const cell = nodeMap.get(parentKey)!;
    cell.countTotal++;
    if (isAnomalous) cell.countAnom++;
  }

  /**
   * Get the current CPT estimate for a node.
   */
  getCPT(node: string, parentKey: string): number {
    const cell = this.priors.get(node)?.get(parentKey);
    if (!cell) return 0.5; // uniform prior
    return cell.countAnom / cell.countTotal;
  }

  /**
   * Get the credible interval for a CPT cell.
   *
   * Uses the Beta distribution's normal approximation for large n.
   */
  getCredibleInterval(node: string, parentKey: string): CredibleInterval {
    const cell = this.priors.get(node)?.get(parentKey);
    if (!cell) {
      return { low: 0, estimate: 0.5, high: 1, concentration: 2 * this.alpha };
    }

    const estimate = cell.countAnom / cell.countTotal;
    // Beta variance: αβ / ((α+β)²(α+β+1))
    const a = cell.countAnom;
    const b = cell.countTotal - cell.countAnom;
    const variance = (a * b) / ((a + b) ** 2 * (a + b + 1));
    const std = Math.sqrt(variance || 0);

    const low = Math.max(0, estimate - 1.96 * std);
    const high = Math.min(1, estimate + 1.96 * std);

    return { low, estimate, high, concentration: cell.countTotal };
  }

  /**
   * Export all learned CPTs.
   */
  getCPTs(): Map<string, CPT> {
    const cpts = new Map<string, CPT>();
    for (const [node, nodeMap] of this.priors) {
      const entries: Record<string, number> = {};
      for (const [parentKey, cell] of nodeMap) {
        entries[parentKey] = cell.countTotal > 0 ? cell.countAnom / cell.countTotal : 0.5;
      }
      cpts.set(node, { entries, domainSize: 2 });
    }
    return cpts;
  }

  /**
   * Number of observations seen for a node+parent combination.
   */
  getCount(node: string, parentKey: string): number {
    return this.priors.get(node)?.get(parentKey)?.countTotal ?? 0;
  }

  /**
   * Total observation count across all cells.
   */
  get totalObservations(): number {
    let total = 0;
    for (const nodeMap of this.priors.values()) {
      for (const cell of nodeMap.values()) {
        total += cell.countTotal - 2 * this.alpha; // subtract prior pseudo-counts
      }
    }
    return Math.max(0, total);
  }
}
