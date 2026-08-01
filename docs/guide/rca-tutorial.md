# Microservice Root Cause Analysis Tutorial

> Using Causality Analyzer to pinpoint failure root causes in microservice architectures.

---

## Overview

This tutorial walks through a complete root cause analysis (RCA) pipeline using
Causality Analyzer. You will learn how to:

1. Model a microservice topology as a causal graph
2. Inject synthetic faults and generate anomalous metrics
3. Use causal discovery to learn the dependency graph from data
4. Apply RCA algorithms to identify root causes
5. Visualize results and interpret findings

## Prerequisites

```bash
npm install @agentix-e/causality-analyzer-core @agentix-e/causality-analyzer-pipeline
```

## Step 1: Model the Microservice Topology

```typescript
import { CausalGraph } from '@agentix-e/causality-analyzer-pipeline';

// A simplified e-commerce microservice topology
const graph = new CausalGraph([
  'Web',           // Frontend
  'API',           // API Gateway
  'Auth',          // Authentication
  'Catalog',       // Product catalog
  'Cart',          // Shopping cart
  'Order',         // Order processing
  'Payment',       // Payment gateway
  'Inventory',     // Inventory management
  'Shipping',      // Shipping service
  'Database',      // Primary database
]);

// Define service dependencies (A → B means "A calls B")
graph.addEdge('Web', 'API');
graph.addEdge('API', 'Auth');
graph.addEdge('API', 'Catalog');
graph.addEdge('API', 'Cart');
graph.addEdge('Cart', 'Order');
graph.addEdge('Order', 'Payment');
graph.addEdge('Order', 'Inventory');
graph.addEdge('Inventory', 'Shipping');
graph.addEdge('Catalog', 'Database');
graph.addEdge('Cart', 'Database');
graph.addEdge('Order', 'Database');
```

## Step 2: Generate Fault Metrics

```typescript
import { generateLinearData } from '@agentix-e/causality-analyzer-pipeline';

const nPoints = 1000;
const { data, nodeNames } = generateLinearData(graph, nPoints, 42);

// Inject a fault at t=500 — Inventory latency spike
const faultIntensity = 2.5;
for (let t = 500; t < nPoints; t++) {
  const invIdx = nodeNames.indexOf('Inventory');
  data[t][invIdx] *= 1 + faultIntensity * (t - 500) / 500;
}
```

## Step 3: Discover the Causal Graph

```typescript
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

const matrix = new Matrix(data);
const { graph: discovered } = pcAlgorithm(matrix, nodeNames, {
  alpha: 0.05,
  stable: true,
});

console.log(`Discovered ${discovered.edges.length} causal edges`);
```

## Step 4: Run RCA

```typescript
import { HeuristicPathRCA, RandomWalkRCA, HTRCA } from '@agentix-e/causality-analyzer-pipeline';

// Heuristic Path RCA — propagation-based scoring
const hpRCA = new HeuristicPathRCA();
hpRCA.train(discovered, new Set(['Inventory', 'Shipping', 'Order']), matrix);
const hpResult = hpRCA.findRootCauses(['Inventory', 'Shipping', 'Order']);

console.log('HeuristicPath RCA Results:');
for (const rc of hpResult.rootCauses) {
  console.log(`  ${rc.name}: score=${rc.score.toFixed(3)} confidence=${rc.confidence.toFixed(2)}`);
}

// Random Walk RCA — graph traversal
const rwRCA = new RandomWalkRCA();
rwRCA.train(discovered, new Set(['Inventory', 'Shipping', 'Order']), matrix);
const rwResult = rwRCA.findRootCauses(['Inventory', 'Shipping', 'Order']);

// HTRCA — hybrid temporal RCA
const htRCA = new HTRCA();
htRCA.train(discovered, new Set(['Inventory', 'Shipping', 'Order']), matrix);
const htResult = htRCA.findRootCauses(['Inventory', 'Shipping', 'Order']);
```

## Step 5: Evaluate RCA Accuracy

```typescript
// Ground truth: the Inventory service is the root cause
const groundTruth = 'Inventory';

function topKAccuracy(result: any, k: number): boolean {
  return result.rootCauses.slice(0, k).some(
    (rc: any) => rc.name === groundTruth
  );
}

console.log(`HeuristicPath Top-1: ${topKAccuracy(hpResult, 1)}`);
console.log(`RandomWalk Top-1:    ${topKAccuracy(rwResult, 1)}`);
console.log(`HTRCA Top-1:         ${topKAccuracy(htResult, 1)}`);
```

## Step 6: Visualize

```typescript
import { CaCausalGraph, CaRootCauseRanking } from '@agentix-e/causality-analyzer-visual';

// Render causal graph in browser
const graphViz = new CaCausalGraph();
graphViz.graph = discovered;
graphViz.highlightedNodes = ['Inventory']; // highlight root cause
document.body.appendChild(graphViz);

// Render ranking
const rankingViz = new CaRootCauseRanking();
rankingViz.data = hpResult.rootCauses;
document.body.appendChild(rankingViz);
```

## Expected Output

```
HeuristicPath RCA Results:
  Web: score=0.000 confidence=0.00
  API: score=0.000 confidence=0.00
  Inventory: score=0.892 confidence=0.94  ← Root cause!
  Shipping: score=0.653 confidence=0.78   ← Symptom (descendant)
  Order: score=0.723 confidence=0.82      ← Symptom (descendant)

HeuristicPath Top-1: true
RandomWalk Top-1:    true
HTRCA Top-1:         true
```

## Full Pipeline with CIRCA

```typescript
import { CIRCAPipeline } from '@agentix-e/causality-analyzer-pipeline';

const circa = new CIRCAPipeline({
  discoveryMethod: 'pc',
  rcaMethod: 'heuristic',
  streamingEnabled: false,
});

const circaResult = await circa.analyze({
  data: matrix,
  nodeNames,
  anomalyNodes: ['Inventory', 'Shipping', 'Order'],
  timestamp: Date.now(),
});

console.log(`CIRCA identified root cause: ${circaResult.rootCauses[0].name}`);
```

## Online Streaming RCA

For real-time monitoring scenarios, use the online PC algorithm with sliding
window and drift detection:

```typescript
import { OnlinePCDiscovery } from '@agentix-e/causality-analyzer-pipeline';

const online = new OnlinePCDiscovery({
  windowSize: 500,
  alpha: 0.05,
  driftThreshold: 3, // σ threshold for structural drift
});

// Feed streaming data
for (const batch of streamingBatches) {
  const { graph, driftDetected } = online.update(batch);
  if (driftDetected) {
    console.log('Graph structure change detected — triggering RCA...');
    // Re-run RCA with updated graph
  }
}
```

---

## Related Documentation

- [Causal Discovery Reference](/docs/reference/causal-discovery.md)
- [Root Cause Analysis Reference](/docs/reference/root-cause-analysis.md)
- [Anomaly Detection Reference](/docs/reference/anomaly-detection.md)
- [API Cookbook](/docs/reference/api-cookbook.md)
