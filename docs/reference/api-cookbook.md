# Causality Analyzer — API Cookbook

## Causal Discovery

### PC: Constraint-Based CPDAG
```ts
import { pcAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
import { Matrix } from 'ml-matrix';

const data = new Matrix(100, 4);
const { graph, sepSet } = pcAlgorithm(data, ['X','Y','Z','W'], { alpha: 0.05, stable: true });
console.log('Edges:', graph.edges.map(e => `${e.source}→${e.target}`));
```

### BOSS: Best Order Score Search (2023)
```ts
import { bossAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
const dag = bossAlgorithm(data, ['A','B','C'], { numStarts: 5, seed: 42 });
console.log('SHD:', computeSHD(dag, truth).shd);
```

### GFCI: Hybrid PAG for Latents
```ts
import { gfciAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
const { graph, pagEdges } = gfciAlgorithm(data, ['X','Y','C'], { alpha: 0.01 });
for (const [pair, type] of pagEdges) console.log(pair, type);
```

### DAGMA: Log-Det Optimization
```ts
import { dagmaAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
const arr = [[1,2],[3,4],[5,6]];
const { graph, h } = dagmaAlgorithm(arr, ['X','Y']);
```

### FASK: Skewness-Based
```ts
import { faskAlgorithm } from '@agentix-e/causality-analyzer-pipeline';
const { graph, orientationConfidence } = faskAlgorithm(data, ['A','B'], { skewThreshold: 0.1 });
```

## Effect Estimation

```ts
import { findBackdoorAdjustmentSet, adjustBackdoor } from '@agentix-e/causality-analyzer-pipeline';
const adj = findBackdoorAdjustmentSet(graph, 'T', 'Y', 'efficient');
const { ate, se } = adjustBackdoor(graph, 'T', 'Y', data, nodeIndex);
console.log(`ATE = ${ate.toFixed(3)} ± ${(1.96 * se).toFixed(3)}`);
```

## Sensitivity

```ts
import { eValueSensitivity, refuteRandomCommonCause } from '@agentix-e/causality-analyzer-pipeline';
const ev = eValueSensitivity(ate, outcomeStd);
const refute = refuteRandomCommonCause(data, treatIdx, outcomeIdx, estimator, { numSimulations: 100 });
```

## Streaming Discovery

```ts
import { OnlinePC } from '@agentix-e/causality-analyzer-pipeline';
const online = new OnlinePC(['cpu','mem','latency'], { windowSize: 500, minBatchSize: 50 });
for await (const batch of dataStream) online.update(batch);
const graph = online.getGraph();
```

## Stability Selection

```ts
import { stabilitySelection, starsSelection } from '@agentix-e/causality-analyzer-pipeline';
const stable = stabilitySelection(data, nodes, pcWrapper, { nSubsamples: 50, edgeThreshold: 0.7 });
const stars = starsSelection(data, nodes, paramFn, [0.01, 0.05, 0.1], { nSubsamples: 20 });
```

## HTTP API

```bash
curl -X POST http://localhost:3000/discover -H 'Content-Type: application/json' \
  -d '{"data":[[1,2],[3,4]],"nodeNames":["X","Y"]}'

curl -X POST http://localhost:3000/estimate -H 'Content-Type: application/json' \
  -d '{"graph":{"nodes":["X","Y","C"],"edges":[{"source":"C","target":"X"},{"source":"C","target":"Y"}]},"treatment":"X","outcome":"Y","data":[[...]]}'
```

## Uplift Modeling

```ts
import { evaluateUplift, upliftAtK } from '@agentix-e/causality-analyzer-pipeline';
const eval = evaluateUplift(observations);
console.log('AUUC:', eval.auuc, 'Uplift@10%:', eval.upliftAt10);
```
