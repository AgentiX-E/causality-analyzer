# @agentix-e/causality-analyzer-visual

> Lit 3 Web Components for causal graph and time series visualization — framework-agnostic, self-describing.

[![npm](https://img.shields.io/badge/version-0.1.0-blue)](https://www.npmjs.com/package/@agentix-e/causality-analyzer-visual)

## Overview

`@agentix-e/causality-analyzer-visual` provides three Web Components for rendering causal analysis results. Built with Lit 3 and HTML Canvas / uPlot, they work in any framework (React, Vue, Svelte, vanilla JS) with zero framework lock-in.

### Components

| Component | Renderer | Purpose |
|-----------|----------|---------|
| `<ca-causal-graph>` | Canvas 2D | DAG/PDAG with BFS layered layout |
| `<ca-time-series>` | uPlot | Time series with anomaly region bands |
| `<ca-root-cause-ranking>` | DOM | Ranked list with propagation paths |

## Installation

```bash
npm install @agentix-e/causality-analyzer-visual
```

## Quick Start

### Causal Graph

```html
<ca-causal-graph></ca-causal-graph>
```

```typescript
import '@agentix-e/causality-analyzer-visual';
import { buildGraphVizData } from '@agentix-e/causality-analyzer-pipeline';

const graphEl = document.querySelector('ca-causal-graph');
const vizData = buildGraphVizData(nodes, edges, rootCauses, anomalousNodes);
graphEl.data = vizData;
```

### Time Series with Anomaly Regions

```html
<ca-time-series></ca-time-series>
```

```typescript
import { buildTimeseriesVizData } from '@agentix-e/causality-analyzer-pipeline';

const chartEl = document.querySelector('ca-time-series');
chartEl.data = buildTimeseriesVizData(metricData, timestamps, anomalousIndices, 'Memory');
// Anomaly regions are automatically rendered as semi-transparent bands
```

### Root Cause Ranking

```html
<ca-root-cause-ranking></ca-root-cause-ranking>
```

```typescript
import { buildRankingVizData } from '@agentix-e/causality-analyzer-pipeline';

const rankingEl = document.querySelector('ca-root-cause-ranking');
rankingEl.data = buildRankingVizData(rootCauses, paths);
```

## Customization

All components use CSS custom properties for theming:

```css
ca-causal-graph {
  --ca-text: #1e293b;
  --ca-anomaly: #dc2626;
  --ca-root-cause: #f59e0b;
  --ca-healthy: #22c55e;
  --ca-edge-weight: #94a3b8;
}
```

## API Reference

### `<ca-causal-graph>`

| Property | Type | Description |
|----------|------|-------------|
| `data` | `GraphVisualizationData` | Nodes + edges for rendering |
| `renderer` | `GraphRenderer` | Pluggable renderer (default: Canvas2D) |

### `<ca-time-series>`

| Property | Type | Description |
|----------|------|-------------|
| `data` | `TimeSeriesChartData` | Series + anomaly regions |

### `<ca-root-cause-ranking>`

| Property | Type | Description |
|----------|------|-------------|
| `data` | `RCARankingData` | Root causes + propagation paths |

## License

MIT
