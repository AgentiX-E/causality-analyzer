/**
 * PluginRegistry — decorator-based plugin discovery and factory.
 *
 * Enables declarative registration of detectors, graph builders,
 * and RCA analyzers. Plugins are auto-discovered at import time
 * via TypeScript decorators — no manual wiring required.
 *
 * @packageDocumentation
 */

/** Registry key type — maps a plugin name to its constructor */
type RegistryEntry<T> = Map<string, new (...args: never[]) => T>;

/** Supported plugin categories */
export enum PluginCategory {
  DETECTOR = 'detector',
  GRAPH = 'graph',
  ANALYZER = 'analyzer',
}

/**
 * Central plugin registry.
 *
 * Plugins are registered via the @RegisterDetector, @RegisterGraph,
 * and @RegisterAnalyzer decorators. The registry is a static
 * singleton — all decorators write to the same shared registry.
 */
export class PluginRegistry {
  private static detectors: RegistryEntry<unknown> = new Map();
  private static graphs: RegistryEntry<unknown> = new Map();
  private static analyzers: RegistryEntry<unknown> = new Map();

  /** Register a detector constructor */
  static registerDetector(name: string, ctor: new (...args: never[]) => unknown): void {
    if (PluginRegistry.detectors.has(name)) {
      throw new Error(`Detector "${name}" is already registered`);
    }
    PluginRegistry.detectors.set(name, ctor);
  }

  /** Register a graph builder constructor */
  static registerGraph(name: string, ctor: new (...args: never[]) => unknown): void {
    if (PluginRegistry.graphs.has(name)) {
      throw new Error(`Graph builder "${name}" is already registered`);
    }
    PluginRegistry.graphs.set(name, ctor);
  }

  /** Register an analyzer constructor */
  static registerAnalyzer(name: string, ctor: new (...args: never[]) => unknown): void {
    if (PluginRegistry.analyzers.has(name)) {
      throw new Error(`Analyzer "${name}" is already registered`);
    }
    PluginRegistry.analyzers.set(name, ctor);
  }

  /** Unregister a plugin by category and name */
  static unregister(category: PluginCategory, name: string): boolean {
    switch (category) {
      case PluginCategory.DETECTOR:
        return PluginRegistry.detectors.delete(name);
      case PluginCategory.GRAPH:
        return PluginRegistry.graphs.delete(name);
      case PluginCategory.ANALYZER:
        return PluginRegistry.analyzers.delete(name);
      default:
        return false;
    }
  }

  /** List all registered detectors */
  static listDetectors(): string[] {
    return Array.from(PluginRegistry.detectors.keys());
  }

  /** List all registered graph builders */
  static listGraphs(): string[] {
    return Array.from(PluginRegistry.graphs.keys());
  }

  /** List all registered analyzers */
  static listAnalyzers(): string[] {
    return Array.from(PluginRegistry.analyzers.keys());
  }

  /** Check if a detector is registered */
  static hasDetector(name: string): boolean {
    return PluginRegistry.detectors.has(name);
  }

  /** Check if a graph builder is registered */
  static hasGraph(name: string): boolean {
    return PluginRegistry.graphs.has(name);
  }

  /** Check if an analyzer is registered */
  static hasAnalyzer(name: string): boolean {
    return PluginRegistry.analyzers.has(name);
  }

  /** Clear all registrations (primarily for testing) */
  static clear(): void {
    PluginRegistry.detectors.clear();
    PluginRegistry.graphs.clear();
    PluginRegistry.analyzers.clear();
  }
}

// ── Decorators ───────────────────────────────────────────────────────────

/**
 * Decorator: register a class as an anomaly detector.
 *
 * @example
 * ```typescript
 * @RegisterDetector('spot')
 * class SPOTDetector extends BaseDetector { ... }
 * ```
 */
export function RegisterDetector(name: string) {
  return function <T extends new (...args: never[]) => unknown>(target: T): T {
    PluginRegistry.registerDetector(name, target);
    return target;
  };
}

/**
 * Decorator: register a class as a causal graph builder.
 *
 * @example
 * ```typescript
 * @RegisterGraph('pc')
 * class PCBuilder extends BaseGraphBuilder { ... }
 * ```
 */
export function RegisterGraph(name: string) {
  return function <T extends new (...args: never[]) => unknown>(target: T): T {
    PluginRegistry.registerGraph(name, target);
    return target;
  };
}

/**
 * Decorator: register a class as an RCA analyzer.
 *
 * @example
 * ```typescript
 * @RegisterAnalyzer('bayesian')
 * class BayesianAnalyzer extends BaseAnalyzer { ... }
 * ```
 */
export function RegisterAnalyzer(name: string) {
  return function <T extends new (...args: never[]) => unknown>(target: T): T {
    PluginRegistry.registerAnalyzer(name, target);
    return target;
  };
}
