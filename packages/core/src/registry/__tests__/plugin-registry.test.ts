/**
 * Unit tests for PluginRegistry and decorators.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  PluginRegistry,
  PluginCategory,
  RegisterDetector,
  RegisterGraph,
  RegisterAnalyzer,
} from '../../index.js';

// Helper: stub classes for testing registration
class FakeDetector {}
class FakeGraph {}
class FakeAnalyzer {}

describe('PluginRegistry', () => {
  beforeEach(() => {
    PluginRegistry.clear();
  });

  // ── Registration ──────────────────────────────────────────────────
  describe('registerDetector', () => {
    it('should register a detector constructor', () => {
      PluginRegistry.registerDetector('test_det', FakeDetector);
      expect(PluginRegistry.hasDetector('test_det')).toBe(true);
    });

    it('should throw on duplicate registration', () => {
      PluginRegistry.registerDetector('dup', FakeDetector);
      expect(() => PluginRegistry.registerDetector('dup', FakeDetector)).toThrow(
        /already registered/,
      );
    });
  });

  describe('registerGraph', () => {
    it('should register a graph builder constructor', () => {
      PluginRegistry.registerGraph('pc', FakeGraph);
      expect(PluginRegistry.hasGraph('pc')).toBe(true);
    });

    it('should throw on duplicate registration', () => {
      PluginRegistry.registerGraph('pc', FakeGraph);
      expect(() => PluginRegistry.registerGraph('pc', FakeGraph)).toThrow(
        /already registered/,
      );
    });
  });

  describe('registerAnalyzer', () => {
    it('should register an analyzer constructor', () => {
      PluginRegistry.registerAnalyzer('bayesian', FakeAnalyzer);
      expect(PluginRegistry.hasAnalyzer('bayesian')).toBe(true);
    });

    it('should throw on duplicate registration', () => {
      PluginRegistry.registerAnalyzer('bayesian', FakeAnalyzer);
      expect(() =>
        PluginRegistry.registerAnalyzer('bayesian', FakeAnalyzer),
      ).toThrow(/already registered/);
    });
  });

  // ── Listing ───────────────────────────────────────────────────────
  describe('listDetectors', () => {
    it('should return all registered detector names', () => {
      PluginRegistry.registerDetector('spot', FakeDetector);
      PluginRegistry.registerDetector('sr', FakeDetector);
      expect(PluginRegistry.listDetectors()).toEqual(
        expect.arrayContaining(['spot', 'sr']),
      );
    });

    it('should return empty array when nothing registered', () => {
      expect(PluginRegistry.listDetectors()).toEqual([]);
    });
  });

  describe('listGraphs', () => {
    it('should return registered graph names', () => {
      PluginRegistry.registerGraph('pc', FakeGraph);
      PluginRegistry.registerGraph('ges', FakeGraph);
      expect(PluginRegistry.listGraphs()).toEqual(
        expect.arrayContaining(['pc', 'ges']),
      );
    });
  });

  describe('listAnalyzers', () => {
    it('should return registered analyzer names', () => {
      PluginRegistry.registerAnalyzer('bn', FakeAnalyzer);
      expect(PluginRegistry.listAnalyzers()).toEqual(['bn']);
    });
  });

  // ── Unregistration ────────────────────────────────────────────────
  describe('unregister', () => {
    it('should unregister a detector', () => {
      PluginRegistry.registerDetector('tmp', FakeDetector);
      const result = PluginRegistry.unregister(PluginCategory.DETECTOR, 'tmp');
      expect(result).toBe(true);
      expect(PluginRegistry.hasDetector('tmp')).toBe(false);
    });

    it('should unregister a graph builder', () => {
      PluginRegistry.registerGraph('tmp', FakeGraph);
      PluginRegistry.unregister(PluginCategory.GRAPH, 'tmp');
      expect(PluginRegistry.hasGraph('tmp')).toBe(false);
    });

    it('should unregister an analyzer', () => {
      PluginRegistry.registerAnalyzer('tmp', FakeAnalyzer);
      PluginRegistry.unregister(PluginCategory.ANALYZER, 'tmp');
      expect(PluginRegistry.hasAnalyzer('tmp')).toBe(false);
    });

    it('should return false when unregistering non-existent plugin', () => {
      const result = PluginRegistry.unregister(
        PluginCategory.DETECTOR,
        'nonexistent',
      );
      expect(result).toBe(false);
    });
  });

  // ── Clear ─────────────────────────────────────────────────────────
  describe('clear', () => {
    it('should remove all registrations', () => {
      PluginRegistry.registerDetector('d1', FakeDetector);
      PluginRegistry.registerGraph('g1', FakeGraph);
      PluginRegistry.registerAnalyzer('a1', FakeAnalyzer);
      PluginRegistry.clear();
      expect(PluginRegistry.listDetectors()).toEqual([]);
      expect(PluginRegistry.listGraphs()).toEqual([]);
      expect(PluginRegistry.listAnalyzers()).toEqual([]);
    });
  });

  // ── Decorators ───────────────────────────────────────────────────
  describe('@RegisterDetector', () => {
    it('should register decorated class automatically', () => {
      @RegisterDetector('my_detector')
      class MyDetector {}
      expect(PluginRegistry.hasDetector('my_detector')).toBe(true);
    });
  });

  describe('@RegisterGraph', () => {
    it('should register decorated class automatically', () => {
      @RegisterGraph('my_graph')
      class MyGraph {}
      expect(PluginRegistry.hasGraph('my_graph')).toBe(true);
    });
  });

  describe('@RegisterAnalyzer', () => {
    it('should register decorated class automatically', () => {
      @RegisterAnalyzer('my_analyzer')
      class MyAnalyzer {}
      expect(PluginRegistry.hasAnalyzer('my_analyzer')).toBe(true);
    });
  });

  describe('decorator conflicts', () => {
    it('should allow same name across different categories', () => {
      @RegisterDetector('shared_name')
      class D {}
      @RegisterGraph('shared_name')
      class G {}
      expect(PluginRegistry.hasDetector('shared_name')).toBe(true);
      expect(PluginRegistry.hasGraph('shared_name')).toBe(true);
    });
  });
});
