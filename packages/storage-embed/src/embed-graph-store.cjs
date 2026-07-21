const og = require('overgraph');

class EmbedGraphStore {
  constructor(opts = {}) {
    const path = (opts.path === ':memory:' || !opts.path) ? ':memory:' : (opts.path || './causality-analyzer-graph');
    // Create via Object.create and manually initialize
    const proto = og.OverGraph.prototype;
    const g = Object.create(proto);
    // The issue is that Object.create doesn't properly bind native methods
    // Let's try a different approach
    this.path = path;
  }
}
module.exports = { EmbedGraphStore };
