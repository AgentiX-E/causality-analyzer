/**
 * Merge per-package coverage into a single coverable report directory.
 * Run after `pnpm -r test` — copies each package's coverage/lcov-report
 * into a merged directory under `coverage/{package}/`.
 */

const fs = require('fs');
const path = require('path');

const PKGS = ['core', 'pipeline', 'storage-embed', 'storage-remote', 'visual'];
const OUT_DIR = 'coverage';

fs.mkdirSync(OUT_DIR, { recursive: true });

const merged = {
  total: {
    lines: { pct: 0, total: 0, covered: 0 },
    statements: { pct: 0, total: 0, covered: 0 },
    functions: { pct: 0, total: 0, covered: 0 },
    branches: { pct: 0, total: 0, covered: 0 },
  },
};

for (const pkg of PKGS) {
  const srcPath = path.join('packages', pkg, 'coverage', 'coverage-summary.json');
  if (!fs.existsSync(srcPath)) continue;

  const cov = JSON.parse(fs.readFileSync(srcPath, 'utf8'));
  for (const key of Object.keys(cov.total)) {
    if (merged.total[key]) {
      merged.total[key].total += cov.total[key].total || 0;
      merged.total[key].covered += cov.total[key].covered || 0;
    }
  }

  // Copy lcov-report
  const lcovSrc = path.join('packages', pkg, 'coverage', 'lcov-report');
  const lcovDst = path.join(OUT_DIR, pkg);
  if (fs.existsSync(lcovSrc)) {
    fs.cpSync(lcovSrc, lcovDst, { recursive: true });
  }
}

for (const key of Object.keys(merged.total)) {
  const t = merged.total[key];
  t.pct = t.total > 0 ? Number(((t.covered / t.total) * 100).toFixed(2)) : 0;
}

fs.writeFileSync(path.join(OUT_DIR, 'coverage-summary.json'), JSON.stringify(merged, null, 2));
console.log(`Coverage merged from ${PKGS.length} packages`);
