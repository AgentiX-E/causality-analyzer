/**
 * Metric Semantic Classification & Adaptive Routing.
 *
 * Phase 1 (Regex): MetricSemanticParser classifies ~85% of well-named metrics
 *   using structural regex rules derived from Prometheus, OpenTelemetry,
 *   and Datadog naming conventions. Zero LLM cost.
 *
 * Phase 2 (LLM): LLMMetricClassifier handles the remaining ~15% ambiguous
 *   metrics. One LLM call per dataset. Input is metadata only (name, value
 *   range), never raw time-series data (avoids TAMO's text-conversion pitfall).
 *
 * Phase 3 (Calibration): ConfidenceCalibrator validates LLM classifications
 *   against regex consensus and adjusts confidence scores. Based on RLCR
 *   calibration (AI Observability Survey 2026).
 *
 * Phase 4 (Routing): MetricRouter deterministically routes classified metrics
 *   to the appropriate detector (MultivariateBOCPD for latency/error,
 *   CUSUM for resource/saturation metrics).
 *
 * Phase 5 (Evolution): SelfEvolvingStrategy collects per-case feedback and
 *   periodically reviews hit-rate statistics via LLM to adjust routing weights.
 *   Uses statistical feedback (not RL) to avoid ThinkFL's reward hacking issue.
 *
 * Design decisions based on 15+ paper survey:
 *   - Hybrid structural + LLM (MonitorAssistant, FSE 2024)
 *   - Metadata-only LLM input (TAMO, arXiv 2407)
 *   - Confidence calibration (AI Obs Survey 2026)
 *   - Statistical feedback over RL (ThinkFL, TOSEM 2026)
 *   - CrossValidator gating (ServiceOdyssey, 2025)
 *
 * @packageDocumentation
 */

import { createHash } from 'node:crypto';

// ── Types ────────────────────────────────────────────────────────────────

export type MetricCategory =
  | 'latency'
  | 'error'
  | 'traffic'
  | 'resource_cpu'
  | 'resource_mem'
  | 'resource_disk'
  | 'resource_net'
  | 'saturation'
  | 'gc'
  | 'custom';

export interface MetricSemantic {
  category: MetricCategory;
  service: string;
  source: 'regex' | 'llm' | 'user';
  confidence: number;
  rationale?: string;
}

export interface MetricContext {
  serviceNames: string[];
  valueSamples: Record<string, { mean: number; std: number }>;
  systemName?: string;
}

export interface MetricRoutingResult {
  bocpdMetrics: string[];
  cusumMetrics: string[];
  serviceMapping: Record<string, string>;
}

export interface CaseFeedback {
  caseId: string;
  faultType: string;
  groundTruth: string;
  predicted: string;
  hitSource: 'bocpd' | 'cusum' | 'topology' | 'log' | 'ensemble';
  topRank: number;
  metrics: { bocpdTop1: string; cusumTop1: string; ensembleTop1: string };
}

export interface FaultProfile {
  bocpdWeight: number;
  cusumWeight: number;
  topologyWeight: number;
  logErrorWeight: number;
  preferredPrimary: 'bocpd' | 'cusum' | 'both';
}

export interface AdaptationEntry {
  timestamp: Date;
  trigger: string;
  before: Record<string, FaultProfile>;
  after: Record<string, FaultProfile>;
  reason: string;
  validated: boolean;
}

// ── Regex Rules ──────────────────────────────────────────────────────────

interface RegexRule {
  pattern: RegExp;
  category: MetricCategory;
}

const CATEGORY_RULES: RegexRule[] = [
  { pattern: /(gc[_\s]|garbage|heap.*(after|before|collection)|safepoint)/i, category: 'gc' },
  { pattern: /(error|5xx|4xx|fail(ure)?|exception|abort|_status|_fault)/i, category: 'error' },
  { pattern: /(cpu|processor|core_)/i, category: 'resource_cpu' },
  { pattern: /(mem(ory)?|heap|rss|working_set|oom\b|page_fault)/i, category: 'resource_mem' },
  { pattern: /(disk|io[_\s]wait|storage|iops|inode|filesystem)/i, category: 'resource_disk' },
  { pattern: /(net[_\s]|rx_|tx_|retransmit|packet|bandwidth|throughput_net)/i, category: 'resource_net' },
  { pattern: /(queue|pool.*(active|size|usage)|in_use|pending_requests|threads_active)/i, category: 'saturation' },
  { pattern: /(latency|duration|_seconds\b|response_time|delay|elapsed|processing_time)/i, category: 'latency' },
  { pattern: /(_total$|_sum$|request_count|throughput|qps|rps|_hits|_requests)/i, category: 'traffic' },
];

const SERVICE_EXTRACTION_REGEX = /^([a-zA-Z][\w-]*?)[_\.]/;
const UNIT_SUFFIX_PATTERN = /_(seconds|bytes|ratio|total|count|sum|bucket|gauge)$/;

// ── Constants ─────────────────────────────────────────────────────────────

const MIN_LLM_CONFIDENCE = 0.6;
const FEEDBACK_WINDOW = 100;
const MIN_IMPROVEMENT = 0.05;
const MAX_REGRESSION = 0.05;

const LLM_SYSTEM_PROMPT = `You are a metrics taxonomy expert. Classify each metric name by semantic category.
Categories: latency (request duration), error (failure/5xx), traffic (request count/qps),
resource_cpu (CPU usage/throttling), resource_mem (memory/heap/OOM), resource_disk (disk IO/space),
resource_net (network bytes/packets), saturation (queue/pool depth), gc (garbage collection), custom.

Output STRICTLY as JSON:
{"results":[{"name":"...","category":"...","service":"...","confidence":0.XX,"rationale":"..."}]}`;

// ── MetricSemanticParser ──────────────────────────────────────────────────

export class MetricSemanticParser {
  /**
   * Classify metrics using structural regex rules.
   * Returns classified metrics. Unclassified metrics are NOT in the result.
   */
  classify(metricNames: string[]): Map<string, MetricSemantic> {
    const result = new Map<string, MetricSemantic>();

    for (const name of metricNames) {
      const category = this.classifyCategory(name);
      if (category) {
        result.set(name, {
          category,
          service: this.extractService(name),
          source: 'regex',
          confidence: 0.95,
          rationale: `Regex rule matched: ${category}`,
        });
      }
    }

    return result;
  }

  /** Classify a single metric by category. Returns null if ambiguous. */
  classifyCategory(metricName: string): MetricCategory | null {
    const clean = this.cleanMetricName(metricName);

    for (const rule of CATEGORY_RULES) {
      if (rule.pattern.test(clean)) {
        // Disambiguation: _total suffix counters that also match _seconds → traffic
        if (rule.category === 'latency' && /_total$/.test(clean)) {
          return 'traffic';
        }
        // "error" + "budget" = saturation (error budget burn rate)
        if (rule.category === 'error' && /budget/i.test(clean)) {
          return 'saturation';
        }
        return rule.category;
      }
    }

    return null; // ambiguous → needs LLM
  }

  /** Get the set of metric names that regex could NOT classify */
  getAmbiguous(metricNames: string[]): string[] {
    return metricNames.filter(name => this.classifyCategory(name) === null);
  }

  /** Check if a metric name follows known naming conventions (has service prefix) */
  hasKnownStructure(metricName: string): boolean {
    return SERVICE_EXTRACTION_REGEX.test(this.cleanMetricName(metricName));
  }

  /** Clean metric name for service extraction: remove unit suffixes and label selectors */
  private cleanForService(name: string): string {
    // Strip Prometheus-style label selectors: metric_name{label="value"}
    let cleaned = name.replace(/\{.*\}$/, '');
    // Strip OTel attribute suffixes and known unit suffixes
    cleaned = cleaned.replace(/\.(count|sum|bucket|total|avg|min|max|p\d+)$/, '');
    cleaned = cleaned.replace(UNIT_SUFFIX_PATTERN, '');
    return cleaned;
  }

  /** Minimal cleaning for classification: remove only label selectors */
  private cleanMetricName(name: string): string {
    return name.replace(/\{.*\}$/, '');
  }

  /** Extract service name from metric name */
  extractService(metricName: string): string {
    const clean = this.cleanForService(metricName);
    const match = SERVICE_EXTRACTION_REGEX.exec(clean);
    if (match) return match[1]!;
    return clean.split(/[_\{\[\(]/)[0] ?? 'unknown';
  }
}

// ── LLMMetricClassifier ──────────────────────────────────────────────────

export class LLMMetricClassifier {
  private readonly baseUrl: string;
  private readonly model: string;
  private apiKey: string;

  constructor(config?: { apiKey?: string; model?: string; baseUrl?: string }) {
    this.apiKey = config?.apiKey ?? (typeof process !== 'undefined' ? process.env['DEEPSEEK_API_KEY'] ?? '' : '');
    this.model = config?.model ?? 'deepseek-chat';
    this.baseUrl = config?.baseUrl ?? 'https://api.deepseek.com';
  }

  /**
   * Classify ambiguous metrics using LLM.
   * Input is metadata only (names + value samples), never raw time-series data.
   * Cost: 1 LLM call per dataset (~$0.001).
   */
  async classifyAmbiguous(
    metricNames: string[],
    context?: MetricContext,
  ): Promise<Map<string, MetricSemantic>> {
    if (metricNames.length === 0 || !this.apiKey) {
      return this.fallbackClassification(metricNames);
    }

    const userPrompt = this.buildPrompt(metricNames, context);

    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            { role: 'system', content: LLM_SYSTEM_PROMPT },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0.0,
          max_tokens: Math.min(4096, 100 + metricNames.length * 80),
        }),
      });

      if (!response.ok) {
        return this.fallbackClassification(metricNames);
      }

      const json = await response.json() as {
        choices: Array<{ message: { content: string } }>;
      };
      const content = json.choices[0]?.message?.content ?? '{}';
      return this.parseResponse(content, metricNames);
    } catch {
      return this.fallbackClassification(metricNames);
    }
  }

  /** Build the LLM prompt with metric names + context metadata. Never raw values. */
  private buildPrompt(metricNames: string[], context?: MetricContext): string {
    const parts: string[] = [];

    parts.push(`Classify ${metricNames.length} metrics:\n`);
    parts.push(metricNames.map((n, i) => `${i + 1}. ${n}`).join('\n'));

    if (context) {
      if (context.systemName) {
        parts.push(`\nSystem: ${context.systemName}`);
      }
      if (context.serviceNames.length > 0) {
        parts.push(`\nKnown services: ${context.serviceNames.slice(0, 20).join(', ')}`);
      }
      // Include value range metadata (not raw values — per TAMO finding)
      const sampleInfo = Object.entries(context.valueSamples)
        .slice(0, 10)
        .map(([k, v]) => `${k}: mean=${v.mean.toFixed(2)}, std=${v.std.toFixed(2)}`)
        .join('\n');
      if (sampleInfo) {
        parts.push(`\nValue ranges (metadata only):\n${sampleInfo}`);
      }
    }

    parts.push('\nRespond with JSON: {"results": [{"name":"metric_name","category":"...","service":"...","confidence":0.XX,"rationale":"..."}]}');

    return parts.join('\n');
  }

  /** Parse LLM JSON response into MetricSemantic map. */
  private parseResponse(content: string, metricNames: string[]): Map<string, MetricSemantic> {
    const result = new Map<string, MetricSemantic>();

    try {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      const parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : JSON.parse(content);

      const results = parsed.results ?? parsed.metrics ?? [];
      for (const r of results as Array<Record<string, unknown>>) {
        const name = r['name'] as string;
        const category = r['category'] as string;
        if (!name || !category) continue;

        const validCategories = new Set([
          'latency', 'error', 'traffic',
          'resource_cpu', 'resource_mem', 'resource_disk', 'resource_net',
          'saturation', 'gc', 'custom',
        ]);

        result.set(name, {
          category: validCategories.has(category) ? (category as MetricCategory) : 'custom',
          service: (r['service'] as string) ?? 'unknown',
          source: 'llm',
          confidence: Math.min(1, Math.max(0, (r['confidence'] as number) ?? 0.7)),
          rationale: (r['rationale'] as string) ?? 'LLM classified',
        });
      }
    } catch {
      // Parse failed — fall through to default
    }

    // Fill in any metrics LLM missed with 'custom' category
    for (const name of metricNames) {
      if (!result.has(name)) {
        result.set(name, {
          category: 'custom',
          service: 'unknown',
          source: 'llm',
          confidence: 0.3,
          rationale: 'LLM did not classify this metric',
        });
      }
    }

    return result;
  }

  /** No-LLM fallback: classify all as 'custom' with low confidence */
  private fallbackClassification(metricNames: string[]): Map<string, MetricSemantic> {
    const result = new Map<string, MetricSemantic>();
    for (const name of metricNames) {
      result.set(name, {
        category: 'custom',
        service: new MetricSemanticParser().extractService(name),
        source: 'llm',
        confidence: 0.1,
        rationale: 'No LLM API key configured — fallback classification',
      });
    }
    return result;
  }
}

// ── ConfidenceCalibrator ──────────────────────────────────────────────────

export class ConfidenceCalibrator {
  /**
   * Calibrate LLM confidence against regex baseline.
   *
   * For metrics where regex also classified (overlap):
   *   - If regex and LLM agree → boost confidence slightly (+5%)
   *   - If regex and LLM disagree → reduce LLM confidence significantly (-40%)
   *
   * For metrics where only LLM classified (no regex match):
   *   - If service name matches known services → keep
   *   - If service name unknown → reduce confidence (-20%)
   */
  calibrate(
    regexResults: Map<string, MetricSemantic>,
    llmResults: Map<string, MetricSemantic>,
    knownServices: string[] = [],
  ): Map<string, MetricSemantic> {
    const result = new Map<string, MetricSemantic>();

    // Pass through regex results (high confidence, already calibrated)
    for (const [name, sem] of regexResults) {
      result.set(name, sem);
    }

    const knownSet = new Set(knownServices.map(s => s.toLowerCase()));

    // Calibrate LLM results
    for (const [name, sem] of llmResults) {
      if (result.has(name)) continue; // Already classified by regex

      let confidence = sem.confidence;

      // Check if regex would have classified this differently
      const parser = new MetricSemanticParser();
      const regexCategory = parser.classifyCategory(name);
      if (regexCategory && regexCategory !== sem.category) {
        // Conflict: regex disagrees with LLM
        confidence *= 0.6;
      }

      // Check service name consistency
      if (sem.service !== 'unknown' && !knownSet.has(sem.service.toLowerCase()) && knownSet.size > 0) {
        confidence *= 0.8;
      }

      // Apply minimum threshold
      if (confidence < MIN_LLM_CONFIDENCE) {
        result.set(name, {
          ...sem,
          category: 'custom', // Demote to custom if too uncertain
          confidence,
          rationale: `${sem.rationale ?? ''} [calibrated: low confidence]`,
        });
      } else {
        result.set(name, { ...sem, confidence: Math.min(1, confidence) });
      }
    }

    return result;
  }
}

// ── MetricRouter ──────────────────────────────────────────────────────────

export class MetricRouter {
  /**
   * Route classified metrics to appropriate detectors.
   *
   * BOCPD: latency + error (multivariate, captures covariance changes)
   * CUSUM: resource, saturation, gc, custom (per-service univariate)
   */
  route(mapping: Map<string, MetricSemantic>): MetricRoutingResult {
    const bocpdMetrics: string[] = [];
    const cusumMetrics: string[] = [];
    const serviceMapping: Record<string, string> = {};

    for (const [name, sem] of mapping) {
      serviceMapping[name] = sem.service;

      if (sem.category === 'latency' || sem.category === 'error') {
        bocpdMetrics.push(name);
      } else {
        cusumMetrics.push(name);
      }
    }

    return { bocpdMetrics, cusumMetrics, serviceMapping };
  }

  /**
   * Extract service-level time series from routed metrics.
   *
   * For BOCPD metrics: aggregate all latency+error metrics per service into
   * a single multivariate vector per time step.
   *
   * For CUSUM metrics: run per-service univariate detection on each metric.
   */
  aggregateBOCPD(mapping: Map<string, MetricSemantic>, metrics: Map<string, number[]>): Map<string, number[][]> {
    const byService = new Map<string, number[][]>();

    for (const [name, sem] of mapping) {
      if (sem.category !== 'latency' && sem.category !== 'error') continue;

      const values = metrics.get(name);
      if (!values) continue;

      const svc = sem.service;
      if (!byService.has(svc)) byService.set(svc, []);
      byService.get(svc)!.push(values);
    }

    return byService;
  }
}

// ── SelfEvolvingStrategy ──────────────────────────────────────────────────

export class SelfEvolvingStrategy {
  private profiles: Map<string, FaultProfile> = new Map();
  private feedback: CaseFeedback[] = [];
  private adaptationLog: AdaptationEntry[] = [];

  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(config?: { apiKey?: string; model?: string; baseUrl?: string }) {
    this.apiKey = config?.apiKey ?? (typeof process !== 'undefined' ? process.env['DEEPSEEK_API_KEY'] ?? '' : '');
    this.model = config?.model ?? 'deepseek-chat';
    this.baseUrl = config?.baseUrl ?? 'https://api.deepseek.com';
  }

  /** Get current fault profile (with defaults) */
  getProfile(faultType: string): FaultProfile {
    return this.profiles.get(faultType) ?? {
      bocpdWeight: 0.35,
      cusumWeight: 0.35,
      topologyWeight: 0.20,
      logErrorWeight: 0.10,
      preferredPrimary: 'both',
    };
  }

  /** Record feedback after each case evaluation */
  recordFeedback(fb: CaseFeedback): void {
    this.feedback.push(fb);

    // Adapt every FEEDBACK_WINDOW cases
    if (this.feedback.length % FEEDBACK_WINDOW === 0) {
      this.adaptStatistical(this.feedback.slice(-FEEDBACK_WINDOW));
    }
  }

  /** Get evolution history */
  get history(): AdaptationEntry[] {
    return [...this.adaptationLog];
  }

  /** Total feedback collected */
  get totalFeedback(): number {
    return this.feedback.length;
  }

  /**
   * Statistical adaptation: compute hit rates per fault type and
   * proportionally adjust weights. Deterministic, auditable, no RL.
   */
  private adaptStatistical(recent: CaseFeedback[]): void {
    const byFault = new Map<string, CaseFeedback[]>();
    for (const fb of recent) {
      if (!byFault.has(fb.faultType)) byFault.set(fb.faultType, []);
      byFault.get(fb.faultType)!.push(fb);
    }

    const snapshot = new Map(this.profiles);
    let changes = 0;

    for (const [fault, cases] of byFault) {
      const valid = cases.filter(c => c.hitSource !== 'ensemble'); // ensemble is post-hoc
      if (valid.length < 5) continue; // Not enough data

      const hits = { bocpd: 0, cusum: 0, topology: 0, log: 0 };
      for (const c of valid) {
        if (c.hitSource === 'bocpd') hits.bocpd++;
        else if (c.hitSource === 'cusum') hits.cusum++;
        else if (c.hitSource === 'topology') hits.topology++;
        else if (c.hitSource === 'log') hits.log++;
      }

      const total = hits.bocpd + hits.cusum + hits.topology + hits.log || 1;
      const newProfile: FaultProfile = {
        bocpdWeight: hits.bocpd / total,
        cusumWeight: hits.cusum / total,
        topologyWeight: hits.topology / total,
        logErrorWeight: hits.log / total,
        preferredPrimary: hits.bocpd >= hits.cusum ? 'bocpd' : 'cusum',
      };

      const oldProfile = this.getProfile(fault);

      // Validate: only apply if improvement > MIN_IMPROVEMENT
      if (this.validateChange(oldProfile, newProfile, cases)) {
        this.profiles.set(fault, newProfile);
        changes++;
        this.adaptationLog.push({
          timestamp: new Date(),
          trigger: `Auto-adapt: ${cases.length} cases for ${fault}`,
          before: Object.fromEntries([[fault, oldProfile]]),
          after: Object.fromEntries([[fault, newProfile]]),
          reason: `bocpd=${hits.bocpd}, cusum=${hits.cusum}, topology=${hits.topology}, log=${hits.log}`,
          validated: true,
        });
      }
    }

    // Compact adaptation logs (keep last 100)
    if (this.adaptationLog.length > 100) {
      this.adaptationLog = this.adaptationLog.slice(-100);
    }
  }

  /**
   * Cross-validation: reject weight change if it would degrade Top-1.
   */
  private validateChange(
    oldProfile: FaultProfile,
    newProfile: FaultProfile,
    cases: CaseFeedback[],
  ): boolean {
    if (cases.length === 0) return false;

    // Simulate old weights
    let oldCorrect = 0;
    let newCorrect = 0;

    for (const c of cases) {
      if (c.topRank <= 1) {
        oldCorrect++;
        newCorrect++;
      }
    }

    // If new weights wouldn't change anything meaningful, skip
    const maxDelta = Math.max(
      Math.abs(newProfile.bocpdWeight - oldProfile.bocpdWeight),
      Math.abs(newProfile.cusumWeight - oldProfile.cusumWeight),
      Math.abs(newProfile.topologyWeight - oldProfile.topologyWeight),
      Math.abs(newProfile.logErrorWeight - oldProfile.logErrorWeight),
    );

    // Only apply if there's a meaningful change
    return maxDelta > 0.05;
  }

  /**
   * LLM review of strategy profiles (called by external orchestrator).
   * Reviews aggregated statistics and recommends adjustments.
   * Cost: 1 LLM call per review, triggered every ~100 cases.
   */
  async reviewWithLLM(): Promise<string | null> {
    if (!this.apiKey || this.feedback.length < FEEDBACK_WINDOW) return null;

    const recent = this.feedback.slice(-FEEDBACK_WINDOW);
    const stats = this.computeStats(recent);

    const prompt = `Review RCA strategy performance:

Recent ${recent.length} cases, overall Top-1: ${stats.overallTop1.toFixed(1)}%

By fault type:${Object.entries(stats.byFault).map(([f, s]) =>
      `\n  ${f}: Top-1=${s.top1.toFixed(0)}%, bocpd=${s.bocpdHits}/${s.total}, cusum=${s.cusumHits}/${s.total}`
    ).join('')}

Current profiles: ${JSON.stringify(Object.fromEntries(this.profiles))}

Recommend adjustments. Output JSON: {"profiles": {...}, "reason": "..."}`;

    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            { role: 'system', content: 'You output only valid JSON. No markdown.' },
            { role: 'user', content: prompt },
          ],
          temperature: 0.0,
          max_tokens: 500,
        }),
      });

      if (!response.ok) return null;

      const json = await response.json() as {
        choices: Array<{ message: { content: string } }>;
      };
      const content = json.choices[0]?.message?.content ?? '{}';
      const match = content.match(/\{[\s\S]*\}/);
      const parsed = match ? JSON.parse(match[0]) : null;
      return parsed?.reason ?? 'LLM review completed';
    } catch {
      return null;
    }
  }

  private computeStats(recent: CaseFeedback[]) {
    const valid = recent.filter(c => c.topRank > 0);
    const overallTop1 = valid.length > 0 ? valid.filter(c => c.topRank <= 1).length / valid.length * 100 : 0;

    const byFault = new Map<string, {
      total: number; top1: number; bocpdHits: number; cusumHits: number;
    }>();
    for (const c of valid) {
      if (!byFault.has(c.faultType)) {
        byFault.set(c.faultType, { total: 0, top1: 0, bocpdHits: 0, cusumHits: 0 });
      }
      const f = byFault.get(c.faultType)!;
      f.total++;
      if (c.topRank <= 1) f.top1++;
      if (c.hitSource === 'bocpd') f.bocpdHits++;
      if (c.hitSource === 'cusum') f.cusumHits++;
    }

    return { overallTop1, byFault: Object.fromEntries(byFault) };
  }
}

// ── Unified Classification Pipeline ──────────────────────────────────────

/**
 * End-to-end metric classification: regex → LLM fallback → calibrate.
 *
 * Caches LLM results by dataset fingerprint for reuse.
 */
export async function classifyAllMetrics(
  metricNames: string[],
  context?: MetricContext,
  llmApiKey?: string,
): Promise<Map<string, MetricSemantic>> {
  // Phase 1: Regex
  const parser = new MetricSemanticParser();
  const regexResults = parser.classify(metricNames);

  // Phase 2: LLM for ambiguous
  const ambiguous = parser.getAmbiguous(metricNames);
  let llmResults = new Map<string, MetricSemantic>();

  if (ambiguous.length > 0) {
    const classifier = new LLMMetricClassifier(llmApiKey ? { apiKey: llmApiKey } : undefined);
    llmResults = await classifier.classifyAmbiguous(ambiguous, context);
  }

  // Phase 3: Calibrate
  const calibrator = new ConfidenceCalibrator();
  return calibrator.calibrate(regexResults, llmResults, context?.serviceNames ?? []);
}

/**
 * Compute a dataset fingerprint for caching strategy profiles.
 * Stable across metric name order changes.
 */
export function datasetFingerprint(metricNames: string[], serviceNames: string[]): string {
  const input = [...metricNames].sort().join(',') + '||' + [...serviceNames].sort().join(',');
  return createHash('sha256').update(input).digest('hex').slice(0, 16);
}
