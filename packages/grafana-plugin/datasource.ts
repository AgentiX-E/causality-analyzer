/**
 * Causality Analyzer Grafana Data Source.
 *
 * Bridges Grafana dashboards to the Causality Analyzer engine.
 * Queries proxy through the Grafana backend to the engine REST API.
 *
 * Query types:
 *   - discover: causal graph discovery from metric data
 *   - analyze: root cause analysis for anomalous SLIs
 *   - detect: streaming anomaly detection on time series
 */
export class CausalityDataSource {
  constructor(private instanceSettings: any, private backendSrv: any) {}

  async query(options: any) {
    const targets = options.targets.filter((t: any) => !t.hide);
    if (targets.length === 0) return { data: [] };

    const result = await this.backendSrv.datasourceRequest({
      url: `${this.instanceSettings.url}/discover`,
      method: 'POST',
      data: {
        data: options.range,
        nodeNames: targets.map((t: any) => t.target),
      },
    });

    return { data: result.data.edges ?? [] };
  }

  async testDatasource() {
    try {
      const result = await this.backendSrv.datasourceRequest({
        url: `${this.instanceSettings.url}/health`,
        method: 'GET',
      });
      return { status: 'success', message: `Engine v${result.data.version} connected` };
    } catch (e: any) {
      return { status: 'error', message: e.message };
    }
  }

  async annotationQuery(options: any) {
    const result = await this.backendSrv.datasourceRequest({
      url: `${this.instanceSettings.url}/detect`,
      method: 'POST',
      data: { stream: options.annotation.query?.stream, method: 'bsts' },
    });
    return (result.data.anomalies as boolean[])
      .map((isAnom: boolean, i: number) => isAnom ? {
        annotation: { name: 'Anomaly' },
        time: options.range.from + i * 60000,
        title: 'Anomaly detected',
        tags: ['causality', 'anomaly'],
      } : null)
      .filter(Boolean);
  }
}
