/**
 * Health Check Module.
 *
 * Provides Kubernetes-compatible liveness/readiness probes
 * and health check aggregation.
 *
 * @packageDocumentation
 */

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime: number;
  version: string;
  checks?: Record<string, HealthCheckResult>;
}

export interface HealthCheckResult {
  status: 'ok' | 'warning' | 'error';
  detail?: string;
  latency?: number;
}

/**
 * Lightweight health checker.
 *
 * Tracks: liveness (process alive), readiness (subsystems initialized).
 * Can be extended with database connectivity checks, queue depth, etc.
 */
export class HealthChecker {
  private _ready = false;
  private _alive = true;
  private checks: Record<string, HealthCheckResult> = {};

  /** Mark the service as ready to accept traffic. */
  markReady(): void {
    this._ready = true;
  }

  /** Mark the service as not ready (e.g., during warm-up or draining). */
  markNotReady(): void {
    this._ready = false;
  }

  /** Set a subsystem check result (e.g., DB connectivity). */
  setCheck(name: string, result: HealthCheckResult): void {
    this.checks[name] = result;
  }

  isReady(): boolean {
    return this._ready;
  }

  isAlive(): boolean {
    return this._alive;
  }

  /**
   * Get aggregated health status.
   */
  getStatus(): HealthStatus {
    const checkValues = Object.values(this.checks);
    const hasError = checkValues.some(c => c.status === 'error');
    const hasWarning = checkValues.some(c => c.status === 'warning');

    return {
      status: hasError ? 'unhealthy' : hasWarning ? 'degraded' : 'healthy',
      uptime: process.uptime() * 1000,
      version: '2.0.0',
      checks: { ...this.checks },
    };
  }
}
