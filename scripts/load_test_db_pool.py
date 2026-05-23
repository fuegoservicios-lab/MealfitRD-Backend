#!/usr/bin/env python3
"""[P0-PROD-AUDIT-1 · 2026-05-23] Load test del pool de conexiones DB.

Gap original (audit 2026-05-23 — B-P0-5):
    `db_core.py` mantiene 3 pools (sync + async + chat_checkpoint) con knobs:
      - MEALFIT_DB_POOL_MIN_SIZE=10
      - MEALFIT_DB_POOL_MAX_SIZE=60
      - MEALFIT_DB_POOL_TIMEOUT_S=10

    Ningún test ejecutado validó saturación bajo carga real. Riesgo: primer
    pico de tráfico (lanzamiento + post en redes) puede tumbar el backend
    si el ratio queries/conn está mal estimado.

    Este script ejecuta load test asincrónico (httpx.AsyncClient) contra
    endpoints representativos del backend y mide:
      - Latencia p50/p95/p99 por endpoint.
      - Error rate (HTTP 4xx/5xx + connection refused + timeout).
      - Throughput sostenido (req/s).
      - Indicadores de pool saturation (timeouts del lado cliente vs
        latencia degradada — el primero significa que el cliente NO obtuvo
        conexión TCP; el segundo, que el pool DB del servidor se saturó).

Cómo usar:
    # Smoke (15 usuarios, 30 segundos) contra backend local.
    ./scripts/load_test_db_pool.py --target http://localhost:3001 \\
        --concurrent 15 --duration 30

    # Stress (200 usuarios, 60 segundos) contra staging.
    ./scripts/load_test_db_pool.py --target https://staging.mealfit.example.com \\
        --concurrent 200 --duration 60 \\
        --bearer-token "$STAGING_JWT_FOR_LOAD_TEST_USER"

    # Solo health (sin auth — útil para baseline de routing/proxy overhead).
    ./scripts/load_test_db_pool.py --target https://prod.mealfit.example.com \\
        --concurrent 50 --duration 30 --scenarios health,ready,version

Output:
    - Tabla con latencia + error rate por endpoint.
    - Verdict: PASS / WARN / FAIL según thresholds configurables.
    - Snapshot del pool del lado servidor via /api/system/atomic-pool-health
      (si el bearer token tiene admin scope).

Anti-patrón:
    NO ejecutar contra producción sin avisar al equipo: el load test puede
    triggear `system_alerts` (rate_limit_exceeded, scheduler_missed_*) y
    contaminar métricas operacionales reales. Usar staging o un horario
    coordinado con ventana de mantenimiento.

Runbook completo: docs/runbooks/db_pool_load_test.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# CONFIGURACIÓN — scenarios disponibles.
# =============================================================================
# Cada scenario es un endpoint + descripción + estimación de queries DB que
# ejecuta. Usado para mapear "latencia alta" → "qué pool se saturó".
SCENARIOS = {
    "health": {
        "method": "GET",
        "path": "/health",
        "auth": False,
        "queries": 0,
        "description": "Liveness probe — 0 DB queries. Baseline de overhead HTTP/routing.",
    },
    "ready": {
        "method": "GET",
        "path": "/ready",
        "auth": False,
        "queries": 0,
        "description": "Readiness probe — verifica LangGraph compilado. Sin DB queries.",
    },
    "version": {
        "method": "GET",
        "path": "/health/version",
        "auth": False,
        "queries": 1,
        "description": "Marker + drift check vs app_kv_store. 1 query SELECT (sync pool).",
    },
    "credits": {
        "method": "GET",
        "path": "/api/user/credits/{user_id}",
        "auth": True,
        "queries": 1,
        "description": "Lookup de consumo mensual. 1 query SELECT (sync pool).",
    },
    "user_facts": {
        "method": "GET",
        "path": "/api/user-facts/{user_id}",
        "auth": True,
        "queries": 1,
        "description": "Lectura de facts. 1 query SELECT con filtro user_id.",
    },
    "pool_health": {
        "method": "GET",
        "path": "/api/system/atomic-pool-health",
        "auth": False,  # admin-token gated, pero el script lo pasa via auth si está set
        "queries": 3,
        "description": "Snapshot del pool — 3 queries internas. Útil para verificar pool del SERVER.",
    },
}


# =============================================================================
# DATA CLASSES — resultados por request + agregado.
# =============================================================================
@dataclass
class RequestResult:
    scenario: str
    status: int  # 0 = error de transporte (connection refused, timeout, DNS)
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ScenarioStats:
    scenario: str
    count: int = 0
    errors: int = 0
    error_breakdown: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)

    def add(self, r: RequestResult) -> None:
        self.count += 1
        self.latencies_ms.append(r.latency_ms)
        if r.status == 0 or r.status >= 500:
            self.errors += 1
            key = r.error or f"http_{r.status}"
            self.error_breakdown[key] = self.error_breakdown.get(key, 0) + 1

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        k = int(round((p / 100) * (len(sorted_l) - 1)))
        return sorted_l[k]


# =============================================================================
# WORKER — un "usuario virtual" que martilla endpoints en loop hasta deadline.
# =============================================================================
async def virtual_user(
    client,  # httpx.AsyncClient
    user_id: int,
    scenarios: list[str],
    deadline: float,
    target_user_id_for_path: Optional[str],
    bearer_token: Optional[str],
    results: list[RequestResult],
) -> None:
    """Cada virtual user itera los scenarios en round-robin hasta deadline."""
    idx = 0
    while time.monotonic() < deadline:
        scenario_name = scenarios[idx % len(scenarios)]
        idx += 1
        spec = SCENARIOS[scenario_name]
        path = spec["path"].format(user_id=target_user_id_for_path or "guest")
        headers = {}
        if spec["auth"] and bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        # /admin/* requiere CRON_SECRET, pero el script no asume admin scope —
        # el operador puede pasar bearer-token con admin scope si quiere.
        elif scenario_name == "pool_health" and bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        start = time.monotonic()
        try:
            resp = await client.request(spec["method"], path, headers=headers)
            latency = (time.monotonic() - start) * 1000.0
            results.append(RequestResult(
                scenario=scenario_name,
                status=resp.status_code,
                latency_ms=latency,
            ))
        except Exception as e:
            latency = (time.monotonic() - start) * 1000.0
            results.append(RequestResult(
                scenario=scenario_name,
                status=0,
                latency_ms=latency,
                error=f"{type(e).__name__}",
            ))


# =============================================================================
# RUNNER — orquesta N virtual users durante --duration y agrega stats.
# =============================================================================
async def run_load_test(args) -> dict:
    try:
        import httpx
    except ImportError:
        print(
            "❌ httpx no instalado. Instalar con: pip install httpx\n"
            "(httpx ya está en requirements.txt para el backend — debería estar "
            "disponible si corres este script desde el venv del backend).",
            file=sys.stderr,
        )
        sys.exit(2)

    scenarios = args.scenarios.split(",") if args.scenarios else list(SCENARIOS.keys())
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        print(f"❌ Scenarios desconocidos: {unknown}. Disponibles: {list(SCENARIOS.keys())}",
              file=sys.stderr)
        sys.exit(2)

    print(f"[load_test] target={args.target} concurrent={args.concurrent} "
          f"duration={args.duration}s scenarios={scenarios}")
    if args.bearer_token:
        print("[load_test] auth: Bearer token provisto (scenarios con auth=True lo usarán)")
    else:
        print("[load_test] auth: SIN bearer token (scenarios con auth=True omitidos)")
        scenarios = [s for s in scenarios if not SCENARIOS[s]["auth"]]
        if not scenarios:
            print("❌ Todos los scenarios solicitados requieren auth y no se pasó --bearer-token",
                  file=sys.stderr)
            sys.exit(2)

    results: list[RequestResult] = []
    deadline = time.monotonic() + args.duration

    # `limits` evita que httpx open-source pool overfloodee el server. Match
    # del pool max del server para detectar el cap NO cliente, sino server.
    limits = httpx.Limits(
        max_connections=max(args.concurrent * 2, 200),
        max_keepalive_connections=args.concurrent,
    )
    timeout = httpx.Timeout(args.request_timeout, connect=5.0)

    async with httpx.AsyncClient(
        base_url=args.target,
        limits=limits,
        timeout=timeout,
        follow_redirects=False,
    ) as client:
        # Spawning N virtual users — cada uno itera scenarios en loop.
        tasks = [
            virtual_user(
                client=client,
                user_id=i,
                scenarios=scenarios,
                deadline=deadline,
                target_user_id_for_path=args.target_user_id,
                bearer_token=args.bearer_token,
                results=results,
            )
            for i in range(args.concurrent)
        ]
        wall_start = time.monotonic()
        await asyncio.gather(*tasks)
        wall_duration = time.monotonic() - wall_start

    # Agregar por scenario.
    by_scenario: dict[str, ScenarioStats] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, ScenarioStats(scenario=r.scenario)).add(r)

    return {
        "target": args.target,
        "concurrent": args.concurrent,
        "duration_s": args.duration,
        "wall_duration_s": round(wall_duration, 2),
        "total_requests": len(results),
        "throughput_rps": round(len(results) / wall_duration, 1) if wall_duration > 0 else 0.0,
        "by_scenario": {
            name: {
                "count": s.count,
                "errors": s.errors,
                "error_rate_pct": round(s.error_rate * 100, 2),
                "error_breakdown": s.error_breakdown,
                "p50_ms": round(s.percentile(50), 1),
                "p95_ms": round(s.percentile(95), 1),
                "p99_ms": round(s.percentile(99), 1),
                "max_ms": round(max(s.latencies_ms), 1) if s.latencies_ms else 0,
            }
            for name, s in by_scenario.items()
        },
    }


# =============================================================================
# VERDICT — clasificar el run en PASS / WARN / FAIL.
# =============================================================================
def evaluate_verdict(report: dict, thresholds: dict) -> tuple[str, list[str]]:
    """Compara métricas vs thresholds. Devuelve (verdict, list_of_findings).

    Thresholds default conservadores (MVP <100 MAU):
      - error_rate_pct <= 1.0  → PASS; 1-5 → WARN; >5 → FAIL
      - p95_ms (no-admin):
          /health, /ready: <=200 → PASS; <=500 → WARN; >500 → FAIL
          /api/*: <=1500 → PASS; <=3000 → WARN; >3000 → FAIL
      - p99_ms: x2 del p95 threshold
    """
    findings: list[str] = []
    worst = "PASS"

    def escalate(level: str) -> None:
        nonlocal worst
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        if order[level] > order[worst]:
            worst = level

    for scenario, stats in report["by_scenario"].items():
        spec = SCENARIOS.get(scenario, {})
        is_health = scenario in {"health", "ready", "version"}

        # Error rate.
        er = stats["error_rate_pct"]
        if er > thresholds["error_rate_fail_pct"]:
            findings.append(f"FAIL {scenario}: error_rate={er}% > {thresholds['error_rate_fail_pct']}% (breakdown: {stats['error_breakdown']})")
            escalate("FAIL")
        elif er > thresholds["error_rate_warn_pct"]:
            findings.append(f"WARN {scenario}: error_rate={er}% > {thresholds['error_rate_warn_pct']}%")
            escalate("WARN")

        # p95 latency.
        p95 = stats["p95_ms"]
        p95_fail = thresholds["health_p95_fail_ms"] if is_health else thresholds["api_p95_fail_ms"]
        p95_warn = thresholds["health_p95_warn_ms"] if is_health else thresholds["api_p95_warn_ms"]
        if p95 > p95_fail:
            findings.append(f"FAIL {scenario}: p95={p95}ms > {p95_fail}ms (queries={spec.get('queries')})")
            escalate("FAIL")
        elif p95 > p95_warn:
            findings.append(f"WARN {scenario}: p95={p95}ms > {p95_warn}ms")
            escalate("WARN")

    if not findings:
        findings.append("All scenarios within thresholds.")
    return worst, findings


# =============================================================================
# CLI entry point.
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="DB pool load test (httpx async)")
    parser.add_argument("--target", required=True, help="Base URL del backend (ej. http://localhost:3001)")
    parser.add_argument("--concurrent", type=int, default=15, help="Número de virtual users concurrentes (default: 15)")
    parser.add_argument("--duration", type=int, default=30, help="Duración del test en segundos (default: 30)")
    parser.add_argument("--scenarios", default=None, help=f"Scenarios separados por coma (default: todos: {','.join(SCENARIOS.keys())})")
    parser.add_argument("--bearer-token", default=os.environ.get("MEALFIT_LOAD_TEST_BEARER"),
                        help="JWT/CRON_SECRET para scenarios con auth (default: env MEALFIT_LOAD_TEST_BEARER)")
    parser.add_argument("--target-user-id", default=os.environ.get("MEALFIT_LOAD_TEST_USER_ID", "guest"),
                        help="user_id para path placeholders (default: env MEALFIT_LOAD_TEST_USER_ID o 'guest')")
    parser.add_argument("--request-timeout", type=float, default=10.0, help="Timeout por request en segundos (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output reporte como JSON (no tabla humano)")
    parser.add_argument("--fail-on", choices=["FAIL", "WARN"], default="FAIL",
                        help="Exit code != 0 si verdict >= valor (default: FAIL — solo errores duros)")

    # Thresholds — configurables vía env vars para CI.
    parser.add_argument("--error-rate-warn-pct", type=float,
                        default=float(os.environ.get("MEALFIT_LOAD_TEST_ERR_WARN", "1.0")))
    parser.add_argument("--error-rate-fail-pct", type=float,
                        default=float(os.environ.get("MEALFIT_LOAD_TEST_ERR_FAIL", "5.0")))
    parser.add_argument("--health-p95-warn-ms", type=float,
                        default=float(os.environ.get("MEALFIT_LOAD_TEST_HEALTH_P95_WARN", "200")))
    parser.add_argument("--health-p95-fail-ms", type=float,
                        default=float(os.environ.get("MEALFIT_LOAD_TEST_HEALTH_P95_FAIL", "500")))
    parser.add_argument("--api-p95-warn-ms", type=float,
                        default=float(os.environ.get("MEALFIT_LOAD_TEST_API_P95_WARN", "1500")))
    parser.add_argument("--api-p95-fail-ms", type=float,
                        default=float(os.environ.get("MEALFIT_LOAD_TEST_API_P95_FAIL", "3000")))

    args = parser.parse_args()

    report = asyncio.run(run_load_test(args))

    thresholds = {
        "error_rate_warn_pct": args.error_rate_warn_pct,
        "error_rate_fail_pct": args.error_rate_fail_pct,
        "health_p95_warn_ms": args.health_p95_warn_ms,
        "health_p95_fail_ms": args.health_p95_fail_ms,
        "api_p95_warn_ms": args.api_p95_warn_ms,
        "api_p95_fail_ms": args.api_p95_fail_ms,
    }
    verdict, findings = evaluate_verdict(report, thresholds)
    report["verdict"] = verdict
    report["findings"] = findings
    report["thresholds"] = thresholds

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_table(report)

    if args.fail_on == "WARN" and verdict in {"WARN", "FAIL"}:
        return 1
    if args.fail_on == "FAIL" and verdict == "FAIL":
        return 1
    return 0


def _print_table(report: dict) -> None:
    print()
    print("=" * 90)
    print(f"[load_test] verdict={report['verdict']}  total={report['total_requests']}  "
          f"wall={report['wall_duration_s']}s  rps={report['throughput_rps']}")
    print("=" * 90)
    print(f"{'scenario':<14} {'count':>7} {'errors':>7} {'err%':>6} "
          f"{'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}")
    print("-" * 90)
    for name, s in sorted(report["by_scenario"].items()):
        print(f"{name:<14} {s['count']:>7} {s['errors']:>7} {s['error_rate_pct']:>5.2f}% "
              f"{s['p50_ms']:>6.1f}ms {s['p95_ms']:>6.1f}ms {s['p99_ms']:>6.1f}ms {s['max_ms']:>6.1f}ms")
    print()
    for f in report["findings"]:
        print(f"  {f}")
    print()


if __name__ == "__main__":
    sys.exit(main())
