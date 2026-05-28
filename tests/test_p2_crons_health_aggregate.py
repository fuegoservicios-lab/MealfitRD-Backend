"""[P2-CRONS-HEALTH-AGGREGATE · 2026-05-27] Endpoint admin
`GET /api/system/admin/crons-status` enriquecido con `consecutive_failures`,
`last_failure_at`, `last_tick_at` per-job.

Gap original (plan P2 del audit prod-readiness 2026-05-27):
    El endpoint `/admin/crons-status` (P3-CRONS-STATUS-ADMIN · 2026-05-15)
    listaba qué jobs estaban REGISTRADOS y su `next_run_time` pero NO
    decia NADA sobre si estaban SANOS. Post-incidente, SRE tenia que
    cruzar 3 fuentes manualmente:
      - `app_kv_store` → `cron_failures_count:<cron_name>` (consecutive
        failures + last_failure_at).
      - `pipeline_metrics` → `MAX(created_at) WHERE node='_<cron>_tick'`
        (ultimo tick observable, liveness real).
      - `scheduler.get_jobs()` → `next_run_time` (cuando intentara
        de nuevo).

El enrichment agrega los 3 en una sola row por job, ordenada por
`consecutive_failures DESC, last_tick_at ASC`. Un curl muestra "que cron
va peor right now" sin abrir SQL editor.

Esta suite verifica:
  (1) Parser-based: el endpoint contiene los 3 campos nuevos +
      sort key + queries a las 2 fuentes (KV failures + pipeline_metrics).
  (2) Funcional: con mocks de execute_sql_query, el endpoint enriquece
      correctamente jobs con las 2 fuentes, sort funciona, best-effort
      maneja fallos de las queries.

Tooltip-anchor: P2-CRONS-HEALTH-AGGREGATE.

NOTA: los 2 P2 originales del plan del audit eran:
  - P2-1 PER-PREFIX-TTL-EN-KV-SWEEP — DESCARTADO como falso positivo
    (ya implementado por P1-CRON-BUNDLE/P2-OPS-BUNDLE: `_KV_SWEEP_PREFIXES`
    es `list[dict]` con `prefix/knob/default_h/clamp` per-prefix).
  - P2-2 CRONS-HEALTH-AGGREGATE — implementado aquí.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SYSTEM_PY = _BACKEND_ROOT / "routers" / "system.py"


def _read_system_py() -> str:
    return _SYSTEM_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser-based: structural invariants
# ---------------------------------------------------------------------------

def test_p2_marker_anchor_present_in_endpoint():
    """`P2-CRONS-HEALTH-AGGREGATE` debe aparecer >=3 veces en system.py
    (comments + docstring + sort fallback) para anclaje contra refactors."""
    text = _read_system_py()
    occurrences = text.count("P2-CRONS-HEALTH-AGGREGATE")
    assert occurrences >= 3, (
        f"P2-CRONS-HEALTH-AGGREGATE marker count en system.py: "
        f"{occurrences}. Esperaba >=3 (docstring + comments + log)."
    )


def test_p2_endpoint_queries_kv_failures_source():
    """El endpoint debe hacer un SELECT contra `app_kv_store` filtrando
    por `cron_failures_count:%` (source 1 del enrichment)."""
    text = _read_system_py()
    assert "cron_failures_count:" in text, (
        "P2-CRONS-HEALTH-AGGREGATE: el endpoint no referencia el prefijo "
        "`cron_failures_count:` — sin esa fuente no hay `consecutive_failures` "
        "per-job."
    )
    # El SELECT exacto contra app_kv_store con LIKE.
    assert re.search(
        r"SELECT\s+key.*?FROM\s+app_kv_store.*?LIKE\s+'cron_failures_count:%'",
        text,
        re.DOTALL,
    ), (
        "P2-CRONS-HEALTH-AGGREGATE: SELECT contra app_kv_store con "
        "`LIKE 'cron_failures_count:%'` no encontrado."
    )


def test_p2_endpoint_queries_pipeline_metrics_tick_source():
    """El endpoint debe hacer un SELECT contra `pipeline_metrics` filtrando
    por `node LIKE '_%_tick'` (source 2 del enrichment, liveness real)."""
    text = _read_system_py()
    assert "pipeline_metrics" in text, (
        "P2-CRONS-HEALTH-AGGREGATE: el endpoint no referencia "
        "`pipeline_metrics` — sin esa fuente no hay `last_tick_at` per-job."
    )
    # SELECT con GROUP BY node + filtro de _tick.
    assert re.search(
        r"SELECT\s+node.*?MAX\(created_at\).*?FROM\s+pipeline_metrics.*?LIKE\s+'\\_%\\_tick'",
        text,
        re.DOTALL,
    ), (
        "P2-CRONS-HEALTH-AGGREGATE: SELECT contra pipeline_metrics con "
        "`MAX(created_at)` y `LIKE '\\_%\\_tick'` no encontrado. "
        "Patron escape '\\' es necesario por LIKE de Postgres (los `_` "
        "son wildcards por default)."
    )


def test_p2_endpoint_returns_three_new_fields_per_job():
    """Cada job en `jobs[]` debe ganar 3 nuevos campos:
    `consecutive_failures`, `last_failure_at`, `last_tick_at`."""
    text = _read_system_py()
    for field in ("consecutive_failures", "last_failure_at", "last_tick_at"):
        assert f'"{field}"' in text or f'{field}"' in text, (
            f"P2-CRONS-HEALTH-AGGREGATE: campo `{field}` no presente en "
            f"system.py. Sin él, el enrichment queda incompleto."
        )


def test_p2_endpoint_has_sort_by_failures_desc():
    """El sort debe ser (`consecutive_failures DESC`, `last_tick_at ASC`).
    Sin sort, SRE tiene que ordenar manualmente — el endpoint pierde
    valor."""
    text = _read_system_py()
    # Buscar la función _sort_key dentro del endpoint.
    assert "_sort_key" in text, (
        "P2-CRONS-HEALTH-AGGREGATE: helper `_sort_key` para ordenar jobs "
        "no encontrado."
    )
    # Verificar que el sort key invierte el signo de failures (DESC).
    assert re.search(
        r"cf_val\s*=\s*-1\s+if\s+cf\s+is\s+None\s+else\s+int\(cf\)",
        text,
    ) or re.search(
        r"-\s*cf_val|-\s*int\(cf\)", text
    ), (
        "P2-CRONS-HEALTH-AGGREGATE: el sort no invierte `consecutive_failures` "
        "(DESC). Sin DESC, los crons sanos aparecen primero — invertido del "
        "valor SRE."
    )


def test_p2_endpoint_uses_best_effort_try_except():
    """Las 2 queries adicionales deben estar en try/except — si fallan,
    el endpoint retorna jobs sin enriquecer (back-compat con clientes
    que solo leen el shape original P3-CRONS-STATUS-ADMIN)."""
    text = _read_system_py()
    # Buscar dos try/except con log debug `[P2-CRONS-HEALTH-AGGREGATE]`.
    pattern = re.compile(
        r"\[P2-CRONS-HEALTH-AGGREGATE\].*?falló\s*\(best-effort\)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    assert len(matches) >= 2, (
        f"P2-CRONS-HEALTH-AGGREGATE: esperaba >=2 try/except con log "
        f"`[P2-CRONS-HEALTH-AGGREGATE] ... falló (best-effort)`. Encontré "
        f"{len(matches)}. Sin best-effort, un fallo de la query KV o "
        f"pipeline_metrics rompe el endpoint entero."
    )


def test_p2_endpoint_preserves_legacy_response_shape():
    """El endpoint debe seguir retornando los campos legacy
    (`has_scheduler`, `jobs`, `jobs_count`, `knobs_kill_switches`,
    `knobs_count_total`) — clientes pre-P2 no deben romperse."""
    text = _read_system_py()
    for legacy_field in (
        '"has_scheduler"',
        '"jobs"',
        '"jobs_count"',
        '"knobs_kill_switches"',
        '"knobs_count_total"',
    ):
        assert legacy_field in text, (
            f"P2-CRONS-HEALTH-AGGREGATE: campo legacy {legacy_field} "
            f"removido del response. Romperia clientes pre-P2."
        )


def test_p2_endpoint_window_caps_pipeline_metrics_query():
    """El SELECT contra `pipeline_metrics` debe tener ventana acotada
    (`INTERVAL '7 days'`) — sin cap, el GROUP BY hace full-scan de una
    tabla que crece monotonicamente."""
    text = _read_system_py()
    assert re.search(
        r"pipeline_metrics.*?INTERVAL\s+'7\s+days'",
        text,
        re.DOTALL,
    ), (
        "P2-CRONS-HEALTH-AGGREGATE: SELECT contra pipeline_metrics sin "
        "ventana `INTERVAL '7 days'`. Sin cap, full-scan de tabla "
        "monotonicamente creciente."
    )


# ---------------------------------------------------------------------------
# Functional: mockear execute_sql_query + scheduler + verificar enrichment
# ---------------------------------------------------------------------------

def _stub_apscheduler():
    if "apscheduler" not in sys.modules:
        for mod_name in (
            "apscheduler",
            "apscheduler.schedulers",
            "apscheduler.schedulers.background",
            "apscheduler.executors",
            "apscheduler.executors.pool",
            "apscheduler.events",
        ):
            sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
        sys.modules["apscheduler.events"].EVENT_JOB_MISSED = 1
        sys.modules["apscheduler.events"].EVENT_JOB_ERROR = 2
        sys.modules["apscheduler.events"].EVENT_JOB_EXECUTED = 4


def _try_import_system_router():
    """Importa routers.system con stubs de deps pesadas. Returns None si
    no es posible (entorno standalone sin Supabase)."""
    _stub_apscheduler()
    sys.modules.setdefault("sentry_sdk", types.ModuleType("sentry_sdk"))
    try:
        from routers import system as system_mod
        return system_mod
    except Exception:
        return None


def test_p2_endpoint_enriches_jobs_from_kv_and_pipeline_metrics():
    """Functional: con execute_sql_query mockeado retornando filas falsas
    de KV failures + pipeline_metrics ticks, verificar que cada job en
    la respuesta gana los 3 campos correctamente."""
    system_mod = _try_import_system_router()
    if system_mod is None:
        pytest.skip("routers.system no importable en entorno test standalone")

    # Mock scheduler.get_jobs() returning 3 jobs.
    fake_job_a = MagicMock()
    fake_job_a.id = "sweep_stale_app_kv_store_prefixes"
    fake_job_a.next_run_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    fake_job_a.trigger = "interval[0:30:00]"
    fake_job_a.coalesce = True
    fake_job_a.max_instances = 1

    fake_job_b = MagicMock()
    fake_job_b.id = "sweep_stale_scheduler_missed_alerts"
    fake_job_b.next_run_time = datetime.now(timezone.utc) + timedelta(hours=1)
    fake_job_b.trigger = "interval[1:00:00]"
    fake_job_b.coalesce = True
    fake_job_b.max_instances = 1

    fake_job_c = MagicMock()
    fake_job_c.id = "cron_without_tracking"
    fake_job_c.next_run_time = datetime.now(timezone.utc) + timedelta(hours=2)
    fake_job_c.trigger = "interval[2:00:00]"
    fake_job_c.coalesce = True
    fake_job_c.max_instances = 1

    fake_scheduler = MagicMock()
    fake_scheduler.get_jobs.return_value = [fake_job_a, fake_job_b, fake_job_c]

    # Mock execute_sql_query: 2 calls (KV failures + pipeline_metrics ticks).
    kv_rows = [
        {
            "key": "cron_failures_count:sweep_stale_app_kv_store_prefixes",
            "value": {
                "count": 5,
                "last_failure_at": "2026-05-27T10:00:00+00:00",
                "last_error": "DB timeout",
            },
            "updated_at": datetime.now(timezone.utc),
        },
        {
            "key": "cron_failures_count:sweep_stale_scheduler_missed_alerts",
            "value": {
                "count": 0,
                "last_failure_at": None,
            },
            "updated_at": datetime.now(timezone.utc),
        },
    ]
    last_tick_iso = datetime.now(timezone.utc) - timedelta(minutes=15)
    tick_rows = [
        {
            "node": "_sweep_stale_app_kv_store_prefixes_tick",
            "last_tick_at": last_tick_iso,
        },
        {
            "node": "_sweep_stale_scheduler_missed_alerts_tick",
            "last_tick_at": datetime.now(timezone.utc) - timedelta(minutes=5),
        },
    ]

    call_counter = {"n": 0}

    def fake_execute_sql_query(sql, params=None, **kwargs):
        call_counter["n"] += 1
        if "cron_failures_count" in sql:
            return kv_rows
        if "pipeline_metrics" in sql:
            return tick_rows
        # Otras queries (knobs registry, etc).
        return []

    # Mock _verify_admin_token + _check_admin_rate_limit (no-op).
    fake_request = MagicMock()
    fake_request.headers = {"authorization": "Bearer test"}

    with patch.object(system_mod, "execute_sql_query", side_effect=fake_execute_sql_query), \
         patch.object(system_mod, "_verify_admin_token"), \
         patch.object(system_mod, "_check_admin_rate_limit"), \
         patch.dict("sys.modules", {
             "app": types.SimpleNamespace(
                 scheduler=fake_scheduler, HAS_SCHEDULER=True
             ),
         }):
        # graph_orchestrator import en el endpoint también — stubeamos.
        # Si el módulo real está cargado, lo dejamos; si no, stub mínimo.
        if "graph_orchestrator" not in sys.modules:
            stub_go = types.ModuleType("graph_orchestrator")
            stub_go.get_knobs_registry_snapshot = lambda: {}
            sys.modules["graph_orchestrator"] = stub_go

        response = system_mod.admin_crons_status(fake_request)

    assert response["success"] is True
    assert response["jobs_count"] == 3
    jobs = response["jobs"]

    # Job A (sweep_stale_app_kv_store_prefixes) debe tener consec=5, last_tick.
    by_id = {j["job_id"]: j for j in jobs}
    j_a = by_id["sweep_stale_app_kv_store_prefixes"]
    assert j_a["consecutive_failures"] == 5
    assert j_a["last_failure_at"] == "2026-05-27T10:00:00+00:00"
    assert j_a["last_error"] == "DB timeout"
    assert j_a["last_tick_at"] is not None

    # Job B debe tener count=0 + last_tick reciente.
    j_b = by_id["sweep_stale_scheduler_missed_alerts"]
    assert j_b["consecutive_failures"] == 0
    assert j_b["last_failure_at"] is None
    assert j_b["last_tick_at"] is not None

    # Job C (sin tracking) debe tener los 3 campos en None.
    j_c = by_id["cron_without_tracking"]
    assert j_c["consecutive_failures"] is None
    assert j_c["last_failure_at"] is None
    assert j_c["last_tick_at"] is None


def test_p2_endpoint_sorts_worst_cron_first():
    """Functional: con jobs mixtos (algunos con failures>0, algunos sin
    tracking, algunos sanos), el sort debe poner los peores primero."""
    system_mod = _try_import_system_router()
    if system_mod is None:
        pytest.skip("routers.system no importable en entorno test standalone")

    # 3 jobs: uno con 10 failures, uno sin tracking, uno sano (0 failures).
    fake_jobs = []
    for jid in ("sano", "broken", "untracked"):
        fj = MagicMock()
        fj.id = jid
        fj.next_run_time = datetime.now(timezone.utc) + timedelta(minutes=30)
        fj.trigger = "interval[0:30:00]"
        fj.coalesce = True
        fj.max_instances = 1
        fake_jobs.append(fj)
    fake_scheduler = MagicMock()
    fake_scheduler.get_jobs.return_value = fake_jobs

    kv_rows = [
        {
            "key": "cron_failures_count:broken",
            "value": {"count": 10, "last_failure_at": "2026-05-27T10:00:00+00:00"},
            "updated_at": datetime.now(timezone.utc),
        },
        {
            "key": "cron_failures_count:sano",
            "value": {"count": 0, "last_failure_at": None},
            "updated_at": datetime.now(timezone.utc),
        },
    ]

    def fake_execute_sql_query(sql, params=None, **kwargs):
        if "cron_failures_count" in sql:
            return kv_rows
        if "pipeline_metrics" in sql:
            return []
        return []

    fake_request = MagicMock()
    fake_request.headers = {"authorization": "Bearer test"}

    with patch.object(system_mod, "execute_sql_query", side_effect=fake_execute_sql_query), \
         patch.object(system_mod, "_verify_admin_token"), \
         patch.object(system_mod, "_check_admin_rate_limit"), \
         patch.dict("sys.modules", {
             "app": types.SimpleNamespace(
                 scheduler=fake_scheduler, HAS_SCHEDULER=True
             ),
         }):
        if "graph_orchestrator" not in sys.modules:
            stub_go = types.ModuleType("graph_orchestrator")
            stub_go.get_knobs_registry_snapshot = lambda: {}
            sys.modules["graph_orchestrator"] = stub_go

        response = system_mod.admin_crons_status(fake_request)

    jobs = response["jobs"]
    # Orden esperado: broken (10 failures), untracked (-1 = "no data"),
    # sano (0). El sort key invierte signo → broken primero, untracked
    # tratado como peor (None=-1=mayor peso negativo invertido).
    # Con `cf_val = -1 if cf is None else int(cf)` y sort por `-cf_val`:
    #   broken: cf=10 → key=(-10, "")
    #   untracked: cf=None → cf_val=-1 → key=(1, "")
    #   sano: cf=0 → key=(0, "")
    # Orden ascendente del tuple: -10 < 0 < 1 → broken, sano, untracked.
    assert jobs[0]["job_id"] == "broken", (
        f"Esperaba `broken` primero (10 failures). Orden actual: "
        f"{[j['job_id'] for j in jobs]}"
    )


def test_p2_endpoint_resilient_to_kv_query_failure():
    """Functional: si la query KV falla (DB blip), el endpoint debe
    retornar jobs con campos None pero el response sigue OK (200).
    Back-compat con clientes pre-P2."""
    system_mod = _try_import_system_router()
    if system_mod is None:
        pytest.skip("routers.system no importable en entorno test standalone")

    fake_job = MagicMock()
    fake_job.id = "any_job"
    fake_job.next_run_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    fake_job.trigger = "interval[0:30:00]"
    fake_job.coalesce = True
    fake_job.max_instances = 1
    fake_scheduler = MagicMock()
    fake_scheduler.get_jobs.return_value = [fake_job]

    def fake_execute_sql_query(sql, params=None, **kwargs):
        # Simular fallo en AMBAS queries.
        raise RuntimeError("DB blip simulated")

    fake_request = MagicMock()
    fake_request.headers = {"authorization": "Bearer test"}

    with patch.object(system_mod, "execute_sql_query", side_effect=fake_execute_sql_query), \
         patch.object(system_mod, "_verify_admin_token"), \
         patch.object(system_mod, "_check_admin_rate_limit"), \
         patch.dict("sys.modules", {
             "app": types.SimpleNamespace(
                 scheduler=fake_scheduler, HAS_SCHEDULER=True
             ),
         }):
        if "graph_orchestrator" not in sys.modules:
            stub_go = types.ModuleType("graph_orchestrator")
            stub_go.get_knobs_registry_snapshot = lambda: {}
            sys.modules["graph_orchestrator"] = stub_go

        # NO debe levantar — best-effort.
        response = system_mod.admin_crons_status(fake_request)

    assert response["success"] is True
    assert response["jobs_count"] == 1
    j = response["jobs"][0]
    assert j["consecutive_failures"] is None
    assert j["last_failure_at"] is None
    assert j["last_tick_at"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
