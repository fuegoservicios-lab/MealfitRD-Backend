"""[P1-PERF-1 · 2026-05-12] Listener `EVENT_JOB_EXECUTED` consulta cache
in-memory antes de hacer PATCH REST a `system_alerts`.

Bug observado (audit 2026-05-11):
    Pre-fix, el listener disparaba 1 PATCH REST por CADA job EXECUTED
    aunque NO existiera alert pendiente (UPDATE no-op de 0 rows). Con 28
    jobs * múltiples ejecuciones/h, ~5000+ PATCH/h. MCP API logs mostraron
    ~1-3 PATCH/seg sostenido. Desperdicio real de REST quota + DB CPU.

Fix:
    Cache `_SCHEDULER_JOBS_WITH_OPEN_ALERTS: set[str]` mantenido por el
    propio listener. add en MISSED/ERROR post-upsert; discard en EXECUTED
    post-éxito. EXECUTED hace PATCH solo si `job_id in cache`. Refresh
    TTL=60s capta cambios out-of-band (resoluciones manuales). Cold cache
    refresh en lifespan startup.

Lo que este test enforza:
    A) Anchor `P1-PERF-1` presente en `app.py`.
    B) `_SCHEDULER_JOBS_WITH_OPEN_ALERTS: set[...]` declarado a nivel
       módulo + `_SCHEDULER_OPEN_ALERTS_LOCK = threading.Lock()` para
       acceso atómico desde threads del scheduler.
    C) Helper `_refresh_scheduler_open_alerts_cache` declarado.
    D) Rama EXECUTED contiene un check `job_id in _SCHEDULER_JOBS_WITH_OPEN_ALERTS`
       ANTES de cualquier `supabase.table("system_alerts").update(`.
    E) Cold-cache refresh invocado en lifespan post-`scheduler.start()`.
    F) Rama MISSED/ERROR popula el cache tras el upsert (sino el primer
       EXECUTED no encontraría el job).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"


@pytest.fixture(scope="module")
def app_src() -> str:
    return _APP_PY.read_text(encoding="utf-8")


def _isolate_listener(src: str) -> str:
    m = re.search(
        r"def\s+_scheduler_alert_listener\b(.*?)(?=^def\s+\w)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None, "_scheduler_alert_listener no encontrado."
    return m.group(1)


def _isolate_lifespan(src: str) -> str:
    m = re.search(
        r"async\s+def\s+lifespan\b(.*?)(?=^(?:async\s+)?def\s+\w)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None, "lifespan no encontrado."
    return m.group(1)


def test_a_anchor_present(app_src: str):
    assert "P1-PERF-1" in app_src, (
        "P1-PERF-1: anchor desapareció. Restaurar el bloque de "
        "documentación del cache."
    )


def test_b_cache_declared_with_lock(app_src: str):
    """El cache `set` está declarado a nivel módulo + Lock para acceso
    multi-thread (BackgroundScheduler usa ThreadPoolExecutor)."""
    assert re.search(
        r"_SCHEDULER_JOBS_WITH_OPEN_ALERTS\s*:\s*set\[",
        app_src,
    ), (
        "P1-PERF-1: `_SCHEDULER_JOBS_WITH_OPEN_ALERTS: set[...]` no "
        "declarado a nivel módulo."
    )
    assert "_SCHEDULER_OPEN_ALERTS_LOCK = threading.Lock(" in app_src, (
        "P1-PERF-1: `_SCHEDULER_OPEN_ALERTS_LOCK` ausente. Sin lock, "
        "race entre threads del scheduler corrompe el set."
    )


def test_c_refresh_helper_defined(app_src: str):
    assert "def _refresh_scheduler_open_alerts_cache(" in app_src, (
        "P1-PERF-1: helper `_refresh_scheduler_open_alerts_cache` ausente."
    )


def test_d_executed_branch_checks_cache_before_patch(app_src: str):
    """En la rama EXECUTED, debe aparecer un check contra el set ANTES
    de invocar `supabase.table('system_alerts').update(`."""
    listener = _isolate_listener(app_src)
    # Aislar la rama EXECUTED.
    executed_match = re.search(
        r"elif\s+code\s*==\s*EVENT_JOB_EXECUTED\s*:(.*?)(?:^\s{8}else\s*:|\Z)",
        listener,
        re.DOTALL | re.MULTILINE,
    )
    assert executed_match is not None, "Rama EVENT_JOB_EXECUTED no aislable."
    branch = executed_match.group(1)

    # Check del cache debe ocurrir antes del primer update().
    cache_idx = branch.find("_SCHEDULER_JOBS_WITH_OPEN_ALERTS")
    update_idx = branch.find('supabase.table("system_alerts").update(')
    if update_idx < 0:
        update_idx = branch.find("supabase.table('system_alerts').update(")
    assert cache_idx >= 0, (
        "P1-PERF-1: la rama EXECUTED no consulta "
        "`_SCHEDULER_JOBS_WITH_OPEN_ALERTS`. PATCHing sigue siendo "
        "incondicional → no se ahorra el costo REST."
    )
    if update_idx >= 0:
        assert cache_idx < update_idx, (
            "P1-PERF-1: el check del cache debe ocurrir ANTES del "
            "PATCH. Sino, el PATCH se ejecuta y el check es irrelevante."
        )


def test_e_cold_cache_refresh_in_lifespan(app_src: str):
    """`_refresh_scheduler_open_alerts_cache(force=True)` invocado en
    lifespan startup. Sin este refresh, el primer ciclo post-restart
    saltaría PATCHes que SÍ deberían ocurrir (alerts persistentes)."""
    lifespan = _isolate_lifespan(app_src)
    assert "_refresh_scheduler_open_alerts_cache(force=True)" in lifespan, (
        "P1-PERF-1: cold-cache refresh no invocado en lifespan. Post-restart "
        "el cache estará vacío y EXECUTED saltará PATCHes legítimos."
    )


def test_f_missed_or_error_branch_populates_cache(app_src: str):
    """Tras el upsert en MISSED/ERROR, el `job_id` debe añadirse al set.
    Sino, el subsiguiente EXECUTED no encontrará el job y NO disparará
    el PATCH (alert quedaría abierta hasta TTL)."""
    listener = _isolate_listener(app_src)
    # Buscar `_SCHEDULER_JOBS_WITH_OPEN_ALERTS.add(` post-upsert.
    assert "_SCHEDULER_JOBS_WITH_OPEN_ALERTS.add(" in listener, (
        "P1-PERF-1: tras upsert MISSED/ERROR, el cache no se popula con "
        "`add(job_id)`. El primer EXECUTED siguiente NO disparará PATCH "
        "(cache miss) y la alert quedará abierta hasta el TTL."
    )


def test_g_executed_branch_removes_from_cache_on_success(app_src: str):
    """Tras PATCH exitoso, remover del cache. Sino, próximos EXECUTED del
    mismo job harían PATCH no-op de nuevo."""
    listener = _isolate_listener(app_src)
    assert "_SCHEDULER_JOBS_WITH_OPEN_ALERTS.discard(" in listener, (
        "P1-PERF-1: post-PATCH exitoso, no se hace `discard(job_id)`. "
        "Próximos EXECUTED dispararán PATCH no-op de nuevo (cache "
        "lookup acierta + DB UPDATE 0 rows)."
    )
