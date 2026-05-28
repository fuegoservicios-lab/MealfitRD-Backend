"""[P2-HEALTHZ-DEEP · 2026-05-12] `/health/version` extendido con 5 keys
para blackbox monitor externo (UptimeRobot/cronitor) sin auth.

Bug original (audit production-readiness 2026-05-12):
    El watchdog interno `_alert_pipeline_metrics_silence` (P2-OBSERVABILITY-1)
    detecta deploy lag silencioso, pero solo si el binary corriendo TIENE
    la lógica del watchdog. Si el binary es PRE-watchdog (caso real
    observado 2026-05-12 — `is_guest` errors crashing a pipeline_metrics
    INSERT loop), el watchdog está dormido y nadie alerta.

    Solución: endpoint público que un poller externo
    (UptimeRobot/StatusCake/cronitor — sin acceso a CRON_SECRET) puede
    consultar para detectar:
      1. drift entre live binary y expected marker (KV)
      2. ausencia de tick reciente del autoheal loop (heartbeat)
      3. ausencia del gate `_is_guest_metrics_enabled` (P0-PROD-1)
      4. ausencia del cache `_SCHEDULER_JOBS_WITH_OPEN_ALERTS` (P1-PERF-1)

    Cierre de la paradoja "el binary roto se vigila a sí mismo".

Diseño:
    Extendí `/health/version` (público, ya existente desde P1-A · 2026-05-08)
    con 5 keys nuevas en lugar de crear `/healthz/deep` separado. Razón:
      - Evita duplicar URL pública para datos casi-iguales.
      - El operador y los pollers consultan la misma URL — un solo SSOT.
      - Cada lectura es best-effort (try/except → None/False) para que el
        endpoint siga respondiendo 200 incluso si la sub-query falla
        (KV unreachable, pipeline_metrics tabla missing, etc.) — el poller
        externo necesita distinguir "binary alive con lectura parcial" de
        "binary down" (timeout/5xx).

Cobertura (parser-estático sobre `app.py`):
    1. Anchor `P2-HEALTHZ-DEEP` presente en docstring + en bloque de código.
    2. Las 5 keys nuevas están en el diccionario de retorno:
       expected_marker, drift, last_pipeline_metrics_tick_at,
       has_p0_prod_1_gate, has_p1_perf_1_cache.
    3. Cada lectura externa (KV + pipeline_metrics) está envuelta en
       try/except (best-effort).
    4. `has_p0_prod_1_gate` se evalúa importando `_is_guest_metrics_enabled`
       desde graph_orchestrator (NO hardcoded True).
    5. `has_p1_perf_1_cache` evalúa la presencia de la variable global
       `_SCHEDULER_JOBS_WITH_OPEN_ALERTS` en `globals()`.
    6. Endpoint sigue siendo público (sin `_verify_admin_token`).
    7. Las 6 keys de P3-2 originales NO removidas (regression guard).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"

_NEW_KEYS = [
    "expected_marker",
    "drift",
    "last_pipeline_metrics_tick_at",
    "has_p0_prod_1_gate",
    "has_p1_perf_1_cache",
]


@pytest.fixture(scope="module")
def app_src() -> str:
    if not _APP_PY.exists():
        pytest.fail(
            f"app.py no encontrado: {_APP_PY}. P2-HEALTHZ-DEEP depende "
            f"de extender el handler `health_version` en este archivo."
        )
    return _APP_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor textual
# ---------------------------------------------------------------------------
def test_anchor_present(app_src: str):
    """Anchor `P2-HEALTHZ-DEEP` debe aparecer al menos 2 veces:
    una en el docstring del endpoint y una en el bloque de código que
    ejecuta las 3 sub-queries best-effort."""
    count = app_src.count("P2-HEALTHZ-DEEP")
    assert count >= 2, (
        f"P2-HEALTHZ-DEEP regresión: anchor textual aparece {count} "
        f"veces en app.py. Esperaba ≥2 (docstring + bloque código). "
        f"Si fue removido, restaurar para `grep -r P2-HEALTHZ-DEEP backend/`."
    )


# ---------------------------------------------------------------------------
# 2. Las 5 keys nuevas están en el dict de retorno
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", _NEW_KEYS)
def test_new_key_in_response_dict(app_src: str, key: str):
    """Cada una de las 5 keys nuevas debe aparecer como string-literal
    en app.py (es decir, dentro del dict del return de health_version)."""
    quoted = f'"{key}"'
    assert quoted in app_src, (
        f"P2-HEALTHZ-DEEP regresión: la key {quoted} desapareció del "
        f"diccionario de retorno de /health/version. Sin esta key, los "
        f"pollers externos (UptimeRobot/cronitor) NO pueden assertir "
        f"sobre el estado de deploy lag → vuelve la paradoja "
        f"'binary roto se vigila a sí mismo'."
    )


# ---------------------------------------------------------------------------
# 3. Best-effort: lecturas externas en try/except
# ---------------------------------------------------------------------------
def test_kv_read_is_best_effort(app_src: str):
    """La lectura de `app_kv_store.expected_last_known_pfix` debe estar
    envuelta en try/except. Si KV falla (DB down, RLS bloquea), el
    endpoint sigue respondiendo 200 con `expected_marker=None` y
    `drift=None` — el poller distingue eso de un 5xx."""
    pattern = re.compile(
        r"expected_last_known_pfix.*?\n.*?except\s+Exception",
        re.DOTALL,
    )
    assert pattern.search(app_src), (
        "P2-HEALTHZ-DEEP regresión: la lectura de "
        "`expected_last_known_pfix` no está protegida por try/except. "
        "Si la query falla, /health/version retorna 500 y los pollers "
        "externos no distinguen 'binary alive con KV down' de 'binary "
        "down' — pierde el contrato del endpoint público."
    )


def test_pipeline_metrics_read_is_best_effort(app_src: str):
    """Lectura de `pipeline_metrics` para `last_pipeline_metrics_tick_at`
    también best-effort."""
    pattern = re.compile(
        r"_hardfloor_autoheal_tick.*?\n.*?except\s+Exception",
        re.DOTALL,
    )
    assert pattern.search(app_src), (
        "P2-HEALTHZ-DEEP regresión: lectura de `pipeline_metrics` "
        "para last_pipeline_metrics_tick_at no está protegida por "
        "try/except."
    )


# ---------------------------------------------------------------------------
# 4. has_p0_prod_1_gate evalúa importabilidad real (NO hardcoded)
# ---------------------------------------------------------------------------
def test_has_p0_prod_1_gate_imports_real_symbol(app_src: str):
    """`has_p0_prod_1_gate` debe asignarse en función de un `import` real
    (no hardcoded `True`/`False`). Si el binary no tiene
    `_is_guest_metrics_enabled` en graph_orchestrator (binary PRE-P0-PROD-1),
    el import falla y la key reporta False."""
    # Patrón flexible: import _is_guest_metrics_enabled + assignment.
    pattern = re.compile(
        r"from\s+graph_orchestrator\s+import\s+_is_guest_metrics_enabled.*?\n.*?has_p0_prod_1_gate\s*=\s*True",
        re.DOTALL,
    )
    assert pattern.search(app_src), (
        "P2-HEALTHZ-DEEP regresión: `has_p0_prod_1_gate` NO usa import "
        "real desde graph_orchestrator. Si pasó a hardcoded True/False, "
        "el endpoint MIENTE sobre la presencia del gate y los pollers "
        "no detectan binary PRE-P0-PROD-1."
    )


def test_has_p1_perf_1_cache_evaluates_globals(app_src: str):
    """`has_p1_perf_1_cache` debe evaluar `'_SCHEDULER_JOBS_WITH_OPEN_ALERTS'
    in globals()` (NO hardcoded). El cache vive como variable de módulo
    en app.py — verificar via `globals()` es la prueba real."""
    pattern = re.compile(
        r"_SCHEDULER_JOBS_WITH_OPEN_ALERTS.*?in\s+globals\s*\(",
        re.DOTALL,
    )
    assert pattern.search(app_src), (
        "P2-HEALTHZ-DEEP regresión: `has_p1_perf_1_cache` NO usa "
        "`'_SCHEDULER_JOBS_WITH_OPEN_ALERTS' in globals()`. Si pasó a "
        "hardcoded, el endpoint miente sobre la presencia del cache."
    )


# ---------------------------------------------------------------------------
# 5. Endpoint sigue público (NO _verify_admin_token)
# ---------------------------------------------------------------------------
def test_health_version_remains_public(app_src: str):
    """`/health/version` NO debe requerir auth — los pollers externos
    no tienen CRON_SECRET. Verificar que dentro del bloque de la función
    `health_version` NO hay llamada a `_verify_admin_token`."""
    # Aislar el bloque de la función desde @app.get("/health/version")
    # hasta el siguiente @app.get o el final del archivo.
    block_pattern = re.compile(
        r'@app\.get\("/health/version"\).*?(?=\n@app\.|^\Z)',
        re.DOTALL | re.MULTILINE,
    )
    block_match = block_pattern.search(app_src)
    assert block_match, "No pude aislar el bloque de health_version"
    block = block_match.group(0)
    assert "_verify_admin_token" not in block, (
        "P2-HEALTHZ-DEEP regresión: `/health/version` ahora requiere "
        "auth (`_verify_admin_token` presente en el bloque del handler). "
        "Esto rompe el contrato del endpoint público — los pollers "
        "externos UptimeRobot/cronitor no tienen CRON_SECRET y empezarían "
        "a recibir 401/403, perdiendo la señal de deploy lag."
    )


# ---------------------------------------------------------------------------
# 6. Regression guard: keys originales P3-2 no removidas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", [
    "git_sha",
    "git_short_sha",
    "deploy_timestamp",
    "process_started_at",
    "process_uptime_s",
    "last_known_pfix",
    "knobs_count",
    "knobs_overrides_count",
    "cron_missed_1h_total",
])
def test_pre_existing_keys_preserved(app_src: str, key: str):
    """Las keys de P1-A + P3-2 NO deben ser removidas. Extender ≠
    reemplazar."""
    quoted = f'"{key}"'
    assert quoted in app_src, (
        f"P2-HEALTHZ-DEEP regresión: la key {quoted} (pre-existente) "
        f"desapareció del response de /health/version. Si la "
        f"refactorización es intencional, actualizar este test y "
        f"`test_p3_2_health_version_extended.py`."
    )
