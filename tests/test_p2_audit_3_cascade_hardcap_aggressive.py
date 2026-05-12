"""[P2-AUDIT-3 · 2026-05-12] `MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS`
default bajado de 6 a 2 + clamp explícito [1, 6].

Contexto:
    Audit production-readiness 2026-05-12 observó `scheduler_cascade_missed`
    (CRITICAL) abierto ~10 min tras un burst transitorio post-restart. El
    hard-cap default de 6h dejaba el alert en dashboards 6h después de
    eventos triviales, generando alert fatigue.

    Decisión: bajar el default a 2h. El detector re-emite el parent cada
    30min (MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN) si la cascada sigue
    activa — el alert NO se pierde, solo su "edad" se resetea a una
    ventana <2h. Operador puede revertir vía
    `MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS=6` sin redeploy.

    Clamp [1, 6] explícito añadido — antes solo había `< 1: hard_cap_h = 1`.
    Operador que setea 99 ahora ve 6 (alineado con el default histórico
    pre-audit).

Lo que este test enforza:
  A) El default leído via `_env_int("MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS",
     2)` — bumpeado de 6 (regresión: si vuelve a 6, alert fatigue).
  B) Clamp `max(1, min(hard_cap_h, 6))` presente — defensa contra valores
     extremos del operador.
  C) Stabilization-min default NO se tocó (test legacy P2-LIVE-1 lo enforza
     a 60 alineado con detector lookback) — sanity check para evitar regresión
     accidental que rompa P2-LIVE-1.

Tooltip-anchor: P2-AUDIT-3-CASCADE-HARDCAP.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


def _extract_function_body(src: str, name: str) -> str:
    """Captura el cuerpo de `def name(...)` hasta el próximo `def ` top-level."""
    pattern = re.compile(
        rf"def {re.escape(name)}\b.*?(?=^def\s)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"P2-AUDIT-3: no localicé `def {name}` en cron_tasks.py."
    return m.group(0)


def test_a_hard_cap_default_is_2_hours(cron_src: str):
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # Acepta default=2 (post-audit). Si vuelve a 6, falla con el mensaje.
    pattern = re.compile(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS["\']\s*,\s*2\s*\)'
    )
    assert pattern.search(body), (
        "P2-AUDIT-3 regresión: `MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS` "
        "ya no usa default=2. El audit 2026-05-12 lo bumpeó de 6→2 para "
        "reducir alert fatigue tras bursts transitorios post-restart. Si "
        "se requiere subir, hacerlo vía env var en el deploy (kill switch "
        "sin redeploy del binary) — NO hardcodear de vuelta a 6.\n"
        "Si la modificación es intencional, actualizar este test."
    )


def test_b_hard_cap_clamp_max_6(cron_src: str):
    """El clamp upper-bound debe ser 6 (sweet-spot pre-audit). Operador que
    setea 99 → 6, no se le permite alert fatigue extremo."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # Acepta `max(1, min(hard_cap_h, 6))` u orden equivalente sobre la
    # misma variable.
    pattern = re.compile(
        r"max\(\s*1\s*,\s*min\(\s*hard_cap_h\s*,\s*6\s*\)\s*\)"
    )
    assert pattern.search(body), (
        "P2-AUDIT-3 regresión: clamp `max(1, min(hard_cap_h, 6))` no "
        "presente. Sin clamp upper, operador puede setear "
        "MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS=99 y el alert vive 99h. "
        "Mantener el techo en 6 (default histórico pre-audit)."
    )


def test_c_stabilization_default_still_60(cron_src: str):
    """P2-LIVE-1 enforza stabilization=60 alineado con detector lookback.
    P2-AUDIT-3 NO debe haber tocado ese valor. Si este test falla,
    P2-LIVE-1 también fallará — fix los dos coherentemente."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    pattern = re.compile(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_CASCADE_STABILIZATION_MIN["\']\s*,\s*60\s*\)'
    )
    assert pattern.search(body), (
        "P2-AUDIT-3 sanity: stabilization default ya no es 60. P2-LIVE-1 "
        "lo enforza alineado con el detector lookback (1h). Si cambias "
        "uno, debes cambiar también `MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN` "
        "y revisar P2-LIVE-1."
    )


def test_d_anchor_present(cron_src: str):
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert "P2-AUDIT-3" in body, (
        "P2-AUDIT-3 regresión: anchor `P2-AUDIT-3` removido del cuerpo de "
        "_resolve_stale_scheduler_alerts. Sin el anchor, un futuro "
        "refactor no entiende por qué el default es 2 y podría devolverlo "
        "a 6 cumpliendo \"mejor observabilidad\"."
    )
