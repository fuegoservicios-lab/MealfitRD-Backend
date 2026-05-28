"""[P0-REGEN-BILLING · 2026-05-24] Regression: `/regen-degraded` debe
loggear api_usage 1×N (no 1×1) — un row por cada chunk re-encolado.

Bug original (audit production-readiness 2026-05-24):
    [`backend/routers/plans.py:9657-9661`] el endpoint `/regen-degraded`
    invocaba `log_api_usage(verified_user_id, "regen_degraded")` **una sola
    vez** por request, aunque `regenerated` (N chunks re-queued) puede ser
    ≥5. Cada chunk re-encolado consume un LLM call independiente cuando el
    worker lo procese — el undercount permitía:

      a) Revenue leak: user regeneraba N chunks pagando 1 cargo (cap
         mensual gratis=15 / basic=50 / plus=200 no reflejaba el costo
         real cuando N=5+ chunks degradados se re-encolaban juntos).
      b) Quota bypass: attacker con plan dead-lettered de muchos chunks
         podía amplificar su quota Nx llamando este endpoint en lugar
         de regenerar individualmente vía /retry-chunk (que sí cobra
         per-call).

Fix:
    Loop `for _ in range(regenerated): log_api_usage(...)` antes del
    response. La función `log_api_usage(user_id, endpoint)` inserta un row
    en `api_usage` cada llamada (`db_profiles.py:458`); el contador
    mensual queda con N rows.

Este test parser-based escanea el source del endpoint
(`api_regen_degraded_chunks`) y exige que:
  1. Exista un `for _ in range(regenerated)` (o equivalente con `regenerated`
     en el `range(...)`) que envuelva la llamada a `log_api_usage`.
  2. La llamada NO esté fuera del loop (regresión sentinel).
  3. El marker `[P0-REGEN-BILLING ...]` esté presente (anchor para
     futuros refactors — si alguien renombra el endpoint, este test
     falla loud).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"


def _read_regen_endpoint_body() -> str:
    """Extrae el cuerpo de `api_regen_degraded_chunks` desde `routers/plans.py`
    via regex (sin importar el módulo, que dispara FastAPI deps + DB init).

    El cuerpo va desde `def api_regen_degraded_chunks(...):` hasta el
    siguiente `^@router.` o EOF.
    """
    text = _PLANS_PY.read_text(encoding="utf-8")
    # Capturar desde `def api_regen_degraded_chunks` hasta el siguiente
    # decorador top-level o EOF.
    m = re.search(
        r"^def api_regen_degraded_chunks\b.*?(?=^@router\.|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m is not None, (
        "No se encontró `def api_regen_degraded_chunks` en "
        f"{_PLANS_PY}. ¿Fue renombrado? Si es intencional, actualizar "
        "este test."
    )
    return m.group(0)


def test_log_api_usage_inside_loop_over_regenerated():
    """El endpoint debe loggear `log_api_usage` dentro de un `for` que
    itere sobre `range(regenerated)` (o equivalente) — 1 row por chunk
    re-encolado, NO 1 row total.

    Pattern aceptado:
        for _ in range(regenerated):
            ...
            log_api_usage(verified_user_id, "regen_degraded")
            ...

    Si alguien revierte al patrón legacy (`log_api_usage` fuera del loop),
    este test falla loud.
    """
    body = _read_regen_endpoint_body()
    # Encontrar la posición del `log_api_usage("regen_degraded")`
    log_call_match = re.search(
        r'log_api_usage\([^)]*"regen_degraded"[^)]*\)', body
    )
    assert log_call_match is not None, (
        "No se encontró `log_api_usage(..., \"regen_degraded\")` en el "
        "endpoint `api_regen_degraded_chunks`. ¿Se eliminó el audit? "
        "Revisar P0-REGEN-BILLING (revenue leak + quota bypass)."
    )
    # Buscar un `for ... in range(regenerated)` antes de la llamada
    # (en la misma función). Debe estar a una profundidad mayor que el
    # `log_api_usage`.
    pre_call_block = body[: log_call_match.start()]
    for_match = re.search(
        r"for\s+\w+\s+in\s+range\(\s*regenerated\s*\)\s*:",
        pre_call_block,
    )
    assert for_match is not None, (
        "El `log_api_usage('regen_degraded')` NO está dentro de un "
        "`for _ in range(regenerated):`. Pre-fix el endpoint loggeaba "
        "1 vez aunque N chunks se re-encolaban → undercount billing + "
        "quota bypass. Restaurar el loop (ancla [P0-REGEN-BILLING])."
    )


def test_marker_anchor_present_in_endpoint():
    """El comentario `[P0-REGEN-BILLING ...]` debe estar presente en el
    endpoint — anchor textual para futuros refactors.

    Sin el anchor, un refactor cosmético podría romper el loop sin que
    el dev entienda POR QUÉ está ahí (history del bug se pierde).
    """
    body = _read_regen_endpoint_body()
    assert "[P0-REGEN-BILLING" in body, (
        "El marker `[P0-REGEN-BILLING ...]` desapareció del endpoint "
        "`api_regen_degraded_chunks`. Restaurar el comentario que "
        "explica POR QUÉ el `log_api_usage` debe estar dentro del loop."
    )


def test_no_log_api_usage_outside_loop_regression_sentinel():
    """Sentinel anti-regresión: no debe haber un `log_api_usage('regen_degraded')`
    en el endpoint que NO esté precedido por un `for ... in range(regenerated)`.

    Si alguien añade una segunda llamada (e.g. para auditoría adicional)
    fuera del loop, también falla — protege el contrato 1:N estrictamente.
    """
    body = _read_regen_endpoint_body()
    log_calls = list(re.finditer(
        r'log_api_usage\([^)]*"regen_degraded"[^)]*\)', body
    ))
    assert len(log_calls) >= 1, "No hay llamadas a log_api_usage en el endpoint"
    for call in log_calls:
        # Para cada llamada, buscar el `for` más cercano antes.
        pre = body[: call.start()]
        last_for = None
        for m in re.finditer(
            r"for\s+\w+\s+in\s+range\(\s*regenerated\s*\)\s*:",
            pre,
        ):
            last_for = m
        assert last_for is not None, (
            f"Llamada a log_api_usage('regen_degraded') en pos {call.start()} "
            "NO está dentro de un `for ... in range(regenerated)`. "
            "Cada llamada DEBE estar bajo el loop — contrato 1:N estricto."
        )
