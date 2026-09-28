"""[P1-PLAN-LOTE-707 · 2026-09-28] Cuánto espera el worker de chunks al pipeline.

El pipeline planifica sus reintentos contra `GLOBAL_PIPELINE_TIMEOUT_S` (900 s en prod), pero el worker lo esperaba
sólo `CHUNK_PIPELINE_TIMEOUT_SECONDS` (600 s): el resultado se tiraba y el hilo huérfano seguía gastando IA hasta sus
900 s, porque `ThreadPoolExecutor.shutdown(wait=False)` no mata un hilo. Esperar más no cuesta recursos (el hilo corre
igual) y el latido del lock se sigue refrescando mientras se espera. Default ON; `MEALFIT_CHUNK_WAIT_COVERS_PIPELINE=false`
vuelve a la espera del knob del chunk tal cual.

tooltip-anchor: P1-PLAN-LOTE-707
"""
from __future__ import annotations


def espera_del_pipeline(timeout_s) -> int:
    """`max(timeout_s, GLOBAL_PIPELINE_TIMEOUT_S + margen)`. Nunca acorta. Fail-open: ante error, `timeout_s`."""
    try:
        base = int(timeout_s)
    except Exception:
        return timeout_s
    try:
        from knobs import _env_bool, _env_int
        if not _env_bool("MEALFIT_CHUNK_WAIT_COVERS_PIPELINE", True):
            return base
        margen = _env_int("MEALFIT_CHUNK_WAIT_MARGIN_S", 60, validator=lambda v: 0 <= v <= 600)
        import graph_orchestrator
        return max(base, int(graph_orchestrator.GLOBAL_PIPELINE_TIMEOUT_S) + int(margen))
    except Exception:
        return base
