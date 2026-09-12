"""[P1-PLAN-LOTE-15 · 2026-09-12] Atribución del coste LLM al PLAN (y usuario) que lo gasta.

Por qué existe: en la cola (el 100 % de los planes desde el 09-04) el placeholder YA tiene id cuando corre el pipeline,
así que el «canje `corr` → `plan_id`» de P1-COST-ATTRIBUTION —pensado para el SSE, donde el id nacía DESPUÉS de generar—
nunca dispara. Medido el 09-12: 0 de 116 filas de `day_generator` con `plan_id` y 0 con `corr`; `swap_meal` 0 de 117 con
`user_id`. Sin esto no hay coste ni latencia por plan entregado, que es lo que pide el canario de ARQ27-P1-06.

Cómo: un ContextVar con el plan (`plan_id_var`), fijado por quien CONOCE el plan antes de gastar —el worker de chunks,
`/swap-meal` (sólo el plan propio) y `/regenerate-day`— y leído por `graph_orchestrator._emit_llm_usage_event_best_effort`.
`user_id_var` sigue viviendo en `graph_orchestrator` (lo usa el router de modelos); aquí sólo se fija junto al plan.

Vive en su módulo y no en `graph_orchestrator.py` porque ese fichero está CONGELADO por tamaño
(`test_p3_shopping_projection_pkg.py`): extraer, no subir el tope. Default `None` → conducta anterior.
tooltip-anchor: plan_id_var
"""
from __future__ import annotations

import contextvars
from typing import Optional

#: Plan al que atribuir las filas de `llm_usage_events` emitidas en este contexto. None = sin atribuir (como antes).
plan_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("plan_id", default=None)


def set_llm_attribution(user_id=None, plan_id=None) -> list:
    """Fija usuario y plan para las filas de coste que emita este contexto.

    Devuelve los tokens para `reset_llm_attribution`: el thread del pool de chunks se reutiliza y un valor stale
    atribuiría el coste del siguiente chunk al plan anterior. Best-effort: nunca revienta al caller.
    """
    toks = []
    try:
        if user_id:
            from graph_orchestrator import user_id_var  # lazy: graph_orchestrator ya está cargado cuando alguien gasta LLM
            toks.append((user_id_var, user_id_var.set(str(user_id))))
        if plan_id:
            toks.append((plan_id_var, plan_id_var.set(str(plan_id))))
    except Exception:
        pass
    return toks


def reset_llm_attribution(toks) -> None:
    """Deshace `set_llm_attribution` (un token de otro contexto no revienta: best-effort)."""
    for var, tok in reversed(list(toks or [])):
        try:
            var.reset(tok)
        except Exception:
            pass


__all__ = ["plan_id_var", "set_llm_attribution", "reset_llm_attribution"]
