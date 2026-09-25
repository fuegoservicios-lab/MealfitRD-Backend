# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-232 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] Los rechazos («no me gusta») tienen guard.

Los RECHAZOS del formulario («¿Qué alimentos no te gustan?» + «Otro») eran la única respuesta sin guard determinista:
alergia, dieta y mercurio tienen el suyo; el rechazo dependía de que el LLM lo respetara y de que el revisor LLM lo viera.
Batería real del 25-sep: el tercer intento de «no me gusta el pescado» traía «Bowl ligero de atún y bulgur con mango»
(200 g de atún en agua) y solo lo paró el revisor. Mismo escáner que el de alérgenos (sinónimos, plurales, límite de
palabra, solo ingredientes) + el texto libre + unos sinónimos propios de lo que no es un alérgeno («Hongos» ⇒
champiñones). El knob `MEALFIT_DISLIKE_HARD_GUARD` y el guard del revisor viven en el grafo.
tooltip-anchor: P1-PLAN-LOTE-232-RECHAZOS
"""
from __future__ import annotations

_DISLIKE_EXTRA_TERMS = {
    "hongos": ("hongo", "champinon", "champinones", "seta", "setas", "portobello", "shiitake"),
    "hongo": ("hongos", "champinon", "champinones", "seta", "setas", "portobello", "shiitake"),
    "cilantro": ("cilantrico",),
    "aguacate": ("guacamole",),
}


def _dislike_declarations(form_data) -> list:
    """Rechazos declarados (chips + «Otro», sin centinela) + sinónimos propios. Puro; nunca lanza."""
    try:
        import graph_orchestrator as _go
        from constants import strip_accents as _sa_dl
        fd = _go.profile_with_free_text(form_data) if isinstance(form_data, dict) else {}
        decl = [str(d).strip() for d in (fd.get("dislikes") or [])
                if str(d).strip() and str(d).strip().lower() not in _go._SENTINEL_NONE_VALUES]
        extra = [t for d in decl for t in _DISLIKE_EXTRA_TERMS.get(_sa_dl(d.lower()), ())]
        return decl + extra
    except Exception:
        return []


def _scan_dislike_violations(plan: dict, form_data) -> list:
    """(comida, ingrediente, término) de cada alimento rechazado en los ingredientes del plan."""
    decl = _dislike_declarations(form_data)
    if not decl or not isinstance(plan, dict):
        return []
    try:
        import graph_orchestrator as _go
        return _go._scan_allergen_violations(plan, decl)
    except Exception:
        return []
