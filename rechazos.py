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


# [P1-PLAN-LOTE-258 · 2026-09-25] Un rechazo que nombra la CLASE («Pescado», «Mariscos», «fish», «lácteos») excluye la
# clase; uno que nombra un ALIMENTO («corvina», «atún», «tilapia») excluye ese alimento. Batería rd257: «no me gusta la
# corvina» escrito a mano quemó dos reintentos rechazando tilapia y bacalao — el guard expandía por clase como con una
# alergia, y la nota del revisor (lote 249) lo leía como «rechazo al pescado». Con la ALERGIA la expansión por clase es la
# prudencia clínica y no se toca; un «no me gusta» es una preferencia. tooltip-anchor: P1-PLAN-LOTE-258-RECHAZO-LITERAL
def _nombra_clase(decl, go) -> bool:
    import re
    a = go._norm_declaracion(decl)
    if not a:
        return False
    for cat in go._ALLERGEN_SYNONYMS:
        cat_n = go._norm_declaracion(cat)
        if re.search(go._patron_termino_alergeno(cat_n), a) or re.search(go._patron_termino_alergeno(a), cat_n):
            return True
        if any(go._declaracion_casa(a, al) for al in go._ALLERGEN_DECLARATION_ALIASES.get(cat, ())):
            return True
    return False


def terminos_de_rechazo(form_data) -> list:
    """Lo que el guard de rechazos busca en el plato: la clase entera si el rechazo la nombra; si no, el alimento
    literal (con su nombre canónico si viene en otro idioma). Puro; nunca lanza."""
    try:
        import graph_orchestrator as go
        out = set()
        for d in _dislike_declarations(form_data):
            if _nombra_clase(d, go):
                out |= set(go._expand_allergy_declarations([d]))
                continue
            a = go._norm_declaracion(d)
            if a:
                out.add(a)
                try:
                    out |= {go._norm_declaracion(c) for c in go._nombres_canonicos_de_alimento(a)}
                except Exception:
                    pass
        return sorted(t for t in out if t)
    except Exception:
        return []


def _scan_dislike_violations(plan: dict, form_data) -> list:
    """(comida, ingrediente, término) de cada alimento rechazado en los ingredientes del plan."""
    terminos = terminos_de_rechazo(form_data)
    if not terminos or not isinstance(plan, dict):
        return []
    try:
        import graph_orchestrator as _go
        return _go._scan_allergen_violations(plan, [], terminos=terminos)
    except Exception:
        return []
