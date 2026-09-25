# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-253 · 2026-09-25] Las comidas que el revisor le SUGIERE al corrector, sin lo que el perfil prohíbe.

Batería rd252, perfil alérgico a lácteos y mariscos: el rechazo determinista «SOBREUSO DE HUEVO» mandaba reemplazar el
huevo por «pollo guisado, pescado, atún, sardina, res molida magra, queso de freír, yogur griego, habichuelas». Es una
ORDEN al corrector: a un alérgico a los lácteos le pedía queso y yogur, a uno al pescado atún y sardina, a una
vegetariana pollo y res; los otros dos textos proponían «revoltillo» a quien no come huevo y «panqueques de avena» a un
celíaco. La guarda de alérgenos lo habría cazado después, a costa de otro intento. Aquí cada sugerencia pasa por los
MISMOS escáneres que el plan (alergia, dieta, rechazo); sin perfil, la lista sale intacta (el texto de siempre, byte a
byte). tooltip-anchor: P1-PLAN-LOTE-253-SUGERENCIAS
"""
from __future__ import annotations

_SENT = frozenset({"", "ninguna", "ninguno", "none", "no", "nada", "n/a"})
# Lo que la sugerencia NOMBRA en genérico y el escáner no reconoce por su nombre.
_COMO_SE_ESCANEA = {"aves": "pollo", "lácteos permitidos": "leche"}


def compatibles(items, form_data) -> list:
    """Las sugerencias que el perfil permite, en su orden. Sin perfil (o si algo falla) devuelve todas."""
    items = [str(i) for i in (items or ())]
    if not isinstance(form_data, dict) or not form_data:
        return items
    try:
        import graph_orchestrator as go
        import rechazos
        fd = go.profile_with_free_text(form_data)
        alergias = [a for a in (fd.get("allergies") or []) if str(a).strip().lower() not in _SENT]
        dieta = fd.get("dietType") or fd.get("diet_type")
    except Exception:
        return items
    out = []
    for it in items:
        txt = _COMO_SE_ESCANEA.get(it, it)
        mini = {"days": [{"meals": [{"name": txt, "ingredients": [f"100 g de {txt}"]}]}]}
        try:
            if alergias and go._scan_allergen_violations(mini, alergias):
                continue
            if dieta and go._scan_diet_violations(mini, dieta):
                continue
            if rechazos._scan_dislike_violations(mini, form_data):
                continue
        except Exception:
            pass
        out.append(it)
    return out


def enumerar(items, ultimo: str = "") -> str:
    """«a, b, c»; con `ultimo` («y» / «o»), «a, b y c»."""
    items = list(items)
    if ultimo and len(items) > 1:
        return ", ".join(items[:-1]) + f" {ultimo} " + items[-1]
    return ", ".join(items)
