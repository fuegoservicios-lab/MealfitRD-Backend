# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-237 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] El plan de emergencia pasa el escáner SSOT.

El plan de emergencia —la red que salta DESPUÉS de un rechazo crítico, a menudo por alérgeno— filtraba con su propio
vocabulario de 13 clases (`_FALLBACK_ALLERGEN_KEYWORDS`) y sus plantillas «neutrales» (tokens vacíos) llevan aguacate y
semillas: a un alérgico al aguacate o al sésamo (o a «fresa» tecleada en «Otra alergia») le tocaba «Arroz con Vegetales
y Aguacate» justo tras rechazar su plan por el alérgeno. Cada plantilla pasa además por el escáner SSOT
(`clinical_backstop_for_meal`: alérgenos con sinónimos, dieta, mercurio en embarazo) y por el de rechazos. El knob
`MEALFIT_FALLBACK_SSOT_SCAN` y el selector viven en el grafo. tooltip-anchor: P1-PLAN-LOTE-237-EMERGENCIA
"""
from __future__ import annotations


def _fallback_template_violations(tmpl, form_data) -> list:
    """Violaciones de una plantilla del fallback contra el formulario (vacía = segura)."""
    if not isinstance(form_data, dict):
        return []
    try:
        import graph_orchestrator as _go
        name, _tk, _desc, ings = tmpl
        fd = _go.profile_with_free_text(form_data)
        meal = {"name": name, "ingredients": list(ings or [])}
        _alg = [a for a in (fd.get("allergies") or []) if str(a).strip().lower() not in _go._SENTINEL_NONE_VALUES]
        out = list(_go.clinical_backstop_for_meal(meal, allergies=_alg, diet_type=fd.get("dietType"), form_data=fd) or [])
        out += [f"rechazo '{t}' en '{i}'" for _m, i, t in _go._scan_dislike_violations({"days": [{"meals": [meal]}]}, fd)]
        return out
    except Exception as _ftv_e:
        return [f"error ({type(_ftv_e).__name__})"]
