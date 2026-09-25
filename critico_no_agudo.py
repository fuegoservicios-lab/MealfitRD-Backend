# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-230 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] G18 para toda condición.

`docs/clinical_enforcement_decisions.md` (G18): SOLO los críticos AGUDOS van al fallback matemático; lo demás se entrega
con banner ámbar. Pero el revisor LLM etiqueta «critical» preocupaciones crónicas que su propio reporte de fact-check
extrapola, y el gate que las degradaba existía solo para DM2 (glucémico) y bariátrico. Batería real del 25-sep: colesterol
alto («6½ yemas en el día 2»), gastritis («ajo, cebolla, vinagre, piña»), hipotiroidismo («yuca y casabe repetidos») →
dos críticos seguidos y el usuario recibía «Pollo y Arroz / Pescado y Batata», 3 comidas, sin sus básicos ni suplementos.
Un crítico SIN ninguna marca aguda (alérgeno, dieta, crudo, embarazo/lactancia, mercurio, celíaca, interacción
farmacológica, renal/potasio/fósforo, hipoglucemia, tiramina) baja a «high»: reintenta con la directiva y, agotados los
intentos, se entrega el plan REAL con banner. El knob `MEALFIT_NON_ACUTE_CRITICAL_SOFT_REJECT` vive en el grafo.
tooltip-anchor: P1-PLAN-LOTE-230-CRITICO-NO-AGUDO
"""
from __future__ import annotations

import unicodedata

# Marcas agudas que se SUMAN a `graph_orchestrator._NON_GLYCEMIC_SAFETY_MARKERS` (el SSOT de los otros dos gates).
_ACUTE_EXTRA_MARKERS = (
    "tiramina", "imao", "crisis hipertensiva", "hipoglucem", "insulina", "sulfonilurea", "litio",
    "anticoagul", "sangrado", "hiperkalem", "hiperpotasem", "atragant", "asfixia",
    "arritmia", "dialisis",   # [P1-PLAN-LOTE-261] con el potasio renal, lo agudo es esto
)

# [P1-PLAN-LOTE-261 · 2026-09-25] G20 (`docs/clinical_enforcement_decisions.md`): en ERC el potasio y el fósforo son
# ORIENTATIVOS (dependen de estadio, diálisis y analíticas; el panel los reporta y el nefrólogo decide). El revisor de IA
# los rechazaba como CRÍTICO («el plan acumula fuentes importantes de potasio en el contexto de enfermedad renal: yautía,
# yuca…») y «renal»/«potasio»/«fósforo» contaban como marca aguda: un reintento y, si el último intento repetía, el plan
# de EMERGENCIA (batería rd260, renal + gota). Un crítico cuyas ÚNICAS marcas son esas tres y que habla de potasio o
# fósforo baja a «high»: reintenta y se entrega el plan real con banner. El techo renal de PROTEÍNA sigue agudo (tiene su
# propio gate duro) y la hiperpotasemia, la arritmia o la diálisis también. tooltip-anchor: P1-PLAN-LOTE-261-KP-RENAL
_SOLO_KP_RENAL = frozenset({"renal", "potasio", "fosforo"})


def _solo_kp_renal(t: str, marcas) -> bool:
    presentes = {m for m in marcas if m in t}
    return bool(presentes) and presentes <= _SOLO_KP_RENAL and ("potasio" in t or "fosforo" in t)


def _sin_acentos(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def _critical_is_non_acute(issues) -> bool:
    """True si NINGUNA razón del crítico lleva una marca de peligro agudo (acentos fuera). Vacío → False."""
    if not issues:
        return False
    try:
        import graph_orchestrator as _go
        marcas = tuple(_go._NON_GLYCEMIC_SAFETY_MARKERS) + _ACUTE_EXTRA_MARKERS
    except Exception:
        return False
    for raw in issues:
        t = _sin_acentos(str(raw).lower())
        if any(m in t for m in marcas):
            if _solo_kp_renal(t, marcas):
                continue          # [P1-PLAN-LOTE-261] G20: K/P renal es orientativo
            return False
    return True
