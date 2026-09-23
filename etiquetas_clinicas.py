# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-173 · 2026-09-23] Lo que el revisor médico tiene que LEER en el plan para aprobarlo, por condición.

El revisor médico (un LLM) rechaza con severidad CRÍTICA cuando el plan no DICE lo que la condición exige, aunque el
alimento sea el correcto: cada rechazo cuesta un intento entero (2-3 min y su gasto) y, si reincide, el usuario
recibe el plan de emergencia. Batería real del 23-sep:
  · embarazo — «pescado sin especie», «queso/leche sin pasteurizar» (→ `embarazo_seguro`, lote 172-173);
  · hipertensión — «queso fresco y atún sin especificar versiones bajas en sodio».

Este módulo es la puerta única: `etiquetar(plan, form_data)` aplica, según las reglas activas del perfil
(`condition_rules.detect_active_rules`), las etiquetas de cada condición. Se llama desde la sustitución clínica
(`graph_orchestrator._apply_condition_substitutions`) y al final del escudo (`db_plans._finalize_plan_data_for_insert`),
porque los cerradores añaden alimentos DESPUÉS de la sustitución. Idempotente. Knob `MEALFIT_CLINICAL_LABELS` (True).
tooltip-anchor: P1-PLAN-LOTE-173-ETIQUETAS-CLINICAS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# HTA: el queso fresco/blando y el pescado en lata, «bajo en sodio». Los curados (parmesano, cheddar…) ya son otra
# conversación (el revisor pide moderarlos, no etiquetarlos).
_QUESO_HTA = re.compile(r"\b(?:queso(?!\s+(?:cheddar|parmesano|gouda|provolone|edam|de\s+papa|de\s+bola|amarillo|suizo|"
                        r"manchego|curado|azul))|ricotta|cottage|reques[oó]n|mozzarella)\b[^,;()]*", re.IGNORECASE)
_LATA_HTA = re.compile(r"\b(?:at[uú]n|sardinas?)\b[^,;()]*", re.IGNORECASE)
_YA_BAJO = re.compile(r"bajo\s+en\s+sodio|baja\s+en\s+sodio|sin\s+sal|sin\s+sodio|reducid[oa]\s+en\s+sodio", re.IGNORECASE)


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CLINICAL_LABELS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _reglas(form_data) -> set:
    try:
        from condition_rules import detect_active_rules
        return {getattr(r, "id", "") for r in detect_active_rules(form_data or {})}
    except Exception:                                                          # noqa: BLE001
        return set()


def _sufijo(s: str, rx, sufijo: str) -> str:
    if _YA_BAJO.search(s):
        return s
    m = rx.search(s)
    if not m:
        return s
    fin = m.end()
    while fin > m.start() and s[fin - 1] == " ":
        fin -= 1
    return s[:fin] + sufijo + s[fin:]


def _linea_hta(s: str) -> str:
    s = _sufijo(s, _QUESO_HTA, " bajo en sodio")
    return _sufijo(s, _LATA_HTA, " bajas en sodio" if re.search(r"\bsardinas\b", s, re.IGNORECASE) else " bajo en sodio")


def _etiquetar_hta(plan: dict) -> int:
    tocadas = 0
    for d in plan.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict):
                continue
            cambio = False
            for campo in ("ingredients", "ingredients_raw"):
                lineas = m.get(campo)
                if not isinstance(lineas, list):
                    continue
                nuevas = [(_linea_hta(x) if isinstance(x, str) else x) for x in lineas]
                if nuevas != lineas:
                    m[campo] = nuevas
                    cambio = True
            if cambio:
                m.pop("_display", None)
                m["_hta_labels"] = True
                tocadas += 1
    return tocadas


def etiquetar(plan: dict, form_data) -> int:
    """Devuelve cuántas comidas tocó (sumando condiciones). Muta `plan`."""
    if not (enabled() and isinstance(plan, dict)):
        return 0
    reglas = _reglas(form_data)
    if not reglas:
        return 0
    n = 0
    if "pregnancy" in reglas:
        try:
            import embarazo_seguro
            n += embarazo_seguro.etiquetar(plan, form_data)
        except Exception as e:                                                 # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-173] etiquetas de embarazo no-op: {type(e).__name__}: {e}")
    if "hta" in reglas or "renal" in reglas:
        n_hta = _etiquetar_hta(plan)
        if n_hta:
            logger.info(f"🧂 [P1-PLAN-LOTE-173] HTA: «bajo en sodio» escrito en {n_hta} comida(s)")
        n += n_hta
    return n
