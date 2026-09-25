# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-257 · 2026-09-25] El techo renal de proteína se cumple en lo que se ENTREGA.

Batería rd252, perfil renal + gota (techo KDIGO 0,8 g/kg = 60 g/día): el plan del modelo pasaba el techo, el recorte por
comida «no convergía» y se entregaba el plan de EMERGENCIA… que también lo pasaba (96/75/75 g). Réplica local pase a pase:

  · el recorte (`_trim_day_protein_to_ceiling`) solo escala las líneas de proteína DOMINANTE — carne y pescado. El huevo
    (su grasa pesa más en kcal) y los lácteos no cuentan, así que 4 huevos seguían enteros y la pechuga bajaba a 35 g
    («Pollo y Arroz» con 35 g de pollo);
  · después, `identidad_plato` devolvía el alimento del nombre a su ración («↑35→90 g de Pechuga de pollo») mirando solo
    el margen de kcal y de grasa del día: el techo de proteína no existía para él.

Aquí: qué es recortable bajo un techo renal (lo dominante + huevo, claras y lácteos) y cuánto margen de proteína le queda
a un día con techo. tooltip-anchor: P1-PLAN-LOTE-257-TECHO-RENAL
"""
from __future__ import annotations

import re
import unicodedata

_HUEVO_LACTEO_RX = re.compile(
    r"\b(?:huevos?|claras?|yemas?|queso|quesos|yogur|yogurt|yogures|requeson|cottage|kefir|ricotta|"
    r"leche(?! de (?:coco|almendra|avena|soya|soja|arroz|marañon|maranon)))\b")


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def recortable(linea, db, incluye_huevo_lacteo: bool = False) -> bool:
    """Lo que el recorte de proteína puede escalar: lo proteína-dominante y, bajo un techo renal, huevo y lácteos."""
    import graph_orchestrator as go
    if go._ingredient_is_protein_dominant(str(linea), db):
        return True
    return bool(incluye_huevo_lacteo and _HUEVO_LACTEO_RX.search(_sa(linea)))


def techo_renal(plan_data) -> float:
    """Gramos del techo renal de proteína del plan; 0 si no hay techo aplicado."""
    cap = (plan_data or {}).get("renal_protein_cap") if isinstance(plan_data, dict) else None
    if not isinstance(cap, dict) or not cap.get("applied"):
        return 0.0
    try:
        return float(cap.get("protein_g") or 0)
    except (TypeError, ValueError):
        return 0.0
