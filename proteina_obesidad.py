# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-211 · 2026-09-24] La proteína en obesidad se mide sobre el peso AJUSTADO aunque no des tu % de grasa.

El objetivo de proteína es un % de las calorías (pérdida de grasa: 35 %) con techo de 2,2 g/kg (`calculate_macros`). Desde
P2-PROTEIN-CEILING-ADJ-WEIGHT el techo usa el peso AJUSTADO (masa magra + 25 % del resto) cuando la grasa pasa del 30 % —
pero sólo si el usuario ESCRIBE su % de grasa, que casi nadie sabe. Sin él, el techo va sobre el peso TOTAL y no muerde:
una mujer de 88 kg y 160 cm con DM2 (IMC 34,4) recibía 153 g/día (35 % de 1.750 kcal), y con ese objetivo el día no
tiene margen: el cerrador mete 300 g de edamame y los platos quedan con su base en 15–25 g (batería rd20, 23-sep).

Aquí:
  · sin % de grasa y con IMC ≥ 30, la grasa se ESTIMA con Deurenberg (1991, validada en adultos):
        % grasa = 1,20 × IMC + 0,23 × edad − 10,8 × sexo(hombre=1) − 5,4
    sólo para el TECHO de proteína — el metabolismo basal sigue con Mifflin-St Jeor (una estimación no reemplaza una
    medición en la ecuación de Katch-McArdle). Mismo umbral (> 30 %) y misma fórmula de peso ajustado que ya existían.
    El caso de arriba: grasa estimada ≈ 44 % (a 45 años) ⇒ peso ajustado ≈ 60 kg ⇒ techo ≈ 132 g.
  · con diabetes, las calorías que libera el techo van a GRASA y no a carbohidratos (hoy iban a carbohidratos: subía la
    carga glucémica justo en quien no la tolera). Mismo criterio que ya aplican el tope renal+diabetes y el bariátrico.
Decisión del dueño (24-sep): «soluciónalo». Knobs `MEALFIT_PROTEIN_OBESITY_BMI_ESTIMATE` y
`MEALFIT_DM2_FREED_KCAL_TO_FAT` (los dos True).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_HOMBRE = ("male", "masculino", "masculina", "m", "hombre")   # el mismo conjunto que `calculate_bmr`
IMC_OBESIDAD = 30.0


def _knob(nombre: str) -> bool:
    try:
        from knobs import _env_bool
        return _env_bool(nombre, True)
    except Exception:
        return True


def estimar_grasa(peso_kg: float, altura_cm: float, edad: int, hombre: bool) -> Optional[float]:
    """% de grasa por Deurenberg. None si faltan datos o salen absurdos."""
    try:
        peso, altura, edad = float(peso_kg), float(altura_cm), int(edad)
    except (TypeError, ValueError):
        return None
    if not (30 <= peso <= 300 and 100 <= altura <= 250 and 18 <= edad <= 100):
        return None
    imc = peso / ((altura / 100.0) ** 2)
    grasa = 1.20 * imc + 0.23 * edad - 10.8 * (1 if hombre else 0) - 5.4
    return round(max(5.0, min(60.0, grasa)), 1)


def grasa_para_techo(body_fat, peso_kg, form_data) -> Optional[float]:
    """La grasa que usa el TECHO de proteína: la del usuario si la dio; si no, y el IMC ≥ 30, la estimada; si no, None
    (conducta previa: peso total)."""
    if body_fat:
        return body_fat
    if not _knob("MEALFIT_PROTEIN_OBESITY_BMI_ESTIMATE"):
        return None
    fd = form_data if isinstance(form_data, dict) else {}
    try:
        altura = float(fd.get("height"))
        edad = int(float(fd.get("age")))
        peso = float(peso_kg)
    except (TypeError, ValueError):
        return None
    if peso / ((altura / 100.0) ** 2) < IMC_OBESIDAD:
        return None
    hombre = str(fd.get("gender") or "").strip().lower() in _HOMBRE
    grasa = estimar_grasa(peso, altura, edad, hombre)
    if grasa is not None:
        logger.info(f"🩺 [P1-PLAN-LOTE-211] IMC ≥ 30 sin % de grasa: estimada {grasa} % (Deurenberg) para el techo de "
                    f"proteína")
    return grasa


def destino_liberado(form_data) -> str:
    """«fats» con diabetes (las calorías que libera el techo de proteína no van a subir la carga glucémica); si no,
    «carbs» (conducta previa)."""
    if not _knob("MEALFIT_DM2_FREED_KCAL_TO_FAT"):
        return "carbs"
    fd = form_data if isinstance(form_data, dict) else {}
    try:
        from constants import DIABETES_CONDITION_TERMS, strip_accents
        blob = strip_accents(" ".join(str(x) for x in (fd.get("medicalConditions") or fd.get("medical_conditions") or []))
                             + " " + str(fd.get("otherConditions") or "")).lower()
    except Exception:
        return "carbs"
    return "fats" if any(t in blob for t in DIABETES_CONDITION_TERMS) else "carbs"
