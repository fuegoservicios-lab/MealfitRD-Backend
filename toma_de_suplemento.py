# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1141 · 2026-11-21] Una toma de suplemento es SIEMPRE su propia fila: nunca se fusiona en una comida.

Caso vivo del dueño (21-nov, 00:44): «Me tomé la creatina y la multivitaminas» 2 min después de anotar la cena. El
modelo llamó `log_consumed_meal` con meal_type «cena» y la fusión por franja (lote 982: misma franja < 30 min) convirtió
las dos tomas en correcciones de la CENA. La creatina la frenó `GUARD-SUPLEMENTO` (4 veces, hasta el corte del bucle) y
la multivitamina ENTRÓ en la cena —«DESIGNS FOR SPORT MULTI + PHYTO» no tiene ninguna palabra de la lista—; el coach
dijo «no pude dejar registradas tus tomas» con una dentro de la cena.

Dos reglas:
  · `es_toma`: el registro es un suplemento si el modelo lo dice (`suplemento=` / meal_type «suplemento»), si es ligero
    y su nombre lleva una palabra de suplemento, o si nombra un pote de SU Alacena. Entonces va como «suplemento» y no
    pasa por la fusión.
  · `nombra_pote_nuevo`: la guarda de `correct_consumed_meal` reconoce también los potes de la Alacena por su nombre.
Fail-open: si la Alacena no se puede leer, solo cuentan las otras señales.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

_TOPE_KCAL_POR_PALABRA = 60   # «proteína» en un plato de 500 kcal es comida; en una toma de 0-60 kcal, suplemento


def _plano(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    for ch in "()[]/,.;:+-·":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def _palabras(texto) -> set:
    return {w for w in _plano(texto).split() if len(w) >= 2}


def nombres_de_su_alacena(user_id: str) -> list:
    try:
        from db_core import execute_sql_query
        filas = execute_sql_query(
            # [SUPLEMENTOS-OK: lee justamente los suplementos]
            "SELECT ingredient_name FROM user_inventory WHERE user_id = %s AND kind = 'supplement'",
            (str(user_id),), fetch_all=True,
        ) or []
        return [str(f.get("ingredient_name") or "") for f in filas if isinstance(f, dict) and f.get("ingredient_name")]
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-1141] Alacena ilegible: {exc!r}")
        return []


def _nombra(nombre: str, pote: str) -> bool:
    """El nombre lleva el pote entero («DESIGNS FOR SPORT MULTI + PHYTO (Designs for Sport)» ⊇ el pote)."""
    p = _palabras(pote)
    return bool(p) and p <= _palabras(nombre)


def es_toma(user_id: str, tool_args: dict, potes: Optional[list] = None) -> bool:
    args = tool_args or {}
    if str(args.get("suplemento") or "").strip():
        return True
    try:
        from tools import _normalize_meal_type, _SUPPLEMENT_KEYWORDS
    except Exception:  # noqa: BLE001
        return False
    if _normalize_meal_type(args.get("meal_type")) == "suplemento":
        return True
    nombre = str(args.get("meal_name") or "")
    try:
        kcal = int(args.get("calories") or 0)
    except (TypeError, ValueError):
        kcal = 0
    plano = _plano(nombre)
    if kcal <= _TOPE_KCAL_POR_PALABRA and any(_plano(kw) in plano for kw in _SUPPLEMENT_KEYWORDS):
        return True
    lista = potes if potes is not None else nombres_de_su_alacena(user_id)
    return any(_nombra(nombre, p) for p in lista)


def nombra_pote_nuevo(user_id: str, nombre_nuevo: str, nombre_original: str, potes: Optional[list] = None) -> bool:
    """¿El nombre nuevo añade un pote de su Alacena que el original no tenía?"""
    lista = potes if potes is not None else nombres_de_su_alacena(user_id)
    return any(_nombra(nombre_nuevo, p) and not _nombra(nombre_original, p) for p in lista)
