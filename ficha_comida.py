# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-720 · 2026-09-28] La ficha de un plato registrado en el diario.

El dueño: «¿no sería interesante que se pudiera ver más detalles de la info de los platos que hemos agregado del
contador de calorías?». La fila del contador pintaba nombre, franja y kcal; el resto (ingredientes, de dónde vino)
o no viajaba o no se guardaba. Spec: docs/superpowers/specs/2026-09-28-ficha-plato-registrado-design.md.

Aquí vive lo que NO es SQL ni HTTP, para que se pueda probar sin base:

  · `origen_de_comida` — el vocabulario de `consumed_meals.source`. Lo desconocido es None, nunca un error: es una
    etiqueta de la ficha, y registrar una comida no puede depender de ella (por eso tampoco hay CHECK en la base).
  · `plan_ref_limpio` — las coordenadas de «Me lo comí», y solo ellas.
  · `desglose_de_ingredientes` — los renglones de la comida, con sus kcal SOLO cuando cuadran con la comida.
  · `ficha_de_comida` — la respuesta de `GET /api/diary/meal/{meal_id}`.

La foto NO pasa por aquí: vive solo en el dispositivo (la Política de Privacidad publicada promete no retenerla).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# El vocabulario del ledger de la Nevera (`inventory_consumption_events.source`: photo, manual, plan_meal, chat) más
# los dos caminos que el ledger no ve porque no descuentan: el estimado por texto libre y «Registrar otra vez».
ORIGENES_DE_COMIDA = frozenset({"photo", "manual", "estimate", "plan_meal", "chat", "repeat"})

# Cuánto pueden separarse la suma de los renglones y las kcal guardadas de la comida antes de callar las kcal por
# renglón. Las dos cifras salen de BASES distintas (la IA estimó el total de una foto; el catálogo, cada renglón):
# si no cuadran, pintar las dos es enseñar dos verdades que no suman.
TOLERANCIA_DESGLOSE = 0.25

_CLAVES_PLAN_REF = ("plan_id", "day_index", "meal_index")


def origen_de_comida(source: Any) -> Optional[str]:
    """`source` normalizado al vocabulario, o None si no es uno de sus valores."""
    s = str(source or "").strip().lower()
    return s if s in ORIGENES_DE_COMIDA else None


def plan_ref_limpio(plan_ref: Any) -> Optional[dict]:
    """`{plan_id, day_index, meal_index}` de «Me lo comí», o None. Nada más entra: la fila no es un sitio para
    guardar lo que el llamador traiga."""
    if not isinstance(plan_ref, dict):
        return None
    plan_id = str(plan_ref.get("plan_id") or "").strip()
    if not plan_id or len(plan_id) > 64:
        return None
    out = {"plan_id": plan_id}
    for clave in ("day_index", "meal_index"):
        try:
            v = int(plan_ref.get(clave))
        except (TypeError, ValueError):
            return None
        if v < 0 or v > 400:
            return None
        out[clave] = v
    return out


def es_columna_inexistente(exc: BaseException) -> bool:
    """¿El error es «la columna no existe» (SQLSTATE 42703)? Es lo que devuelve un INSERT con `source`/`plan_ref`
    contra una base a la que aún no llegó la migración del lote 720."""
    if getattr(exc, "sqlstate", None) == "42703":
        return True
    return type(exc).__name__ == "UndefinedColumn"


def desglose_de_ingredientes(ingredients: Any, kcal_comida: Any, db: Any) -> dict:
    """Los renglones de una comida para la ficha.

    Devuelve `{"lineas": [{"texto", "gramos"?, "kcal"?}], "con_kcal": bool}`. Las kcal por renglón salen SOLO si
    todos los renglones resuelven contra el catálogo y su suma queda a ±`TOLERANCIA_DESGLOSE` de las kcal guardadas
    de la comida; si no, cada renglón va sin cifras (el texto ya dice su cantidad: «150 g de Pechuga de pollo»).
    """
    if not isinstance(ingredients, list):
        return {"lineas": [], "con_kcal": False}
    textos = [str(x).strip() for x in ingredients if isinstance(x, str) and str(x).strip()]
    if not textos:
        return {"lineas": [], "con_kcal": False}

    resueltos = []
    if db is not None and hasattr(db, "macros_from_ingredient_string"):
        for texto in textos:
            try:
                m = db.macros_from_ingredient_string(texto)
            except Exception:
                m = None
            resueltos.append(m if isinstance(m, dict) and m.get("kcal") is not None else None)
    else:
        resueltos = [None] * len(textos)

    con_kcal = False
    try:
        total = float(kcal_comida or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if total > 0 and resueltos and all(r is not None for r in resueltos):
        suma = sum(float(r["kcal"]) for r in resueltos)
        con_kcal = abs(suma - total) <= TOLERANCIA_DESGLOSE * total

    lineas = []
    for texto, r in zip(textos, resueltos):
        linea = {"texto": texto}
        if con_kcal and r is not None:
            linea["kcal"] = int(round(float(r["kcal"])))
            try:
                linea["gramos"] = int(round(float(r.get("grams"))))
            except (TypeError, ValueError):
                pass
        lineas.append(linea)
    return {"lineas": lineas, "con_kcal": con_kcal}


def _iso(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return v.isoformat()
    except AttributeError:
        return str(v)


def ficha_de_comida(row: dict, db: Any = None) -> dict:
    """La respuesta de `GET /api/diary/meal/{meal_id}` a partir de la fila (ya filtrada por dueño)."""
    kcal = row.get("calories") or 0
    return {
        "id": str(row.get("id")),
        "meal_name": row.get("meal_name") or "",
        "meal_type": row.get("meal_type"),
        "calories": kcal,
        "protein": row.get("protein") or 0,
        "carbs": row.get("carbs") or 0,
        "healthy_fats": row.get("healthy_fats") or 0,
        "consumed_at": _iso(row.get("consumed_at")),
        "created_at": _iso(row.get("created_at")),
        "source": origen_de_comida(row.get("source")),
        "plan_ref": plan_ref_limpio(row.get("plan_ref")),
        "ingredientes": desglose_de_ingredientes(row.get("ingredients"), kcal, db),
    }
