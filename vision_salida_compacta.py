# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1046 · A-12 · 2026-11-16] Salida COMPACTA del escáner de platos: los mismos datos, menos tokens.

Medido el 16-nov (12 fotos del dueño, gemini-3.8-flash): 8,3 s p50 y ~900 tokens de salida; sin razonamiento casi
igual (7,8 s): el tiempo es ESCRIBIR el JSON. En una salida típica las `dudas` son el 60 % de los caracteres —cada
opción repite `supuesta`, `nombre_plato` (casi siempre vacío) y 4 macros con nombre largo— y los `items` otro 20 %.

Aquí el modelo escribe claves cortas y las macros como lista `[kcal, proteína, carbos, grasa]`; `expandir()` lo
devuelve EXACTAMENTE con la forma de `_MealVisionResult.model_dump()`, así que `_coerce_meal_scan` y todo lo de
después no cambian. Knob `MEALFIT_VISION_SALIDA_COMPACTA` (False por defecto: se enciende tras medir).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from food_label_nutrients import FoodLabel
from knobs import _env_bool
from supplement_labels import SupplementFacts

_MACROS = ("calories", "protein", "carbs", "healthy_fats")


def activo() -> bool:
    return _env_bool("MEALFIT_VISION_SALIDA_COMPACTA", False)


class _CItem(BaseModel):
    n: str = Field(default="", description="name")
    q: float = Field(default=1.0, description="quantity")
    u: str = Field(default="unidad", description="unit")
    m: list[float] = Field(default_factory=list, description="[calories, protein, carbs, healthy_fats] de ESTE componente; [] en 'items'")


class _COpcion(BaseModel):
    t: str = Field(default="", description="texto")
    s: bool = Field(default=False, description="supuesta")
    n: str = Field(default="", description="nombre_plato (vacío si no cambia)")
    a: list[float] = Field(default_factory=list, description="ajuste del plato entero [calories, protein, carbs, healthy_fats]")


class _CDuda(BaseModel):
    o: str = Field(default="", description="sobre: el name EXACTO del item")
    p: str = Field(default="", description="pregunta")
    x: list[_COpcion] = Field(default_factory=list, description="opciones")


class ResultadoCompacto(BaseModel):
    """Espejo compacto de `_MealVisionResult` (vision_agent)."""
    k: str = Field(description="photo_kind: plato | items | etiqueta | otro")
    f: bool = Field(description="is_food")
    nm: str = Field(default="", description="meal_name")
    d: str = Field(default="", description="description (inventario de componentes)")
    m: list[float] = Field(default_factory=list, description="totales [calories, protein, carbs, healthy_fats]")
    i: list[_CItem] = Field(default_factory=list, description="items")
    q: list[_CDuda] = Field(default_factory=list, description="dudas")
    food_facts: Optional[FoodLabel] = None
    supplement_facts: Optional[SupplementFacts] = None


LEYENDA = (
    " FORMATO DE SALIDA COMPACTO (obligatorio; mismas reglas de arriba, solo cambian los nombres de las claves): "
    "k=photo_kind, f=is_food, nm=meal_name, d=description, m=[calories,protein,carbs,healthy_fats] del total, "
    "i=items con n=name, q=quantity, u=unit y m=[calories,protein,carbs,healthy_fats] de ese componente "
    "(m=[] en photo_kind 'items'); q=dudas con o=sobre (el n EXACTO del item), p=pregunta y x=opciones con "
    "t=texto, s=supuesta, n=nombre_plato (omitelo o vacio si no cambia) y a=ajuste [calories,protein,carbs,healthy_fats] "
    "(a=[0,0,0,0] en la supuesta). food_facts y supplement_facts igual que antes. Sin espacios ni texto fuera del JSON."
)


def _macros(lista: Any) -> dict:
    vals = list(lista or [])[:4] if isinstance(lista, (list, tuple)) else []
    vals += [0.0] * (4 - len(vals))
    out = {}
    for k, v in zip(_MACROS, vals):
        try:
            out[k] = float(v or 0)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def expandir(r: Any) -> dict:
    """`ResultadoCompacto` (o su dict) → dict con la forma de `_MealVisionResult.model_dump()`."""
    d = r.model_dump() if hasattr(r, "model_dump") else dict(r or {})
    tot = _macros(d.get("m"))
    items = []
    for it in d.get("i") or []:
        items.append({"name": it.get("n") or "", "quantity": it.get("q") if it.get("q") is not None else 1.0,
                      "unit": it.get("u") or "unidad", **_macros(it.get("m"))})
    dudas = []
    for du in d.get("q") or []:
        dudas.append({"sobre": du.get("o") or "", "pregunta": du.get("p") or "",
                      "opciones": [{"texto": op.get("t") or "", "supuesta": bool(op.get("s")),
                                    "nombre_plato": op.get("n") or "", "ajuste": _macros(op.get("a"))}
                                   for op in (du.get("x") or [])]})
    return {"photo_kind": d.get("k") or "otro", "is_food": bool(d.get("f")), "meal_name": d.get("nm") or "",
            "description": d.get("d") or "", **tot, "items": items, "dudas": dudas,
            "food_facts": d.get("food_facts"), "supplement_facts": d.get("supplement_facts")}
