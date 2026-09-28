# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-801 · 2026-09-28] El huevo se cuenta, no se pesa: la identidad del plato lo sube en unidades enteras.

Corpus de la cola 744 (426 planes de la batería RD): 140 líneas «60 g de huevo», «60 g de clara de huevo», «90 g de
huevo»…, y 137 las escribió la restauración de identidad (`identidad_plato`), que sube lo que da nombre al plato hasta el
piso de su categoría (proteínas, 60 g) y lo escribe EN GRAMOS: «1 huevo» (50 g) → «60 g de huevo», «1 clara» (33 g) →
«60 g de clara de huevo». Esa restauración corre después del humanizador, así que el usuario lee los gramos tal cual
mientras los pasos dicen «mide 1 huevo» o «cocina 3 huevos y 2 claras»; y una línea en gramos se escapa del tope diario de
huevos enteros (`_WHOLE_EGG_LINE_RE` sólo cuenta «N huevos»).

Para el huevo, la clara y la yema: el piso se redondea a unidades ENTERAS (al menos una: un huevo ya es la ración de un
plato «…con huevo»; la clara, dos), la línea nueva se escribe en unidades («2 claras de huevo»), lo que no cabe entero no
sube, y un huevo ENTERO sólo se suma si el día tiene sitio bajo `MEALFIT_EGG_DAY_MAX_WHOLE` (la identidad de la cola corre
DESPUÉS del tope). La compensación de migajas no toca el huevo. Knob `MEALFIT_DISH_IDENTITY_EGG_UNITS` (True).
tooltip-anchor: P1-PLAN-LOTE-801
"""
from __future__ import annotations

import contextlib
import contextvars
import math
import os
import re
import sys
import unicodedata
from typing import Optional

# tipo → (singular, plural, gramos por unidad de respaldo; manda el del catálogo)
_TIPOS = {
    "entero": ("huevo", "huevos", 50.0),
    "clara": ("clara de huevo", "claras de huevo", 33.0),
    "yema": ("yema de huevo", "yemas de huevo", 17.0),
}
# Sólo el alimento crudo: «Huevos rellenos» es un plato preparado y se queda en lo suyo.
_NOMBRES = {
    "huevo": "entero", "huevos": "entero", "huevo entero": "entero", "huevos enteros": "entero",
    "clara": "clara", "claras": "clara", "clara de huevo": "clara", "claras de huevo": "clara",
    "clara de huevos": "clara", "yema": "yema", "yemas": "yema", "yema de huevo": "yema", "yemas de huevo": "yema",
    "yema de huevos": "yema",
}
# El mismo patrón que cuenta el tope diario (`graph_orchestrator._WHOLE_EGG_LINE_RE`), más la línea en gramos.
_ENTEROS_EN_UNIDADES = re.compile(r"^\s*(\d+)\s*huevos?\b(?![^,;(]*\bclaras?\b)", re.IGNORECASE)
_ENTEROS_EN_GRAMOS = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+huevos?(?:\s+enteros?)?\s*$", re.IGNORECASE)
_LIBRES: contextvars.ContextVar = contextvars.ContextVar("p1_plan_lote_801_huevos_enteros_libres", default=None)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DISH_IDENTITY_EGG_UNITS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def tipo(canon) -> Optional[str]:
    """«Huevo» → "entero", «Clara de huevo» → "clara", «Yema de huevo» → "yema"; lo demás → `None`."""
    if not on():
        return None
    return _NOMBRES.get(re.sub(r"\s+", " ", _sa(canon)).strip())


def peso(canon, db) -> float:
    """Gramos de UNA unidad, del catálogo (`density_g_per_unit`: 50 / 33 / 17); la tabla sólo si el catálogo no lo sabe."""
    t = tipo(canon) or "entero"
    sing, _plur, respaldo = _TIPOS[t]
    try:
        g = float(db.grams_from_ingredient_string(f"1 {sing}") or 0)
        if 5.0 <= g <= 100.0:
            return g
    except Exception:                                                          # noqa: BLE001
        pass
    return respaldo


def piso_en_gramos(canon, piso_g, db) -> float:
    """El piso redondeado a unidades enteras, en gramos: al menos una unidad (huevo 60 → 50; clara 60 → 66)."""
    w = peso(canon, db)
    return max(1, int(round(float(piso_g) / w))) * w


def linea(canon, n: int) -> str:
    sing, plur, _w = _TIPOS[tipo(canon) or "entero"]
    return f"1 {sing}" if int(n) == 1 else f"{int(n)} {plur}"


def _tope() -> int:
    go = sys.modules.get("graph_orchestrator")
    if go is not None and getattr(go, "EGG_DAY_MAX_WHOLE", None):
        return int(go.EGG_DAY_MAX_WHOLE)
    try:
        return max(1, min(12, int(os.environ.get("MEALFIT_EGG_DAY_MAX_WHOLE", "3"))))
    except ValueError:
        return 3


def enteros_del_dia(meals) -> float:
    """Huevos enteros que ya lleva el día (lo que lee el tope, y también las líneas en gramos que el tope no ve)."""
    n = 0.0
    for m in meals or []:
        for s in ((m.get("ingredients") or []) if isinstance(m, dict) else []):
            if not isinstance(s, str):
                continue
            mu = _ENTEROS_EN_UNIDADES.match(s)
            if mu:
                n += int(mu.group(1))
                continue
            mg = _ENTEROS_EN_GRAMOS.match(_sa(s))
            if mg:
                n += float(mg.group(1).replace(",", ".")) / _TIPOS["entero"][2]
    return n


@contextlib.contextmanager
def presupuesto_del_dia(meals):
    """Abre el sitio que le queda al día bajo el tope de huevos enteros mientras la identidad lo recorre."""
    tok = _LIBRES.set({"libres": max(0.0, _tope() - enteros_del_dia(meals))} if on() else None)
    try:
        yield
    finally:
        _LIBRES.reset(tok)


def unidades_objetivo(canon, g_cur: float, objetivo_g: float, db) -> Optional[tuple]:
    """`(n, gramos)`: las unidades ENTERAS hasta `objetivo_g` que superan lo que hay (`g_cur`), recortadas al sitio del
    día si son huevos enteros; `None` si no cabe ni una más."""
    w = peso(canon, db)
    n_cur = float(g_cur or 0) / w
    n = int(math.floor(float(objetivo_g) / w + 1e-6))
    ctx = _LIBRES.get()
    if tipo(canon) == "entero" and ctx is not None:
        n = min(n, int(math.floor(n_cur + ctx["libres"] + 1e-6)))
    if n < 1 or n <= n_cur + 1e-6:
        return None
    return n, n * w


def gastar(canon, g_antes: float, g_despues: float, db) -> None:
    """Descuenta del sitio del día los huevos enteros que la identidad acaba de sumar."""
    ctx = _LIBRES.get()
    if ctx is None or tipo(canon) != "entero":
        return
    w = peso(canon, db)
    ctx["libres"] = max(0.0, ctx["libres"] - max(0.0, (float(g_despues) - float(g_antes or 0)) / w))
