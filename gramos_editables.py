# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1096 · 2026-11-18] Cualquier ingrediente con gramos se puede corregir en la ficha.

El dueño, con la ficha de un mangú con huevos, manteca, aceite, salami y pollo: «¿por qué hay ingredientes que no se
pueden editar y otros sí? Quiero que se pueda editar cualquier cantidad: si la IA no fue precisa con algún
ingrediente, lo corregimos». Solo el huevo y el pollo tenían −/+.

Por qué: el −/+ del lote 1039 reusaba la regla del SUELO DE PROTEÍNA (`macros_por_gramos.ajustables`), que descarta a
propósito los platos compuestos (mangú), los derivados (manteca, aceite, salami) y lo muy denso (aceite) — para poner
un suelo hace falta la fila exacta de la pieza. Para CORREGIR una cantidad basta con que el renglón se mueva
exactamente lo que ese renglón cambia según el catálogo: los totales se desplazan la diferencia entre el renglón con
los gramos viejos y con los nuevos, la misma cifra que la ficha pinta a la derecha.

Y un error de datos que esto destapaba: «manteca de cerdo» resolvía a la fila de CARNE de cerdo (17 kcal por 10 g; la
manteca es grasa pura, ~90). Las grasas puras sin fila propia se cuentan como lo que son (`_GRASAS_PURAS`).

Puro (el catálogo se inyecta) y fail-open: cualquier fallo ⇒ None / sin cambio.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

from macros_por_gramos import GRAMOS_MAX, GRAMOS_MIN, _con_gramos, ajustables, ajustar_gramos, componente

# Grasas puras que el catálogo no tiene como fila propia y que resolvían a otra cosa (la carne de su animal).
# kcal y grasa por GRAMO (USDA: manteca de cerdo 902 kcal/100 g, 100 g de grasa; sebo igual).
_GRASAS_PURAS = (
    (re.compile(r"\bmanteca\s+de\s+(?:cerdo|puerco|res)\b|\bmanteca\b(?!\s+de\s+(?:mani|cacahuate|cacao|karite))|\bsebo\b|\blard\b"),
     "Manteca", 9.02, 1.0),
)


def _plano(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


def macros_de_renglon(texto: str, db: Any) -> Optional[dict]:
    """{name, grams, kcal, protein, carbs, fats} de un renglón tal como lo pinta la ficha, o None."""
    if not isinstance(texto, str) or not texto.strip():
        return None
    c = componente(texto)
    # la PIEZA (sin su «con …»/«en …»): «huevos fritos en manteca» es huevo, no manteca
    pieza = _plano((c or {}).get("pieza") or "")
    for patron, nombre, kcal_g, grasa_g in _GRASAS_PURAS:
        if pieza and patron.search(pieza):
            g = (c or {}).get("gramos")
            if not g:
                return None
            return {"name": nombre, "grams": float(g), "kcal": round(kcal_g * g, 1), "protein": 0.0, "carbs": 0.0,
                    "fats": round(grasa_g * g, 1)}
    if db is None or not hasattr(db, "macros_from_ingredient_string"):
        return None
    try:
        r = db.macros_from_ingredient_string(texto)
    except Exception:
        return None
    return r if isinstance(r, dict) and r.get("kcal") is not None else None


def editables(textos: list, db: Any) -> dict:
    """{índice: gramos} de los renglones que se pueden corregir: los del suelo (1039) y, además, cualquier renglón
    con gramos explícitos que el catálogo resuelve."""
    out = {}
    try:
        out.update(ajustables(textos, db))
    except Exception:
        pass
    for i, texto in enumerate(textos):
        if i in out:
            continue
        try:
            c = componente(texto)
            g = (c or {}).get("gramos")
            if g and GRAMOS_MIN <= g <= GRAMOS_MAX and macros_de_renglon(texto, db):
                out[i] = int(round(g))
        except Exception:
            continue
    return dict(sorted(out.items()))


def ajustar(row: dict, indice: Any, gramos_nuevos: Any, db: Any) -> Optional[dict]:
    """Como `macros_por_gramos.ajustar_gramos` (que se usa tal cual para las piezas simples) y, para el resto de
    renglones con gramos, los totales se mueven la diferencia del renglón entre los gramos viejos y los nuevos."""
    try:
        textos = [x for x in (row.get("ingredients") or []) if isinstance(x, str) and x.strip()]
        indice = int(indice)
        if not (0 <= indice < len(textos)):
            return None
        if indice in ajustables(textos, db):
            return ajustar_gramos(row, indice, gramos_nuevos, db)
        if indice not in editables(textos, db):
            return None
        g = int(min(GRAMOS_MAX, max(GRAMOS_MIN, round(float(gramos_nuevos)))))
        viejo = textos[indice]
        nuevo_txt = _con_gramos(viejo, g)
        antes = macros_de_renglon(viejo, db)
        despues = macros_de_renglon(nuevo_txt, db)
        if not antes or not despues:
            return None
        delta = {k: float(despues.get(k) or 0) - float(antes.get(k) or 0) for k in ("kcal", "protein", "carbs", "fats")}
        nuevos = list(textos)
        nuevos[indice] = nuevo_txt
        return {
            "calories": max(0, round(float(row.get("calories") or 0) + delta["kcal"])),
            "protein": max(0, round(float(row.get("protein") or 0) + delta["protein"])),
            "carbs": max(0, round(float(row.get("carbs") or 0) + delta["carbs"])),
            "healthy_fats": max(0, round(float(row.get("healthy_fats") or 0) + delta["fats"])),
            "ingredients": nuevos,
            "gramos": g,
        }
    except Exception:
        return None
