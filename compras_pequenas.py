# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-660 · 2026-09-28] Con la Nevera exigida, una o dos compras pequeñas no tumban el bloque.

Decisión del dueño (28-sep). Producción 14-27 sep: 12 de los 36 rechazos del revisor fueron «ERRORES DE DESPENSA». Tras
el lote 199 (el edamame del cerrador) los que quedan son compras mínimas —«15 g de pepino», «½ plátano maduro», «1 rebanada
de pan de trigo (30 g), ¼ pechuga de pollo (≈44 g)»— y cada una costaba 3 intentos del pipeline y, en el worker, más
reintentos y la PAUSA del bloque («Actualiza tu nevera para continuar»): un menú entero perdido por 15 g de pepino.

Aquí la regla, una sola para todas las guardas del bloque (revisor, existencia y cantidades del worker, validación viva,
post-merge): `tolerar(resultado)` convierte en aprobado el resultado de `validate_ingredients_against_pantry` cuando su
ÚNICO fallo son ingredientes que no están en la Nevera y `pequenas` los acepta —como mucho
`MEALFIT_PANTRY_SMALL_PURCHASES_MAX` (2) alimentos, cada uno ≤ `MEALFIT_PANTRY_SMALL_PURCHASE_MAX_G` (120 g) en total o
UNA pieza como mucho («½ plátano maduro», «1 pepino»: la compra suelta del colmado)—, y ninguno es una proteína animal o
un marisco (la proteína es el centro del plato, no una
compra menor: «¼ pechuga de pollo» sigue rechazándose). Las CANTIDADES de lo que sí está en la Nevera no se relajan
(el fallo «CANTIDADES» pasa intacto). Envuelve el resultado en vez de cambiar la firma del validador: sus mocks en los tests no se enteran.
`marcar` deja en el plato «🛒 Compra pequeña: …»; la lista de compras ya la trae (resta la Nevera, no la ignora).
Knob a 0 = conducta previa. tooltip-anchor: P1-PLAN-LOTE-660
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_INEXISTENTES = re.compile(r"INEXISTENTES en inventario: (.*?)\.\n")
_MASA_LIDER = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?|ml)\b", re.IGNORECASE)
_MASA_PAREN = re.compile(r"\(\s*(?:≈|~|aprox\.?)?\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?|ml)\s*\)", re.IGNORECASE)
_CUENTA = re.compile(r"^\s*(?P<n>\d+(?:[.,]\d+)?)?\s*(?P<f>[½¼¾⅓⅔])?\s")
_FRACCION = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
# un volumen o un envase no es «una pieza»: sin gramos, no se puede decir que sea pequeño
_NO_PIEZA = re.compile(r"\b(?:tazas?|cdas?|cucharadas?|cdtas?|cucharaditas?|latas?|potes?|paquetes?|bolsas?|cajas?|"
                       r"botellas?|libras?|lb|kg|kilos?|litros?|l|onzas?|oz|pu[nñ]ados?)\b", re.IGNORECASE)
_NOTA = "🛒 Compra pequeña"
# la proteína es el centro del plato, no una compra menor: nunca se tolera (test_p0_4: «100 g de camarones» falla)
_PROTEINA = re.compile(r"\b(?:pollo|pechugas?|muslos?|pavo|res|carnes?|cerdo|chuletas?|lomo|pernil|pescados?|filetes?|"
                       r"tilapia|mero|chillo|dorado|salmon|atun|sardinas?|bacalao|arenque|camarones|camaron|mariscos?|"
                       r"langosta|pulpo|calamar(?:es)?|cangrejo|jaiba|lambi|jamon|longaniza|salami|salchich\w*|"
                       r"chorizo|tocino|chicharron)\b")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _max_alimentos() -> int:
    try:
        from knobs import _env_int
        return _env_int("MEALFIT_PANTRY_SMALL_PURCHASES_MAX", 2, validator=lambda v: 0 <= v <= 5)
    except Exception:
        return 2


def _max_gramos() -> float:
    try:
        from knobs import _env_float
        return _env_float("MEALFIT_PANTRY_SMALL_PURCHASE_MAX_G", 120.0, validator=lambda v: 10.0 <= v <= 500.0)
    except Exception:
        return 120.0


#: piezas chicas cuyo peso el catálogo no siempre resuelve («1 rebanada de pan integral familiar»)
_PIEZA_G = (("rebanada", 30.0), ("tortilla", 45.0), ("lonja", 20.0), ("loncha", 20.0), ("hoja", 5.0),
            ("diente", 5.0), ("ramita", 3.0))


def _gramos(linea: str):
    s = str(linea or "")
    m = _MASA_LIDER.match(s) or _MASA_PAREN.search(s)
    if m:
        return float(m.group(1).replace(",", "."))
    try:
        import graph_orchestrator as go
        g = go._resolve_line_food_grams(s)[1]
        if g:
            return float(g)
    except Exception:
        pass
    n = _piezas(s)
    if n is not None:
        for unidad, g in _PIEZA_G:
            if re.search(r"\b" + unidad + r"s?\b", _sa(s)):
                return n * g
    return None


def _piezas(linea: str):
    """Piezas de una línea de conteo («½ plátano maduro», «1 rebanada de pan»); `None` si es volumen/envase o no se lee."""
    s = str(linea or "")
    if _NO_PIEZA.search(_sa(s)) or _MASA_LIDER.match(s):
        return None                                        # «15 g de pepino» son gramos, no 15 piezas
    m = _CUENTA.match(s + " ")
    if not m or not (m.group("n") or m.group("f")):
        return None
    return float((m.group("n") or "0").replace(",", ".")) + _FRACCION.get(m.group("f") or "", 0.0)


def _alimento(linea: str) -> str:
    try:
        from constants import normalize_ingredient_for_tracking
        a = normalize_ingredient_for_tracking(str(linea))
        if a:
            return _sa(a)
    except Exception:
        pass
    t = re.sub(r"\([^)]*\)", "", _sa(linea))
    t = re.sub(r"^[\d\s.,½¼¾⅓⅔/]+(?:[a-z]+\s+de\s+)?", "", t).strip()
    return t


def pequenas(lineas) -> "list | None":
    """Las `lineas` si TODAS juntas son compras pequeñas; `None` si no (o con el knob a 0)."""
    tope = _max_alimentos()
    lineas = [str(x).strip() for x in (lineas or []) if str(x).strip()]
    if tope <= 0 or not lineas:
        return None
    # por alimento: gramos y piezas sumados; `None` en cuanto una línea no los dice
    por = {}
    for ln in lineas:
        a = _alimento(ln)
        if not a or _PROTEINA.search(_sa(ln)):
            return None
        g, p = _gramos(ln), _piezas(ln)
        if g is None and p is None:
            return None                                # no se sabe cuánto es: no se da por pequeña
        e = por.setdefault(a, {"g": 0.0, "p": 0.0})
        e["g"] = None if (e["g"] is None or g is None) else e["g"] + g
        e["p"] = None if (e["p"] is None or p is None) else e["p"] + p
    if len(por) > tope:
        return None
    lim = _max_gramos()
    # pequeña: hasta el tope de gramos, o UNA pieza como mucho (½ plátano maduro pesa ~140 g con el catálogo cargado y es
    # la compra suelta del colmado)
    if any(not ((e["g"] is not None and e["g"] <= lim) or (e["p"] is not None and e["p"] <= 1.0)) for e in por.values()):
        return None
    return lineas


def faltantes(resultado) -> list:
    """Las líneas que `validate_ingredients_against_pantry` dio por INEXISTENTES (su mensaje de error)."""
    if not isinstance(resultado, str):
        return []
    m = _INEXISTENTES.search(resultado)
    return [x.strip() for x in m.group(1).split(", ") if x.strip()] if m else []


def tolerar(resultado):
    """`True` si el único fallo del resultado de la validación de despensa son compras pequeñas; si no, el resultado."""
    try:
        if resultado is True or not isinstance(resultado, str) or "CANTIDADES" in resultado:
            return resultado
        lineas = faltantes(resultado)
        if lineas and pequenas(lineas):
            logger.info(f"🛒 [P1-PLAN-LOTE-660] compra(s) pequeña(s) aceptada(s) con la Nevera exigida: {lineas}")
            return True
        return resultado
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-660] no-op: {type(e).__name__}: {e}")
        return resultado


def marcar(days, nevera, country: str = "DO") -> int:
    """Nº de platos que llevan su «🛒 Compra pequeña»: sólo si lo que falta en TODOS los `days` es pequeño."""
    try:
        if not nevera:
            return 0
        from constants import validate_ingredients_against_pantry as _vip
        por_plato, todas = [], []
        for d in days or []:
            for m in ((d.get("meals") or []) if isinstance(d, dict) else []):
                if not isinstance(m, dict):
                    continue
                lineas = [x for x in (m.get("ingredients") or []) if isinstance(x, str) and x.strip()]
                if not lineas:
                    continue
                falta = faltantes(_vip(lineas, nevera, strict_quantities=False, country=country, probe_only=True))
                if falta:
                    por_plato.append((m, falta))
                    todas += falta
        if not todas or not pequenas(todas):
            return 0
        for m, falta in por_plato:
            m["_compra_pequena"] = falta
            nota = f"{_NOTA} (no está en tu Nevera; va en tu lista de compras): {', '.join(falta)}."
            rec = m.get("recipe") if isinstance(m.get("recipe"), list) else None
            if rec is not None and not any(isinstance(p, str) and p.startswith(_NOTA) for p in rec):
                rec.append(nota)
                m.pop("_display", None)
        return len(por_plato)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-660] marcar no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["pequenas", "faltantes", "tolerar", "marcar"]
