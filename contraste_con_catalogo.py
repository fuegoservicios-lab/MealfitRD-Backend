# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-984 · 2026-11-11] Un registro del chat que se queda muy corto contra el catálogo de platos cocinados.

Caso vivo del dueño (11-nov): «una taza y media de arroz, 3 pedazos y medio de pechuga, 6 pedacitos de maduro frito y
una taza de lentejas» quedó en 940 kcal / 22 g de grasa. Con los gramos que el propio coach escribió (240 + 175 + 100 +
200) los platos COCINADOS del catálogo (`food_search.load_dishes`: `per_100g` del plato terminado, con su aceite)
suman ≈1.227 kcal / 36 g. La IA estimaba arroz y lentejas «en agua» y el maduro sin el aceite que absorbe.

`log_consumed_meal` rechaza UNA vez (por usuario, 120 s) cuando lo que el catálogo reconoce de la lista —solo una
parte de ella— ya supera en ≥25 % (y ≥150 kcal) el total que puso el modelo, y le da la cifra línea por línea. Si
reintenta igual, pasa: el registro jamás se pierde por esto. Solo casa platos COCINADOS, nunca alimentos crudos del
catálogo (la base del número, lotes 282-286). Knob `MEALFIT_CATALOG_FLOOR_GUARD` (True).
tooltip-anchor: P1-PLAN-LOTE-984
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Iterable, Optional

_VENTANA_S = 120.0
_YA_AVISADO: dict = {}
RATIO = 0.80          # el modelo por debajo del 80 % de lo reconocido ⇒ se queda corto
MIN_DIFERENCIA = 150  # kcal: debajo de eso no merece un rechazo

_VACIAS = {"de", "del", "la", "el", "los", "las", "con", "y", "o", "a", "al", "en", "tipo", "estilo", "un", "una"}
# El método «neutro» de un guiso o un grano servido: «lentejas cocidas» es el mismo plato que «lentejas guisadas».
_NEUTRAS = {"cocido", "cocida", "guisado", "guisada", "cocinado", "cocinada", "hervido", "hervida", "dominicano",
            "dominicana", "criollo", "criolla"}
_GRAMOS = re.compile(r"(?:≈|~|aprox\.?\s*)\s*(\d+(?:[.,]\d+)?)\s*g\b|\b(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b")
_PARENTESIS = re.compile(r"\(([^)]*)\)")


def _norm(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("es") and w[-3] not in "aeiou":
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


# Medidas y recipientes: «1 taza de arroz», «4 rodajas de maduro», «1 presa de pollo» — el alimento va después.
_MEDIDAS = {"taza", "rodaja", "presa", "unidad", "cucharada", "cucharadita", "pedazo", "pedacito", "tajada",
            "porcion", "lasca", "plato", "vaso", "trozo", "racion", "pieza", "media", "medio", "aprox", "estimada",
            "estimado", "gramo"}


def _lista(texto: str) -> list:
    out = []
    for w in re.findall(r"[a-z]+", _norm(texto)):
        s = _singular(w)
        if w in _VACIAS or s in _NEUTRAS or s in _MEDIDAS or len(w) <= 2:
            continue
        out.append(s)
    return out


def _tokens(texto: str) -> frozenset:
    return frozenset(_lista(texto))


def _claves_del_plato(label: str) -> list:
    """El nombre sin paréntesis y, si el paréntesis dice qué es («Pollo al horno (pechuga)»), su variante."""
    base = _PARENTESIS.sub(" ", label)
    claves = [_tokens(base)]
    for par in _PARENTESIS.findall(label):
        # «Tostones (plátano verde frito)» se nombra por su paréntesis; «Pollo al horno (pechuga)», por paréntesis
        # más la cabeza del nombre.
        cabeza = (_lista(base) or [""])[0]
        claves.append(_tokens(par) if len(_tokens(par)) > 1 else _tokens(par + " " + cabeza))
        claves.append(_tokens(base + " " + par))   # «mangú de plátano verde majado»
    return [c for c in claves if c]


_INDICE: Optional[list] = None


def _indice() -> list:
    global _INDICE
    if _INDICE is None:
        import food_search
        out = []
        for _slug, d in (food_search.load_dishes() or {}).items():
            per = (d or {}).get("per_100g") or {}
            if not d.get("label") or not per.get("kcal"):
                continue
            for clave in _claves_del_plato(d["label"]):
                out.append((clave, d["label"], per))
        out.sort(key=lambda x: -len(x[0]))   # el más específico primero
        _INDICE = out
    return _INDICE


def _gramos(linea: str) -> Optional[float]:
    m = _GRAMOS.search(str(linea or "").lower())
    if not m:
        return None
    g = float((m.group(1) or m.group(2)).replace(",", "."))
    return g if 5 <= g <= 1500 else None


def plato_de(linea: str, indice: Optional[list] = None):
    """`(label, per_100g)` del plato cocinado que la línea nombra, o None. Una clave de UNA palabra («Mangú») solo
    casa si la línea no dice nada más («mangú de auyama» no es el mangú de plátano), y la primera palabra de la línea
    (su cabeza) tiene que ser del plato."""
    lista = _lista(_PARENTESIS.sub(" ", str(linea)))
    if not lista:
        return None
    toks, cabeza = frozenset(lista), lista[0]
    for clave, label, per in (indice if indice is not None else _indice()):
        # La CABEZA de la línea tiene que ser del plato: «sancocho con plátano verde» no es plátano verde.
        if cabeza in clave and ((len(clave) == 1 and toks == clave) or (len(clave) > 1 and clave <= toks)):
            return label, per
    return None


def contraste(ingredientes: Iterable, indice: Optional[list] = None) -> dict:
    """Lo que el catálogo reconoce de la lista: `{"kcal", "grasa", "lineas": [(texto, label, g, kcal)]}`."""
    lineas, kcal, grasa = [], 0.0, 0.0
    for ing in ingredientes or []:
        txt = ing.get("name") if isinstance(ing, dict) else ing
        g = _gramos(str(txt or ""))
        if not g:
            continue
        hit = plato_de(str(txt), indice)
        if not hit:
            continue
        label, per = hit
        k = per["kcal"] * g / 100.0
        kcal += k
        grasa += float(per.get("fats") or 0) * g / 100.0
        lineas.append((str(txt).strip()[:70], label, g, k))
    return {"kcal": kcal, "grasa": grasa, "lineas": lineas}


def se_queda_corto(calorias, ingredientes, indice: Optional[list] = None) -> Optional[dict]:
    try:
        cal = float(calorias or 0)
    except (TypeError, ValueError):
        return None
    c = contraste(ingredientes, indice)
    if cal <= 0 or not c["lineas"]:
        return None
    if cal < RATIO * c["kcal"] and c["kcal"] - cal >= MIN_DIFERENCIA:
        return c
    return None


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CATALOG_FLOOR_GUARD", True)
    except Exception:                                                          # noqa: BLE001
        return True


def rechazo(user_id, meal_name, calorias, healthy_fats, ingredientes, *, ahora: float = None,
            indice: Optional[list] = None):
    """Texto del rechazo, o None (= registrar). Una sola vez por usuario en 120 s."""
    if not activo():
        return None
    c = se_queda_corto(calorias, ingredientes, indice)
    if not c:
        return None
    t = time.time() if ahora is None else ahora
    for k, ts in list(_YA_AVISADO.items()):
        if t - ts > _VENTANA_S:
            _YA_AVISADO.pop(k, None)
    clave = str(user_id)
    if clave in _YA_AVISADO:
        return None
    _YA_AVISADO[clave] = t
    detalle = "; ".join(f"{txt} → {label}: ≈{int(round(k))} kcal" for txt, label, _g, k in c["lineas"])
    return ("⚠️ NO REGISTRADO TODAVÍA: tu estimado se queda corto. Con los gramos que escribiste, la tabla de platos "
            f"COCINADOS del sistema (con el aceite y el sofrito de la cocina dominicana) da: {detalle}. Solo eso suma "
            f"≈{int(round(c['kcal']))} kcal y ≈{int(round(c['grasa']))} g de grasa, y tú pusiste {int(round(float(calorias)))} kcal "
            f"y {int(round(float(healthy_fats or 0)))} g de grasa para TODO. Recalcula las calorías y macros (usa esas "
            "cifras para esas líneas y suma el resto) y vuelve a llamar a `log_consumed_meal`. Si el usuario dijo que "
            "algo se hizo sin aceite o en agua, mantén tu cifra y vuelve a llamar igual. No le preguntes nada al "
            "usuario por esto ni le digas que hubo un rechazo.")
