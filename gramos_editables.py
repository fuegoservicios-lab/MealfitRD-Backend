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


# ───────────────────────────────────────────────────────────── [P1-PLAN-LOTE-1098] cantidades sin gramos
# «1 taza de arroz», «2 unidad de huevo», «Leche (≈1 vaso)»: 1 de cada 5 renglones reales no trae gramos (replay
# 18-nov en el servidor: 768 renglones de 30 días, 163 sin gramos, ~100 con número y unidad que el catálogo resuelve).
# Se corrige el NÚMERO de su unidad. Solo si ese número es de verdad la cantidad: al duplicarlo, el catálogo da el doble
# de energía (si no, el número era otra cosa —«leche 2 %»— y no se toca).
CANTIDAD_MIN, CANTIDAD_MAX = 0.25, 30.0
_RE_NUM_UNIDAD = re.compile(r"(\d+(?:[.,]\d+)?|\d+/\d+)\s*([a-záéíóúñ]+)?", re.I)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def cantidad_de(texto: str) -> Optional[dict]:
    """{valor, unidad, inicio, fin} del número de la cantidad (el del paréntesis si lo hay, si no el primero), o None si
    el renglón trae gramos (eso lo corrige `editables`)."""
    if not isinstance(texto, str) or not texto.strip():
        return None
    c = componente(texto)
    if c and c.get("gramos"):
        return None
    m_paren = re.search(r"\(([^)]*)\)", texto)
    base = m_paren.start(1) if m_paren else 0
    zona = m_paren.group(1) if m_paren else texto
    m = _RE_NUM_UNIDAD.search(zona)
    if not m:
        return None
    crudo = m.group(1)
    try:
        if "/" in crudo:
            a, b = crudo.split("/")
            valor = float(a) / float(b)
        else:
            valor = float(crudo.replace(",", "."))
    except (ValueError, ZeroDivisionError):
        return None
    if not (CANTIDAD_MIN <= valor <= CANTIDAD_MAX):
        return None
    return {"valor": valor, "unidad": (m.group(2) or "").lower(), "inicio": base + m.start(1), "fin": base + m.end(1)}


# [P1-PLAN-LOTE-1104] «Jugo (2 vasos, ≈500 ml)»: el «≈N ml» acompaña a la cantidad. Sin escalarlo, «4 vasos, ≈500 ml»
# le daba al catálogo los mismos 500 ml (la misma energía) y el renglón no se podía corregir.
_RE_APROX = re.compile(r"(≈\s*)(\d+(?:[.,]\d+)?)(\s*(?:ml|g|oz|l)\b)", re.I)


def con_cantidad(texto: str, valor: float) -> str:
    q = cantidad_de(texto)
    if not q:
        return texto
    nuevo = texto[:q["inicio"]] + _fmt(valor) + texto[q["fin"]:]
    factor = float(valor) / q["valor"] if q["valor"] else 1.0
    if factor == 1.0:
        return nuevo
    desde = q["inicio"] + len(_fmt(valor))
    resto = nuevo[desde:]
    cierre = resto.find(")")
    zona = resto if cierre < 0 else resto[:cierre]
    zona = _RE_APROX.sub(lambda m: m.group(1) + _fmt(round(float(m.group(2).replace(",", ".")) * factor))
                         + m.group(3), zona, count=1)
    return nuevo[:desde] + zona + (resto[cierre:] if cierre >= 0 else "")


def cantidades(textos: list, db: Any) -> dict:
    """{índice: {valor, unidad}} de los renglones sin gramos cuya cantidad se puede corregir."""
    out = {}
    for i, texto in enumerate(textos):
        try:
            q = cantidad_de(texto)
            if not q:
                continue
            uno = macros_de_renglon(texto, db)
            dos = macros_de_renglon(con_cantidad(texto, q["valor"] * 2), db)
            if not uno or not dos or not (float(uno.get("kcal") or 0) > 0):
                continue
            razon = float(dos.get("kcal") or 0) / float(uno["kcal"])
            if 1.6 <= razon <= 2.4:
                out[i] = {"valor": q["valor"], "unidad": q["unidad"]}
        except Exception:
            continue
    return out


def ajustar_cantidad(row: dict, indice: Any, valor_nuevo: Any, db: Any) -> Optional[dict]:
    """Como `ajustar`, para la cantidad de un renglón sin gramos: los totales se mueven la diferencia del renglón."""
    try:
        textos = [x for x in (row.get("ingredients") or []) if isinstance(x, str) and x.strip()]
        indice = int(indice)
        if not (0 <= indice < len(textos)) or indice not in cantidades(textos, db):
            return None
        v = round(min(CANTIDAD_MAX, max(CANTIDAD_MIN, float(valor_nuevo))) * 4) / 4   # de cuarto en cuarto
        viejo = textos[indice]
        nuevo_txt = con_cantidad(viejo, v)
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
            "cantidad": v,
        }
    except Exception:
        return None
