# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-529 · 2026-09-27] «Maní tostado, 20 g» → «20 g de maní tostado», al entrar en `assemble_plan_node`.

Batería real del dueño (compra única, día 28): la IA escribió TODO el día con el alimento primero y la cantidad detrás
—«Arroz blanco cocido, listo para calentar, 1½ tazas», «Maní tostado, 20 g», «Apio, 1 taza picado», «Limones, 0.5
unidad»—; el motor de macros no ve cantidad al principio, antepone la suya y la línea queda con DOS («15 g de maní
tostado, 20 g», «¼ taza de yogurt griego entero, 200 g», «2 limones, 0.5 unidad») y el paso «Añade arroz blanco cocido,
listo para calentar, 1½ tazas al lado». En las baterías guardadas: 23 de 862 días, 112 líneas. Se endereza la línea
ANTES del motor —y su copia literal en los pasos—; lo que ya empieza por una cantidad no se toca.
tooltip-anchor: P1-PLAN-LOTE-529
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_FRAC = {0.25: "¼", 0.5: "½", 0.75: "¾", 1 / 3: "⅓", 2 / 3: "⅔"}
_UNIDADES = (r"unidad(?:es)?|g|gr|gramos|kg|ml|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|dientes?|ramitas?|ramas?"
             r"|rebanadas?|lonjas?|latas?|pizcas?|hojas?|filetes?|pechugas?|tortas?")
_INVERTIDA = re.compile(
    r"^\s*(?P<nombre>[^\d½¼¾⅓⅔,][^\d½¼¾⅓⅔]*?)\s*,\s*"
    r"(?P<q>\d+\s*[½¼¾⅓⅔]|\d+(?:[.,]\d+)?|[½¼¾⅓⅔])\s*"
    r"(?:(?P<u>" + _UNIDADES + r")\b\.?)?"
    r"(?P<resto>\s+[^\d½¼¾⅓⅔,()]+)?\s*$", re.IGNORECASE)


def _cantidad(q: str) -> str:
    """«0.5» → «½», «1.5» → «1½», «2» → «2»; lo demás tal cual."""
    t = q.replace(" ", "").replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return q.replace(" ", "")
    ent = int(v)
    for f, s in _FRAC.items():
        if abs((v - ent) - f) < 0.01:
            return (str(ent) if ent else "") + s
    return str(ent) if abs(v - ent) < 0.01 else f"{v:g}"


def _minuscula(nombre: str) -> str:
    """«Maní tostado» → «maní tostado»; «Corn Flakes» (marca: dos mayúsculas) se queda."""
    if len(re.findall(r"\b[A-ZÁÉÍÓÚÑ]", nombre)) >= 2:
        return nombre
    return nombre[:1].lower() + nombre[1:]


def _singular(nombre: str) -> str:
    """«Limones» → «limón», «Tomates» → «tomate» (sólo la primera palabra); lo que no sabe, igual."""
    cab, sep, resto = nombre.partition(" ")
    low = cab.lower()
    if low.endswith("ones") and len(low) > 5:
        cab = cab[:-4] + "ón"
    elif low.endswith("s") and len(low) > 3 and low[-2] in "aeiouáéíó":
        cab = cab[:-1]
    return cab + sep + resto


def _partes(linea):
    m = _INVERTIDA.match(str(linea or ""))
    if not m:
        return None
    nombre = m.group("nombre").strip()
    if not nombre or len(nombre) > 70:
        return None
    return nombre, m.group("q"), (m.group("u") or "").strip(), (m.group("resto") or "").rstrip()


def enderezar(linea) -> str | None:
    """La línea con la cantidad delante, o None si no es de la forma «alimento, cantidad [unidad] [resto]»."""
    p = _partes(linea)
    if not p:
        return None
    nombre, q_txt, u, resto = p
    q = _cantidad(q_txt)
    de = re.match(r"^\s+de\s+(?P<x>.+)$", resto, re.IGNORECASE)
    if u and de and not re.match(r"unidad", u, re.IGNORECASE):
        # «Limón, 1 cucharada de jugo» → «1 cucharada de jugo de limón»; «Granada, ½ taza de semillas» → «… de semillas de granada»
        return f"{q} {u} de {de.group('x').strip()} de {_minuscula(nombre)}"
    if not u or re.match(r"unidad", u, re.IGNORECASE):
        try:
            uno_o_menos = float(q_txt.replace(" ", "").replace(",", ".")) <= 1
        except ValueError:
            uno_o_menos = q in ("½", "¼", "¾", "⅓", "⅔", "1")
        base = _singular(nombre) if uno_o_menos else nombre
        return f"{q} {_minuscula(base)}{resto}"
    return f"{q} {u} de {_minuscula(nombre)}{resto}"


def normaliza_dias(days) -> int:
    """Endereza en sitio las líneas de `ingredients` (y `ingredients_raw`) y su copia literal en los pasos. Devuelve cuántas
    líneas cambió; 0 ante cualquier error (fail-open: la línea se queda como vino)."""
    n = 0
    try:
        for d in days or []:
            for m in (d.get("meals") or []) if isinstance(d, dict) else []:
                if not isinstance(m, dict):
                    continue
                cambios = {}
                for k in ("ingredients", "ingredients_raw"):
                    lista = m.get(k)
                    if not isinstance(lista, list):
                        continue
                    for i, x in enumerate(lista):
                        if not isinstance(x, str):
                            continue
                        nueva = enderezar(x)
                        if nueva and nueva != x:
                            lista[i] = nueva
                            partes = _partes(x)
                            cambios[x.strip()] = (nueva, _minuscula(partes[0]) + partes[3])
                            n += k == "ingredients"
                rec = m.get("recipe")
                if cambios and isinstance(rec, list):
                    for j, p in enumerate(rec):
                        if isinstance(p, str):
                            q = p
                            for viejo, (nueva, solo_nombre) in cambios.items():
                                def _uno(mm, _n=nueva, _s=solo_nombre):
                                    # «mide ½ taza de yogurt griego natural, 170 g»: el paso ya dice cuánto → sólo el nombre
                                    antes = mm.string[:mm.start()]
                                    if re.search(r"(?:\d|[½¼¾⅓⅔])\s*[A-Za-záéíóúñ]*\.?\s+de\s+$", antes):
                                        return _s
                                    return _n
                                q = re.sub(re.escape(viejo), _uno, q, flags=re.IGNORECASE)
                            rec[j] = q
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-529] no-op: {type(e).__name__}: {e}")
    if n:
        logger.info(f"🧾 [P1-PLAN-LOTE-529] {n} línea(s) «alimento, cantidad» enderezada(s)")
    return n


__all__ = ["enderezar", "normaliza_dias"]
