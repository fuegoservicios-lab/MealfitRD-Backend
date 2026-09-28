# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-636 · 2026-09-28] El yogur que el texto nombra es el de la lista.

Validación del 592 (estudiante, merienda del día 2): «Casabe crujiente con mantequilla de maní, canela y yogurt natural
entero» con «20 g de yogurt natural entero» en la lista y «Acompaña con yogurt natural entero y yogurt griego entero» en
el Montaje: el motor llegó a poner dos yogures, la fusión de líneas duplicadas los juntó en UNA (todo yogur es un solo
artículo de compra, P3-YOGURT-CONSOLIDATE) y el texto siguió sirviendo los dos. Replay de 5.042 comidas guardadas: 255
nombran un yogur de otra CLASE que el de la lista —griego ≠ natural para el catálogo (P1-YOGURT-NATURAL), y los
vegetales—, entre ellas las que sirven dos. Como el 520 con el queso: con UN yogur en la lista, la mención de otra clase
pasa a nombrar el de la lista («yogurt griego», o «yogurt» si la línea no dice clase) y la repetición que eso deja («X y
X», dos frases iguales seguidas) se va. «entero», «sin azúcar», «light» no cambian de producto y no se tocan. Con dos
yogures en la lista no se toca nada, ni las notas (⚠️ ⚕️ 💡). tooltip-anchor: P1-PLAN-LOTE-636
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_YOG = r"yog(?:h)?urt?"
_YOG_LINEA = re.compile(r"\b" + _YOG + r"\b", re.IGNORECASE)
_VARIANTE = (r"griego|natural|enter[oa]|descremad[oa]|semidescremad[oa]|light|bebible|sin\s+az[uú]car|bajo\s+en\s+grasa|"
             r"sin\s+grasa|0\s*%|de\s+coco|vegetal|de\s+soya|de\s+almendras?|pasteurizad[oa]")
_MENCION = re.compile(r"\b(?P<y>" + _YOG + r")(?P<cola>(?:\s+(?:" + _VARIANTE + r"))*)(?![\w%])", re.IGNORECASE)
_NOTAS = ("⚠", "⚕", "💡", "🤰", "🧊", "🛒")
_CANT = re.compile(r"^\s*(?:[\d.,/½¼¾⅓⅔]+\s*(?:g|gr|gramos|ml|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|potes?|"
                   r"vasos?|porci[oó]n(?:es)?|unidades?)?\.?\s*(?:de\s+)?)?", re.IGNORECASE)


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _variantes(cola: str) -> set:
    return {re.sub(r"\s+", " ", v) for v in re.findall(_VARIANTE, _sa(cola))}


def _clase(cola: str):
    """Lo que el catálogo distingue: griego ≠ natural (P1-YOGURT-NATURAL), y los vegetales. «entero», «sin azúcar»,
    «light»… no cambian de producto: `None` = sin clase (compatible con cualquiera)."""
    v = _variantes(cola)
    for veg in ("de coco", "de soya", "de almendra", "de almendras", "vegetal"):
        if veg in v:
            return "vegetal:" + veg.replace("de ", "").rstrip("s")
    if "griego" in v:
        return "griego"
    if "natural" in v:
        return "natural"
    return None


def _nombre_linea(linea: str) -> str:
    x = re.sub(r"\([^)]*\)", "", str(linea or ""))
    x = _CANT.sub("", x, count=1).strip(" ,.")
    return x[:1].lower() + x[1:] if x else ""


def alinear(meal) -> int:
    """Nº de textos corregidos; 0 ante cualquier error o si no aplica."""
    try:
        if not isinstance(meal, dict):
            return 0
        yogs = [str(x) for x in (meal.get("ingredients") or []) if isinstance(x, str) and _YOG_LINEA.search(x)]
        if len(yogs) != 1:
            return 0
        de_lista = _nombre_linea(yogs[0])
        m0 = _MENCION.match(de_lista)
        if not m0 or m0.end() != len(de_lista):
            return 0                                   # «yogurt con frutas», «yogur bebible de fresa»: no es un nombre limpio
        c_lista = _clase(m0.group("cola"))
        textos = [meal.get(k) for k in ("name", "desc", "description")] + [
            p for p in (meal.get("recipe") or []) if isinstance(p, str) and not p.lstrip().startswith(_NOTAS)]
        clases = {c for t in textos if isinstance(t, str) for mm in _MENCION.finditer(t) if (c := _clase(mm.group("cola")))}
        if c_lista is None and len(clases) < 2:
            return 0                                   # lista genérica y el texto sirve UN yogur: no hay contradicción
        if c_lista is not None and not (clases - {c_lista}):
            return 0
        # lo que se escribe: el yogur de la lista por su CLASE («yogurt griego», «yogurt de coco», o «yogurt» si la línea no
        # la dice), sin los adjetivos de la línea: «entero», «sin azúcar» o «pasteurizado» ya los dice la lista, y copiarlos
        # en una frase que ya los lleva deja «que el yogurt pasteurizado sea pasteurizado»
        cabeza = m0.group("y")
        if c_lista is None:
            nuevo = cabeza
        elif c_lista.startswith("vegetal:"):
            veg = next(v for v in re.findall(r"\s+(?:" + _VARIANTE + r")", m0.group("cola"), flags=re.IGNORECASE)
                       if _clase(v) == c_lista)
            nuevo = cabeza + veg
        else:
            nuevo = cabeza + " " + c_lista

        def _sub(m):
            c = _clase(m.group("cola"))
            if c is None or c == c_lista:
                return m.group(0)
            pz = re.search(r"\s+pasteurizad[oa]", m.group("cola"), flags=re.IGNORECASE)
            t = nuevo + (pz.group(0) if pz else "")    # «pasteurizado» es la pista de seguridad del embarazo: se queda
            return (t[:1].upper() + t[1:]) if m.group(0)[:1].isupper() else t

        par = re.compile(r"(?P<a>" + _MENCION.pattern + r")(?P<sep>\s*,\s*|\s+y\s+)(?P<b>\b" + _YOG +
                         r"(?:\s+(?:" + _VARIANTE + r"))*)(?![\w%])", re.IGNORECASE)

        def _junta(mm):
            ca, cb = _clase(mm.group("cola")), _clase(re.sub(r"^\S+", "", mm.group("b")))
            if {ca, cb} <= {c_lista, None}:
                return mm.group("a")                   # «yogurt natural entero y yogurt natural» → uno
            return mm.group(0)

        def _arreglar(t: str) -> str:
            q = _MENCION.sub(_sub, t)
            if q != t:
                while True:
                    r = par.sub(_junta, q)
                    if r == q:
                        break
                    q = r
            return q

        cambios = 0
        for k in ("name", "desc", "description"):
            t = meal.get(k)
            if isinstance(t, str):
                q = _arreglar(t)
                if q != t:
                    meal[k] = q
                    cambios += 1
        rec = meal.get("recipe")
        if isinstance(rec, list):
            for i, p in enumerate(rec):
                if not isinstance(p, str) or p.lstrip().startswith(_NOTAS):
                    continue
                q = _arreglar(p)
                if q != p:
                    rec[i] = q
                    cambios += 1
        if cambios:
            __import__("pasos_cantidades").frases_repetidas(meal)   # «Sirve yogurt al lado. Sirve yogurt al lado.»
            meal.pop("_display", None)
            logger.info(f"🥣 [P1-PLAN-LOTE-636] «{str(meal.get('name'))[:50]}»: el texto nombra el yogur de la lista "
                        f"({de_lista})")
        return cambios
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-636] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["alinear"]
