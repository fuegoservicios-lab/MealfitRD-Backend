# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-520 · 2026-09-27] El queso que el texto nombra es el de la lista.

Cadena completa re-encadenada desde la salida guardada de la IA (173 planes): en 58 comidas el nombre o un paso nombran
una VARIEDAD de queso que la lista no trae, y la lista trae UN solo queso, otro: «Yogurt natural con guineo fresco y
queso cottage… Termina con queso cottage» con «40 g de queso bajo en sodio» en la lista (HTA), «Mango con zanahoria… y
queso mozzarella» con «15 g de queso pasteurizado» (embarazo), «Acompaña con queso cottage pasteurizado» con «30 g de
queso blanco fresco pasteurizado». El usuario compra y sirve lo que dice la lista; el texto pasa a nombrar ese queso.
Con dos quesos en la lista no se toca (no se sabe cuál es cuál), ni las notas (⚠️ ⚕️ 💡).
tooltip-anchor: P1-PLAN-LOTE-520
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_QUESO_LINEA = re.compile(r"\b(?:quesos?|ricotta|cottage|mozzarella|reques[oó]n)\b", re.IGNORECASE)
_VARIEDADES = ("cottage", "mozzarella", "ricotta", "requeson", "parmesano", "gouda", "cheddar", "provolone", "suizo",
               "crema", "edam", "manchego", "feta", "de hoja", "de freir", "amarillo")
_CALIF = (r"(?:\s+(?:pasteurizad[oa]s?|bajos?\s+en\s+sodio|bajas?\s+en\s+sodio|sin\s+sal|fresc[oa]s?|light|"
          r"descremad[oa]s?|desmenuzad[oa]s?|rallad[oa]s?|en\s+cubos|en\s+l[aá]minas))*")
_MENCION = re.compile(r"\b(?P<q>queso\s+)?(?P<v>cottage|mozzarella|ricotta|reques[oó]n|parmesano|gouda|cheddar|provolone|"
                      r"suizo|crema|edam|manchego|feta|de\s+hoja|de\s+fre[ií]r|amarillo)\b" + _CALIF, re.IGNORECASE)
_NOTAS = ("⚠", "⚕", "💡", "🤰", "🧊", "🛒")
_CANT = re.compile(r"^\s*(?:[\d.,/½¼¾⅓⅔]+\s*(?:g|gr|gramos|ml|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|"
                   r"lonjas?(?:/pedazos?)?|pedazos?|rebanadas?|porci[oó]n(?:es)?|unidades?)?\.?\s*(?:de\s+)?)?",
                   re.IGNORECASE)


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _variedad(texto_sa: str) -> str:
    for v in _VARIEDADES:
        if re.search(r"\b" + re.escape(v) + r"\b", texto_sa):
            return v
    return ""


def _nombre_linea(linea: str) -> str:
    """«40 g de queso bajo en sodio» → «queso bajo en sodio»; «¼ taza de queso ricotta pasteurizado» → «queso ricotta
    pasteurizado». Sin paréntesis de peso."""
    x = re.sub(r"\([^)]*\)", "", str(linea or ""))
    x = _CANT.sub("", x, count=1).strip(" ,.")
    return x[:1].lower() + x[1:] if x else ""


def alinear(meal) -> int:
    """Nº de textos corregidos; 0 ante cualquier error o si no aplica."""
    try:
        if not isinstance(meal, dict):
            return 0
        quesos = [str(x) for x in (meal.get("ingredients") or []) if isinstance(x, str) and _QUESO_LINEA.search(x)]
        if len(quesos) != 1:
            return 0
        de_lista = _nombre_linea(quesos[0])
        v_lista = _variedad(_sa(de_lista))
        if not de_lista or not de_lista.lower().startswith(("queso", "ricotta", "cottage", "mozzarella", "reques")):
            return 0

        def _sub(m):
            v = _sa(m.group("v"))
            v = re.sub(r"\s+", " ", v)
            if v == v_lista or (v_lista and v in v_lista) or (v and v_lista and v_lista in v):
                return m.group(0)
            if not m.group("q") and v in ("crema", "amarillo", "de hoja", "de freir", "suizo"):
                return m.group(0)          # «crema» suelta no es un queso («crema de yogurt»)
            nuevo = de_lista
            return (nuevo[:1].upper() + nuevo[1:]) if m.group(0)[:1].isupper() else nuevo

        cambios = 0
        for k in ("name", "desc", "description"):
            t = meal.get(k)
            if isinstance(t, str):
                q = _MENCION.sub(_sub, t)
                if q != t:
                    meal[k] = q
                    cambios += 1
        rec = meal.get("recipe")
        if isinstance(rec, list):
            for i, p in enumerate(rec):
                if not isinstance(p, str) or p.lstrip().startswith(_NOTAS):
                    continue
                q = _MENCION.sub(_sub, p)
                if q != p:
                    rec[i] = q
                    cambios += 1
        if cambios:
            meal.pop("_display", None)
        return cambios
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-520] no-op: {type(e).__name__}: {e}")
        return 0
