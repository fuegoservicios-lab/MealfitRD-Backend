# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-910 · 2026-09-29] El marisco COCIDO que añade el cerrador recibe su paso de calentar.

Leídas enteras las 5 baterías reales del 28/29-sep (rdv801-868, 60 comidas): en dos almuerzos de EMBARAZO el cerrador de
proteína añadió «100 g de camarones cocidos» y el único texto que los nombra es «Acompaña con camarones.» — mientras la
nota 🤰 del mismo plato dice «cocina el pescado y los mariscos POR COMPLETO». El edamame cocido sí trae su «Calienta el
edamame cocido en agua hirviendo 2-3 minutos» (425); el camarón, no. Corpus del VPS (5.336 comidas, salida de la cola con
el 888): 50 con camarones cocidos en la lista, 23 sin ningún paso que los caliente, 6 de embarazo o lactancia.

Aquí, con un marisco que la lista compra COCIDO (no de lata) y que ningún paso de fuego nombra: «Calienta los camarones
cocidos en una sartén 2-3 minutos, hasta que humeen.» al final del último «El Toque de Fuego» (o en uno nuevo antes del
Montaje). El Montaje los sigue sirviendo. Un plato frío (ensalada, ceviche, cóctel) los sirve fríos… salvo en embarazo o
lactancia (la nota 🤰 está en el plato), donde se calientan siempre. El camarón crudo no es de este lote (lo cuecen el 445
y el 377). Knob `MEALFIT_COOKED_SEAFOOD_HEAT_STEP` (True). tooltip-anchor: P1-PLAN-LOTE-910
"""
from __future__ import annotations

import re
import unicodedata

_NOTAS = ("⚠", "💡", "🌱", "⚕", "🤰")
_MARISCO_RE = re.compile(r"\b(camar[oó]n(?:es)?|langostinos?|calamar(?:es)?|pulpo|mejill[oó]n(?:es)?|almejas?|cangrejo|jaiba|"
                         r"lamb[ií])\b", re.IGNORECASE)
_COCIDO_RE = re.compile(r"\b(?:pre)?cocid[oa]s?\b")
_LATA_RE = re.compile(r"\blatas?\b|\benlatad[oa]s?\b|\ben\s+(?:agua|aceite|conserva|salmuera)\b")
_FRIO_RE = re.compile(r"\b(?:ensaladas?|ceviche|coctel|fri[oa]s?|fresc[oa]s?)\b")
_FEMENINOS = ("almeja", "jaiba")


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_COOKED_SEAFOOD_HEAT_STEP", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _es_nota(p) -> bool:
    return not isinstance(p, str) or any(e in p for e in _NOTAS)


def _pilar(p: str) -> str:
    t = _sa(p).lstrip()
    for k in ("mise en place", "el toque de fuego", "montaje"):
        if t.startswith(k):
            return k
    return ""


def _frase(nombre: str) -> str:
    w = _sa(nombre)
    plural = w.endswith("s")
    fem = w.rstrip("s") in _FEMENINOS
    art = ("las" if fem else "los") if plural else ("la" if fem else "el")
    adj = "cocid" + (("as" if fem else "os") if plural else ("a" if fem else "o"))
    return f"Calienta {art} {nombre.lower()} {adj} en una sartén 2-3 minutos, hasta que {'humeen' if plural else 'humee'}."


def calentar(meal) -> int:
    """1 si añadió el paso de calentar; 0 si no había nada que hacer o ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list) or not rec:
            return 0
        nombre = None
        for linea in meal.get("ingredients") or []:
            if not isinstance(linea, str):
                continue
            m = _MARISCO_RE.search(linea)
            ls = _sa(linea)
            if m and _COCIDO_RE.search(ls) and not _LATA_RE.search(ls):
                nombre = m.group(1)
                break
        if not nombre:
            return 0
        raiz = _sa(nombre)[:5]
        al_fuego = " ".join(_sa(p) for p in rec if isinstance(p, str) and not _es_nota(p)
                            and _pilar(p) not in ("montaje", "mise en place"))
        if re.search(r"\b" + re.escape(raiz), al_fuego):
            return 0                                 # un paso ya lo pone al fuego (o lo incorpora al guiso)
        embarazo = any(isinstance(p, str) and "🤰" in p for p in rec) or bool(meal.get("_pregnancy_labels"))
        if not embarazo and _FRIO_RE.search(_sa(meal.get("name"))):
            return 0                                 # una ensalada o un ceviche lo sirven frío
        frase = _frase(nombre)
        k = max((j for j, p in enumerate(rec) if isinstance(p, str) and not _es_nota(p)
                 and _pilar(p) == "el toque de fuego"), default=None)
        if k is not None:
            t = rec[k].rstrip()
            rec[k] = t + ("" if t.endswith((".", "!", "…")) else ".") + " " + frase
        else:
            j = next((i for i, p in enumerate(rec) if isinstance(p, str) and _pilar(p) == "montaje"), None)
            if j is None:
                j = next((i for i, p in enumerate(rec) if _es_nota(p)), len(rec))
            rec.insert(j, "El Toque de Fuego: " + frase)
        meal["recipe"] = rec
        meal.pop("_display", None)
        return 1
    except Exception:                                                          # noqa: BLE001
        return 0
