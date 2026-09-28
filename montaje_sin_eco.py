# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-582 · 2026-09-27] El Montaje no vuelve a servir lo que ya sirvió.

Batería real y corpus: «Sirve la ensalada con el gouda y las arepitas calientes. Acompaña con queso gouda.»; «…sirve el
mangú con la cebolla salteada y el queso blanco pasteurizado; acompaña con aguacate. Acompaña con queso blanco
pasteurizado.»; «Acompaña con filete de pescado blanco. Acompaña con filete de pescado blanco.». La frase final la pega
quien comprueba que cada ingrediente se sirva, antes de que otra pasada reescriba la mención, y la repite. Aquí, al final
de la cola, la ÚLTIMA frase «Acompaña/Termina con X.» de un Montaje se va si X ya está servido antes en ese paso: la
frase entera del alimento (sin «entero», «fresco»…) o, para un queso o un yogur, «el gouda»/«la ricotta». Palabras
sueltas no bastan (la primera versión quitó «yogurt natural» porque ponía «mantequilla de maní natural»), y el agua no
se toca nunca (el agua de un aderezo no es servir agua). tooltip-anchor: P1-PLAN-LOTE-582
"""
from __future__ import annotations

import re
import unicodedata

_ECO_RE = re.compile(r"\s*(?:Acompaña|Termina)\s+con\s+(?:el\s+|la\s+|los\s+|las\s+)?(?P<obj>[^.;:]{3,60}?)\.\s*$")
_COLA_GENERICA = {"entero", "entera", "enteros", "enteras", "fresco", "fresca", "frescos", "frescas", "picado", "picada"}
_CABEZAS = {"queso", "quesos", "yogur", "yogurt"}


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _ya_servido(obj: str, antes: str) -> bool:
    toks = re.findall(r"[a-z]+", _sa(obj))
    while toks and toks[-1] in _COLA_GENERICA:
        toks.pop()
    if not toks or toks == ["agua"]:
        return False
    frase = " ".join(toks)
    if re.search(r"\b" + re.escape(frase) + r"s?\b", antes):
        return True
    return (toks[0] in _CABEZAS and len(toks) >= 2
            and bool(re.search(r"\b(?:el|la|los|las)\s+" + re.escape(toks[1]) + r"\b", antes)))


# [P1-PLAN-LOTE-789 · 2026-09-28] Dos huecos del 582, medidos en el bloque 3 REAL de 6594aae1 y en el corpus de la cola 744:
#   · sólo miraba la ÚLTIMA frase: «…coloca la mozzarella sobre las tostadas… Acompaña con queso mozzarella fresco bajo en
#     sodio. Espolvorea las semillas de linaza por encima.» — la siembra de micros (709) ya había cerrado el Montaje;
#   · exigía la frase entera: «coloca encima el pescado desmenuzado… Acompaña con filete de pescado blanco.», «sirve las
#     tortitas con el pollo desmenuzado encima… Acompaña con pechuga de pollo cocida y desmenuzada.».
# Aquí, cualquier «Acompaña con X.» del Montaje (no «Termina con»: suele ser el acabado) se va si lo que IDENTIFICA a X —un
# nombre de proteína o de queso concreto, no «queso», «filete» ni «pechuga»— ya se sirvió antes en ese paso con artículo
# («el pescado», «la mozzarella»). Con varios alimentos («con pollo y aguacate») sólo si todos lo están; lo que sigue a «y»
# puede ser un adjetivo («cocida y desmenuzada»). El agua nunca. tooltip-anchor: P1-PLAN-LOTE-789
_ACOMPANA_789_RE = re.compile(r"(?:(?<=[.;:])|^)\s*Acompaña\s+con\s+(?:el\s+|la\s+|los\s+|las\s+)?"
                              r"(?P<obj>[^.;:]{3,80}?)\.(?=\s|$)")
_IDENTIDAD_789 = ("pescado", "pollo", "pavo", "res", "cerdo", "atun", "sardina", "tilapia", "camaron", "huevo",
                  "mozzarella", "gouda", "ricotta", "cottage", "cheddar", "parmesano", "provolone", "edamame")
_ADJETIVOS_789 = {"cocido", "cocida", "cocidos", "cocidas", "desmenuzado", "desmenuzada", "desmenuzados", "desmenuzadas",
                  "picado", "picada", "rallado", "rallada", "tostado", "tostada", "fresco", "fresca", "frio", "fria",
                  "caliente", "entero", "entera", "bajo", "sodio", "en", "agua", "blanco", "blanca", "claro"}


def _identidad_789(parte: str):
    toks = re.findall(r"[a-z]+", _sa(parte))
    ids = [t for t in toks if t in _IDENTIDAD_789 or (t.endswith("s") and t[:-1] in _IDENTIDAD_789)
           or (t.endswith("es") and t[:-2] in _IDENTIDAD_789)]
    if not ids:
        return "" if toks and all(t in _ADJETIVOS_789 for t in toks) else None
    return ids[0][:-2] if ids[0][:-2] in _IDENTIDAD_789 else (ids[0][:-1] if ids[0][:-1] in _IDENTIDAD_789 else ids[0])


def _servido_789(obj: str, antes: str) -> bool:
    partes = [x for x in re.split(r",|\s+y\s+|\s+e\s+", _sa(obj)) if x.strip()]
    if not partes or any(re.fullmatch(r"\s*agua\s*", x) for x in partes):
        return False
    vistos = 0
    for x in partes:
        ident = _identidad_789(x)
        if ident is None:
            return False                       # un alimento sin identidad reconocible: no se toca
        if ident == "":
            continue                           # sólo adjetivos («… y desmenuzada»)
        if not re.search(r"\b(?:el|la|los|las)\s+" + re.escape(ident) + r"(?:s|es)?\b", antes):
            return False
        vistos += 1
    return vistos > 0


def _limpiar_789(p: str) -> str:
    q = p
    for m in reversed(list(_ACOMPANA_789_RE.finditer(q))):
        if _servido_789(m.group("obj"), _sa(q[:m.start()])):
            q = (q[:m.start()].rstrip() + (" " if q[m.end():].strip() else "") + q[m.end():].lstrip()).rstrip()
    return q

def limpiar(meal) -> int:
    """Nº de frases quitadas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or not _sa(p).lstrip().startswith("montaje"):
                continue
            q = p
            while True:
                m = _ECO_RE.search(q)
                if not m or not _ya_servido(m.group("obj"), _sa(q[:m.start()])):
                    break
                q = q[:m.start()].rstrip()
                if q and q[-1] not in ".!?":
                    q += "."
                n += 1
            q789 = _limpiar_789(q)  # [P1-PLAN-LOTE-789] también en medio del Montaje y por lo que identifica al alimento
            if q789 != q:
                q = q789
                n += 1
            if q != p:
                rec[i] = q
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["limpiar"]
