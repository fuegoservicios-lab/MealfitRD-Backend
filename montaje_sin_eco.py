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
            if q != p:
                rec[i] = q
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["limpiar"]
