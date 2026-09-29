# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-796 · 2026-09-28] El patrón con el que el escáner de alérgenos busca un término dentro del plato.

Lo llama `graph_orchestrator._patron_termino_alergeno` (el backstop, las declaraciones, los rechazos y el catálogo
cerrado comparten este patrón). Frontera de palabra y plural español: `-z` → `-ces` y `s|es` opcional.

Antes el plural sólo se aceptaba en la ÚLTIMA palabra, así que un término compuesto no casaba cuando se pluralizaba
su núcleo: «tortilla de harina» no veía «2 tortillas de harina» y «tortilla integral» no veía «2 tortillas integrales»
(200 líneas en el corpus de la batería) — un celíaco recibía tortillas de trigo con el backstop en verde. Ahora cada
palabra con contenido admite su plural; los conectores («de», «con», «para»…) y las palabras de 1-2 letras quedan
literales. Sólo AÑADE formas del mismo término: todo lo que casaba antes sigue casando.
tooltip-anchor: P1-PLAN-LOTE-796-PLURAL-COMPUESTO
"""
from __future__ import annotations

import re

_CONECTORES = frozenset({"de", "del", "con", "para", "en", "la", "el", "los", "las", "y", "a", "al", "sin", "tipo",
                         "o", "u", "e"})


def _con_plural(w: str) -> str:
    if w.endswith("z"):
        return re.escape(w[:-1]) + r"(?:z|ces)"
    return re.escape(w) + r"(?:s|es)?"


def patron(termino) -> str:
    """Regex con frontera y plural español (en cada palabra con contenido) para un término clínico normalizado.

    La última palabra conserva la regla de siempre (plural aunque sea corta: es la del término de una palabra); las
    internas lo admiten salvo conectores y palabras de 1-2 letras."""
    t = str(termino or "")
    if not t:
        return r"(?!)"
    *internas, ultima = t.split(" ")
    cuerpo = [re.escape(w) if (len(w) <= 2 or w in _CONECTORES) else _con_plural(w) for w in internas]
    return r"\b" + " ".join(cuerpo + [_con_plural(ultima)]) + r"\b"
