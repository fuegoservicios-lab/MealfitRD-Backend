# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-880 · 2026-09-29] «sirve la papa majadas»: el participio sigue al alimento que lo lleva, en número y
género.

Batería real (mujer que pierde grasa, 29-sep, código 868): «Montaje: sirve la papa majadas con el revoltillo encima» —
el escudo bajó «las papas» a «la papa» porque la lista compra una, y el participio siguió en plural. Es la clase del lote
809 («el huevo duros») para cualquier alimento: corpus, «la papa escurridas/doradas/horneadas», «la remolacha asadas».
Sólo cuando el participio va PEGADO a un alimento en singular con su artículo y ese alimento no es el segundo de una
pareja («el pepino y el tomate aliñados» concuerda con los dos, y se queda). Sólo pasos (no notas) y nunca la lista.
Knob `MEALFIT_PARTICIPLE_AGREEMENT` (True). tooltip-anchor: P1-PLAN-LOTE-880
"""
from __future__ import annotations

import re

#: sustantivos que no son el alimento del que habla el participio («la mezcla batidos», «la salsa…»): se dejan
_NO_ALIMENTO = {"taza", "cucharada", "cucharadita", "pizca", "porcion", "porción", "mitad", "parte", "vez", "olla",
                "sarten", "sartén", "bandeja", "capa", "base", "mezcla", "masa", "salsa", "crema", "pasta", "preparacion",
                "preparación", "ensalada", "mano", "fuente", "tapa", "hoja", "hojas", "receta", "guarnicion", "guarnición",
                "carne", "proteina", "proteína",
                # recipientes y lugares: «Sirve 2-3 arepitas en el plato acompañadas de…» — el participio es de las arepitas
                "plato", "bol", "bowl", "tazon", "tazón", "vaso", "recipiente", "centro", "lado", "horno", "fondo", "borde",
                "medio", "momento", "fuego", "punto", "final", "molde", "comal", "caldero", "microondas", "airfryer",
                "agua", "resto", "total", "conjunto", "mesa"}
_RX = re.compile(r"(?P<art>\b(?:la|el|una|un)\s+)(?P<n>[a-záéíóúñü]+)(?P<cad>(?:\s+(?:y\s+)?[a-záéíóúñü]+(?:ad|id)[oa]s)+)\b",
                 re.IGNORECASE)
_PART_RE = re.compile(r"(?P<b>[a-záéíóúñü]+(?:ad|id))(?P<g>[oa])s\b", re.IGNORECASE)
#: «el pepino y el tomate aliñados», «el ajo, la cebolla y el ají salteados»: el alimento es el último de una serie
_COORD_RE = re.compile(r"(?:\by|,)\s*$", re.IGNORECASE)
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|💡(?!\s*cocci)|nota\b)", re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PARTICIPLE_AGREEMENT", True)
    except Exception:                                                          # noqa: BLE001
        return True


def concordar(texto: str) -> str:
    def _sub(m):
        n = m.group("n")
        if n.lower() in _NO_ALIMENTO or n.lower().endswith("s") or _COORD_RE.search(texto[:m.start()]):
            return m.group(0)
        fem = m.group("art").strip().lower() in ("la", "una")
        g = "a" if fem else "o"
        cad = _PART_RE.sub(lambda p: p.group("b") + (g.upper() if p.group("g").isupper() else g), m.group("cad"))
        return m.group("art") + n + cad
    return _RX.sub(_sub, texto)


def concordar_pasos(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _NOTA_RE.search(p):
                continue
            q = concordar(p)
            if q != p:
                rec[i] = q
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0
