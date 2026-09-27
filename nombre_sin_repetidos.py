# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-564 · 2026-09-27] El nombre del plato no repite un alimento.

Batería real (perfil tipo dueño, día 1): «Pescado blanco a la plancha con nabo salteado al limón, casabe y pescado
blanco» — el cerrador puso atún, el tope de sodio lo cambió por filete fresco y renombró «…y atún» → «…y pescado blanco»
sin mirar que el plato ya se llamaba así. En el corpus (4.710 comidas) hay 9 nombres así, casi todos con queso:
«Wrap dominicano de queso blanco, nabo con aguacate y queso blanco y queso blanco», «Yautía Majada Caliente con Queso
Blanco, Queso Blanco, Vegetales…». Aquí, al final de la cola, cada elemento de la enumeración del nombre (separada por
comas e «y») que ya está dicho ANTES en el nombre se va: «…con queso blanco fresco, aguacate, queso blanco cuajado y
queso blanco» → «…con queso blanco fresco y aguacate». Se compara el núcleo (las dos primeras palabras con significado),
con límite de palabra y sin acentos: «queso crema» tras «queso blanco» se queda. Sólo el nombre; la descripción y los
pasos no se tocan. tooltip-anchor: P1-PLAN-LOTE-564
"""
from __future__ import annotations

import re
import unicodedata

_SEP = re.compile(r"(,\s+|\s+y\s+|\s+e\s+)")
_VACIAS = {"de", "del", "la", "el", "los", "las", "con", "al", "a", "en", "un", "una"}


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _nucleo(item: str) -> str:
    palabras = [w for w in re.findall(r"[a-z]+", _sa(item)) if w not in _VACIAS]
    return " ".join(palabras[:2])


def nombre(name):
    """El nombre sin elementos repetidos, o `None` si no hay nada que quitar."""
    try:
        if not isinstance(name, str) or not name.strip():
            return None
        trozos = _SEP.split(name)
        items = trozos[0::2]
        if len(items) < 2:
            return None
        quedan = [items[0]]
        for it in items[1:]:
            nuc = _nucleo(it)
            antes = _sa(" ".join(quedan))
            if nuc and len(nuc) >= 3 and re.search(r"\b" + re.escape(nuc) + r"s?\b", antes):
                continue
            quedan.append(it)
        if len(quedan) == len(items):
            return None
        if len(quedan) == 1:
            return quedan[0].strip()
        return (", ".join(q.strip() for q in quedan[:-1]) + " y " + quedan[-1].strip()).strip()
    except Exception:
        return None


def limpiar(meal) -> int:
    """1 si el nombre cambió; 0 si no (o ante cualquier error)."""
    try:
        if not isinstance(meal, dict):
            return 0
        nuevo = nombre(meal.get("name"))
        if not nuevo or nuevo == meal.get("name"):
            return 0
        meal["name"] = nuevo
        meal.pop("_display", None)
        return 1
    except Exception:
        return 0


__all__ = ["nombre", "limpiar"]
