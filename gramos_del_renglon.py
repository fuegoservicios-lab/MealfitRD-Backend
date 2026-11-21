# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1141 · 2026-11-21] El descuento de la Nevera lee la cantidad del renglón como lo escribe el coach.

El coach está OBLIGADO (lote 984) a escribir cada ingrediente como «arroz blanco cocido (≈200 g)» o «huevo frito
(1 unidad, ≈50 g)», pero `_parse_quantity` (compartido con la lista de compras) solo entiende la cantidad AL PRINCIPIO.
Ningún renglón del coach se leía: todos caían en la porción típica inventada (`P1-PANTRY-INFER`) — un paquete de Lay's
de 20 g restaba 150 g de papa, y «huevo frito en aceite de oliva» restaba aceite y no el huevo (cena del dueño, 21-nov).

`al_frente` reescribe «nombre (… N unidad … ≈G g)» como «N nombre» (piezas) o «G g nombre» (peso), que el parser sí lee.
Sin paréntesis con cantidad, el renglón sale tal cual.
"""
from __future__ import annotations

import re

_PAREN_FINAL = re.compile(r"^(?P<nombre>.*?)\s*\((?P<dentro>[^()]*)\)\s*$")
_PESO = re.compile(r"≈?\s*(?P<n>\d+(?:[.,]\d+)?)\s*(?P<u>kg|g|gr|ml|l)\b", re.IGNORECASE)
_TROZOS = re.compile(r"^\s*\d+(?:[.,]\d+)?\s+(?:rodajas?|lascas?|trozos?|pedazos?|tiras?|rebanadas?)\s+(?:de\s+)?",
                     re.IGNORECASE)
_PIEZAS = re.compile(r"^\s*(?P<n>\d+(?:[.,]\d+)?)\s*(?:unidad(?:es)?|huevos?|piezas?)\b", re.IGNORECASE)


def al_frente(renglon: str) -> str:
    s = str(renglon or "").strip()
    m = _PAREN_FINAL.match(s)
    if not m or not m.group("nombre").strip():
        return s
    nombre, dentro = m.group("nombre").strip(), m.group("dentro")
    if re.match(r"^\s*\d", nombre):   # ya trae la cantidad delante
        # «8 rodajas de plátano (≈180 g)»: las rodajas no son piezas enteras; manda el peso
        sin_trozos = _TROZOS.sub("", nombre)
        peso = _PESO.search(dentro)
        if sin_trozos != nombre and peso:
            return f"{peso.group('n').replace(',', '.')} {peso.group('u').lower()} {sin_trozos.strip()}"
        return s
    piezas = _PIEZAS.match(dentro)
    if piezas:
        return f"{piezas.group('n').replace(',', '.')} {nombre}"
    peso = _PESO.search(dentro)
    if peso:
        return f"{peso.group('n').replace(',', '.')} {peso.group('u').lower()} {nombre}"
    return s
