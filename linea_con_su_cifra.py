# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-917 · 2026-09-29] La línea de la lista que perdió su cifra la recupera.

Batería real de embarazo rdv864 (día 1, cena): la lista visible decía «g de nabo pelado y cortado en rodajas de 1 cm» y la
del motor «264.35 g de nabo pelado y cortado en rodajas de 1 cm». La cifra se pierde DENTRO del grafo (ya falta en
`pipeline_result`, antes del escudo) y no se ha localizado quién: ni el humanizador, ni el pulido, ni el contrato, ni el
escudo la reproducen desde el crudo, en local ni en el VPS con el catálogo real. La otra («G de queso blanco fresco
(extensor opcional)», rdb524) sí tiene dueño: el redondeo de lo «opcional», que arregla el 916.

Sin cifra nadie puede leer la línea: el contrato no sincroniza el paso («corta 250 g de nabo» contra 264 g) y el escudo
no la reescala. Red de seguridad al PRINCIPIO del contrato, antes de todo lo que lee la lista: una línea visible que
empieza por la unidad («g de…», «ml de…») recupera la cifra de la línea del motor del mismo alimento, redondeada como
redondea el repo (`quantize_ingredient_string`); si el motor tampoco la tiene, del paso que lo mide («añade 30 g de queso
blanco fresco…»), y entonces vuelve a las dos listas. Con dos líneas del motor del mismo alimento, o dos cantidades
distintas en los pasos, no se adivina. Knob `MEALFIT_LIST_LINE_KEEPS_NUMBER` (True). tooltip-anchor: P1-PLAN-LOTE-917
"""
from __future__ import annotations

import re
import unicodedata

_NOTAS = ("⚠", "💡", "🌱", "⚕", "🤰", "🛡")
_SIN_RE = re.compile(r"^\s*(?P<u>g|gr|gramos|ml|kg)\s+de\s+(?P<resto>\S.*)$", re.IGNORECASE)
_CON_RE = re.compile(r"^\s*(?P<q>\d+(?:[.,]\d+)?)\s*(?P<u>g|gr|gramos|ml|kg)\s+de\s+(?P<resto>\S.*)$", re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_LIST_LINE_KEEPS_NUMBER", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower().strip()


def _del_motor(raw, unidad: str, resto: str):
    """La línea del motor del mismo alimento y la misma unidad, si es UNA; si no, `None`."""
    if not isinstance(raw, list):
        return None
    iguales = []
    for r in raw:
        m = _CON_RE.match(r) if isinstance(r, str) else None
        if m and m.group("u").lower() == unidad and _sa(m.group("resto")) == _sa(resto):
            iguales.append(r)
    return iguales[0] if len(iguales) == 1 else None


def _del_paso(rec, unidad: str, resto: str):
    """La cantidad con la que un paso mide ese alimento, si los pasos dan UNA sola; si no, `None`."""
    nombre = _sa(re.split(r"[(,]", resto, maxsplit=1)[0])
    if len(nombre) < 3 or not isinstance(rec, list):
        return None
    rx = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*" + re.escape(unidad) + r"\s+de\s+" + re.escape(nombre) + r"\b")
    vistas = set()
    for p in rec:
        if not isinstance(p, str) or any(e in p for e in _NOTAS):
            continue
        for m in rx.finditer(_sa(p)):
            vistas.add(m.group(1).replace(",", "."))
    return next(iter(vistas)) if len(vistas) == 1 else None


def restaurar(meal) -> int:
    """Nº de líneas que recuperaron su cifra; 0 si no había nada que hacer o ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict):
            return 0
        ings = meal.get("ingredients")
        if not isinstance(ings, list) or not ings:
            return 0
        raw = meal.get("ingredients_raw")
        n = 0
        for i, linea in enumerate(ings):
            m = _SIN_RE.match(linea) if isinstance(linea, str) else None
            if not m:
                continue
            unidad, resto = m.group("u").lower(), m.group("resto").strip()
            motor = _del_motor(raw, unidad, resto)
            if motor:
                from nutrition_db import quantize_ingredient_string
                nueva = quantize_ingredient_string(motor)[0]
                if not _CON_RE.match(str(nueva)):
                    continue
                ings[i] = nueva
                n += 1
                continue
            cifra = _del_paso(meal.get("recipe"), unidad, resto)
            if not cifra:
                continue
            nueva = f"{cifra} {unidad} de {resto}"
            ings[i] = nueva
            if isinstance(raw, list):
                for j, r in enumerate(raw):
                    mr = _SIN_RE.match(r) if isinstance(r, str) else None
                    if mr and mr.group("u").lower() == unidad and _sa(mr.group("resto")) == _sa(resto):
                        raw[j] = nueva
            n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0
