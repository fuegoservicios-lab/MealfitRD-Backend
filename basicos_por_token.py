# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-857 · 2026-09-29] Los básicos repetidos entre días, por FRONTERA DE PALABRA.

`graph_orchestrator._count_staple_repetitions` (los «staples cross-día» de la autocrítica, del gate
P1-STAPLE-REPEAT-GATE y de `reeleccion_dia`) y su espejo del armador determinista (`deterministic_day._basicos_de`)
casaban `_STAPLE_INGREDIENT_ALIASES` por SUBCADENA: «pina» dentro de «esPINAca». G24 DO salió con `pina: 2` sin
una sola piña — dos días con espinacas —, y la autocrítica pedía rotar una fruta que no estaba. Es la misma clase
que P1-FRUIT-SEEDER-GATE-CONTRACT cerró en el gate de frutas del mismo día y que el 855 cerró en el contador de
proteínas cross-día.

Resolvedor: el del gate same-day (`culinary_context._name_has_token`, frontera de palabra INICIAL): «piñas» y
«yogurt» (por «yogur») siguen contando; «espinaca» deja de ser piña. Knob `MEALFIT_STAPLE_TOKEN_MATCH` (False ⇒ la
subcadena de antes, byte a byte). Puro; nunca lanza. tooltip-anchor: P1-PLAN-LOTE-857-BASICOS-POR-TOKEN
"""
from __future__ import annotations

import unicodedata

from culinary_context import _name_has_token
from knobs import _env_bool

STAPLE_TOKEN_MATCH = _env_bool("MEALFIT_STAPLE_TOKEN_MATCH", True)


def _norm(s) -> str:
    return unicodedata.normalize("NFD", str(s or "").lower()).encode("ascii", "ignore").decode("ascii")


def basicos_en_texto(texto_norm: str, alias_por_basico: dict) -> set:
    """Los básicos de `alias_por_basico` ({básico: [alias]}) presentes en `texto_norm` (minúscula, sin acentos)."""
    try:
        encontrar = _name_has_token if STAPLE_TOKEN_MATCH else (lambda a, t: a in t)
        return {lbl for lbl, als in (alias_por_basico or {}).items()
                if any(encontrar(_norm(a), texto_norm) for a in (als or ()) if a)}
    except Exception:                                                  # noqa: BLE001
        return set()


def basicos_de_comida(comida, alias_por_basico: dict) -> set:
    """Los básicos de una comida (nombre + ingredientes)."""
    if not isinstance(comida, dict):
        return set()
    return basicos_en_texto(_norm(" " + str(comida.get("name") or "") + " "
                                  + " ".join(str(i) for i in (comida.get("ingredients") or []))), alias_por_basico)


def dias_por_basico(days: list, alias_por_basico: dict, min_dias: int = 2) -> dict:
    """En cuántos días distintos aparece cada básico; sólo los que aparecen en ≥ `min_dias`."""
    cuenta: dict = {}
    for day in days or []:
        blob = ""                       # el mismo texto que armaba `_count_staple_repetitions` (knob off = byte a byte)
        for meal in (day or {}).get("meals", []) or []:
            if not isinstance(meal, dict):
                continue
            blob += " " + str(meal.get("name", "") or "")
            for ing in meal.get("ingredients", []) or []:
                blob += " " + str(ing)
        for lbl in basicos_en_texto(_norm(blob), alias_por_basico):
            cuenta[lbl] = cuenta.get(lbl, 0) + 1
    return {k: v for k, v in cuenta.items() if v >= min_dias}
