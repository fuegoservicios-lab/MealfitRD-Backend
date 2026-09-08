# -*- coding: utf-8 -*-
"""[P1-RECIPE-LIBRARY-SELECT · 2026-09-08] Recetas escritas UNA vez, seleccionadas en vez de inventadas.

Hoy el motor le pide a un LLM la receta de cada plato **en el momento**: cuesta cada vez, se espera,
y el mismo plato sale escrito distinto cada vez. `recipe_library_do_v1.json` tiene las 140 recetas
del registry de RD ya escritas y revisadas; este módulo es el enganche.

## El estado, medido y sin adornos (2026-09-08)

La cadena que propone platos del catálogo —política → blueprint → rebanada → CandidateSet— **funciona
de punta a punta**: montada a mano sobre un formulario sintético propone 61 platos del registry para
7 días, y **los 61 tienen su receta escrita**. Cero huecos.

Lo que falta no es maquinaria, son tres interruptores:

  1. `MEALFIT_PLAN_POLICY_MODE` nace en `off`, así que hoy no hay blueprint ni CandidateSet y el
     modelo inventa libre. Verificado: 94 de 95 planes vivos sin sello de política.
  2. El prompt dice «elige uno de estos **o una variante equivalente**». Con esa frase el modelo
     puede irse del catálogo aunque la política esté encendida — y entonces el plato no casa con
     ninguna plantilla y no hay receta que enganchar. `library_select_enabled()` la endurece.
  3. Este módulo, que resuelve NOMBRE de plato → `template_id` → receta.

## Por qué nace apagado

Encenderlo cambia lo que el usuario lee en cada plato. El knob (`MEALFIT_RECIPE_LIBRARY_SELECT`,
default `False`) deja el código vivo y el comportamiento intacto, que es la convención de la casa:
lo que puede necesitar revertirse sin redeploy va como knob, no como hardcode.

Con el knob apagado, este módulo no se llama desde ninguna ruta de generación.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent / "data" / "registry"


def library_select_enabled() -> bool:
    """Knob maestro. Apagado ⇒ el motor sigue pidiéndole la receta al LLM, byte-idéntico a hoy."""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_RECIPE_LIBRARY_SELECT", False)
    except Exception:
        return False


def _norm(s) -> str:
    """Normaliza un nombre de plato para compararlo: sin acentos, sin puntuación, minúsculas.

    NO usa `pantry_names_match`: aquí se compara el nombre de un PLATO contra el de una plantilla,
    no la identidad de un alimento. Confundir las dos capas es la clase de error que este repo pagó
    en `P1-PANTRY-NAME-RESOLUTION`.
    """
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


@lru_cache(maxsize=8)
def _library(country: str = "DO") -> dict:
    p = _DIR / f"recipe_library_{str(country).lower()}_v1.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("por_id") or {}
    except Exception as e:
        logger.warning(f"[P1-RECIPE-LIBRARY-SELECT] biblioteca {country} ilegible: {e!r}")
        return {}


@lru_cache(maxsize=8)
def _name_index(country: str = "DO") -> dict:
    """`nombre normalizado` → `template_id`, sólo para las plantillas que TIENEN receta."""
    p = _DIR / f"dish_registry_{str(country).lower()}_v1.json"
    if not p.exists():
        return {}
    try:
        reg = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    lib = _library(country)
    idx = {}
    for t in reg.get("templates") or []:
        tid = t.get("template_id")
        if tid not in lib:
            continue
        nom = ((t.get("editorial") or {}).get("display_name") or {}).get("es") or t.get("name")
        if nom:
            idx[_norm(nom)] = tid
    return idx


def recipe_for_dish_name(name, country: str = "DO") -> Optional[list]:
    """Los pasos escritos para ese plato, o `None` si no es una plantilla con receta.

    Coincidencia EXACTA sobre el nombre normalizado, a propósito. Un parecido del 70 % serviría la
    receta de otro plato — y servir la receta equivocada es peor que no servir ninguna, que es la
    doctrina que este repo ya aplica a la lista de compras (`_remove_one_raw_line_by_food`).
    """
    if not library_select_enabled():
        return None
    tid = _name_index(country).get(_norm(name))
    if not tid:
        return None
    pasos = (_library(country).get(tid) or {}).get("pasos")
    return list(pasos) if isinstance(pasos, list) and pasos else None


def coverage(country: str = "DO") -> dict:
    """Cuántas plantillas del registry tienen receta. Para telemetría y para el test."""
    return {"con_receta": len(_name_index(country)), "recetas": len(_library(country))}
