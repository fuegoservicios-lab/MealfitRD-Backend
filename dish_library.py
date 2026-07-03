"""[P1-NEXT-LEVEL-BATCH · 2026-07-02] Biblioteca de platos — creatividad por RECOMBINACIÓN.

La creatividad del motor dependía 100% del LLM + prompt: los guards (raw-staple, dish-quality)
evitan disparates pero nada COMPONE. Esta biblioteca (data/dish_templates.json, ~85 plantillas
DD curadas con slot/proteína/base/técnica/transform) le da al day-generator un espacio
verificado del cual ELEGIR Y ADAPTAR — los platos transformados que el owner pidió
(panqueques de avena, bollitos de yuca, pastelón, arepitas) se vuelven DATA, no prompt-fe.

Integración: `build_dish_library_context(skeleton_day, day_num)` produce un bloque compacto
por día — muestra determinista (seed=day_num → mismo prompt para el mismo día = prompt-cache
preservado) filtrada por el pool de proteínas asignado por el planner y por slot (los slots
de cada plantilla RESPETAN el SSOT de constants.SLOT_INAPPROPRIATE_FOODS: cero arroz en
cena/desayuno, sopones solo almuerzo). El bloque es INSPIRACIÓN ("elige/adapta o crea uno
equivalente"), no obligación — el LLM conserva libertad creativa.

Fail-open total: sin archivo / JSON corrupto / knob OFF → ''. Knob: MEALFIT_DISH_LIBRARY
(default ON — prompt-aditivo, ~100-150 tokens por día).
tooltip-anchor: P1-NEXT-LEVEL-LIBRARY. Test: test_p1_next_level_batch.py.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re

from knobs import _env_bool, _env_int

logger = logging.getLogger(__name__)

DISH_LIBRARY_ENABLED = _env_bool("MEALFIT_DISH_LIBRARY", True)
DISH_LIBRARY_PER_SLOT = _env_int("MEALFIT_DISH_LIBRARY_PER_SLOT", 2, validator=lambda v: 1 <= v <= 5)

_TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dish_templates.json")
_CACHE: list | None = None

_SLOT_LABELS = {"desayuno": "Desayuno", "almuerzo": "Almuerzo", "cena": "Cena", "merienda": "Merienda"}


def load_dish_templates() -> list:
    """Carga (una vez) las plantillas. Fail-open → []."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with open(_TEMPLATES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        templates = data.get("templates") or []
        _CACHE = [t for t in templates if isinstance(t, dict) and t.get("name") and t.get("slots")]
    except Exception as _e:
        logger.warning(f"[P1-NEXT-LEVEL-LIBRARY] no se cargaron plantillas: {type(_e).__name__}: {_e}")
        _CACHE = []
    return _CACHE


def _protein_matches_pool(template_protein: str, pool_ascii: str) -> bool:
    """¿La proteína de la plantilla es compatible con el pool asignado por el planner?
    'none'/'mixta'/'queso'/'legumbre'/'huevo' son siempre compatibles (el SSOT del
    day-generator ya permite huevo/queso/legumbres como diversificadores)."""
    p = str(template_protein or "none").lower()
    if p in ("none", "mixta", "queso", "legumbre", "huevo"):
        return True
    return bool(re.search(r"\b" + re.escape(p), pool_ascii))


def sample_templates_for_slot(slot_key: str, pool_ascii: str, k: int, seed: int,
                              avoid_tokens: tuple = ()) -> list:
    """Muestra DETERMINISTA (seed) de hasta k plantillas del slot compatibles con el pool.
    Prioriza transformadas (transform=True) — son la creatividad que los staples no dan."""
    cands = []
    for t in load_dish_templates():
        if slot_key not in (t.get("slots") or []):
            continue
        if not _protein_matches_pool(t.get("protein"), pool_ascii):
            continue
        name_low = str(t.get("name", "")).lower()
        if any(tok and tok in name_low for tok in avoid_tokens):
            continue
        cands.append(t)
    if not cands:
        return []
    rng = random.Random(int(seed) * 1000003 + sum(ord(c) for c in slot_key))
    transformed = [t for t in cands if t.get("transform")]
    plain = [t for t in cands if not t.get("transform")]
    rng.shuffle(transformed)
    rng.shuffle(plain)
    # al menos 1 transformada si existe (la mitad del valor de la biblioteca es el transform)
    picked = (transformed[:max(1, k - 1)] + plain)[:k]
    return picked


def build_dish_library_context(skeleton_day: dict, day_num: int) -> str:
    """Bloque de inspiración por día para el prompt del day-generator. '' si knob OFF /
    sin plantillas compatibles. Determinista por (día, pool) → prompt-cache friendly."""
    if not DISH_LIBRARY_ENABLED or not isinstance(skeleton_day, dict):
        return ""
    try:
        from constants import strip_accents
        pool_ascii = strip_accents(", ".join(
            str(x) for x in (skeleton_day.get("protein_pool") or [])).lower())
        meal_types = skeleton_day.get("meal_types") or ["Desayuno", "Almuerzo", "Merienda", "Cena"]
        lines = []
        for mt in meal_types:
            slot = strip_accents(str(mt).strip().lower())
            if slot not in _SLOT_LABELS:
                continue
            picks = sample_templates_for_slot(slot, pool_ascii, int(DISH_LIBRARY_PER_SLOT), int(day_num or 1))
            if not picks:
                continue
            entries = "; ".join(
                f"{t['name']} ({t.get('technique', 'libre')})" for t in picks
            )
            lines.append(f"   • {_SLOT_LABELS[slot]}: {entries}")
        if not lines:
            return ""
        return (
            "\n🍽️ INSPIRACIÓN DOMINICANA (biblioteca curada — ELIGE Y ADAPTA una, o crea un plato "
            "equivalente en espíritu; ajusta porciones a los macros del día):\n"
            + "\n".join(lines)
            + "\n   💡 Prioriza preparaciones TRANSFORMADAS (masas, guisos, rellenos, horneados) "
              "sobre staples sueltos — un plato con nombre propio se disfruta y se repite.\n"
        )
    except Exception as _e:
        logger.debug(f"[P1-NEXT-LEVEL-LIBRARY] contexto no-op: {type(_e).__name__}: {_e}")
        return ""
