# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-709 · 2026-09-28] Lo que siembra el cerrador de micros se usa en el Montaje, no sólo en una nota.

El cerrador de micros (`graph_orchestrator`, P1-MICRO-SEED / P2-SEED-STEP-NOTE) añade «10 g de semillas de linaza» o
«½ zanahoria» a la lista y su única instrucción en una nota 🌱 («espolvorea semillas de linaza sobre el plato al servir —
cierra tu omega-3 del día»). Las notas no son pasos (P1-PLAN-LOTE-24): en el replay del corpus, 176 de las 229 líneas
de la lista sin paso eran esto. Aquí, en el contrato final: el Montaje (o el último paso) gana «Espolvorea las semillas
de linaza por encima.» / «Acompaña con la zanahoria rallada.», y la nota se queda con el porqué. Sin cantidades: la
cantidad vive en la línea de la lista (P3-9). Si algún paso ya usa el alimento, o no se conoce su artículo, no se toca.
Knob `MEALFIT_SIEMBRA_EN_EL_MONTAJE` (True). tooltip-anchor: P1-PLAN-LOTE-709
"""
from __future__ import annotations

import re
import unicodedata

_NOTA_RE = re.compile(
    r"^🌱\s*Nota del Nutricionista AI:\s*(?P<verbo>espolvorea|acompaña el plato con)\s+(?P<food>.+?)"
    r"(?:\s+sobre (?:el plato|la bebida) al servir|\s+al servir)?\s+—\s+cierra tu\s+(?P<micro>.+?)\s+del día\.?\s*$")

# Artículo y número de lo que el cerrador siembra (`_MICRO_SEED_SOURCES`): conjunto cerrado; lo demás no se toca.
_ARTICULO = (
    ("semillas", "las", True), ("nueces", "las", True), ("almendras", "las", True), ("espinacas", "las", True),
    ("mani", "el", False), ("zanahoria", "la", False), ("auyama", "la", False),
)


def _sa(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def _on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SIEMBRA_EN_EL_MONTAJE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _es_nota(paso) -> bool:
    try:
        from recipe_contract import _es_nota as _n
        return _n(paso)
    except Exception:                                                          # noqa: BLE001
        s = str(paso or "")
        return "⚠" in s or "💡" in s or "🌱" in s


def _articulo(food: str):
    cab = _sa(food.lower()).split()[:1]
    for pal, art, plural in _ARTICULO:
        if cab and cab[0] == pal:
            return art, plural
    return None


def _usado(food: str, pasos: str) -> bool:
    claves = [w for w in _sa(food.lower()).split() if len(w) >= 4 and w not in ("semillas", "rallada", "cubos", "fileteadas")]
    return any(re.search(r"\b" + re.escape(w[:5]), pasos) for w in claves)


def integrar(meal) -> int:
    """Nº de siembras llevadas al Montaje; 0 ante cualquier error."""
    try:
        if not _on() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, paso in enumerate(list(rec)):
            m = _NOTA_RE.match(str(paso).strip())
            if not m:
                continue
            # la nota puede traer ya el artículo («espolvorea las semillas de linaza»): se quita y se repone el propio
            food = re.sub(r"^(?:las|los|la|el)\s+", "", m.group("food").strip(), flags=re.IGNORECASE)
            art = _articulo(food)
            if not art:
                continue
            pasos = _sa(" ".join(str(p) for p in rec if not _es_nota(p)).lower())
            if _usado(food, pasos):
                continue
            destino = next((j for j, s in enumerate(rec) if str(s).strip().lower().startswith("montaje")), None)
            if destino is None:
                destino = next((j for j in range(len(rec) - 1, -1, -1) if not _es_nota(rec[j])), None)
            if destino is None:
                continue
            articulo, plural = art
            accion = (f"Espolvorea {articulo} {food} por encima." if m.group("verbo") == "espolvorea"
                      else f"Acompaña con {articulo} {food}.")
            base = str(rec[destino]).rstrip()
            if not base.endswith((".", "!", "?")):
                base += "."
            rec[destino] = f"{base} {accion}"
            rec[i] = (f"🌱 Nota del Nutricionista AI: {articulo} {food} {'cierran' if plural else 'cierra'} tu "
                      f"{m.group('micro')} del día.")
            n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0
