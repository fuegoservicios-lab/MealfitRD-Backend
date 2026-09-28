# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-630 · 2026-09-27] Los «panqueques salados» del almuerzo y la cena son tortitas.

Producción, 20-27 sep (plan con Nevera exigida: la harina de trigo era su carbohidrato): cuatro rechazos «COMIDA FUERA DE
HORARIO… es comida de desayuno en la cena (cereal/panqueque/waffle/avena)» sobre «Tilapia a la plancha con panqueques
salados de harina y tayota al limón», «Panqueques de harina con pescado blanco a la plancha cítrica…», y cada rechazo
REGENERÓ el plan entero. La harina de trigo rota como base a propósito (P1-FLOURS-POOLS: «con la harina haces panqueques,
bollos, arepas») y los propios pasos dicen «cocina tortitas finas 2-3 min por lado»: el plato es el mismo, lo que
confunde es la palabra, que el horario lee como desayuno. En la batería del 592 la cena «Tortitas saladas de trigo con
revuelto de huevo…» pasó sin tropiezo.

Aquí, antes del motor (junto a la autocorrección de frituras de la cena): en el almuerzo o la cena, un plato SALADO que
se llama «panqueques»/«pancakes» pasa a «tortitas» en el nombre y en los pasos, con la concordancia del alimento nuevo
(lote 612: «salados» → «saladas», «voltéalos» → «voltéalas»). Unos panqueques DULCES (miel, fruta dulce, chocolate…)
son desayuno de verdad: no se tocan y el horario los sigue señalando. Knob `MEALFIT_DINNER_PANCAKE_RENAME` (True).
tooltip-anchor: P1-PLAN-LOTE-630
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_PANQ = re.compile(r"\b(?P<w>[Pp]anqueques?|[Pp]ancakes?)\b")
_DULCE = re.compile(r"\b(?:miel|az[uú]car|chocolate|cacao|sirope|jarabe|mermelada|dulce|guineo maduro|pl[aá]tano maduro|"
                    r"mango|lechosa|papaya|fresas?|pi[ñn]a|guayaba|mel[oó]n|kiwi|manzana|pera|uvas?|ar[aá]ndanos?|canela|"
                    r"vainilla|nutella|leche condensada|yogu?rt?\w*)\b", re.IGNORECASE)
_SALADO = re.compile(r"\b(?:salad[oa]s?|pollo|pescado|tilapia|mero|at[uú]n|sardinas?|huevos?|revoltillo|revuelto|jam[oó]n|"
                     r"pavo|res|carne|cerdo|camarones|queso de hoja|queso de fre[ií]r|tayota|vegetales|cebolla|ajo|"
                     r"berenjena|espinacas?|habichuelas?|lentejas?|guiso|guisad[oa]s?|plancha)\b", re.IGNORECASE)


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DINNER_PANCAKE_RENAME", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _tortita(m) -> str:
    w = m.group("w")
    nuevo = "tortitas" if w.lower().endswith("s") else "tortita"
    return nuevo[:1].upper() + nuevo[1:] if w[:1].isupper() else nuevo


def _renombra(texto: str) -> str:
    import concordancia_sustituto as cs
    nuevo = _PANQ.sub(_tortita, texto)
    if nuevo == texto:
        return texto
    # el artículo de delante: «los panqueques» → «las tortitas», «un panqueque» → «una tortita»
    nuevo = re.sub(r"\b([Ll])os(\s+tortitas)\b", r"\1as\2", nuevo)
    nuevo = re.sub(r"\b([Uu])nos(\s+tortitas)\b", r"\1nas\2", nuevo)
    nuevo = re.sub(r"\b([Ee])l(\s+tortita)\b", lambda m: ("La" if m.group(1) == "E" else "la") + m.group(2), nuevo)
    nuevo = re.sub(r"\b([Uu])n(\s+tortita)\b", r"\1na\2", nuevo)
    for viejo, nombre in (("panqueques", "tortitas"), ("panqueque", "tortita")):
        nuevo = cs.concordar(nuevo, nombre, viejo)
    return nuevo


def renombrar(days) -> int:
    """Nº de platos renombrados; 0 ante cualquier error."""
    hechos = 0
    try:
        if not activo():
            return 0
        from constants import canonical_slot_key
        for d in days or []:
            for meal in ((d.get("meals") or []) if isinstance(d, dict) else []):
                if not isinstance(meal, dict) or canonical_slot_key(meal.get("meal", "")) not in ("almuerzo", "cena"):
                    continue
                nombre = str(meal.get("name") or "")
                if not _PANQ.search(nombre):
                    continue
                todo = " ".join([nombre] + [str(x) for x in (meal.get("ingredients") or [])])
                if _DULCE.search(_sa(todo)) or not _SALADO.search(_sa(nombre)):
                    continue                           # panqueques dulces: desayuno de verdad
                meal["name"] = _renombra(nombre)
                rec = meal.get("recipe")
                if isinstance(rec, list):
                    meal["recipe"] = [_renombra(p) if isinstance(p, str) else p for p in rec]
                meal["_slot_autofix_applied"] = "pancake_tortitas"
                meal.pop("_display", None)
                hechos += 1
        if hechos:
            logger.info(f"🫓 [P1-PLAN-LOTE-630] {hechos} «panqueques» salados de almuerzo/cena → «tortitas» (horario).")
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-630] no-op: {type(e).__name__}: {e}")
    return hechos


__all__ = ["renombrar"]
