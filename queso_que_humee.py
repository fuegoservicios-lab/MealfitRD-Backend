# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-807 · 2026-09-29] En el embarazo, el queso blanco que se puede dorar recibe su paso, no sólo la nota.

Lote 193: el revisor médico exigió (guía de los CDC para el queso fresco estilo latino) que el queso fresco o blando se
caliente hasta que humee, y se añadió la nota «⚠️ … caliéntalo hasta que humee (74 °C por dentro)». Pero la receta seguía
sirviéndolo frío: 2.ª batería real de embarazo del 29-sep, «Guiso ligero de lentejas… y queso blanco fresco» («sirve las
lentejas con la remolacha y el queso blanco pasteurizado en cubos») y «Batata asada con queso blanco fresco…»; corpus: de
70 comidas de embarazo con esa nota, en 24 el queso es de los que se doran (blanco, fresco, de hoja, mozzarella) y ningún
paso lo calienta. El queso frito es de lo más dominicano que hay: aquí, cuando se pone la nota, se inserta antes del
Montaje «💪 Dora el queso blanco en la sartén caliente… hasta que humee». No en un plato frío de vaso o batido (ahí el
queso caliente no tiene sitio: queda la nota), ni en el cottage/ricotta (no se doran), ni si un paso ya lo calienta.
Knob `MEALFIT_PREGNANCY_CHEESE_HEAT_STEP` (True). tooltip-anchor: P1-PLAN-LOTE-807
"""
from __future__ import annotations

import re
import unicodedata

_DORABLE_RE = re.compile(r"\bqueso\s+(?:blanco|fresco|de\s+hoja|de\s+freir|para\s+freir)|\bmozzarella\b")
_QUESO_RE = re.compile(r"\bqueso\b|\bmozzarella\b")
_CALOR_RE = re.compile(r"\b(?:dor\w*|plancha|sarten|calient\w*|calienta\w*|gratin\w*|derrit\w*|fund\w*|horne\w*|horno|asa\b|"
                       r"asal\w*|fri[eo]\w*|frit\w*|tuesta\w*|airfryer|microondas|humee|burbuje\w*|cocin\w*|salte\w*)")
#: platos fríos por construcción: un queso caliente no tiene sitio (queda la nota)
_FRIO_RE = re.compile(r"\b(?:vasito|vaso|batido|licuado|smoothie|parfait|frio|fria|frios|frias|helad\w*)\b")
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|nota\b)", re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PREGNANCY_CHEESE_HEAT_STEP", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


#: el queso entra en una masa o mezcla que luego se cuece entera («integra el queso…, forma pastelitos; al airfryer»)
_MEZCLA_RE = re.compile(r"\b(?:mezcl\w*|integr(?!al)\w*|incorpor\w*|combin\w*|amas\w*|masa|rellen\w*|pastelitos?|"
                        r"tortitas?|arepitas?|croquetas?|bollitos?|empanad\w*)")


def _calentado(pasos: list) -> bool:
    en_mezcla = False
    for p in pasos:
        if not isinstance(p, str) or _NOTA_RE.search(p):
            continue
        for cl in re.split(r"(?<=[.;])\s+", _sa(p)):   # [P1-PLAN-LOTE-52] con espacio: no corta «1.5 tazas»
            queso = bool(_QUESO_RE.search(cl))
            if (queso or en_mezcla) and _CALOR_RE.search(cl):
                return True
            if queso and _MEZCLA_RE.search(cl):
                en_mezcla = True
    return False


def insertar_paso(meal: dict) -> bool:
    """Inserta el paso de dorar el queso antes del Montaje si hace falta. True si lo insertó."""
    try:
        if not on() or not isinstance(meal, dict):
            return False
        pasos = meal.get("recipe")
        if not isinstance(pasos, list) or _FRIO_RE.search(_sa(meal.get("name"))):
            return False
        lista = " ; ".join(_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str))
        m = _DORABLE_RE.search(lista)
        if not m or _calentado(pasos):
            return False
        if "mozzarella" in m.group(0):
            paso = ("💪 Calienta la mozzarella pasteurizada sobre el pan caliente o en la sartén hasta que se derrita y humee "
                    "(74 °C por dentro).")
        else:
            nombre = "el queso de hoja pasteurizado" if "de hoja" in m.group(0) else "el queso blanco pasteurizado"
            paso = (f"💪 Dora {nombre} en la sartén caliente, 1-2 minutos por lado, hasta que humee y esté bien caliente "
                    f"por dentro (74 °C).")
        i = next((k for k, p in enumerate(pasos) if isinstance(p, str) and p.strip().lower().startswith("montaje")),
                 len(pasos))
        meal["recipe"] = pasos[:i] + [paso] + pasos[i:]
        return True
    except Exception:                                                          # noqa: BLE001
        return False
