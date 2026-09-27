# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-616 · 2026-09-27] Sin sustituto para la fruta de un plato salado, la fruta va al lado.

Producción, 27-sep 04:30-04:51 UTC (plan 3957a669, bloques de la semana 8-9, Nevera exigida con 41 artículos): el primer
intento de 4 de 5 generaciones cayó por «PAREO CHOCANTE FRUTA+SALADO» y cada rechazo REGENERÓ el plan entero (5 de 6
generaciones de 72 h necesitaron 3 intentos: 6,5-8,5 min y el triple de gasto). La autocorrección determinista
(`_fruit_savory_autofix`, la de antes del motor y la tardía de P1-FRUIT-SAVORY-BURN-FIX) cambia la fruta por aguacate,
tomate (lote 613) o batata… pero con la Nevera exigida ninguno estaba en la nevera, y sin sustituto no hacía NADA: el
revisor rechazaba lo mismo tres veces.

Aquí, cuando no hay sustituto admitido, la fruta se separa del plato como lo acepta el detector (lote 26:
«un componente SEPARADO dicho en el nombre no es una mezcla»): el nombre, que termina en la fruta, pasa a «… y mango al
lado», y el Montaje lo dice si no lo decía («Sirve el mango aparte.»). Sólo si ningún paso la cocina, la mezcla, la
licúa, la usa de decoración o la pone encima del plato (entonces la fruta ES parte del plato y se deja para el revisor), y
sólo si el nombre termina en ella. Los ingredientes y los macros no cambian. tooltip-anchor: P1-PLAN-LOTE-616
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_ARTICULO = {"mango": "el", "melon": "el", "mamey": "el", "zapote": "el", "pina": "la", "lechosa": "la", "papaya": "la",
             "guayaba": "la", "sandia": "la"}
_VISIBLE = {"melon": "melón", "pina": "piña", "sandia": "sandía"}
_COLA = r"(?:\s+(?:fresc[oa]s?|madur[oa]s?|picad[oa]s?|trocead[oa]s?|en\s+cubos|en\s+rodajas|en\s+trozos|en\s+l[aá]minas|" \
        r"en\s+gajos|natural(?:es)?))*"
_DENTRO = re.compile(r"\b(?:saltea|sofr[ií]e|cocina|hierve|hornea|mezcla|incorpora|a[ñn]ade|agrega|integra|rellena|lic[uú]a|"
                     r"bate|glasea|carameliza|asa|dora|tuesta|cuece|revuelve|combina|decora\w*|encima|sobre\s+(?:el|la|los|"
                     r"las)\b|por\s+encima|corona|cubre|dentro)\b", re.IGNORECASE)
_APARTE = re.compile(r"\b(?:aparte|al\s+lado|por\s+separado|acompa[ñn]a\w*|de\s+postre|como\s+postre)\b", re.IGNORECASE)
_NOTAS = ("⚠", "💡", "🤰", "⚕", "🧊", "🛒", "🍽", "❄", "⏱", "🌱", "🥬", "🍠", "💪", "ℹ", "🛡")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _fruta_rx(fruta: str):
    # «pina» cubre «piña»: el nombre visible se compara sin acentos, posición a posición (NFD de un carácter compuesto
    # conserva la longitud tras quitar la marca)
    return re.compile(r"\b" + re.escape(fruta) + r"s?\b" + _COLA, re.IGNORECASE)


def _separable(meal, fruta) -> bool:
    for p in meal.get("recipe") or []:
        if not isinstance(p, str) or p.lstrip().startswith(_NOTAS):
            continue
        for c in re.split(r"(?<=[.;])\s+", p):
            sc = _sa(c)
            if re.search(r"\b" + re.escape(fruta) + r"s?\b", sc) and _DENTRO.search(sc) and not _APARTE.search(sc):
                return False
    return True


def _renombrar(nombre: str, fruta: str):
    sa = _sa(nombre)
    m = None
    for m in _fruta_rx(fruta).finditer(sa):
        pass
    if not m or sa[m.end():].strip(" .") != "":
        return None                                    # la fruta no cierra el nombre: no se reescribe
    return nombre[:m.end()].rstrip() + " al lado"


def separar(days) -> int:
    """Nº de platos separados; 0 ante cualquier error."""
    hechos = 0
    try:
        from culinary_context import _meal_has_sweet_savory_clash, _SWEET_DOMINANT_FRUITS, _name_has_token
        for d in days or []:
            for meal in ((d.get("meals") or []) if isinstance(d, dict) else []):
                if not isinstance(meal, dict) or not _meal_has_sweet_savory_clash(meal):
                    continue
                nombre = str(meal.get("name") or "")
                fruta = next((f for f in _SWEET_DOMINANT_FRUITS if _name_has_token(f, _sa(nombre))), None)
                if not fruta or not _separable(meal, fruta):
                    continue
                nuevo = _renombrar(nombre, fruta)
                if not nuevo:
                    continue
                antes = (meal.get("name"), list(meal.get("recipe") or []))
                meal["name"] = nuevo
                rec = meal.get("recipe")
                if isinstance(rec, list) and not any(isinstance(p, str) and re.search(r"\b" + re.escape(fruta), _sa(p))
                                                     and _APARTE.search(_sa(p)) for p in rec):
                    frase = f"Sirve {_ARTICULO.get(fruta, 'la')} {_VISIBLE.get(fruta, fruta)} aparte."
                    i = next((k for k in range(len(rec) - 1, -1, -1) if isinstance(rec[k], str)
                              and rec[k].lstrip().lower().startswith("montaje")), None)
                    if i is None:
                        i = next((k for k in range(len(rec) - 1, -1, -1) if isinstance(rec[k], str)
                                  and not rec[k].lstrip().startswith(_NOTAS)), None)
                    if i is None:
                        rec.append(frase)
                    else:
                        rec[i] = rec[i].rstrip() + (" " if rec[i].rstrip().endswith(".") else ". ") + frase
                if _meal_has_sweet_savory_clash(meal):
                    meal["name"], meal["recipe"] = antes[0], antes[1]      # el detector no lo acepta: se deshace
                    continue
                meal["_slot_autofix_applied"] = "fruit_side"
                meal.pop("_display", None)
                hechos += 1
        if hechos:
            logger.info(f"🍈 [P1-PLAN-LOTE-616] {hechos} pareo(s) fruta+salado sin sustituto admitido: la fruta, al lado.")
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-616] no-op: {type(e).__name__}: {e}")
    return hechos


__all__ = ["separar"]
