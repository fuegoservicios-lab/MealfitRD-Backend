# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-614 · 2026-09-27] La leche de una cucharadita no es un ingrediente.

El motor de macros encoge la leche para cuadrar el día y el lote 311 completa la avena con agua: queda «5 ml de leche
descremada» junto a «80 ml de agua» (la validación del 592: «20 g de avena, 5 ml de leche descremada… cocina la avena con
la leche descremada, el agua y la canela»). Corpus de baterías: 22 comidas recientes (1-2 por plan, casi todas avenas y
batidos) y 104 antiguas con 1-10 ml de leche; una cucharadita que además entra a la lista de compras.

Aquí, en la cola del contrato (tras el agua del 311; en la cadena de assemble corre antes de la compra): una leche o
bebida vegetal de ≤ 10 ml sale de la lista (y de `ingredients_raw`, por alimento) y de los pasos —«y 5 ml de leche», «la
leche, el agua y la canela», «licúa el mango y la leche», «acompaña con la leche»—, y «cocina la avena con la leche» pasa
a «con el agua» cuando el plato tiene su línea de agua. Solo si queda otra base líquida: agua, yogur o hielo, o huevo en
la masa de unos panqueques o tortitas; la avena que se cocina exige su agua. TODO O NADA: si tras retirar queda una
mención de la leche en un paso (no una nota) o una frase coja («añade deja reposar», «con incorpora»), el plato queda
como estaba. Nunca con café o té, con otra leche en la lista, ni «unas gotas de leche» (eso es la receta). Los macros no
se recalculan: ≤ 6 kcal. Knob `MEALFIT_LECHE_INFIMA` (True). tooltip-anchor: P1-PLAN-LOTE-614
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

UMBRAL_ML = 10.0
_ML = {"ml": 1.0, "cdta": 5.0, "cdtas": 5.0, "cucharadita": 5.0, "cucharaditas": 5.0}
_LINEA = re.compile(r"^\s*(?P<q>\d+(?:[.,]\d+)?|½|¼)\s*(?P<u>ml|cdtas?|cucharaditas?)\s+de\s+(?P<f>(?:leche|bebida)\b.*)$",
                    re.IGNORECASE)
_LECHE_EN_LISTA = re.compile(r"\b(?:leche|bebida\s+(?:de|vegetal))\b", re.IGNORECASE)
_CAFE_TE = re.compile(r"\b(?:caf[eé]|t[eé]\b|infusi[oó]n|chocolate caliente|capuchino|latte)", re.IGNORECASE)
_AGUA = re.compile(r"\bagua\b", re.IGNORECASE)
_YOGUR = re.compile(r"\byogu?rt?\w*", re.IGNORECASE)
_HIELO = re.compile(r"\bhielo\b", re.IGNORECASE)
_HUEVO = re.compile(r"\b(?:huevos?|claras?)\b", re.IGNORECASE)
_MASA = re.compile(r"\b(?:panqueques?|pancakes?|tortitas?|crepes?|crepas?|waffles?|wafles?)\b", re.IGNORECASE)
_GRANO = r"(?:avena|quinoa|quinua|arroz|cebada|bulgur|trigo|ma[ií]z)"
_GRANO_RX = re.compile(r"\b" + _GRANO + r"\b", re.IGNORECASE)
_MEZCLA = re.compile(r"\b(?:mezcla|remoja|combina|incorpora|reposa|refrigera|hidrata|vierte|integra)\w*", re.IGNORECASE)
_FUEGO_RX = re.compile(r"\b(?:cocina|cocinar|cuece|cocer|hierve|hervir|hierva|calienta|lleva|microondas|a fuego|olla|"
                       r"hervor)\b", re.IGNORECASE)


class _Cocida:
    """¿Alguna frase nombra un grano y lo cocina, en cualquier orden?"""

    @staticmethod
    def search(texto) -> bool:
        return any(_GRANO_RX.search(c) and _FUEGO_RX.search(c) for c in re.split(r"(?<=[.;])\s+", str(texto or "")))


_COCIDA = _Cocida()
_REMOJO = re.compile(r"\b(?:remoj\w*|repos\w*|noche|nocturna|nevera|refrigera\w*|fr[ií]a|overnight|hidrat\w*)\b",
                     re.IGNORECASE)
_NOTAS = ("⚠", "💡", "🤰", "⚕", "🧊", "🛒", "🍽", "❄", "⏱", "🌱", "🥬", "🍠", "💪", "ℹ", "🛡")
#: la frase de la leche: artículo o cantidad y calificativos; la leche sin artículo tras «de» («unas gotas de leche») no
_NP = (r"(?:(?:(?:la|las|los|el)\s+)?(?:\d+(?:[.,]\d+)?|½|¼)\s*(?:ml|cdtas?|cucharaditas?)\s+de\s+|(?:la|las|los|el)\s+|"
       r"(?<!de\s))(?:leche|bebida\s+(?:de\s+\w+|vegetal))(?:\s+(?:descremada|semidesnatada|semidescremada|desnatada|entera|"
       r"evaporada|pasteurizada|uht|ultrapasteurizada|light|deslactosada|sin\s+lactosa|de\s+(?:soya|soja|almendras?|coco|"
       r"avena|arroz)|sin\s+az[uú]car)){0,3}\b")
_VERBO = (r"(?:incorpora|a[ñn]ade|agrega|cocina|mezcla|bate|lic[uú]a|vierte|deja|sirve|corta|pela|calienta|forma|coloca|"
          r"procesa|remueve|revuelve|tapa|lleva|hornea|dora|tuesta|reserva|retira|espolvorea|decora|termina|mide|pesa|"
          r"tritura|muele|unta|rellena|cuece|hierve|separa|lava|pica|ralla|exprime|rebana|desmenuza|casca|rompe|"
          r"escurre|enjuaga|sazona|acompa[ñn]a|licua)")
_REGLAS = [
    (re.compile(r"(?:,\s*|\s+y\s+)acompa[ñn]a\s+con\s+" + _NP, re.IGNORECASE), ""),       # «… y acompaña con la leche»
    (re.compile(r"\s+con\s+" + _NP + r"\s+y\s+(?=" + _VERBO + r"\b)", re.IGNORECASE), " y "),  # «bate el huevo con la leche y mezcla»
    (re.compile(r"\s+con\s+" + _NP + r"(?=\s*,\s*" + _VERBO + r"\b)", re.IGNORECASE), ""),   # «bate el huevo con la leche, incorpora»
    (re.compile(r"\s*,\s*" + _NP + r"(?=\s+y\s)", re.IGNORECASE), ""),                       # «A, la leche y B» → «A y B»
    (re.compile(_NP + r"\s*,\s+", re.IGNORECASE), ""),                                       # «la leche, el agua»
    (re.compile(_NP + r"\s+y\s+", re.IGNORECASE), ""),                                       # «la leche y el agua»
    # [P1-PLAN-LOTE-914 · 2026-09-29] «A, B y la leche» → «A y B» es para una ENUMERACIÓN de alimentos: B no lleva verbo ni
    # otra «y». Batería real rdv801: «…derretida, espolvorea la canela y acompaña con la guayaba fresca y la leche» se
    # volvía «…derretida y espolvorea la canela y acompaña con la guayaba fresca» (una frase coja más) y el TODO O NADA
    # dejaba «5 ml de leche pasteurizada» en el plato. tooltip-anchor: P1-PLAN-LOTE-914
    (re.compile(r",\s*(?P<x>(?:(?!\s+y\s)(?!\b" + _VERBO + r"\b)[^,.;])+?)\s+y\s+" + _NP, re.IGNORECASE), r" y \g<x>"),
    (re.compile(r"\s+y\s+" + _NP, re.IGNORECASE), ""),                                       # «A y la leche»
    (re.compile(r"\s+con\s+" + _NP, re.IGNORECASE), ""),                                     # «con 5 ml de leche»
]
_VERBO_Y = re.compile(r"\b(?P<v>a[ñn]ade|agrega|incorpora|vierte)\s+" + _NP + r"\s+y\s+(?=" + _VERBO + r"\b)(?P<w>[a-záéíóúñ])",
                      re.IGNORECASE)
_MIDE_Y_VERBO = re.compile(r"\b(?:mide|pesa|prepara|ten\s+(?:a\s+mano|list[oa]s?))\s+" + _NP + r"\s*,\s*(?=" + _VERBO + r"\b)",
                           re.IGNORECASE)
#: «cocina la avena con la leche [a fuego | en el microondas | y …]»: en una frase de grano SIN agua, la leche ínfima era
#: el líquido nombrado; el agua del plato (lote 311) ocupa su lugar
_SOLO_LIQUIDO = re.compile(r"\bcon\s+" + _NP + r"(?=\s+(?:a\s+fuego|en\s+|y\s+)|\s*[,.;])", re.IGNORECASE)
#: una frase coja tras retirar: dos verbos seguidos, preposición o conjunción ante un verbo o ante la puntuación
_COJA = re.compile(r"\b(?:con|de|y|a|en|al|del|" + _VERBO + r")\s+" + _VERBO + r"\b|\b(?:con|de|y|a|en)\s*[.,;]|"
                   r"\s[,.;]|,\s*[.;]|,\s*,", re.IGNORECASE)


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_LECHE_INFIMA", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _ml(linea):
    m = _LINEA.match(str(linea or ""))
    if not m:
        return None
    q = {"½": 0.5, "¼": 0.25}.get(m.group("q")) or float(m.group("q").replace(",", "."))
    return q * _ML.get(m.group("u").lower(), 1.0)


def _es_nota(p) -> bool:
    return not isinstance(p, str) or p.lstrip().startswith(_NOTAS)


def _sin_leche(paso: str, hay_agua: bool) -> str:
    s = paso
    if hay_agua:
        for clausula in re.split(r"(?<=[.;])\s+", paso):
            if re.search(r"\b" + _GRANO + r"\b", clausula, re.IGNORECASE) and not _AGUA.search(clausula) and \
                    _SOLO_LIQUIDO.search(clausula):
                s = s.replace(clausula, _SOLO_LIQUIDO.sub("con el agua", clausula, count=1), 1)
    s = _VERBO_Y.sub(lambda m: m.group("w").upper() if m.group("v")[:1].isupper() else m.group("w"), s)
    s = _MIDE_Y_VERBO.sub("", s)
    for rx, rep in _REGLAS:
        s = rx.sub(rep, s)
    return re.sub(r"\s{2,}", " ", s)


def _base_liquida(meal, ings) -> bool:
    lista = " ".join(str(x) for x in ings)
    pasos = [str(p) for p in (meal.get("recipe") or []) if not _es_nota(p)]
    texto = " ".join([str(meal.get("name") or "")] + pasos)
    hay_agua = bool(_AGUA.search(lista))
    if _MASA.search(str(meal.get("name") or "")):
        return bool(_HUEVO.search(lista) or _YOGUR.search(lista))   # la masa se liga con huevo o yogur
    if _COCIDA.search(texto):
        return hay_agua                                # el grano que se cocina necesita su agua (lote 311)
    if _REMOJO.search(texto):                          # remojo: el yogur o el agua van en la frase que mezcla la leche
        for p in pasos:
            for c in re.split(r"(?<=[.;])\s+", p):
                if _GRANO_RX.search(c) and re.search(r"\bleche\b", c, re.IGNORECASE) and _MEZCLA.search(c) and \
                        not (_YOGUR.search(c) or _AGUA.search(c)):
                    return False
    return hay_agua or bool(_YOGUR.search(lista)) or bool(_HIELO.search(lista))


def quitar(meal) -> int:
    """1 si quitó la leche ínfima del plato; 0 si no aplica o ante cualquier error."""
    try:
        if not activo() or not isinstance(meal, dict):
            return 0
        ings = meal.get("ingredients")
        rec = meal.get("recipe")
        if not isinstance(ings, list) or not isinstance(rec, list):
            return 0
        leches = [(i, x) for i, x in enumerate(ings) if isinstance(x, str) and _LECHE_EN_LISTA.search(x)]
        if len(leches) != 1:
            return 0                                   # ninguna, o dos leches: «la leche» del paso sería ambigua
        idx, linea = leches[0]
        ml = _ml(linea)
        if ml is None or ml > UMBRAL_ML:
            return 0
        if _CAFE_TE.search(" ".join([str(meal.get("name") or "")] + [str(p) for p in rec if not _es_nota(p)])):
            return 0                                   # (la nota clínica nombra el café sin servirlo)
        resto = [x for i, x in enumerate(ings) if i != idx]
        if not _base_liquida(meal, resto):
            return 0
        hay_agua = any(isinstance(x, str) and _AGUA.search(x) for x in resto)
        nuevos = [p if _es_nota(p) else _sin_leche(p, hay_agua) for p in rec]
        for a, b in zip(rec, nuevos):
            if _es_nota(b):
                continue
            if re.search(r"\bleche\b|\bbebida\s+(?:de|vegetal)\b", b, re.IGNORECASE) or \
                    len(_COJA.findall(b)) > len(_COJA.findall(a)):
                return 0                               # todo o nada: queda la leche (aun en un paso intacto) o una frase coja
        texto_nuevo = " ".join(str(p) for p in nuevos if not _es_nota(p))
        if _COCIDA.search(texto_nuevo) and not _AGUA.search(texto_nuevo) and \
                not _MASA.search(str(meal.get("name") or "")):
            return 0                                   # el grano se cocina y ya ningún paso nombra su agua
        import graph_orchestrator as go
        go._remove_one_raw_line_by_food(meal, linea, idx)   # la de raw por ALIMENTO, antes de tocar la lista visible
        ings.pop(idx)
        meal["recipe"] = nuevos
        meal.pop("_display", None)
        meal["_leche_infima_614"] = linea
        logger.info(f"🥛 [P1-PLAN-LOTE-614] «{linea}» fuera de «{str(meal.get('name'))[:50]}»")
        return 1
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-614] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["quitar"]
