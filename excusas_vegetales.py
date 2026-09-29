# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-247 · 2026-09-25] «Maní molido hasta obtener una crema» no es un lácteo.

Batería real del 25-sep (vegana, México): la guarda de dieta rechazó CRÍTICO «25 g de maní molido hasta obtener una
crema» como LÁCTEO por la palabra «crema» y quemó un intento (y si reincide, el siguiente paso es el plan de
emergencia). Los escáneres de dieta y de alérgenos solo perdonaban el análogo vegetal DETRÁS del término («crema de
maní», `_PLANT_ADJ_EXCUSE_RX`); aquí el vegetal va DELANTE y la «crema» es el RESULTADO de procesarlo.

Acotado a propósito: solo «crema»/«mantequilla», solo si lo inmediatamente anterior dice que es el resultado
(«hasta obtener/formar/hacer una…», «convertido en…») y solo si antes, en la MISMA línea, hay un fruto seco, semilla o
legumbre. «Fresas con almendras y crema batida» sigue siendo lácteo. tooltip-anchor: P1-PLAN-LOTE-247-CREMA-VEGETAL
"""
from __future__ import annotations

import re

_TERMINOS = ("crema", "mantequilla")
_VEGETAL_RX = re.compile(
    r"\b(?:mani|cacahuate|cacahuete|almendras?|nuez|nueces|avellanas?|maranon|anacardos?|cajuil|pistachos?|"
    r"semillas?|ajonjoli|sesamo|girasol|calabaza|chia|linaza|coco|soya|soja|garbanzos?|avena|aguacate|tahini|"
    r"habichuelas?|frijol(?:es)?|lentejas?|guandules?|gandules?|arvejas?|auyama|batata|yuca|papas?)\b")
_RESULTADO_RX = re.compile(
    r"(?:obtener|formar|hacer|lograr|conseguir|quede|queden|quedar|convertid[oa]s?\s+en|convertir(?:lo|la|los|las)?"
    r"\s+en|reducid[oa]s?\s+a|hecho\s+una?|hecha\s+una?)\s+(?:una?\s+)?$")


def prefijo_vegetal_excusa(termino: str, antes: str) -> bool:
    """True si `termino` (sin acentos) es la crema/mantequilla que RESULTA de moler un vegetal nombrado en `antes`."""
    try:
        t = str(termino or "").strip().lower()
        if not any(t == x or t.startswith(x) for x in _TERMINOS):
            return False
        a = str(antes or "").lower()
        return bool(_RESULTADO_RX.search(a) and _VEGETAL_RX.search(a))
    except Exception:
        return False


# ─── [P1-PLAN-LOTE-262 · 2026-09-25] el ADJETIVO «tostada(s)» y el wrap hecho DE hojas ──────────────────────────────
# Batería final (sin gluten + huevo) y medido contra el código: el escáner de alérgenos marcaba como GLUTEN «almendras/
# nueces/avellanas tostadas», «semillas de sésamo tostadas», «pepitas de auyama tostadas», «wrap de lechuga» y «hojas de
# lechuga para wrap» — el plan del celíaco se rechazaba (reintento y, si se repetía, el de EMERGENCIA; el F2 lo dejó
# escrito como coste aceptado). Peor aún: la sustitución proactiva del gluten (tostada→Casabe, por subcadena) convertía
# «10 g de semillas de sésamo tostadas» en «10 g de Casabe» y los pasos en «mide 10 g de semillas de sésamo casabe».
# Los TÉRMINOS se quedan («1 tostada», «tostadas integrales», «1 wrap», «wrap integral» siguen siendo pan): se excusa solo
# el adjetivo pegado a un fruto seco o una semilla, y el wrap hecho DE hojas sin tortilla, pan, harina ni trigo en la
# línea. Acotado al término, como la sémola y la tostada de casabe (lote 77). tooltip-anchor: P1-PLAN-LOTE-262-ADJETIVO
_SEMILLA = (r"almendras?|nuez|nueces|avellanas?|pistachos?|anacardos?|maranon(?:es)?|cajuil(?:es)?|cacahuates?|"
            r"cacahuetes?|mani(?:es)?|pepitas?|semillas?|ajonjoli|sesamo|linaza|chia|girasol|calabaza|auyama|coco|"
            r"macadamias?|pecanas?|castanas?|pinon(?:es)?|cacao|amapola")
_TOSTADA_ADJETIVO_RX = re.compile(
    r"\b(?:" + _SEMILLA + r")(?:\s+de\s+(?:" + _SEMILLA + r"))?"
    r"(?:\s+[a-z]+(?:adas?|idas?|ados?|idos?)|\s+enteras?)?\s*$")
_HOJA = r"lechugas?|repollos?|col(?:es)?|acelgas?|berzas?|nori"
_WRAP_DE_HOJA_RX = re.compile(r"^\s*de\s+(?:hojas?\s+de\s+)?(?:" + _HOJA + r")\b")
_HOJA_PARA_WRAP_RX = re.compile(
    r"\b(?:" + _HOJA + r")(?:\s+[a-z]+)?\s+(?:para|como|en\s+forma\s+de|a\s+modo\s+de)\s+"
    r"(?:(?:el|los|la|las|un|unos|una|unas|hacer|armar|formar)\s+)*$")
_PAN_RX = re.compile(r"\b(?:tortillas?|pan(?:es)?|harinas?|trigo|pitas?|arabes?|integral(?:es)?|wheat)\b")
_TOSTADA_RX = re.compile(r"(?<![a-z0-9])tostadas?(?![a-z0-9])")
# [P1-PLAN-LOTE-269] Alternativas vegetales de la carne oculta y de la miel, ACOTADAS al término (una excusa genérica
# «de maple» absolvería al «tocino de maple»).
_GELATINA_VEGETAL_RX = re.compile(r"^\s*(?:de\s+)?(?:agar(?:[\s-]*agar)?|pectina|vegetal|vegana|origen\s+vegetal)\b")
_MIEL_VEGETAL_RX = re.compile(r"^\s*de\s+(?:agave|cana|maple|arce|palma|datil(?:es)?|coco|yacon)\b")
# [P1-PLAN-LOTE-796 · 2026-09-28] «manchego» suelto es el queso («30 g de manchego»), pero en «pisto manchego» es el
# gentilicio de un sofrito de verduras sin lácteo: lo marcaba como lácteo/lactosa y como no vegano.
_PISTO_RX = re.compile(r"\bpistos?\s+$")
# Y «macha» es una almeja chilena (vocabulario_mar), pero la «salsa macha» mexicana es aceite de chile con cacahuate.
_SALSA_RX = re.compile(r"\bsalsas?\s+$")
# El chocolate o el chile «para mole» (alias de la fila «Chocolate de mesa») es un INGREDIENTE del mole, no su pasta:
# «2 cdas de pasta para mole» sigue siendo mole. [ronda 3] Y la «pasta/salsa de chiles para mole» también: es la pasta
# (la comercial lleva cacahuate, ajonjolí y pan), no el chile suelto.
_PARA_MOLE_RX = re.compile(r"\b(?:chocolates?|chiles?)(?:\s+(?!pastas?\b|salsas?\b|y\b|e\b|con\b)[a-z]+){0,3}\s+para\s+$")
_PASTA_DE_RX = re.compile(r"\b(?:pastas?|salsas?|mezclas?|polvos?|bases?|concentrados?|preparados?)\s+(?:de\s+)?$")


def _para_mole(antes: str) -> bool:
    m = _PARA_MOLE_RX.search(antes)
    return bool(m) and not _PASTA_DE_RX.search(antes[:m.start()])


# [P1-PLAN-LOTE-796 · ronda 3 · 2026-09-28] El capuchino, el latte o la bechamel hechos con leche VEGETAL: el nombre del
# preparado marcaba lácteo (al vegano y al alérgico) aunque la línea dijera con qué leche iba («1 capuchino con leche de
# avena», «1 latte con bebida de almendras», «bechamel de coliflor»). Se excusa el NOMBRE del preparado cuando la misma
# línea nombra la leche/bebida vegetal que lo hace; la leche de vaca que también nombre la línea se sigue marcando por su
# propio término. La bechamel es además GLUTEN (la harina del roux): con leche vegetal sólo se le excusa a la dieta
# (aquí, `dieta=True`) y al alérgico a lácteos (`termino_compartido.excusa_de_su_clase`); la «de coliflor» (puré de
# verdura, sin roux ni leche) vale para todos. tooltip-anchor: P1-PLAN-LOTE-796-PREPARADO-VEGETAL
_PREPARADOS = ("capuchino", "cappuccino", "latte", "bechamel")
_LECHE_VEGETAL_RX = re.compile(
    r"\b(?:leches?|bebidas?|cremas?)\s+(?:de\s+)?(?:vegetal(?:es)?|avena|almendras?|soya|soja|coco|arroz|nuez|nueces|"
    r"avellanas?|anacardos?|maranon(?:es)?|cajuil|mani|cacahuates?|cacahuetes?|guisantes?|arvejas?|chufa|quinoa|"
    r"linaza|ajonjoli|sesamo|pistachos?|macadamias?|canamo)\b")
_BECHAMEL_DE_VERDURA_RX = re.compile(r"^\s*de\s+(?:coliflor(?:es)?|brocoli|auyama|calabaza|calabacin|papas?|patatas?|"
                                     r"yuca|batata|anacardos?|maranon(?:es)?)\b")


def leche_vegetal_en(linea: str) -> bool:
    """¿La línea (sin acentos, minúsculas) nombra una leche, bebida o crema VEGETAL?"""
    return bool(_LECHE_VEGETAL_RX.search(str(linea or "")))


# [ronda 3] En DO «tortilla de yuca» es también el CASABE (pan de yuca, sin huevo): «1 porción de casabe (tortilla de
# yuca)» es línea real de la batería. La de huevo con yuca rallada no nombra el casabe.
_CASABE_RX = re.compile(r"\bcasabes?\b")
# [ronda 3] «nueces y semillas» es la mezcla (maní); «nueces y semillas de linaza» nombra la nuez y UNA semilla.
_DE_SEMILLA_RX = re.compile(r"^\s*de\s+(?:linaza|chia|girasol|calabaza|auyama|sesamo|ajonjoli|canamo|amapola)\b")


def excusa_contextual(termino: str, linea: str, ini: int, fin: int, dieta: bool = False) -> bool:
    """¿El término de alérgeno que casó en `linea[ini:fin]` (sin acentos, minúsculas) es inocuo por su CONTEXTO?

    Reúne la excusa del 247 (la crema que resulta de moler un vegetal) y las del 262 (el adjetivo «tostada(s)» tras un
    fruto seco o una semilla; el wrap hecho DE hojas). Acotada al TÉRMINO: nunca absuelve a otro que case en la línea.
    `dieta=True` lo llama el escáner de dieta, que sólo pregunta por el producto animal (la bechamel con leche de avena
    no lo lleva, aunque siga llevando harina)."""
    try:
        t = str(termino or "").strip().lower()
        s = str(linea or "")
        if prefijo_vegetal_excusa(t, s[:ini]):
            return True
        if t == "tostada":
            return bool(_TOSTADA_ADJETIVO_RX.search(s[:ini]))
        if t == "wrap":
            if _PAN_RX.search(s):
                return False
            return bool(_WRAP_DE_HOJA_RX.match(s[fin:]) or _HOJA_PARA_WRAP_RX.search(s[:ini]))
        if t in ("gelatina", "grenetina"):                     # [P1-PLAN-LOTE-269] gelatina de agar/pectina
            return bool(_GELATINA_VEGETAL_RX.match(s[fin:]))
        if t == "miel":                                        # [P1-PLAN-LOTE-269] miel de agave/caña/maple
            return bool(_MIEL_VEGETAL_RX.match(s[fin:]))
        if t == "manchego":                                    # [P1-PLAN-LOTE-796] el pisto manchego
            return bool(_PISTO_RX.search(s[:ini]))
        if t == "macha":                                       # [P1-PLAN-LOTE-796] la salsa macha no es la almeja
            return bool(_SALSA_RX.search(s[:ini]))
        if t == "mole":                                        # [P1-PLAN-LOTE-796] chocolate para mole
            return _para_mole(s[:ini])
        if t == "bechamel":                                    # [ronda 3] bechamel de coliflor · con leche vegetal
            return bool(_BECHAMEL_DE_VERDURA_RX.match(s[fin:]) or (dieta and leche_vegetal_en(s)))
        if t in _PREPARADOS:                                   # [ronda 3] capuchino con leche de avena
            return leche_vegetal_en(s)
        if t == "tortilla de yuca":                            # [ronda 3] el casabe
            return bool(_CASABE_RX.search(s))
        if t == "nueces y semillas":                           # [ronda 3] nueces y semillas de linaza
            return bool(_DE_SEMILLA_RX.match(s[fin:]))
        return False
    except Exception:
        return False


def sustitucion_excusada(sub, linea: str) -> bool:
    """[P1-PLAN-LOTE-262] True si en `linea` (sin acentos, minúsculas) la ÚNICA razón para aplicar la sustitución `sub`
    es un «tostada(s)» que es adjetivo de un fruto seco o semilla: «semillas de sésamo tostadas» no es pan tostado."""
    try:
        toks = [str(x) for x in ((sub or {}).get("tokens") or ()) if str(x) in linea]
        if not toks or any(not x.startswith("tostada") for x in toks):
            return False
        apariciones = list(_TOSTADA_RX.finditer(linea))
        return bool(apariciones) and all(excusa_contextual("tostada", linea, m.start(), m.end()) for m in apariciones)
    except Exception:
        return False
