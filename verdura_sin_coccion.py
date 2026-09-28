# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-540 · 2026-09-27] La verdura de la lista que ningún paso cocina.

Batería real del 27-sep leída entera (DM2 con insulina, celíaco): «corta 40 g de berenjena y 120 g de vainitas» y el
Toque de Fuego sólo añade la berenjena al guiso —las vainitas llegan crudas al plato—; «separa ½ taza de brócoli» en unas
lentejas guisadas que nunca lo reciben; «lava ½ taza de espinacas» en un pollo guisado que nunca las lleva. El lote 444
cubre los víveres; la verdura no tenía reparador. Aquí, en un plato con Toque de Fuego:
- vainitas, berenjena, tayota, molondrones (no se comen crudas) y brócoli/coliflor (fuera de una ensalada): una
  «💡 Cocción previa» tras el Mise en place con su hervor o salteado —su verbo la vuelve «cocida» para el detector:
  idempotente—;
- espinacas en un guiso o una salsa: «Añade las espinacas… en los últimos 2-3 minutos» antes del Montaje.
Una verdura que ALGUNA frase de los pasos cocina (con verbo o minutos) no se toca. tooltip-anchor: P1-PLAN-LOTE-540
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_NOTAS = ("⚠", "💡", "🤰", "⚕", "🧊", "🛒", "🍽", "❄", "⏱", "🌱", "🥬", "🍠")

# clave → (patrón en la lista, nombre con artículo, texto de la cocción previa); `None` = sólo al guiso
_VERDURAS_540 = (
    ("vainitas", r"\bvainitas?\b|\bhabichuelas?\s+tiernas?\b", "las vainitas",
     "hierve las vainitas 6-8 minutos en agua con sal, hasta que estén tiernas, y escúrrelas"),
    ("berenjena", r"\bberenjenas?\b", "la berenjena",
     "saltea la berenjena en una sartén con unas gotas de aceite 8-10 minutos, hasta que esté tierna"),
    ("tayota", r"\btayotas?\b", "la tayota", "hierve la tayota 10-12 minutos, hasta que esté tierna, y escúrrela"),
    ("molondron", r"\bmolondron(?:es)?\b", "los molondrones",
     "hierve los molondrones 8-10 minutos, hasta que estén tiernos, y escúrrelos"),
    ("brocoli", r"\bbrocolis?\b", "el brócoli",
     "hierve el brócoli 4-5 minutos (o cocínalo al vapor), hasta que esté tierno, y escúrrelo"),
    ("coliflor", r"\bcoliflor(?:es)?\b", "la coliflor", "hierve la coliflor 6-8 minutos, hasta que esté tierna, y escúrrela"),
)
# en una ensalada sí se comen crudos. [P1-PLAN-LOTE-560 · 2026-09-27] + tayota: batería real con «Nada» de tiempo,
# «Pollo cítrico a la plancha con maíz y ensalada fresca de tayota» («corta la tayota en láminas finas… combina la
# tayota, el tomate y los rábanos en una ensalada») recibía «💡 hierve la tayota 10-12 minutos» y el plato pasaba
# de 10 a 20 min contra lo que respondió el usuario. La tayota cruda en láminas es comestible (vainitas, berenjena y
# molondrones no). tooltip-anchor: P1-PLAN-LOTE-560
_CRUDO_OK_540 = ("brocoli", "coliflor", "tayota")
_ENSALADA_560 = re.compile(r"\bensalada\b|\bcrud[oa]s?\b|\bfresc[oa]s?\b")
_ESPINACA_540 = re.compile(r"\bespinacas?\b")
_COCCION_540 = re.compile(
    r"\b(?:hierv\w*|herv\w*|cocin\w*|coce\w*|cuec\w*|salte\w*|sofri\w*|sofre\w*|asa|asal\w*|asad\w*|horne\w*|dora|dor[ae]n?"
    r"|guis\w*|sancoch\w*|vapor|microondas|calient\w*|escald\w*|rehog\w*|marchit\w*|ablande\w*)\b"
    r"|\b\d+(?:\s*[-–]\s*\d+)?\s*min", re.IGNORECASE)
# un guiso o una salsa DE VERDAD: «tuesta el pan en una sartén seca» no es donde van las espinacas
_GUISO_540 = re.compile(r"\b(?:guis\w*|salsa|sofri\w*|sofre\w*)\b", re.IGNORECASE)
# [P1-PLAN-LOTE-632 · 2026-09-28] Un participio o un adjetivo describe un ESTADO, no cuece. Validación del 592
# (estudiante, día 3): «Lentejas guisadas… con plátano maduro, brócoli al vapor y pechuga de pollo» —el Mise separa el
# brócoli, ningún paso lo cocina y el Montaje lo sirve «al lado»— quedaba sin su cocción previa porque esa misma frase
# del Montaje decía «las lentejas guisadas»: `guis\w*` contaba el adjetivo de OTRO alimento como la cocción del brócoli.
# Lo mismo con «las tortitas horneadas», «el pollo asado», «sirve caliente» y el «al vapor» que sólo repite el nombre en
# la frase que sirve. tooltip-anchor: P1-PLAN-LOTE-632
_DESCRIPTIVO_632 = re.compile(r"^(?:\w+(?:ad|id)[oa]s?|calientes?)$")
_SIRVE_632 = re.compile(r"\b(?:montaje|sirve\w*|acompan\w*|emplat\w*)\b")


def _cuece(f: str) -> bool:
    """[P1-PLAN-LOTE-632] ¿La frase `f` (ya sin acentos) cuece algo? Un verbo o unos minutos sí; un participio
    («guisadas», «horneado», «hervida») o «caliente» no, y «vapor» tampoco en la frase que sirve."""
    servir = bool(_SIRVE_632.search(f))
    for m in _COCCION_540.finditer(f):
        t = m.group(0).lower()
        if _DESCRIPTIVO_632.match(t) or (t == "vapor" and servir):
            continue
        return True
    return False


def _servida_cruda_663(pat: str, f: str) -> bool:
    """[P1-PLAN-LOTE-663 · 2026-09-28] ¿La frase sirve ESTA verdura cruda? Una ensalada, o «crudo/fresco» pegado a la
    verdura («el brócoli crudo», «coliflor fresca»). Batería real del 28-sep sobre el 636 (estudiante, día 3): «Canoas…
    rellenas de queso fresco y brócoli al ajo» con «…con el queso blanco fresco y el brócoli al ajo como relleno» — el
    «fresco» era del QUESO y el brócoli (250 g, que ningún paso cocinaba) salía crudo sin su cocción previa.
    tooltip-anchor: P1-PLAN-LOTE-663"""
    if re.search(r"\bensalada\b", f):
        return True
    return bool(re.search(r"(?:" + pat + r")(?:\s+[a-z]+){0,2}?\s+(?:crud|fresc)[oa]s?\b", f)
                or re.search(r"\b(?:crud|fresc)[oa]s?\s+(?:de\s+)?(?:" + pat + r")", f))


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _es_nota(p: str) -> bool:
    return str(p).lstrip().startswith(_NOTAS)


def _frases(rec) -> list:
    """Las frases de los pasos (sin notas, salvo las «💡 Cocción previa», que sí cuecen)."""
    out = []
    for p in rec:
        if not isinstance(p, str):
            continue
        t = _sa(p)
        if _es_nota(p) and "coccion previa" not in t:
            continue
        out += [f for f in re.split(r"(?<=[.])\s+", t) if f.strip()]
    return out


def cocer(meal) -> int:
    """Nº de pasos añadidos; 0 ante cualquier error (fail-open)."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        ings = [_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        if not isinstance(rec, list) or not rec or not ings:
            return 0
        toque = [p for p in rec if isinstance(p, str) and _sa(p).lstrip().startswith("el toque de fuego")]
        if not toque or not any(_COCCION_540.search(_sa(p)) for p in toque):
            return 0                                           # plato frío: nada que cocer aquí
        frases = _frases(rec)
        # un Toque que nombra la verdura y cocina la cuenta como cocida aunque el verbo esté en otra frase: «mezcla el
        # calabacín, el brócoli… Asa en una bandeja… hasta que los vegetales estén tiernos» (replay del corpus)
        frases += [_sa(p) for p in toque if _COCCION_540.search(_sa(p))]
        nombre_plato = _sa(" ".join(str(meal.get(k) or "") for k in ("name", "desc", "description")))
        ensalada = bool(re.search(r"\bensalada\b|\bcrud[oa]s?\b", nombre_plato))
        previas, finales = [], []
        for clave, pat, con_art, texto in _VERDURAS_540:
            if not any(re.search(pat, x) for x in ings):
                continue
            # [P1-PLAN-LOTE-560] la ensalada también la dicen los PASOS («combina la tayota… en una ensalada»)
            if clave in _CRUDO_OK_540 and (ensalada or any(re.search(pat, f) and _servida_cruda_663(pat, f)
                                                           for f in frases)):
                continue
            # la receta tiene que usarla (el Mise la corta, un paso la nombra): una línea que ningún paso nombra es otra
            # clase —un añadido huérfano—, no una cocción que falta
            if not any(re.search(pat, f) for f in frases):
                continue
            if any(re.search(pat, f) and _cuece(f) for f in frases):   # [P1-PLAN-LOTE-632] verbo, no participio
                continue
            previas.append("💡 Cocción previa: " + texto + ".")
        # las espinacas que el paso sirve frescas («Acompaña con las espinacas frescas», «al lado», «crudas») se quedan así
        frescas = any(_ESPINACA_540.search(f) and re.search(r"\bfresc|\bcrud|\bal lado\b|\bacompa\w*\s+con\s+(?:las\s+)?espinaca", f)
                      for f in frases)
        if any(_ESPINACA_540.search(x) for x in ings) and not ensalada and not frescas \
                and any(_GUISO_540.search(_sa(p)) for p in toque) \
                and any(_ESPINACA_540.search(f) for f in frases) \
                and not any(_ESPINACA_540.search(f) and _cuece(f) for f in frases):
            finales.append("🥬 Añade las espinacas a la sartén o al guiso en los últimos 2-3 minutos, hasta que se "
                           "ablanden, y mézclalas antes de servir.")
        previas = [t for t in previas if t not in rec]
        finales = [t for t in finales if t not in rec]
        if not previas and not finales:
            return 0
        if previas:
            i = next((k for k, p in enumerate(rec) if isinstance(p, str) and _sa(p).lstrip().startswith("mise en place")), -1)
            while i + 1 < len(rec) and isinstance(rec[i + 1], str) and "coccion previa" in _sa(rec[i + 1]):
                i += 1                                          # tras las cocciones previas que ya hubiera
            rec[i + 1:i + 1] = previas
        if finales:
            j = next((k for k, p in enumerate(rec) if isinstance(p, str) and _sa(p).lstrip().startswith("montaje")), len(rec))
            rec[j:j] = finales
        meal["recipe"] = rec
        meal.pop("_display", None)
        return len(previas) + len(finales)
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-540] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["cocer"]
