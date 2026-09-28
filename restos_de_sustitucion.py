# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-731 · 2026-09-28] Lo que sólo se le hace a un huevo o a un queso no se le hace a lo que los sustituyó.

Batería real sobre el código de producción 659 (adulto mayor con HTA): el sustituto del huevo (tope de huevos,
`_EGG_REPLACEMENT_LADDER`) dejó «Wok criollo de yautía en trozos con pechuga de pollo **bien cuajado**» y «Empuja los
vegetales a un lado, **vierte** la pechuga de pollo en tiras»; el del queso por yogur dejó «**desmenuza** los 40 g de
yogurt griego» y «desmenuza yogurt… por encima **para que se funda** con el calor residual». La reescritura cambia el
alimento (y el 612 concuerda el género: «bien cuajado» ya iba con «pollo») pero no los verbos que sólo tienen sentido
con el alimento viejo.

Aquí, en la cola del contrato, sólo cuando la lista ya no tiene ese alimento:
- sin huevo en la lista: «bien cuajado/a(s)» → «bien cocido/a(s)» y «vierte el/la <proteína>» → «añade el/la …»;
- sin queso en la lista y con yogur: «desmenuza … yogurt» → «añade … yogurt» y se va «para que se funda (con el calor
  residual)» de la frase del yogur.
Las notas (⚠️ 💡 …) no se tocan. tooltip-anchor: P1-PLAN-LOTE-731
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_HUEVO = re.compile(r"\b(?:huevos?|claras?|yemas?)\b")
_QUESO = re.compile(r"\b(?:queso|quesos|ricotta|cottage|mozzarella|requeson)\b")
_YOGUR = re.compile(r"\byog(?:h)?urt?\b")
_PROTE = r"(?:pechugas?(?:\s+de\s+(?:pollo|pavo))?|pollo|pavo|pescado|filetes?(?:\s+de\s+\w+)?|carne(?:\s+de\s+res)?|res|atun|tofu)"
_BIEN_CUAJADO = re.compile(r"\bbien\s+cuajad(?P<fin>[oa]s?)\b", re.IGNORECASE)
_VIERTE = re.compile(r"\b(?P<v>[Vv])ierte(?P<resto>\s+(?:la|el|los|las)\s+" + _PROTE + r")\b", re.IGNORECASE)
_DESMENUZA_YOGUR = re.compile(r"\b(?P<verbo>desmenuza|ralla)(?P<resto>\s+(?:los\s+\d+\s+g\s+de\s+|el\s+|la\s+)?yog(?:h)?urt?)",
                              re.IGNORECASE)
_SE_FUNDA = re.compile(r"\s+para\s+que\s+se\s+(?:funda|derrita)(?:\s+con\s+el\s+calor\s+residual)?", re.IGNORECASE)
_FUNDE_DE_VERDAD = re.compile(r"\b(?:mantequilla|margarina|chocolate|coco|queso)\b")
_NOMBRE_CUAJADO = re.compile(r"\s+bien\s+cuajad[oa]s?\b", re.IGNORECASE)
_NOTAS = ("⚠", "💡", "🤰", "⚕", "🌱", "🛒", "🧊")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def limpiar(meal) -> int:
    """Nº de textos corregidos (nombre o pasos); 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        lista = _sa(" ".join(str(x) for x in (meal.get("ingredients") or [])))
        sin_huevo = not _HUEVO.search(lista)
        yogur_sin_queso = bool(_YOGUR.search(lista)) and not _QUESO.search(lista)
        if not sin_huevo and not yogur_sin_queso:
            return 0

        def _arreglar(t: str, mise: bool = False) -> str:
            q = t
            if sin_huevo:
                q = _BIEN_CUAJADO.sub(lambda m: "bien cocid" + m.group("fin"), q)
                q = _VIERTE.sub(lambda m: ("A" if m.group("v") == "V" else "a") + "ñade" + m.group("resto"), q)
            if yogur_sin_queso:
                verbo = "mide" if mise else "añade"               # en la mise en place se MIDE, no se añade

                def _verbo(m):
                    v = verbo[:1].upper() + verbo[1:] if m.group("verbo")[:1].isupper() else verbo
                    return v + m.group("resto")
                q = _DESMENUZA_YOGUR.sub(_verbo, q)
                frases = re.split(r"(?<=[.;])\s+", q)
                q = " ".join(_SE_FUNDA.sub("", f) if _YOGUR.search(_sa(f)) and not _FUNDE_DE_VERDAD.search(_sa(f)) else f
                             for f in frases)
            return q

        n = 0
        nombre = meal.get("name")
        if isinstance(nombre, str):
            # en el NOMBRE el punto de cocción sobra: «…con pechuga de pollo bien cuajado, brócoli» → «…con pechuga de
            # pollo, brócoli» (un «bien cocido» ahí concordaría con «pollo» y no con «pechuga»)
            nuevo = _NOMBRE_CUAJADO.sub("", nombre) if sin_huevo else nombre
            if nuevo != nombre:
                meal["name"] = nuevo
                n += 1
        rec = meal.get("recipe")
        if isinstance(rec, list):
            for i, p in enumerate(rec):
                if not isinstance(p, str) or p.lstrip().startswith(_NOTAS):
                    continue
                nuevo = _arreglar(p, mise=_sa(p).lstrip().startswith("mise en place"))
                if nuevo != p:
                    rec[i] = nuevo
                    n += 1
        if n:
            meal.pop("_display", None)
            logger.info(f"🧽 [P1-PLAN-LOTE-731] «{str(meal.get('name'))[:50]}»: {n} texto(s) sin los verbos del alimento "
                        f"sustituido")
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-731] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["limpiar"]
