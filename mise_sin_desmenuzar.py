# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-884 · 2026-09-29] La mise en place no desmenuza la carne que se cocina en la «💡 Cocción previa».

Medidores sobre las baterías pasadas por el escudo actual (29-sep): el aviso más frecuente (V7f, ~15 % de los planes)
era «Pechuga de pollo: un paso lo usa ya cocido sin cocerlo antes» — «Mise en place: … desmenuza 230 g de pechuga de
pollo…» y, DESPUÉS, «💡 Cocción previa: cocina la pechuga de pollo… 74 °C…; déjala reposar y desmenúzala». Carne cruda
no se desmenuza. Corpus: 54 comidas así con el código actual (37 con la cocción previa del modelo, 17 con la que añade el
lote 407). Aquí, si una «💡 Cocción previa» cuece esa proteína, la mise en place sólo la tiene a mano («mide 230 g de
pechuga de pollo», «ten a mano ½ pechuga de pollo») — sin «ya cocida» ni el «(verifica 74 °C…)» — y el desmenuzado queda
en la cocción previa (la del pescado y la de la carne, que no lo decían, lo dicen al final). Knob
`MEALFIT_MISE_SIN_DESMENUZAR` (True). tooltip-anchor: P1-PLAN-LOTE-884
"""
from __future__ import annotations

import re
import unicodedata

_FAMILIAS = {
    "ave": r"(?:pechugas?\s+de\s+(?:pollo|pavo)|pollo|pavo)",
    "pez": r"(?:filetes?\s+de\s+pescado(?:\s+blanco)?|pescado(?:\s+blanco)?)",
    "carne": r"(?:carne\s+de\s+res|cerdo)",
}
_CANT = r"(?P<cant>(?:(?:\d+\s*[½¼¾⅓⅔]|\d+(?:[.,]\d+)?|[½¼¾⅓⅔])\s*(?:g\s+de\s+)?)?)"


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_MISE_SIN_DESMENUZAR", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _desm_re(prot: str):
    # el estado «cocido» que la mise en place le supone («ya cocida», «cocida y enfriada», «cocido listo para
    # consumir», «(verifica 74 °C…)») se va con el verbo: lo cuece la cocción previa
    return re.compile(r"(?:escurre\s+y\s+)?(?:desmenuza|deshebra)\s+(?P<art>(?:el|la|los|las)\s+)?" + _CANT
                      + r"(?P<prot>" + prot + r")(?P<peso>(?:\s*\((?!verifica)[^)]*\))*)"
                      r"(?:\s+(?:ya\s+)?cocid[oa]s?(?:\s+y\s+enfriad[oa]s?)?(?:\s+listos?\s+para\s+consumir)?)?"
                      r"(?:\s*\(verifica[^)]*\))?", re.IGNORECASE)


def _tener(m) -> str:
    cant = m.group("cant") or ""
    if re.search(r"\bg\s+de\s+$", cant, re.IGNORECASE):
        return f"mide {cant}{m.group('prot')}{m.group('peso') or ''}"
    return f"ten a mano {m.group('art') or ''}{cant}{m.group('prot')}{m.group('peso') or ''}"


# ── [P1-PLAN-LOTE-887 · 2026-09-29] La mise en place no pela ni corta el huevo que hierve la «💡 Cocción previa» ────────
# Corpus + replays (5.281 comidas únicas): 6 así, casi todas mangú con huevo — «Mise en place: … pela y rebana 3 huevos y
# 1 clara de huevo duros ya cocidos…» y DESPUÉS «💡 Cocción previa: hierve los huevos 10-12 min, pásalos a agua fría y
# pélalos». Un huevo crudo no se pela ni se rebana. La mise en place sólo los tiene a mano y el corte pasa al final de la
# cocción previa («Luego córtalos por la mitad.»). Knob `MEALFIT_MISE_HUEVO_SIN_CORTAR` (True).
# tooltip-anchor: P1-PLAN-LOTE-887
_HUEVO_MISE_887_RE = re.compile(
    r"(?P<verbo>(?:(?:pela|lava)\s+y\s+)?(?:corta|rebana|pica|parte|pela|trocea|lamina)"
    r"(?:\s+y\s+(?:corta|rebana|pica|parte|trocea|lamina))?)\s+"
    r"(?P<obj>(?:(?:el|los)\s+)?(?:\d+|[½¼¾])\s+huevos?(?:\s+y\s+(?:\d+|[½¼¾])\s+claras?\s+de\s+huevo)?)"
    r"(?P<estado>(?:\s+(?:dur[oa]s?|bien\s+cocid[oa]s?|ya\s+cocid[oa]s?|cocid[oa]s?))*)"
    r"(?P<forma>\s+(?:en\s+(?:mitades|rodajas|cuartos|cubos|trozos)|por\s+la\s+mitad))?", re.IGNORECASE)
_CORTE_887 = (("rebana", "rebána"), ("lamina", "lamína"), ("trocea", "trocéa"), ("pica", "píca"), ("corta", "córta"),
              ("parte", "párte"))


def on_huevo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_MISE_HUEVO_SIN_CORTAR", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _huevo_887(rec: list) -> int:
    if not on_huevo():
        return 0
    mise = [i for i, p in enumerate(rec) if isinstance(p, str) and _sa(p).lstrip().startswith("mise en place")]
    if not mise:
        return 0
    previas = [i for i, p in enumerate(rec) if i > mise[0] and isinstance(p, str)
               and _sa(p).lstrip().startswith("💡 coccion previa") and re.search(r"\bhierve\s+(?:los|el)\s+huevos?\b", _sa(p))]
    if not previas:
        return 0
    cortes = []

    def _tener(m):
        verbo = _sa(m.group("verbo"))
        plural = bool(re.search(r"huevos|claras", _sa(m.group("obj"))))
        for raiz, imperativo in _CORTE_887:
            if re.search(r"\b" + raiz + r"\b", verbo):
                cortes.append(imperativo + ("los" if plural else "lo") + (m.group("forma") or ""))
                break
        return "ten a mano " + m.group("obj")
    q = _HUEVO_MISE_887_RE.sub(_tener, rec[mise[0]])
    if q == rec[mise[0]]:
        return 0
    rec[mise[0]] = q
    j = previas[0]
    t = rec[j].rstrip()
    if cortes and not re.search(r"\b(?:cortalos|cortalo|rebanalos|rebanalo|picalos|picalo|laminalos|trocealos)\b", _sa(t)):
        rec[j] = t + ("" if t.endswith(".") else ".") + f" Luego {cortes[0]}."
    return 1


def huevo(meal) -> int:
    """[P1-PLAN-LOTE-887] Va DESPUÉS de `huevo_duro_de_la_lista` (409), que es quien pone la cocción previa del huevo.
    Nº de cambios (0 o 1); 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        n = _huevo_887(rec)
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0


def ordenar(meal) -> int:
    """Nº de cambios; 0 ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list):
            return 0
        n = 0
        for fam, prot in _FAMILIAS.items():
            previas = [i for i, p in enumerate(rec) if isinstance(p, str)
                       and _sa(p).lstrip().startswith("💡 coccion previa") and re.search(prot, _sa(p))]
            if not previas:
                continue
            rx = _desm_re(prot)
            cambio = False
            for i, p in enumerate(rec):
                if isinstance(p, str) and _sa(p).lstrip().startswith("mise en place"):
                    q = rx.sub(_tener, p)
                    if q != p:
                        rec[i] = q
                        cambio = True
                        n += 1
            if cambio and fam != "ave":
                # el pescado y la carne no decían que se desmenuzan después: lo dicen al final de su cocción previa
                j = previas[0]
                t = rec[j].rstrip()
                if not re.search(r"desmen[uú]za(?:lo|la)", _sa(t)):
                    rec[j] = t + ("" if t.endswith(".") else ".") + (" Luego desmenúzalo." if fam == "pez"
                                                                     else " Luego desmenúzala.")
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0
