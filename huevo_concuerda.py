# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-809 · 2026-09-29] «el huevo duros»: el participio sigue al número del huevo.

Corpus (5.334 comidas): 23 pasos con «acompaña con el huevo duros», «el huevo cocido pelados y cortados», «añade el huevo
bien cocidos desmenuzados», «desmenuza 1 huevo bien cocidos», «hasta que el huevo estén cuajados» y, al revés, «prepara 3
huevos bien cocido». Ya estaban así ANTES del escudo: el pase `P1-STEP-UNIT-PLURAL` del grafo baja «los huevos» a «el
huevo» cuando la lista compra uno, y la concordancia de piezas del contrato (lote 31) deja de singularizar en la primera
palabra que no es plural («bien»). Aquí la cadena de participios PEGADA al huevo («[bien] cocido pelados y cortados»), y
el «estén» que lo sigue sin nada en medio, toman el número del huevo. Sólo lo pegado: en «el huevo y las claras cocidas» o
«vierte el huevo y revuelve hasta que estén cuajados» el plural puede hablar de más cosas, y no se toca. Sólo los pasos (no
las notas, que otros pases reconocen por su texto exacto) y nunca la lista. Knob `MEALFIT_EGG_NUMBER_AGREEMENT` (True).
tooltip-anchor: P1-PLAN-LOTE-809
"""
from __future__ import annotations

import re

_PART = r"(?:cocid|dur|pelad|cortad|partid|picad|hervid|batid|rallad|trocead|laminad|rebanad|desmenuzad|escalfad|pochad|" \
        r"revuelt|frit|cuajad|guisad)"
_ADV = r"(?:(?:bien|ya|muy|reci[eé]n)\s+)?"
#: «el huevo [bien] cocido pelados y cortados» — la cadena que sigue a UN huevo
_SING_RE = re.compile(r"(?P<cab>\b(?:el|un|1|este|ese|cada|tu|su)\s+huevo\s+" + _ADV + r")"
                      r"(?P<cad>" + _PART + r"os?(?:\s+(?:y\s+)?" + _PART + r"os?)*)\b"
                      r"(?:(?P<enc>\s+(?:para|y)\s+\w+?)(?P<los>los)\b)?", re.IGNORECASE)  # «…cocidos para calentarlos»
#: «hasta que el huevo estén cuajados»
_ESTEN_RE = re.compile(r"(?P<cab>\b(?:el|un|1|este|ese|cada|tu|su)\s+huevo\s+)(?P<v>est[eé]n)(?P<resto>\s+" + _ADV + r")"
                       r"(?P<cad>" + _PART + r"os?(?:\s+(?:y\s+)?" + _PART + r"os?)*)\b", re.IGNORECASE)
#: «prepara 3 huevos bien cocido» — la cadena que sigue a VARIOS huevos
_PLUR_RE = re.compile(r"(?P<cab>\b(?:los|unos|[2-9]|dos|tres|cuatro|cinco|seis)\s+huevos\s+" + _ADV + r")"
                      r"(?P<cad>" + _PART + r"os?(?:\s+(?:y\s+)?" + _PART + r"os?)*)\b", re.IGNORECASE)
_PALABRA_RE = re.compile(_PART + r"(?P<n>os?)\b", re.IGNORECASE)
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|💡|nota\b)", re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_EGG_NUMBER_AGREEMENT", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _cadena(cad: str, plural: bool) -> str:
    def _uno(m):
        base = m.group(0)[:m.start("n") - m.start()]
        o = m.group("n")[0]
        return base + o + (("S" if o.isupper() else "s") if plural else "")
    return _PALABRA_RE.sub(_uno, cad)


def concordar(texto: str) -> str:
    texto = _ESTEN_RE.sub(lambda m: m.group("cab") + ("Esté" if m.group("v")[:1].isupper() else "esté") + m.group("resto")
                          + _cadena(m.group("cad"), False), texto)
    texto = _SING_RE.sub(lambda m: m.group("cab") + _cadena(m.group("cad"), False)
                         + ((m.group("enc") + m.group("los")[:2]) if m.group("enc") else ""), texto)
    return _PLUR_RE.sub(lambda m: m.group("cab") + _cadena(m.group("cad"), True), texto)


def concordar_pasos(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _NOTA_RE.search(p):
                continue
            q = concordar(p)
            if q != p:
                rec[i] = q
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0
