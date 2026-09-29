# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-920 · 2026-09-29] «desmenuza los el queso blanco restante restante»: la segunda mención de un alimento
repartido pasa a «el X restante» sin chocar con el artículo que ya tenía ni repetir «restante».

Batería real (el formulario del dueño, 29-sep, código 888 + 889): «desmenuza los el queso blanco restante restante por
encima». Corpus + replays: 8 de 5.301 comidas — «las el agua durante unos restante minutos», «los el bulgur restante»,
«las el pan integral restante», «los la mantequilla de maní restante», «las el claras restante». El sincronizador de
cantidades (P2-QTYSYNC-MULTIUSE) cambia la 2.ª mención cuantificada de un alimento repartido («los 12 g de queso…») por
«<artículo> <alimento> restante», pero (1) dejaba el artículo que ya iba delante, (2) añadía «restante» aunque el paso ya
lo dijera, (3) el nombre del alimento se tragaba «durante unos» y (4) «claras» salía en singular. Knob
`MEALFIT_RESTANTE_SIN_CHOQUE` (True). tooltip-anchor: P1-PLAN-LOTE-920
"""
from __future__ import annotations

import re
import unicodedata

_ART = r"(?P<art0>\b(?:el|la|los|las|lo|unos|unas|un|una)\s+)?"
#: lo que no es nombre de alimento aunque la regex de menciones lo coma («agua durante unos»)
_CORTE = re.compile(r"\s+(?:durante|unos|unas|aproximadamente|mientras|junto|antes|despu[eé]s|luego|bien)\b", re.I)
_YA_RESTANTE = re.compile(r"^\s*restantes?\b", re.I)
_CACHE: dict = {}


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_RESTANTE_SIN_CHOQUE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def patron(mencion_re):
    """La regex de menciones con el artículo que la precede dentro del match: al reescribir, el artículo viejo se va."""
    if not on():
        return mencion_re
    p = _CACHE.get(mencion_re.pattern)
    if p is None:
        p = re.compile(_ART + "(?:" + mencion_re.pattern + ")", mencion_re.flags)
        _CACHE[mencion_re.pattern] = p
    return p


_CHOQUE_RE = re.compile(r"\b(el|la|los|las)\s+(el|la|los|las)\s+([^.;,]{1,40}?)\s+(restantes?)\b", re.I)
_DOBLE_RE = re.compile(r"\b(restantes?)\s+restantes?\b", re.I)
#: «el agua durante unos restante minutos», «el agua durante 15 restante-18 minutos»
_UNOS_RE = re.compile(r"\b((?:el|la)\s+\w+)\s+((?:durante|por)\s+(?:unos|unas|\d+))\s+restantes?(\s*-\s*\d+)?"
                      r"(\s+(?:minutos?|segundos?|min)\b)", re.I)
#: «dora el pan integral 1 restante minuto» / «1 restante-2 min»: el número del tiempo quedó detrás del «restante»
_NUM_RE = re.compile(r"(?<!durante)(?<!por)(?<!unos)(?<!unas)\s(\d+)\s+(restantes?)(\s*-\s*\d+|\s+a\s+\d+)?"
                     r"(\s+(?:min(?:utos?)?|segundos?)\b)", re.I)
_DE_RE = re.compile(r"\bde\s+(restantes?)\s+(?=\d)", re.I)
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|nota\b)", re.I)


def _choque(m) -> str:
    a1, a2, nucleo, _r = m.groups()
    primera = _sa(nucleo.split()[0]) if nucleo.split() else ""
    if primera.endswith("s") and len(primera) > 3:                          # «las el claras restante»: plural
        out = f"{'las' if primera.endswith('as') else 'los'} {nucleo} restantes"
    else:
        out = f"{a2.lower()} {nucleo} restante"
    return out[:1].upper() + out[1:] if a1[:1].isupper() else out


def limpiar(texto: str) -> str:
    """Lo que ya salió roto: «los el queso restante» → «el queso restante»; «las el claras restante» → «las claras
    restantes»; «restante restante» → «restante»; «el agua durante unos restante minutos» → «el agua restante durante unos
    minutos»; «el pan integral 1 restante minuto» → «el pan integral restante 1 minuto»; «a el» → «al»."""
    t = _UNOS_RE.sub(lambda m: f"{m.group(1)} restante {m.group(2)}{m.group(3) or ''}{m.group(4)}", texto)
    t = _NUM_RE.sub(r" \2 \1\3\4", t)
    t = _DE_RE.sub(r"\1 de ", t)
    t = _CHOQUE_RE.sub(_choque, t)
    t = _DOBLE_RE.sub(r"\1", t)
    if t != texto:
        t = re.sub(r"\b([aA])\s+el\s+(?=\w)", r"\1l ", t)
        t = re.sub(r"\b([dD]e)\s+el\s+(?=\w)", lambda m: m.group(1)[0] + "el ", t)
    return t


def limpiar_pasos(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict) or not isinstance(meal.get("recipe"), list):
            return 0
        n = 0
        for i, p in enumerate(meal["recipe"]):
            if isinstance(p, str) and not _NOTA_RE.search(p):
                q = limpiar(p)
                if q != p:
                    meal["recipe"][i] = q
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0


def resto(mm, nucleo: str, cola: str) -> str:
    """«el queso blanco restante», «las claras restantes», «el agua restante durante unos» — sin «restante» doble."""
    ya = False
    if on():
        r = re.search(r"\s+restantes?\b", nucleo, re.I)   # «12 g de queso blanco restante»: la regex se comió «restante»
        if r:
            nucleo, cola, ya = nucleo[:r.start()], nucleo[r.end():] + cola, True
        m = _CORTE.search(nucleo)
        if m:
            nucleo, cola = nucleo[:m.start()], nucleo[m.start():] + cola
    primera = _sa(nucleo.split()[0]) if nucleo.split() else ""
    plural = on() and primera.endswith("s") and len(primera) > 3
    if plural:
        art, palabra = ("las" if primera.endswith("as") else "los"), "restantes"
    else:
        art = "la" if (primera.endswith("a") and not _sa(nucleo).startswith("agua")) else "el"
        palabra = "restante"
    ya = ya or (on() and bool(_YA_RESTANTE.match(mm.string[mm.end():])))
    return f"{art} {nucleo}" + ("" if ya else f" {palabra}") + cola
