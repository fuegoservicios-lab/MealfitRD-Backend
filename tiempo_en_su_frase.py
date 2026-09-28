# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-662 · 2026-09-28] El tiempo que falta va en la frase que cocina con esa técnica, no al final del paso.

Batería real del 28-sep sobre el código 636 (estudiante, día 2): «…agrega el tomate… y cocina hasta formar un guiso.
Incorpora las claras de huevo y 3 huevos, removiendo suavemente hasta que cuajen (~18-20 min a 180 °C).» — el tiempo de
HORNO que el backstop (`_inject_recipe_time_temp_defaults`) eligió por «asa las rodajas de plátano… o en el horno» se
pegaba al final del paso, detrás de unos huevos en sartén; y en el día 1 «…Escúrrela y deja que se enfríe (~12-15 min en
agua hirviendo)». Corpus: de 30 pasos con el tiempo inyectado al final, ~15 lo cuelgan de otra frase («sin dejar que el
yogurt hierva (~12-15 min en agua hirviendo)», «guisa… (~12-15 min en agua hirviendo)», «cocina el huevo… (~2-3 min en
el microondas)»). Aquí el tiempo se inserta al final de la PRIMERA frase del paso que hace la técnica de ese tiempo
(«asa… o en el horno hasta que estén tiernas y doradas (~18-20 min a 180 °C). …»); si ninguna la hace, al final, como
antes. tooltip-anchor: P1-PLAN-LOTE-662
"""
from __future__ import annotations

import re
import unicodedata

#: técnica del tiempo por defecto (texto de `_TIMETEMP_*_DEFAULT(S)`) → raíces de los verbos que la hacen (sin acentos)
_RAICES = (
    ("a 180 °c", r"\bhorn\w*|\bgratin\w*|\brostiz\w*|\basa\b|\basal[oa]s?\b|\bhornea\w*"),
    ("en agua hirviendo", r"\bhierv\w*|\bherv\w*|\bcuec\w*|\bcuece\b|\bcocer\w*|\bsancoch\w*"),
    ("tapado", r"\bguis\w*|\bestof\w*|\bbrase\w*"),
    ("al vapor", r"\bvapor\b|\bvaporera\b"),
    ("hasta dorar", r"\bfri\w*|\bfreidora\b|\bairfryer\b|\btuest\w*|\btost\w*|\bdora\w*|\bdoral\w*"),
    # antes que «por lado a fuego medio», que es subcadena de «…por lado a fuego medio-alto». Sin «sartén»: el recipiente
    # sale en frases que no son la del tiempo («calienta el aceite en una sartén, añade el huevo… hasta que cuaje» no es
    # la que va «por lado»; replay del corpus)
    ("fuego medio-alto", r"\bplanch\w*|\bsalte\w*|\bsofri\w*|\bsell\w*|\bdora\w*"),
    ("por lado a fuego medio", r"\brevuelv\w*|\brevolt\w*|\bpanqueq\w*|\barepit\w*|\btortitas?\b|\bcuaj\w*|\bdora\w*"),
    ("de licuado", r"\blicu\w*|\bproces\w*|\bbate\b|\bbatido\b"),
    ("en el microondas", r"\bmicroondas\b"),
    ("a fuego medio", r"\bcocin\w*|\bcalient\w*|\bsalte\w*|\bsofri\w*|\bdora\w*|\bhierv\w*|\bcuec\w*|\bguis\w*"),
)
_FRASES = re.compile(r"(?<=[.;])\s+")
_UTENSILIO = re.compile(r"\b(?:papel|toallas?|hilo|tijeras?|utensilios?|guantes?)\s+de\s+cocina\b")
_CABEZA = re.compile(r"^(El Toque de Fuego[^:]*:\s*)", re.IGNORECASE)


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def insertar(paso: str, tiempo: str) -> str:
    """`paso` con « (~tiempo)» al final de la primera frase que hace la técnica de `tiempo`; si ninguna, al final."""
    final = str(paso).rstrip().rstrip(".") + f" (~{tiempo})."
    try:
        t = _sa(tiempo)
        raiz = next((r for clave, r in _RAICES if clave in t), None)
        if not raiz:
            return final
        m = _CABEZA.match(paso)
        cabeza, cuerpo = (m.group(1), paso[m.end():]) if m else ("", paso)
        frases = _FRASES.split(cuerpo.strip())
        for k, f in enumerate(frases):
            # «papel de cocina» no cocina (replay: «bate 6 claras… con papel de cocina» recibía el tiempo de fuego)
            if re.search(raiz, _UTENSILIO.sub(" ", _sa(f))):
                if k == len(frases) - 1:
                    return final                       # ya es la última: el sitio de siempre
                cierre = f[-1] if f[-1] in ".;" else ""
                frases[k] = f[:len(f) - len(cierre)].rstrip() + f" (~{tiempo})" + cierre
                return cabeza + " ".join(frases)
        return final
    except Exception:
        return final


__all__ = ["insertar"]
