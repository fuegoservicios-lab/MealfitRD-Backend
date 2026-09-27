# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-447 · 2026-09-27] «aguacate fresca»: el género de los alimentos masculinos que el modelo escribe en
femenino.

Replay de la cola sobre 322 planes: 87 pasos con «acompaña con aguacate fresca», «½ aguacate mediana y córtala en
mitades», «aguacate fresca cortada en cubos», y la misma familia con el maní («maní fileteado tostadas»), el arroz («arroz
integral cocida»), el mango, el pan («pan integral tostada», también en el NOMBRE del plato) y el guineo («guineo
troceada»): ≈120 frases, unas 4 por plan de 30 días. El modelo trata «aguacate» como femenino (la palta) y el adjetivo le
sigue. Aquí el adjetivo que va PEGADO al alimento (sin un «de» delante: «tortas de casabe tostadas» habla de las tortas)
pasa a masculino, con el número del alimento, y el pronombre del verbo que lo sigue («y córtala») también. Sólo el texto
de los pasos y el nombre: la lista de ingredientes es un identificador (compra, Nevera) y no se toca.
tooltip-anchor: P1-PLAN-LOTE-447
"""
from __future__ import annotations

import re

_NUCLEO_447 = (r"(?P<n>(?P<base>aguacates?|mangos?|guineos?|arroz|pan|queso|man[ií]|casabe)"
               r"(?:\s+(?:integral|blanco|moreno|mozzarella|fileteado|maduros?|verdes?))?)")
#: un adjetivo de la cadena que sigue al alimento («aguacate fresca cortada en cubos»), de cualquier género
_ADJ_447 = re.compile(r"\s+(?P<adj>fresc|median|madur|pequeñ|cortad|picad|rallad|laminad|pelad|cocid|asad|tostad|enter|"
                      r"triturad|machacad|majad|hervid|trocead|rebanad|dorad|tibi|frí|fri)(?P<a>[aoAO])(?P<s>[sS]?)\b",
                      re.IGNORECASE)
_RX_447 = re.compile(r"\b" + _NUCLEO_447, re.IGNORECASE)
#: «tortas de casabe tostadas», «rodajas de aguacate frescas»: el adjetivo puede ser del sustantivo de delante (las
#: medidas —«170 g de arroz integral cocida», «½ taza de…»— no cuentan: el adjetivo es del alimento)
_CONTENEDOR_447 = re.compile(r"\b(?:tortas?|rodajas?|l[aá]minas?|lonjas?|tiras?|hojas?|rebanadas?|tostadas?|mitad(?:es)?|"
                             r"galletas?|bolitas?|cremas?|salsas?|ensaladas?|pur[eé]s?|arepitas?|tortillas?)\s+de\s+$",
                             re.IGNORECASE)
_PRON_447 = re.compile(r"\b(aguacates?\s+\w+\s+y\s+)(c[oó]rta|p[eé]la|l[aá]va|p[ií]ca|s[ií]rve|mach[aá]ca|m[aá]ja|troc[eé]a|"
                       r"lam[ií]na|p[aá]rte|reb[aá]na)(la)(s?)\b", re.IGNORECASE)


def concordar(texto: str) -> str:
    out, pos = [], 0
    for m in _RX_447.finditer(texto):
        if m.start() < pos or _CONTENEDOR_447.search(texto[max(0, m.start() - 30):m.start()]):
            continue
        plural = m.group("base").lower().endswith("s")
        j, cadena = m.end(), []
        while True:
            a = _ADJ_447.match(texto, j)
            if not a:
                break
            cadena.append(a)
            j = a.end()
        if not any(a.group("a") in "aA" for a in cadena):
            continue
        out.append(texto[pos:m.end()])
        k = m.end()
        for a in cadena:
            o = "O" if a.group("a").isupper() else "o"
            s = ("S" if (a.group("s") or a.group("a")).isupper() else "s") if plural else ""
            out.append(texto[k:a.start("a")] + o + s)
            k = a.end()
        pos = k
    out.append(texto[pos:])
    return _PRON_447.sub(lambda m: m.group(1) + m.group(2) + "lo" + ("s" if m.group(1).lower().startswith("aguacates")
                                                                     else ""), "".join(out))


def concordar_masculinos(meal) -> int:
    """Nº de textos corregidos (pasos + nombre); 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        n = 0
        rec = meal.get("recipe")
        if isinstance(rec, list):
            for i, p in enumerate(rec):
                if isinstance(p, str):
                    q = concordar(p)
                    if q != p:
                        rec[i] = q
                        n += 1
        nombre = meal.get("name")
        if isinstance(nombre, str):
            q = concordar(nombre)
            if q != nombre:
                meal["name"] = q
                n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0
