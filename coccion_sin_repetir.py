# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-586 · 2026-09-27] La proteína que el plato ya cuece no se cuece otra vez.

Batería real: «Tilapia guisada con vainitas…» — «Añade la tilapia…; tapa y guisa 8-10 min, hasta que el pescado alcance
63 °C. Añade tilapia fresca al guiso y cocínala a fuego medio 5-7 minutos…»; «Huevos al horno sobre tostada…» — «Casca 1
huevo dentro del molde…; hornea a 180 °C 10-12 minutos… Cocina huevo a la plancha o hervido y sírvelo como proteína del
plato.» El cerrador de proteína añade su frase cuando no ve la proteína trabajada en los pasos; si otra pasada la
renombra después, o la ve en una cláusula y la cocción va en la siguiente, queda una segunda cocción del mismo alimento.

Aquí, al final de la cola, la frase del cerrador se va SOLO si otra CLÁUSULA de un paso de cocina (no el Montaje, no una
nota) nombra ese alimento y lo cuece con verbo Y tiempo o temperatura. El escáner de cocción (V7f) NO sirve para esto: está
afinado para no acusar en falso, así que es generoso con el «cocido» («Sirve el filete a la plancha», «bate los huevos» y
luego «cocina la cebolla») — usado para BORRAR, dejaba la proteína cruda (replay del corpus: 23 de 33 retiradas eran
así, y a nivel de oración «…cocina 3 min; incorpora también pechuga de pavo» dejaba el pavo crudo). Borrar de más deja un alimento sin cocer; borrar de menos, una frase repetida: la regla es estricta a propósito.
También se va la segunda copia EXACTA de una frase del cerrador. Corpus: 30 de 4.868 comidas. tooltip-anchor:
P1-PLAN-LOTE-586
"""
from __future__ import annotations

import re
import unicodedata

_FRASES = (
    re.compile(r"\s*(?:💪\s*)?Cocina (?P<x>[^.;:]{2,60}?) a la plancha o hervid[oa]s? y sírvel[oa]s? como proteína del "
               r"plato\.?"),
    re.compile(r"\s*(?:💪\s*)?Añade (?P<x>[^.;:]{2,60}?) al guiso y cocínal[oa]s? a fuego medio [^.]*(?:\.|$)"),
)
_NOTA = ("⚠", "🤰", "🌱", "⚕", "💡", "🛡", "ℹ")
_VERBO = re.compile(r"\b(?:hornea\w*|horneal\w*|guisa\w*|guisal\w*|cocina|cocinal\w*|hierve|hiervel\w*|sofrie|saltea\w*|"
                    r"asa|asal\w*|dora|doral\w*|frie|cuece|cuaja|cuajal\w*|sella|sellal\w*|airfryer)\b")
_TIEMPO = re.compile(r"\d+\s*(?:-\s*\d+\s*)?(?:min|minutos?)\b|\d+\s*°\s*c\b")
_SERVIR_ANTES = re.compile(r"\b(?:reserva|sirve|acompana|corona|decora|presenta)\s+(?:\w+\s+){0,3}$")
_GENERICAS = {"filete", "filetes", "blanco", "fresco", "fresca", "magra", "magro", "tiras", "cocido", "cocida", "agua",
              "claro", "carne", "muslo", "pieza", "plancha", "hervido", "hervida", "horno", "vapor", "parrilla"}
_HIPERONIMO = (("tilapia", "pescado"), ("mero", "pescado"), ("dorado", "pescado"), ("chillo", "pescado"),
               ("salmon", "pescado"), ("merluza", "pescado"), ("pechuga", "pollo"), ("muslo", "pollo"),
               ("bistec", "res"), ("res", "carne"), ("cerdo", "carne"))


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _nombres(x: str) -> set:
    """El NÚCLEO del alimento de la frase (su primera palabra propia) y su hiperónimo: «salmón previamente congelado y
    apto para consumo crudo» es {salmon, pescado} — con todas sus palabras, «para» casaba «apto para microondas»."""
    toks = [t for t in re.findall(r"[a-zñ]+", _sa(x)) if (len(t) >= 4 or t == "res") and t not in _GENERICAS]
    if not toks:
        return set()
    propios = {toks[0]}
    for a, b in _HIPERONIMO:
        if toks[0].startswith(a):
            propios.add(b)
    return propios


def _nombra(oracion: str, nombres: set) -> bool:
    if "huevo" in nombres and re.search(r"\bclaras?\b", oracion) and re.search(r"\byemas?\b", oracion):
        return True                      # «hasta que la clara esté cuajada y la yema firme»: es el huevo entero
    for n in nombres:
        raiz = n[:-1] if n.endswith("s") and len(n) > 4 else n
        for m in re.finditer(r"\b" + re.escape(raiz) + r"(?:s|es)?\b", oracion):
            if not _SERVIR_ANTES.search(oracion[:m.start()]):
                return True
    return False


def _la_cuece_otra(rec: list, i: int, quitada: str, nombres: set) -> bool:
    for j, p in enumerate(rec):
        if not isinstance(p, str) or p.lstrip().startswith(_NOTA) or _sa(p).lstrip().startswith("montaje"):
            continue
        texto = p.replace(quitada, " ") if j == i else p
        for o in re.split(r"(?<=[.])\s+", texto):
            if _FRASES[0].search(o) or _FRASES[1].search(o):
                continue
            # la CLÁUSULA (hasta «;» o «.»), no la oración: «…cocina 3 min; incorpora también pechuga de pavo» no la cuece
            for c in re.split(r"[;:]", _sa(o)):
                if _VERBO.search(c) and _TIEMPO.search(c) and _nombra(c, nombres):
                    return True
    return False


def quitar(meal, index=None) -> int:
    """Nº de frases quitadas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        vistas = set()
        for i in range(len(rec)):
            for rx in _FRASES:
                p = rec[i]
                if not isinstance(p, str):
                    break
                for m in list(rx.finditer(p)):
                    frase = m.group(0)
                    clave = _sa(frase).strip(" .💪")
                    nombres = _nombres(m.group("x"))
                    if clave in vistas or (nombres and _la_cuece_otra(rec, i, frase, nombres)):
                        rec[i] = rec[i].replace(frase, "", 1)
                        n += 1
                    vistas.add(clave)
        if n:
            meal["recipe"] = [re.sub(r"\s{2,}", " ", p).strip() if isinstance(p, str) else p for p in rec
                              if not (isinstance(p, str) and not re.sub(r"[\s💪.]", "", p))]
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["quitar"]
