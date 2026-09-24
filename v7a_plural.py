# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-212 · 2026-09-24] V7a: el paso que pide MENOS piezas de las que compra la lista se alinea, con plural.

La decisión V7a (14-sep, `docs/decisiones_dueno_2026_09_14.md`) dejaba sin corregir «el paso pide MENOS de lo que compra
la lista»: el contrato de la receta (`recipe_contract.reconcile_step_quantities`) ya alineaba casi todo en las dos
direcciones, pero cuando el número cruza el singular/plural («corta ½ tomate» con «1½ tomates» en la lista) cambiar sólo
el número deja «1½ tomate», y no sabía pluralizar. Medido: 6 de 13 descuadres reales de pasos eran ése. El 24-sep el
dueño cambió la decisión: «soluciónalo».

Aquí vive la dirección que faltaba, espejo de `_concordar_pieza` (que sólo sabe ir al singular):
  · número → el de la lista, con el «de» partitivo fuera («½ de tomate» → «1½ tomates»);
  · el sustantivo y los adjetivos que lo acompañan, al plural por las reglas del español (vocal +s; consonante +es;
    -z → -ces; aguda en -ón/-án/-ín pierde la tilde: «limón» → «limones», «cebollín» → «cebollines»);
  · el artículo («el ½ tomate» → «los 1½ tomates») y el clítico de la misma cláusula («córtalo» → «córtalos»).
Lo que no sabe pluralizar sin inventar (una llana en -n: «orden» → «órdenes»; un calificativo que no está en el léxico)
devuelve None y el llamador lo sigue declarando `gramatical`. Knob `MEALFIT_CONTRACT_V7A` (True).
"""
from __future__ import annotations

import re

_SIN_TILDE = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
_PREPOSICIONES = {"de", "del", "con", "en", "para", "sin", "al"}
_ART_PLURAL = {"el": "los", "la": "las", "los": "los", "las": "las"}
_ART_ANTES_RE = re.compile(r"\b(el|la|los|las)\s+$", re.IGNORECASE)
_CLITICO_SINGULAR_RE = re.compile(r"(?<![\wáéíóúñü])([\wáéíóúñü]*[áéíóú][\wáéíóúñü]*[aeiou]l[oa])\b")
_PALABRA_TRAS_RE = re.compile(r"(\s+)([\wáéíóúñü]+)")


def activo() -> bool:
    """tooltip-anchor: MEALFIT_CONTRACT_V7A"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CONTRACT_V7A", True)
    except Exception:
        return True


def _silabas(palabra: str) -> int:
    return max(1, len(re.findall(r"[aeiouáéíóúü]+", palabra.lower())))


def plural(palabra: str) -> "str | None":
    """Plural español de un sustantivo/adjetivo de receta, o None si no se puede sin inventar una tilde."""
    w = str(palabra or "")
    if not w or not w[-1].isalpha():
        return None
    ult = w[-1].lower()
    if ult in "aeiou":
        return w + "s"
    if ult in "áéó":
        return w + "s"
    if ult in "íú":
        return w + "es"
    if ult == "z":
        return w[:-1] + ("C" if w[-1].isupper() else "c") + "es"
    if ult in "sx":
        return None
    if ult == "n":
        m = re.search(r"([áéíóú])n$", w, re.IGNORECASE)
        if m:
            return w[:m.start(1)] + _SIN_TILDE[m.group(1).lower()] + "n" + "es"
        return w + "es" if _silabas(w) == 1 else None          # «orden» → «órdenes»: la tilde no se inventa
    if ult in "lrdjy":
        return w + "es"
    return None


def _adjetivos_singulares() -> dict:
    """{adjetivo singular normalizado: letras del plural}: el léxico de `recipe_contract._PLURAL_RECORTE`, al revés."""
    from recipe_contract import _PLURAL_RECORTE
    return {k[:-n]: n for k, n in _PLURAL_RECORTE.items()}


def _plural_adjetivo(palabra: str, lexico: dict) -> "str | None":
    from recipe_contract import _norm
    n = lexico.get(_norm(palabra))
    if not n:
        return None
    return palabra + ("s" if n == 1 else "es")


def plural_span(span: str) -> "str | None":
    """«tomate» → «tomates», «plátano verde» → «plátanos verdes», «pechuga de pollo» → «pechugas de pollo». El núcleo y
    sus calificativos del léxico hasta la primera preposición; lo que sigue a la preposición no se toca. None si un
    calificativo no está en el léxico (no se adivina: «ají morrón» se queda como estaba)."""
    toks = span.split()
    if not toks:
        return None
    nucleo = plural(toks[0])
    if nucleo is None:
        return None
    lexico = _adjetivos_singulares()
    out = [nucleo]
    tras_prep = False
    for t in toks[1:]:
        if tras_prep or t.lower() in _PREPOSICIONES:
            tras_prep = True
            out.append(t)
            continue
        p = _plural_adjetivo(t, lexico)
        if p is None:
            return None
        out.append(p)
    return " ".join(out)


def _pluraliza_adjetivos_tras(texto: str, pos: int, maximo: int = 2) -> str:
    lexico = _adjetivos_singulares()
    for _ in range(maximo):
        m = _PALABRA_TRAS_RE.match(texto, pos)
        if not m:
            break
        p = _plural_adjetivo(m.group(2), lexico)
        if p is None:
            break
        texto = texto[:m.start(2)] + p + texto[m.end(2):]
        pos = m.start(2) + len(p)
    return texto


def pluralizar_pieza(paso: str, m: dict, objetivo: float, index: dict) -> "str | None":
    """«corta ½ tomate en cubos y resérvalo» con «1½ tomates» en la lista → «corta 1½ tomates en cubos y resérvalos».
    Sólo hacia el plural (objetivo > 1). None cuando no sabe hacerlo sin dejar un residuo peor."""
    if not activo():
        return None
    from recipe_contract import _catalog_food_spans, _norm, clause_bounds, formatear_cantidad
    fi, ff = m.get("food_ini"), m.get("food_fin")
    if fi is None or ff is None or objetivo <= 1.0 or fi < m["fin"]:
        return None
    entre = paso[m["fin"]:fi].split()
    if entre and _norm(entre[0]) == "de":
        entre = entre[1:]                                           # «½ de tomate»: el partitivo sobra con el plural
    lexico = _adjetivos_singulares()
    nuevos_entre = []
    for w in entre:
        p = _plural_adjetivo(w, lexico)
        if p is None:
            return None                                             # «½ taza de…»: otra familia, no se toca
        nuevos_entre.append(p)
    plur = plural_span(paso[fi:ff])
    if plur is None:
        return None
    prefijo = paso[:m["ini"]]
    art = _ART_ANTES_RE.search(prefijo)
    if art:
        a = art.group(1)
        nuevo_art = _ART_PLURAL.get(a.lower(), a)
        prefijo = prefijo[:art.start(1)] + (nuevo_art.capitalize() if a[:1].isupper() else nuevo_art) + " "
    cuerpo = formatear_cantidad(objetivo) + " " + "".join(w + " " for w in nuevos_entre) + plur
    nuevo = prefijo + cuerpo
    pos_fin = len(nuevo)
    nuevo += paso[ff:]
    nuevo = _pluraliza_adjetivos_tras(nuevo, pos_fin)              # «tomate maduro» → «tomates maduros»
    try:
        fin_cl = next((b_ for a_, b_ in clause_bounds(nuevo) if a_ <= pos_fin < b_), len(nuevo))
        ini_cl = max([a_ for a_, b_ in clause_bounds(nuevo) if a_ <= pos_fin] or [0])
        foods = {n for _, _, n in _catalog_food_spans(nuevo[ini_cl:fin_cl], index)}
        if foods == {m["food"]}:
            seg = nuevo[pos_fin:fin_cl]
            seg2 = _CLITICO_SINGULAR_RE.sub(r"\1s", seg)             # «resérvalo» → «resérvalos»
            if seg2 != seg:
                nuevo = nuevo[:pos_fin] + seg2 + nuevo[fin_cl:]
    except Exception:
        pass
    return nuevo if nuevo != paso else None
