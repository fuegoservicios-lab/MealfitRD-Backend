# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-888 · 2026-09-29] El ave cruda que un paso sólo calienta un momento pasa a cocción real, en el paso.

Revisor del 857 (6d) y corpus: cuando el autofix de repetición cambia pescado por pollo, la pechuga hereda el TIEMPO del
pescado. Copia de un plan real (nocturno sin tiempo, «pescado->pollo»): «Seca 255 g de pechuga de pollo…» y «Calienta
pechuga de pollo en una sartén a fuego bajo durante 1-2 minutos» — la sardina de lata sólo se calentaba; la pechuga cruda
no. También «añade el pollo al guiso en los últimos minutos» (el pescado que no se deshace). `ave_a_74` (392) sólo sube
los °C que ya estaban; el 441 sólo cubre «bien caliente»; el 737 pone al FINAL la nota «debe cocinarse por completo» y el
paso seguía diciendo 1-2 minutos.

Aquí, con ave CRUDA en la lista (no cocida, lata, ahumada, rostizada, desmenuzada, jamón) y ningún 74 °C en los pasos (las
notas no cuentan) ni «💡 Cocción previa» del ave: la PRIMERA cláusula que la calienta o la cocina con ≤5 minutos (o «en
los últimos minutos») pasa a «cocina… 8-10 minutos» («6-7 minutos por lado» si es por lado; «a fuego medio» si decía
bajo), «hasta que el pollo alcance 74 °C por dentro». Una cláusula que ya dice cuándo está hecha no se toca. Knob
`MEALFIT_POULTRY_SAFE_TIME` (True). tooltip-anchor: P1-PLAN-LOTE-888
"""
from __future__ import annotations

import re
import unicodedata

_AVE_LISTA_RE = re.compile(r"\b(?:pollo|pechugas?|pavo|muslos?|contramuslos?)\b")
_YA_HECHA_RE = re.compile(r"cocid|\blatas?\b|enlatad|ahumad|rostizad|desmenuzad|jamon|salchich|embutid|caldo|consome|"
                          r"cubito|sazon|polvo")
_AVE_RE = re.compile(r"\b(?:pollo|pechugas?|pavo|muslos?|contramuslos?)\b", re.IGNORECASE)
_PEZ_RE = re.compile(r"\b(?:pescados?|tilapia|merluza|salm[oó]n|at[uú]n|sardinas?|bacalao|camarones?|mero|chillo|"
                     r"corvina|pargo)\b", re.IGNORECASE)
_HUEVO_RE = re.compile(r"\b(?:huevos?|claras?|yemas?|tortitas?)\b", re.IGNORECASE)
_VERBO_RE = re.compile(r"\b(?:calienta|cocina|cuece|saltea|sella|dora|asa|incorpora|añade|agrega|guisa|sofr[ií]e|fr[ií]e|"
                       r"pon|coloca)\b", re.IGNORECASE)
_PUNTO_RE = re.compile(r"7[0-9]\s*°|sin\s+partes\s+rosadas|no\s+(?:quede|est[eé]|tenga)n?\s+rosad|jugos?\s+(?:salgan\s+)?"
                       r"claros|bien\s+cocid|completamente\s+cocid|por\s+completo|hasta\s+que\s+(?:est[eé]|quede)n?\s+"
                       r"cocid", re.IGNORECASE)
_TIEMPO_RE = re.compile(r"(?P<t>(?P<pre>durante\s+|por\s+)?(?P<a>\d+)(?:\s*-\s*(?P<b>\d+))?\s*(?:min(?:utos?)?)\b)"
                        r"(?P<mas>\s+m[aá]s)?(?P<lado>\s+por\s+lado)?", re.IGNORECASE)
#: una cocción POSTERIOR que sigue cociendo el ave (el guiso tapado 15 min, «cocina 6-8 min más»): la primera no manda
_LUEGO_RE = re.compile(r"\b(?:guis|cocin|cuec|hierv|horne|salte|sofr|tap)\w*\b[^.;]*?\b(?P<a>\d+)(?:\s*-\s*(?P<b>\d+))?\s*"
                       r"min", re.IGNORECASE)
#: las señales de punto del PESCADO que heredó el ave («hasta que se desmenuce fácilmente»)
_PUNTO_PEZ_RE = re.compile(r"^\s*,?\s*hasta\s+que\s+se\s+desmenuce\s+f[aá]cilmente", re.IGNORECASE)
#: la razón del pescado que ya no aplica («…en los últimos minutos para que no se deshaga»)
_DESHAGA_RE = re.compile(r",?\s+para\s+que\s+no\s+se\s+(?:deshaga|desbarate|rompa|desmorone)n?\b", re.IGNORECASE)
_ULTIMOS_RE = re.compile(r"\s+(?:en\s+)?(?:los\s+)?[uú]ltimos\s+minutos", re.IGNORECASE)
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|nota\b)", re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_POULTRY_SAFE_TIME", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _quien(clausula: str) -> str:
    return "el pavo" if re.search(r"\bpavo\b", _sa(clausula)) else "el pollo"


def _reparar(cl: str) -> str:
    """La cláusula con cocción real; la misma si no hay nada que reparar."""
    ave = _AVE_RE.search(cl)
    if not ave or _PEZ_RE.search(cl) or _HUEVO_RE.search(cl) or _PUNTO_RE.search(cl) or not _VERBO_RE.search(cl):
        return cl
    punto = f"hasta que {_quien(cl)} alcance 74 °C por dentro"
    ult = _ULTIMOS_RE.search(cl, ave.end())
    if ult:
        pron = "la" if re.search(r"\bpechugas?\b", ave.group(0), re.IGNORECASE) else "lo"
        resto = _DESHAGA_RE.sub("", cl[ult.end():])
        nuevo = cl[:ult.start()] + f" y cocína{pron} 10-12 minutos, {punto}" + resto
    else:
        t = _TIEMPO_RE.search(cl, ave.end())
        if not t or int(t.group("b") or t.group("a")) > 5:
            return cl
        if _sigue_cociendo([cl[t.end():]]):      # «dóralo 4 minutos, agrega el agua… y guisa tapado 15 min»: en la misma frase
            return cl
        lado = t.group("lado") or ""
        tiempo = ("6-7 minutos" if lado else "8-10 minutos") + (" más" if t.group("mas") else "") + lado
        resto = _PUNTO_PEZ_RE.sub("", cl[t.end():])
        if re.match(r"\s*,?\s*hasta\s+que\s", resto, re.IGNORECASE):    # «…hasta que esté opaco» → «… 74 °C y esté opaco»
            resto = re.sub(r"^\s*,?\s*hasta\s+que\s+", " y ", resto, flags=re.IGNORECASE)
        elif re.match(r"\s+\w", resto):                                    # «… 74 °C por dentro, con el orégano…»
            resto = "," + resto
        nuevo = cl[:t.start()] + f"{t.group('pre') or ''}{tiempo}, {punto}" + resto
    nuevo = re.sub(r"\bCalienta\b(?=\s+(?:la\s+|el\s+|\d|½|¼|¾)?(?:pechugas?|pollo|pavo)\b)", "Cocina", nuevo)
    nuevo = re.sub(r"\bcalienta\b(?=\s+(?:la\s+|el\s+|\d|½|¼|¾)?(?:pechugas?|pollo|pavo)\b)", "cocina", nuevo)
    nuevo = re.sub(r"\ba\s+fuego\s+bajo\b", "a fuego medio", nuevo)
    nuevo = re.sub(r"\b(pollo|pavo|pechugas?(?:\s+de\s+(?:pollo|pavo))?)\s+(?:ya\s+)?cocid[oa]s?\b", r"\1", nuevo)  # la lista: cruda
    return re.sub(r",\s*,", ",", nuevo)


def _sigue_cociendo(textos: list) -> bool:
    """¿Una cláusula posterior (antes del Montaje, sin notas) sigue cociendo el AVE ≥6 min — la nombra, o guisa/tapa la
    olla, o cuece «el guiso / la mezcla / todo / la salsa»? Entonces el primer tiempo corto no manda: no se toca. Otra
    cocción cualquiera («hierve la yuca 15 min») no cuenta."""
    for t in textos:
        if _NOTA_RE.search(t):
            continue
        if re.match(r"\s*montaje", _sa(t)):
            break
        for c in re.split(r"(?<=[.;])\s+", t):
            sc = _sa(c)
            if not (_AVE_RE.search(c) or re.search(r"\bguis|\btap", sc) or re.search(r"\b(?:guiso|mezcla|todo|salsa)\b", sc)):
                continue
            for m in _LUEGO_RE.finditer(c):
                if int(m.group("b") or m.group("a")) >= 6:
                    return True
    return False


def asegurar(meal) -> int:
    """Nº de pasos corregidos (0 o 1); 0 ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = [_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        aves = [l for l in lineas if _AVE_LISTA_RE.search(l)]
        if not aves or any(_YA_HECHA_RE.search(l) for l in aves):
            return 0
        pasos = " . ".join(_sa(p) for p in rec if isinstance(p, str) and not _NOTA_RE.search(p))
        if re.search(r"\b7[0-9]\s*°", pasos) or re.search(r"coccion previa[^.]*\b(?:pollo|pechuga|pavo)", pasos):
            return 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _NOTA_RE.search(p) or _sa(p).lstrip().startswith("mise en place"):
                continue
            trozos = re.split(r"((?<=[.;])\s+)", p)
            for k in range(0, len(trozos), 2):
                cl = trozos[k]
                if not (_AVE_RE.search(cl) and _VERBO_RE.search(cl)):
                    continue
                despues = trozos[k + 2::2] + [x for x in rec[i + 1:] if isinstance(x, str)]
                if _sigue_cociendo(despues):
                    return 0                     # «dora el pollo 3 min…; guisa tapado 15 min»: el guiso lo termina
                nuevo = _reparar(cl)
                if nuevo == cl:
                    return 0                     # la PRIMERA cocción del ave ya está bien (o no es reparable): nada
                trozos[k] = nuevo
                rec[i] = "".join(trozos)
                meal["recipe"] = rec
                meal.pop("_display", None)
                return 1
        return 0
    except Exception:                                                          # noqa: BLE001
        return 0
