# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-545 · 2026-09-27] Con poco tiempo, la legumbre lenta llega cocida (de lata), no seca.

Verificador formulario↔plan sobre las 17 baterías reales del 27-sep: en 11 perfiles con «30 min» o «Nada» de tiempo un
plato de habichuelas/garbanzos SECOS sumaba 50-143 min —«💡 Cocción previa: remoja las habichuelas rojas secas 8-12 h y
hiérvelas 60-90 min»— contra lo que el usuario respondió (y 119 min con «1 hora»). Si el formulario dice «Nada», «30 min»
o «1 hora» y no cocina por tandas a menudo: la habichuela, el frijol, el garbanzo, el gandul y el haba secos pasan a
cocidos de lata, con el MISMO factor con el que el lote 343 convierte seco→cocido (kcal de la fila / kcal del cocido de
su familia) — los macros no cambian. Con «Nada», también las lentejas. Corre al entrar en `assemble_plan_node` (antes del
motor) y otra vez en `finalize_plan_data_coherence` (lo que la autocrítica rehízo después; sin «cocina por tandas» allí:
de lata nunca contradice al formulario). Sin catálogo o sin gramos, la línea se queda como vino.
tooltip-anchor: P1-PLAN-LOTE-545
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_LENTAS = r"(habichuelas?|frijol(?:es)?|garbanzos?|gandules|guandules|habas?)"
_LENTEJA = r"(lentejas?)"
_SECO = re.compile(r"\b(sec[oa]s?|crud[oa]s?)\b", re.IGNORECASE)
_YA_LISTA = re.compile(r"\b(cocid[oa]s?|de lata|enlatad[oa]s?|en lata|precocid[oa]s?|list[oa]s?|tostad[oa]s?|para comer|"
                       r"harina|crema|hummus|humus|pasta|snack|germinad[oa]s?|brotes?|sopa|pure)\b", re.IGNORECASE)
_COLOR = r"(?:rojas?|rojos?|negras?|negros?|blancas?|blancos?|pintas?|pintos?|rosadas?|rosados?|verdes?)"
_FEM = ("habichuela", "haba", "lenteja")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def aplica(form_data) -> str:
    """'' (no toca), 'lentas' (30 min) o 'todas' («Nada»: también lentejas)."""
    fd = form_data if isinstance(form_data, dict) else {}
    ct = str(fd.get("cookingTime") or "").strip().lower()
    if str(fd.get("batchCooking") or "").strip().lower() == "often":
        return ""
    if ct == "none":
        return "todas"
    if ct in ("30min", "1hour"):                   # remojo + 60-90 min de hervor no caben ni en una hora
        return "lentas"
    return ""


def _redondea(g: float) -> int:
    return int(round(g)) if g < 20 else int(5 * round(g / 5.0))


# Lo que los pasos ya decían de la legumbre SECA (batería real + corpus de 161 platos convertidos): el aviso «⚠️ remoja las
# habichuelas secas y hiérvelas… al menos 10 minutos» (etiquetas_clinicas, puesto ANTES de convertir: 97), la «💡 Cocción
# previa… 8-12 h… 60-90 min» (17), el cerrador «Cocina habichuelas rojas secas en agua hasta que ablanden e incorpóralas al
# plato», «si usas habichuelas secas, remójalas…», «(remojados desde la noche anterior)», «20 g de habichuelas rojas secas».
_PREP = re.compile(r"remoj|hi[eé]rv|cocci[oó]n previa|precoc|ablanden|noche anterior|\bsec[oa]s\b|en crudo", re.IGNORECASE)
_CUALQUIERA = re.compile(r"habichuel|frijol|garbanz|gandul|guandul|\bhabas?\b|lentej|jud[ií]a|alubia|poroto")
_NOTA = ("⚠", "💡")
_CERRADOR = re.compile(r"\b(?:Cocina|Incorpora)\s+(?:(?:los|las)\s+)?(?P<x>[^.;:]{2,60}?)\s+sec[oa]s\s*,?\s*(?:remojad[oa]s\s+"
                       r"desde\s+la\s+noche\s+anterior\s*,?\s*)?en\s+agua\s+hasta\s+que\s+ablanden\s+e\s+incorp\w+\s+al\s+plato",
                       re.IGNORECASE)
_PARENTESIS = re.compile(r"\s*\([^()]*(?:remoj|hervid|noche anterior|crudo)[^()]*\)", re.IGNORECASE)


def _enjuagadas(cocid: str) -> str:
    return "enjuagadas y escurridas" if cocid == "cocidas" else "enjuagados y escurridos"


def _limpia_pasos(meal: dict, cab: str, base: str, cocid: str, g_cocido: int, unica: bool) -> None:
    """Los pasos dejan de preparar la legumbre seca: fuera los avisos de remojo/hervor de ESA legumbre y, en el texto, lo
    seco pasa a la lata (con los gramos de la línea si es la única de su clase)."""
    rec = meal.get("recipe")
    if not isinstance(rec, list):
        return
    raiz = cab[:6]
    art = "las" if cocid == "cocidas" else "los"
    de_lata = f"{g_cocido} g de {base} {cocid} de lata" if unica else f"{art} {base} {cocid} de lata"
    # el aviso de etiquetas_clinicas dice «habichuelas» también para frijoles: fuera si NO queda legumbre seca en el plato
    queda_seca = any(isinstance(x, str) and _CUALQUIERA.search(_sa(x)) and not _YA_LISTA.search(_sa(x))
                     for x in (meal.get("ingredients") or []))
    pasos = []                                             # por paso: str intacto o lista de cláusulas
    mencionada = False
    for p in rec:
        if isinstance(p, str) and any(e in p for e in _NOTA):
            if not queda_seca and _CUALQUIERA.search(_sa(p)) and _PREP.search(_sa(p)):
                continue                                   # el aviso de remojo/hervor de la seca ya no aplica
            pasos.append(p)
            continue
        if not isinstance(p, str) or raiz not in _sa(p):
            pasos.append(p)
            continue
        p = _CERRADOR.sub(lambda m: f"Incorpora {art} {base} de lata, {_enjuagadas(cocid)}, al plato", p)
        clausulas = []
        for c in re.split(r"(?<=[;.])\s+", p):
            sc = _sa(c)
            if raiz not in sc:
                clausulas.append(("", c))
                continue
            fin = c[-1] if c[-1:] in ";." else ""
            cab_c = re.match(r"^\s*(?:mise en place|el toque de fuego|montaje)\s*:\s*", c, re.IGNORECASE)
            pre = cab_c.group(0) if cab_c else ""
            cuerpo = c[len(pre):].rstrip(";.")
            if _PREP.search(sc) and re.match(r"^(?:si\s|remoja\b|rem[oó]jal)", _sa(cuerpo)):
                clausulas.append(("cond", (pre, fin)))     # «si usas habichuelas secas, remójalas…»: se decide al final
                continue
            cuerpo = _PARENTESIS.sub("", cuerpo)
            cuerpo = re.sub(r",?\s*remojad[oa]s\s+desde\s+la\s+noche\s+anterior", "", cuerpo, flags=re.IGNORECASE)
            cuerpo = re.sub(r"(\b" + re.escape(raiz) + r"\w*)\s+remojad[oa]s\b", r"\1 de lata", cuerpo, flags=re.IGNORECASE)
            # la MENCIÓN seca entera («¼ taza de garbanzos secos (60 g) cocidos», «315 g de frijoles negros secos
            # (cocidos)») pasa a la cocida de la línea; sin cantidad, sólo cambia la palabra
            _mencion = re.compile(
                r"(?P<q>(?:\d+\s*[¼½¾⅓⅔⅛]|\d+(?:[.,]\d+)?|[¼½¾⅓⅔⅛])\s*(?:g|gr|gramos|tazas?|cdas?|cucharadas?)?\s+de\s+)?"
                r"(?P<n>(?:(?:los|las)\s+)?" + re.escape(raiz) + r"\w*(?:\s+" + _COLOR + r")?)\s+(?:sec[oa]s|crud[oa]s)\b"
                r"(?:\s*\([^()]*\))?(?:\s+cocid[oa]s)?", re.IGNORECASE)

            def _cambia(mm):
                n_ = re.sub(r"^(?:los|las)\s+", "", mm.group("n"), flags=re.IGNORECASE)
                if mm.group("q") and unica:
                    return f"{g_cocido} g de {n_} {cocid}"
                return f"{mm.group('q') or ''}{mm.group('n')} {cocid}"
            cuerpo = _mencion.sub(_cambia, cuerpo)
            if unica:                                      # los gramos del paso, los de la línea
                cuerpo = re.sub(r"\b\d+(?:[.,]\d+)?\s*g\s+de\s+(?=" + re.escape(raiz) + ")", f"{g_cocido} g de ", cuerpo,
                                count=1, flags=re.IGNORECASE)
            cuerpo = re.sub(r"\b(" + re.escape(cocid) + r")\s+(?:\(\s*)?" + re.escape(cocid) + r"\)?", r"\1", cuerpo)
            clausulas.append(("", pre + cuerpo + fin))
            mencionada = True
        pasos.append(clausulas)
    nuevos = []
    for p in pasos:
        if not isinstance(p, list):
            nuevos.append(p)
            continue
        out = []
        for tipo, c in p:
            if tipo != "cond":
                out.append(c)
                continue
            pre, fin = c
            if mencionada:                                 # otra cláusula ya la mide o la usa: la condición sobra
                if pre and not out:
                    out.append(pre.rstrip())
                elif out and fin == "." and out[-1].endswith(";"):
                    out[-1] = out[-1][:-1] + "."
                continue
            out.append(f"{pre}{'E' if not pre and not out else 'e'}scurre y enjuaga {de_lata}{fin}")
            mencionada = True
        texto = " ".join(x for x in out if x).strip()
        if texto and not re.fullmatch(r"(?:mise en place|el toque de fuego|montaje)\s*:", texto, re.IGNORECASE):
            nuevos.append(texto)
    meal["recipe"] = nuevos


def a_lata(days, form_data, db=None) -> int:
    """Nº de líneas convertidas; 0 ante cualquier error (fail-open)."""
    modo = aplica(form_data)
    if not modo:
        return 0
    n = 0
    try:
        import cocido_en_catalogo as _cc
        from nutrition_db import _split_qty_unit_name
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
        pat = re.compile(r"\b" + (_LENTAS if modo == "lentas" else r"(?:" + _LENTAS + "|" + _LENTEJA + r")") + r"\b")
        for day in days or []:
            for m in (day.get("meals") or []) if isinstance(day, dict) else []:
                ings = m.get("ingredients") if isinstance(m, dict) else None
                if not isinstance(ings, list):
                    continue
                for i, linea in enumerate(list(ings)):
                    if not isinstance(linea, str):
                        continue
                    low = _sa(linea)
                    if not pat.search(low) or _YA_LISTA.search(low):
                        continue
                    _q, _u, nombre = _split_qty_unit_name(linea)
                    info = db.lookup(nombre)
                    fam = _cc.familia(getattr(info, "name", "")) if info else None
                    kcal = float(getattr(info, "kcal", 0) or 0)
                    if not fam or kcal <= 0 or kcal / fam[0] < 1.5:
                        continue                                   # la fila ya está en cocido: nada que convertir
                    if not _SECO.search(low) and not re.match(r"^\s*[\d.,]+\s*(?:g|gr|gramos)\b", low):
                        continue                                   # sin «seca» y sin gramos: no sé en qué base está
                    g_seco = db.grams_from_ingredient_string(linea)
                    if not g_seco or float(g_seco) <= 0:
                        continue
                    g_cocido = _redondea(float(g_seco) * kcal / fam[0])
                    base = re.sub(r"\([^()]*\)", " ", str(nombre)).split(",")[0]      # «(43 g), remojados desde…» fuera
                    base = re.sub(r"\bremojad[oa]s?\b.*$", " ", base, flags=re.IGNORECASE)
                    base = _SECO.sub("", base).strip(" ,")
                    base = re.sub(r"\s+", " ", base)
                    cab = _sa(base.split(" ")[0])
                    cocid = "cocidas" if any(cab.startswith(f) for f in _FEM) else "cocidos"
                    nueva = f"{g_cocido} g de {base[:1].lower() + base[1:]} {cocid} (de lata, enjuagadas y escurridas)"
                    if cocid == "cocidos":
                        nueva = nueva.replace("enjuagadas y escurridas", "enjuagados y escurridos")
                    ings[i] = nueva
                    raw = m.get("ingredients_raw")
                    if isinstance(raw, list):
                        cabeza = pat.search(low).group(0)
                        for j, r in enumerate(raw):                # su pareja por ALIMENTO (la misma legumbre, aún seca)
                            if isinstance(r, str) and cabeza in _sa(r) and not _YA_LISTA.search(_sa(r)):
                                raw[j] = nueva
                                break
                    unica = sum(1 for x in ings if isinstance(x, str) and cab[:6] in _sa(x)) == 1
                    _limpia_pasos(m, cab, base[:1].lower() + base[1:], cocid, g_cocido, unica)
                    m.setdefault("_legumbre_a_lata", []).append(f"{linea} → {nueva}")
                    m.pop("_display", None)
                    n += 1
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-545] no-op: {type(e).__name__}: {e}")
    if n:
        logger.info(f"🥫 [P1-PLAN-LOTE-545] {n} legumbre(s) seca(s) → cocida(s) de lata (tiempo de cocina corto)")
    return n


__all__ = ["aplica", "a_lata"]
