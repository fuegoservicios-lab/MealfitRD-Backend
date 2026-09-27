# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-311 · 2026-09-25] La avena COCIDA lleva líquido para cocinarse.

El motor de macros encoge la leche de la avena para cuadrar el día: «30 g de avena» con «15 ml de leche descremada». Los
macros son exactos, pero con 15 ml no se cocina una avena. Hasta el lote 302 el paso seguía diciendo «mide 30 g de avena y
250 ml de leche» (y el plato comido llevaba ~80 kcal más de las contadas); desde el 302 el paso sigue a la lista y la
receta sería imposible. Baterías del 25-sep: 9 desayunos en 25 planes (30-60 g de avena con 0-65 ml de líquido).

El líquido que falta es AGUA: 0 kcal, fuera de la lista de compras (P2-PDF-2) y en la lista blanca del guard de coherencia
(P3-GUARD-BLIND-WATER-WHITELIST), así que ni los macros ni la compra cambian. Sólo avena COCIDA (un paso la cocina, hierve,
calienta o la lleva al microondas; nunca la remojada en la nevera ni la cruda en un batido), y nunca si ya hay líquido
suficiente (≥ 3 ml por g). Se completa a 4 ml por g, en la línea de agua del plato si ya existe o en una nueva, y el paso que
mide la leche dice también el agua. Knob `MEALFIT_OAT_LIQUID_FILL` (True). tooltip-anchor: P1-PLAN-LOTE-311
"""
from __future__ import annotations

import math
import re
import unicodedata

RATIO_MIN = 3.0          # ml de líquido por g de avena por debajo de los cuales no se cocina
RATIO_OBJ = 4.0          # a cuánto se completa
G_POR_TAZA_AVENA = 85.0  # hojuelas
ML = {"ml": 1.0, "l": 1000.0, "litro": 1000.0, "litros": 1000.0, "taza": 240.0, "tazas": 240.0, "cda": 15.0,
      "cdas": 15.0, "cucharada": 15.0, "cucharadas": 15.0, "cdta": 5.0, "cdtas": 5.0, "cucharadita": 5.0,
      "cucharaditas": 5.0}
_FR = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
_LINEA_RE = re.compile(r"^\s*(?P<q>\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s*(?P<u>[A-Za-z]+)\.?\s+(?:de\s+)?(?P<food>.+)$")
_AVENA_RE = re.compile(r"\bavena\b")
#: [P1-PLAN-LOTE-428 · 2026-09-26] «1¾ tazas de avena COCIDA» ya está hidratada: el plan de emergencia recibía 600 ml de
#: agua encima («Completa el líquido con 600 ml de agua para que la avena se cocine», 24 de 36 casos del corpus).
_NO_HOJUELA_RE = re.compile(r"\b(harina|leche|bebida|salvado|galletas?|barras?|granola|batido|cocid[ao]s?)\b")
_LIQUIDO_RE = re.compile(r"\b(leche|bebida|agua)\b")
_NO_LIQUIDO_RE = re.compile(r"\b(en polvo|condensada|evaporada)\b")
_AGUA_RE = re.compile(r"^\s*agua\b")
#: un paso que COCINA la avena como gacha: «cocina la avena con la leche…», «la avena … en el microondas / a fuego medio
#: hasta que espese». Sobre texto sin acentos y en minúscula.
_COCCION_RE = re.compile(
    r"\b(?:cocina|cuece|hierve|calienta|cocinar|cocer|hervir|calentar)\s+(?:la\s+|las\s+)?(?:[a-z]+\s+)?avena\b"
    r"|\bavena\b[^.;]{0,60}\b(?:microondas|a fuego|en una olla|en la olla|espese|espesa|hierva|hervor)\b"
    # [P1-PLAN-LOTE-318 · 2026-09-25] la FRASE que nombra la avena y la cocina, en cualquier orden y a cualquier distancia
    # («en un tazón apto para microondas mezcla la avena con la leche y cocina 2-3 minutos … hasta que espese»: el verbo
    # quedaba a >60 caracteres y la avena de la batería de cierre salía con 5 ml de leche). tooltip-anchor: P1-PLAN-LOTE-318
    r"|\bavena\b[^.;]*\b(?:cocina|cuece|hierve|microondas|a fuego|olla|espese|espesa|hervor)\b"
    r"|\b(?:cocina|cuece|hierve|microondas|a fuego|olla)\b[^.;]*\bavena\b")
#: lo que NO es una gacha: remojada en frío, masa, horneado o licuado
_FRIO_RE = re.compile(r"\b(remoja|remojar|remojada|toda la noche|overnight|en la nevera|refrigera|refrigerar|reposar|"
                      r"panqueques?|pancakes?|tortillas? de avena|galletas?|muffins?|waffles?|crepas?|arepas? de avena|"
                      r"masa|empaniz\w*|rebozad\w*|licua\w*|hornea\w*|horno|"
                      r"(?:tuesta|tostar|dora|dorar)\s+(?:la\s+)?avena|avena\s+tostada|avena\s+fria|avena\s+cruda)\b")
#: [P1-PLAN-LOTE-318] el BATIDO es un plato (su nombre), no un participio: «incorpora el huevo batido» en una avena cremosa
#: la excluía (batería de cierre, perfil con warfarina: 30 g de avena con ¼ taza de leche)
_NOMBRE_BATIDO_RE = re.compile(r"\b(batid[oa]s?|smoothies?|licuados?)\b")
#: el peso entre paréntesis de la línea de avena («1 taza de avena (70 g)») manda sobre la taza estimada
_HINT_G_RE = re.compile(r"\(\s*[≈~]?\s*(\d+(?:[.,]\d+)?)\s*g\s*\)")
#: una línea de agua que no se deja medir («½ agua»): no se adivina cuánto líquido hay
_AGUA_SIN_MEDIDA_RE = re.compile(r"^\s*[\d.,½¼¾⅓⅔\s]+agua\s*$")
_NOTA = ("⚠", "💡", "🌱", "⚕")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def activo() -> bool:
    try:
        from knobs import _env_bool
        return bool(_env_bool("MEALFIT_OAT_LIQUID_FILL", True))
    except Exception:
        return True


def _num(q: str):
    q = str(q).strip().replace(",", ".")
    if q in _FR:
        return _FR[q]
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([½¼¾⅓⅔])?$", q)
    if not m:
        return None
    return float(m.group(1)) + (_FR[m.group(2)] if m.group(2) else 0.0)


def _fmt(v: float) -> str:
    return str(int(round(v)))


def completar(meal) -> int:
    """ml de agua añadidos (0 si no aplica o ante cualquier error). Muta `ingredients`, `ingredients_raw` y `recipe`."""
    try:
        if not activo() or not isinstance(meal, dict):
            return 0
        ings = meal.get("ingredients")
        rec = meal.get("recipe")
        if not isinstance(ings, list) or not isinstance(rec, list) or not rec:
            return 0
        pasos = [p for p in rec if isinstance(p, str) and not any(e in p for e in _NOTA)]
        texto = _sa(" ".join(pasos) + " " + str(meal.get("name") or ""))
        if not _AVENA_RE.search(texto) or _FRIO_RE.search(texto) or _NOMBRE_BATIDO_RE.search(_sa(meal.get("name"))):
            return 0
        if not any(_COCCION_RE.search(_sa(p)) for p in pasos):
            return 0
        avena_g = 0.0
        liquido = 0.0
        agua_idx = None
        leche_linea = None
        for i, ln in enumerate(ings):
            if _AGUA_SIN_MEDIDA_RE.match(_sa(ln)):
                return 0
            m = _LINEA_RE.match(str(ln))
            if not m:
                continue
            v = _num(m.group("q"))
            u = _sa(m.group("u"))
            food = _sa(m.group("food"))
            if v is None:
                continue
            if _AVENA_RE.search(food) and not _NO_HOJUELA_RE.search(food):
                hint = _HINT_G_RE.search(str(ln))
                if u in ("g", "gr", "gramos", "gramo"):
                    avena_g += v
                elif hint:
                    avena_g += float(hint.group(1).replace(",", "."))
                elif u in ("taza", "tazas"):
                    avena_g += v * G_POR_TAZA_AVENA
            elif _LIQUIDO_RE.search(food) and not _NO_LIQUIDO_RE.search(food) and u in ML:
                liquido += v * ML[u]
                if _AGUA_RE.match(food) and u in ML and agua_idx is None:     # [P1-PLAN-LOTE-428] «1 taza de agua» también
                    agua_idx = i
                elif not _AGUA_RE.match(food) and leche_linea is None:
                    leche_linea = str(ln).strip()
        if avena_g <= 0 or liquido >= RATIO_MIN * avena_g:
            return 0
        faltan = max(30.0, math.ceil((RATIO_OBJ * avena_g - liquido) / 10.0) * 10.0)   # a la decena, hacia arriba
        raw = meal.get("ingredients_raw") if isinstance(meal.get("ingredients_raw"), list) else None
        if agua_idx is not None:
            viejo = str(ings[agua_idx])
            mv = _LINEA_RE.match(viejo)
            nuevo_total = (_num(mv.group("q")) or 0.0) * ML.get(_sa(mv.group("u")), 1.0) + faltan   # [P1-PLAN-LOTE-428]
            nueva = f"{_fmt(nuevo_total)} ml de {mv.group('food').strip()}"
            ings[agua_idx] = nueva
            if raw is not None:
                hits = [j for j, r in enumerate(raw) if isinstance(r, str) and r.strip() == viejo.strip()]
                if len(hits) == 1:
                    raw[hits[0]] = nueva
            agua_txt = f"{_fmt(nuevo_total)} ml de agua"
            _reescribir_agua(meal, viejo, agua_txt)
        else:
            nueva = f"{_fmt(faltan)} ml de agua"
            ings.append(nueva)
            if raw is not None:
                raw.append(nueva)
            _anadir_agua_al_paso(meal, leche_linea, nueva)
        meal["_avena_liquido"] = int(faltan)
        meal.pop("_display", None)      # DELETE-on-write: `_display[locale]` espeja la lista y los pasos
        return int(faltan)
    except Exception:
        return 0


def _reescribir_agua(meal, viejo_linea: str, agua_txt: str) -> None:
    """Hay línea de agua y creció: la primera mención cuantificada del agua en los pasos pasa al total nuevo."""
    rx = re.compile(r"\b\d+(?:[.,]\d+)?\s*ml\s+de\s+agua\b", re.IGNORECASE)
    rec = meal.get("recipe") or []
    mv = _LINEA_RE.match(str(viejo_linea or ""))
    if mv:                                    # [P1-PLAN-LOTE-428] «1 taza de agua» del paso → el total nuevo, en ml
        rx_vieja = re.compile(r"(?<![\w½¼¾⅓⅔.,])" + re.escape(mv.group("q").strip()) + r"\s*" + re.escape(mv.group("u"))
                              + r"\.?\s+de\s+agua\b", re.IGNORECASE)
        hits = [i for i, p in enumerate(rec) if isinstance(p, str) and not any(e in p for e in _NOTA) and rx_vieja.search(p)]
        if hits:
            for i in hits:
                rec[i] = rx_vieja.sub(agua_txt, rec[i])
            return
    for i, p in enumerate(rec):
        if isinstance(p, str) and not any(e in p for e in _NOTA) and rx.search(p):
            rec[i] = rx.sub(agua_txt, p, count=1)
            return
    _anadir_agua_al_paso(meal, None, agua_txt)


def _anadir_agua_al_paso(meal, leche_linea, agua_txt: str) -> None:
    """El paso que mide la leche dice también el agua («mide 30 g de avena y 15 ml de leche» → «…, 15 ml de leche y 105 ml
    de agua»); si ningún paso la mide, el paso que cocina la avena acaba con «Completa con N ml de agua»."""
    rec = meal.get("recipe") or []
    if leche_linea:
        m = _LINEA_RE.match(leche_linea)
        if m:
            q = re.escape(m.group("q").strip())
            u = re.escape(m.group("u"))
            primera = _sa(m.group("food")).split()[0] if m.group("food").split() else ""
            if primera:
                rx = re.compile(r"(?i)(?<![\w½¼¾⅓⅔.,])" + q + r"\s*" + u + r"\.?\s+de\s+" + primera
                                + r"[^,.;:()]{0,40}?(?=\s+y\s|[,.;:]|$)")
                for i, p in enumerate(rec):
                    if not isinstance(p, str) or any(e in p for e in _NOTA):
                        continue
                    base = _sa(p)
                    if len(base) != len(p):
                        continue
                    mm = rx.search(base)
                    if mm:
                        rec[i] = p[:mm.end()] + f" y {agua_txt}" + p[mm.end():]
                        return
    # [P1-PLAN-LOTE-317 · 2026-09-25] «cocina la avena con agua a fuego medio»: el agua que el paso ya nombra sin medida
    # recibe la suya ahí («con 200 ml de agua»), en vez de una frase al final del paso («…deja que se enfríe. Completa el
    # líquido con 200 ml de agua…», batería renal del 25-sep). tooltip-anchor: P1-PLAN-LOTE-317
    rx_agua = re.compile(r"\b(con|en)\s+(?:el\s+)?agua\b")
    for i, p in enumerate(rec):
        if not isinstance(p, str) or any(e in p for e in _NOTA) or not _COCCION_RE.search(_sa(p)):
            continue
        base = _sa(p)
        av = _AVENA_RE.search(base)
        if not av or len(base) != len(p):
            continue
        mm = rx_agua.search(base, av.end(), av.end() + 60)       # el agua de la AVENA, no la de otro alimento del paso
        if mm:
            rec[i] = p[:mm.start()] + p[mm.start(1):mm.end(1)] + f" {agua_txt}" + p[mm.end():]
            return
    if _agua_donde_se_nombra_el_liquido(rec, agua_txt):          # [P1-PLAN-LOTE-428]
        return
    for i, p in enumerate(rec):
        if isinstance(p, str) and not any(e in p for e in _NOTA) and _COCCION_RE.search(_sa(p)):
            s = p.rstrip()
            rec[i] = s + ("" if s.endswith(".") else ".") + f" Completa el líquido con {agua_txt} para que la avena se cocine."
            return


# ── [P1-PLAN-LOTE-428 · 2026-09-26] El agua va donde se nombra el líquido, no al final del paso ─────────────────────────
# Batería REAL sobre el 424 (familia de 4, desayuno del día 2): «lleva la leche descremada con la canela…; añade la avena y
# cocina 8-10 minutos… En una sartén… saltea la pera y el mango… Completa el líquido con 160 ml de agua para que la avena se
# cocine»: el agua llegaba DESPUÉS de cocinar la avena y de saltear la fruta. La mise en place nombraba la leche sin medida
# y la frase del lote 311 no encontraba dónde ponerla. Ahora entra con la leche que el paso nombra («lleva la leche
# descremada y 160 ml de agua…») o en «con el líquido» («cocina la avena con 600 ml de agua…»); si una receta ya trae la
# frase al final, `ubicar_agua` la lleva a su sitio. tooltip-anchor: P1-PLAN-LOTE-428
_LIQUIDO_NOMBRADO_428_RE = re.compile(r"\bcon el liquido\b")
_LECHE_NOMBRADA_428_RE = re.compile(
    r"\b(?:la|el)\s+(?:leche|bebida)(?:\s+(?:de\s+(?:almendras?|soya|coco|avena|arroz)|evaporada|descremada|"
    r"semidescremada|entera|deslactosada|light|vegetal|sin\s+lactosa))*")
_COMPLETA_428_RE = re.compile(r"\s*Completa el líquido con (?P<agua>\d+ ml de agua) para que la avena se cocine\.")


def _agua_donde_se_nombra_el_liquido(rec: list, agua_txt: str) -> bool:
    """«con el líquido» → «con N ml de agua»; si no, «la leche descremada» → «la leche descremada y N ml de agua» (con
    «, N ml de agua y» cuando a la leche la sigue otro «y»). Sólo en un paso que cocina la avena. True si lo puso."""
    for i, p in enumerate(rec):
        if not isinstance(p, str) or any(e in p for e in _NOTA) or not _COCCION_RE.search(_sa(p)):
            continue
        base = _sa(p)
        if len(base) != len(p):
            continue
        mm = _LIQUIDO_NOMBRADO_428_RE.search(base)
        if mm:
            rec[i] = p[:mm.start()] + f"con {agua_txt}" + p[mm.end():]
            return True
        ml = _LECHE_NOMBRADA_428_RE.search(base)
        if ml:
            despues = base[ml.end():]
            if re.match(r"\s+y\s", despues):
                rec[i] = p[:ml.end()] + f", {agua_txt}" + p[ml.end():]
            else:
                rec[i] = p[:ml.end()] + f" y {agua_txt}" + p[ml.end():]
            return True
        av = _AVENA_RE.search(base)                       # «cocina la avena con agua…»: el agua sin medida recibe la suya
        ma = re.compile(r"\b(con|en)\s+(?:el\s+)?agua\b").search(base, av.end(), av.end() + 60) if av else None
        if ma:
            rec[i] = p[:ma.start()] + p[ma.start(1):ma.end(1)] + f" {agua_txt}" + p[ma.end():]
            return True
    return False


_AVENA_COCIDA_428_RE = re.compile(r"\bavena\s+cocida\b")
_COCINA_CON_LIQUIDO_428_RE = re.compile(
    r"\b([Cc])ocina la avena con el líquido a fuego medio 5 minutos, removiendo hasta cremosa")


def ubicar_agua(meal) -> int:
    """La frase «Completa el líquido con N ml de agua para que la avena se cocine.» que un paso trae al final pasa a donde
    el paso nombra el líquido; y la avena que la lista compra COCIDA, sin líquido en la lista, se calienta en vez de
    cocinarse «con el líquido» (plantilla del plan de emergencia). 1 si cambió algo; 0 si no, o ante cualquier error."""
    try:
        if not activo() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list):
            return 0
        lista = [_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        if any(_AVENA_COCIDA_428_RE.search(x) for x in lista) and not any(
                _LIQUIDO_RE.search(x) and not _AVENA_RE.search(x) for x in lista):
            for i, p in enumerate(rec):
                if isinstance(p, str) and not any(e in p for e in _NOTA) and _COCINA_CON_LIQUIDO_428_RE.search(p):
                    rec[i] = _COCINA_CON_LIQUIDO_428_RE.sub(
                        lambda m: ("C" if m.group(1) == "C" else "c") + "alienta la avena cocida a fuego medio 2-3 minutos, "
                        "removiendo hasta que esté cremosa", p)
                    meal.pop("_display", None)
                    return 1
        for i, p in enumerate(rec):
            if not isinstance(p, str) or any(e in p for e in _NOTA):
                continue
            mm = _COMPLETA_428_RE.search(p)
            if not mm:
                continue
            prueba = list(rec)
            prueba[i] = (p[:mm.start()] + p[mm.end():]).rstrip()
            if _agua_donde_se_nombra_el_liquido(prueba, mm.group("agua")):
                meal["recipe"] = prueba
                meal.pop("_display", None)
                return 1
            return 0
        return 0
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-448 · 2026-09-27] El agua que el Mise en place mide, la cocción también la usa ─────────────────────────
# Replay de la cola sobre 322 planes: 92 avenas con «mide 1 taza de avena (50 g), 5 ml de leche descremada y 455 ml de
# agua» y luego «cocina la avena con la leche descremada y la canela… 7-9 min»: el agua (la que el lote 311 completa para
# que la avena se cocine, o la que el modelo puso en la lista) se mide y ningún paso la echa a la olla — la avena se
# cocinaría con 5 ml de leche. `_anadir_agua_al_paso` la escribe en el paso que MIDE la leche y ahí termina. Aquí el paso
# que cocina la avena la nombra junto a la leche («con la leche descremada, el agua y la canela») o, sin leche nombrada,
# tras la avena («cocina la avena con el agua a fuego medio»). tooltip-anchor: P1-PLAN-LOTE-448
_AGUA_LINEA_448_RE = re.compile(r"^\s*[\d.,½¼¾⅓⅔\s]+(?:ml|l|litros?|tazas?|cdas?|cucharadas?)\.?\s+de\s+agua\b")
_TRAS_AVENA_448_RE = re.compile(r"\b(?:la|las)\s+avena\b(?:\s+(?:cocida|en\s+hojuelas|integral|instantanea))?")
_ADJ_LECHE_448 = r"(?:\s+(?!(?:y|e|o|a|al|en|con|durante|hasta|de|del|la|el|los|las|por|sin|sobre|removiendo)\b)[a-z]+)*"


def agua_en_la_coccion(meal) -> int:
    """1 si el paso que cocina la avena pasó a nombrar el agua de la lista; 0 si no, o ante cualquier error."""
    try:
        if not activo() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list):
            return 0
        lista = [_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        if not any(_AGUA_LINEA_448_RE.match(x) for x in lista) or not any(_AVENA_RE.search(x) for x in lista):
            return 0
        pasos = [p for p in rec if isinstance(p, str) and not any(e in p for e in _NOTA)]
        texto = _sa(" ".join(pasos) + " " + str(meal.get("name") or ""))
        if _FRIO_RE.search(texto) or _NOMBRE_BATIDO_RE.search(_sa(meal.get("name"))):
            return 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or any(e in p for e in _NOTA) or _pilar_448(p) == "mise en place":
                continue
            base = _sa(p)
            if len(base) != len(p) or not _COCCION_RE.search(base) or not _AVENA_RE.search(base):
                continue
            if "agua" in base:
                return 0                                   # ya la nombra (o una frase del 311/428 la dice)
            ml = _LECHE_NOMBRADA_428_RE.search(base)
            if ml:
                # la leche con TODOS sus adjetivos («la leche pasteurizada»), no sólo los que conoce el 428
                fin = ml.end() + re.match(_ADJ_LECHE_448, base[ml.end():]).end()
                despues = base[fin:]
                rec[i] = p[:fin] + (", el agua" if re.match(r"\s*,|\s+y\s", despues) else " y el agua") + p[fin:]
            else:
                ta = _TRAS_AVENA_448_RE.search(base)
                if not ta:
                    return 0
                if base[ta.end():].startswith(" con "):      # «la avena con la canela» → «con el agua y la canela»
                    rec[i] = p[:ta.end() + 5] + "el agua y " + p[ta.end() + 5:]
                else:
                    rec[i] = p[:ta.end()] + " con el agua" + p[ta.end():]
            meal.pop("_display", None)
            return 1
        return 0
    except Exception:
        return 0


def _pilar_448(p: str) -> str:
    t = _sa(p).lstrip()
    return "mise en place" if t.startswith("mise en place") else ""
