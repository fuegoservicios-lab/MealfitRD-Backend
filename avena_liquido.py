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
_NO_HOJUELA_RE = re.compile(r"\b(harina|leche|bebida|salvado|galletas?|barras?|granola|batido)\b")
_LIQUIDO_RE = re.compile(r"\b(leche|bebida|agua)\b")
_NO_LIQUIDO_RE = re.compile(r"\b(en polvo|condensada|evaporada)\b")
_AGUA_RE = re.compile(r"^\s*agua\b")
#: un paso que COCINA la avena como gacha: «cocina la avena con la leche…», «la avena … en el microondas / a fuego medio
#: hasta que espese». Sobre texto sin acentos y en minúscula.
_COCCION_RE = re.compile(
    r"\b(?:cocina|cuece|hierve|calienta|cocinar|cocer|hervir|calentar)\s+(?:la\s+|las\s+)?(?:[a-z]+\s+)?avena\b"
    r"|\bavena\b[^.;]{0,60}\b(?:microondas|a fuego|en una olla|en la olla|espese|espesa|hierva|hervor)\b")
#: lo que NO es una gacha: remojada en frío, masa, horneado o licuado
_FRIO_RE = re.compile(r"\b(remoja|remojar|remojada|toda la noche|overnight|en la nevera|refrigera|refrigerar|reposar|"
                      r"panqueques?|pancakes?|tortillas? de avena|galletas?|muffins?|waffles?|crepas?|arepas? de avena|"
                      r"masa|empaniz\w*|rebozad\w*|batido|smoothie|licua\w*|hornea\w*|horno|"
                      r"(?:tuesta|tostar|dora|dorar)\s+(?:la\s+)?avena|avena\s+tostada)\b")
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
        if not _AVENA_RE.search(texto) or _FRIO_RE.search(texto):
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
                if _AGUA_RE.match(food) and u == "ml" and agua_idx is None:
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
            nuevo_total = (_num(mv.group("q")) or 0.0) + faltan
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
    for i, p in enumerate(rec):
        if isinstance(p, str) and not any(e in p for e in _NOTA) and _COCCION_RE.search(_sa(p)):
            s = p.rstrip()
            rec[i] = s + ("" if s.endswith(".") else ".") + f" Completa el líquido con {agua_txt} para que la avena se cocine."
            return
