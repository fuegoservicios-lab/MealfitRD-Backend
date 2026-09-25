# -*- coding: utf-8 -*-
"""Las cantidades de los PASOS de una receta contra la lista del plato (lotes 302-309, 2026-09-25).

El sincronizador de pasos (`graph_orchestrator._sync_recipe_step_quantities`, P1-RECIPE-QTY-SYNC) y el contrato final de
receta (`recipe_contract`, P1-PLAN-LOTE-23) ya ponían la lista por encima de los pasos. Las baterías del 25-sep (318
comidas, 27 planes) enseñaron por dónde se les escapaba: 15 planes con al menos un paso que contradice a su propia lista
(«mide 45 g de avena y 200 ml de leche» con 60 ml en la lista; «desmenuza los 12 g de queso» con 25 g; «bate 4 3
huevos»). Aquí vive la parte nueva, fuera del god-file:

  · 304 — `misma_porcion`: la misma porción nombrada dos veces no es un reparto;
  · 308 — `sincronizar_exacto`: el sincronizador exacto corre también como PRIMER paso del contrato final (que tolera
    ±25 % en gramos, la tolerancia del medidor V4, y dejaba «50 g de mango» con 60 en la lista);
  · 309 — `otra_base`: lo COCIDO de un paso no se iguala a la base cruda/seca de la lista («25 g de quinoa cocida» no
    son «30 g de quinoa cocida» porque la lista compre 30 g en seco).

Todo es fail-closed: ante la duda, la conducta de antes.
"""
from __future__ import annotations

import re
import unicodedata

_PALABRA_RE = re.compile(r"[^\wáéíóúñü]+")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn")


def _toks(txt) -> list:
    """Tokens de ≥4 letras, sin acentos y en minúscula — el mismo criterio que el sincronizador."""
    return [t for t in _PALABRA_RE.split(_sa(str(txt or "").lower())) if len(t) >= 4]


# ── [P1-PLAN-LOTE-304 · 2026-09-25] La misma porción nombrada dos veces no es un reparto ─────────────────────────────
# «Mise en place: mide 12 g de queso blanco fresco…» y «Montaje: desmenuza los 12 g de queso blanco fresco», con 25 g en
# la lista (el piso de porción lo subió después): el sincronizador veía DOS menciones del queso y, para no duplicar un
# ingrediente repartido («12 g en la masa y 12 g encima»), no tocaba ninguna. Pero la segunda es la MISMA porción: la
# nombra con artículo («los 12 g») o la mise en place la midió y un paso posterior la usa. Mismo caso con el casabe de la
# bariátrica («ten listos 15 g de casabe» … «sirve con los 15 g de casabe», lista 20). Con cualquier marca de reparto
# (mitad, resto, restante, más, extra, adicional, cada, otra parte) sigue sin tocarse. tooltip-anchor: P1-PLAN-LOTE-304
_ART_ANTES_RE = re.compile(r"\b(?:el|la|los|las)\s+$", re.IGNORECASE)
_REPARTO_ANTES_RE = re.compile(r"\b(?:mitad|resto|restantes?|otras?|otros?|parte|cada)\b[^.;:]{0,40}$", re.IGNORECASE)
_REPARTO_DESPUES_RE = re.compile(r"^[^.;:,]{0,40}?\b(?:restantes?|m[aá]s|adicional(?:es)?|extra)\b", re.IGNORECASE)
_MISE_RE = re.compile(r"^\s*mise\s+en\s+place\b", re.IGNORECASE)


def misma_porcion(pasos, token, menciones, total, rx) -> bool:
    """¿Las menciones cuantificadas de `token` en `pasos` son la MISMA porción nombrada otra vez (y no un reparto)?

    `menciones`: [(cantidad_float, unidad_normalizada)] en orden de aparición (las que contó el sincronizador);
    `total`: (cantidad, unidad, display) de la línea de la lista, o None; `rx`: la regex de menciones del sincronizador
    (con grupos `qty`, `unit`, `food`). True sólo si todas dicen la misma cantidad en la unidad de la lista, ninguna
    lleva marca de reparto, y (a) alguna posterior es una referencia con artículo o (b) la primera está en la «Mise en
    place» y las demás en pasos posteriores. Con True el sincronizador lleva cada mención a la lista (si ya coincide, no
    cambia nada: tampoco pasa por el «restante» del reparto)."""
    try:
        if not menciones or len(menciones) < 2 or not total:
            return False
        vals = [v for v, _u in menciones]
        unis = {u for _v, u in menciones}
        if any(v is None for v in vals) or len(unis) != 1:
            return False
        v0 = float(vals[0])
        if any(abs(float(v) - v0) > 1e-6 for v in vals):
            return False
        if next(iter(unis)) != total[1]:
            return False
        # igual a la lista también cuenta: así la misma porción no pasa por el «restante» del reparto (que escribía «sirve
        # con los la lechosa fresca aparte restante» sobre «los 105 g de lechosa»), y el paso ya coincide con la lista
        vistos = []   # (índice del paso, ¿mise en place?, ¿artículo delante?, ¿marca de reparto?)
        for i, paso in enumerate(pasos or []):
            if not isinstance(paso, str):
                continue
            for mm in rx.finditer(paso):
                if token not in _toks(mm.group("food"))[:2]:
                    continue
                antes, despues = paso[:mm.start()], paso[mm.end("unit"):]   # «los 12 g de queso restantes»: el
                # nombre del alimento ya se come hasta tres palabras, así que la marca se busca desde la unidad
                reparto = bool(_REPARTO_ANTES_RE.search(antes[-60:]) or _REPARTO_DESPUES_RE.search(despues))
                vistos.append((i, bool(_MISE_RE.match(paso)), bool(_ART_ANTES_RE.search(antes)), reparto))
        if len(vistos) < 2 or any(r for _i, _m, _a, r in vistos):
            return False
        if any(art for _i, _m, art, _r in vistos[1:]):
            return True
        return bool(vistos[0][1]) and all(i > vistos[0][0] for i, _m, _a, _r in vistos[1:])
    except Exception:
        return False


# ── [P1-PLAN-LOTE-309 · 2026-09-25] Lo COCIDO de un paso no es la base de la lista ──────────────────────────────────────
# La lista compra en la base del catálogo (granos y legumbres en SECO, carnes en CRUDO: lotes 282-284 y 301); el paso
# puede nombrar la forma cocida («mide 25 g de quinoa cocida», «desmenuza 150 g de pechuga cocida»). El sincronizador
# igualaba el número sin mirar la forma: «30 g de quinoa» en la lista volvía el paso «30 g de quinoa cocida» (que son
# ~10 g secos) y «200 g de pechuga» lo volvía «200 g de pechuga cocida» (~270 g crudos). Si la mención del paso y la línea
# de la lista no están en la MISMA forma, el paso no se iguala. tooltip-anchor: P1-PLAN-LOTE-309
_COCCION_RE = re.compile(r"\b(?:cocid|hervid|sancochad|guisad|asad|hornead|desmenuzad|frit)[oa]s?\b")
_CORTE_MENCION_RE = re.compile(r"[,;:.()]|\by\b|\bcon\b|\bhasta\b|\bpara\b")
_LIDER_RE = re.compile(r"^\s*[\d.,½¼¾⅓⅔]+(?:\s*[½¼¾⅓⅔])?\s*[A-Za-zÁÉÍÓÚÑÜáéíóúñü.]*\s+(?:de\s+|del\s+)?")


def _cocido(txt) -> bool:
    return bool(_COCCION_RE.search(_sa(str(txt or "").lower())))


def _forma_huevo(txt) -> str:
    """«clara»/«yema» si el texto nombra esa forma del huevo; «» si no. Batería real del 25-sep: «200 g de clara de huevo»
    del paso se emparejaba por su SEGUNDA palabra con la línea «60 g de huevo» (entero) y pasaba a «60 g de clara de
    huevo» — la forma del huevo es otra base, como lo cocido."""
    t = _sa(str(txt or "").lower())
    return "clara" if re.search(r"\bclaras?\b", t) else ("yema" if re.search(r"\byemas?\b", t) else "")


def otra_base(paso, ini, fin, ings, token) -> bool:
    """True si la mención `paso[ini:fin]` (con su cola hasta la próxima coma, «y», «con»…) y la línea de la lista cuyo
    alimento empieza por `token` no están en la misma forma (una cocida y la otra no). Sin línea: False."""
    try:
        cola = str(paso)[fin:fin + 40]
        corte = _CORTE_MENCION_RE.search(cola)
        mencion = str(paso)[ini:fin] + (cola[:corte.start()] if corte else cola)
        linea = None
        for ln in ings or []:
            cuerpo = _LIDER_RE.sub("", str(ln), count=1)
            t = _toks(re.sub(r"\(.*?\)", " ", cuerpo))
            if t and t[0] == token:
                linea = str(ln)
                break
        if linea is None:
            return False
        limpia = re.sub(r"\(.*?\)", " ", linea)
        return _cocido(mencion) != _cocido(limpia) or _forma_huevo(mencion) != _forma_huevo(limpia)
    except Exception:
        return False


# ── [P1-PLAN-LOTE-310 · 2026-09-25] El peso entre paréntesis de un paso sigue al de la MISMA línea de la lista ────────
# El sincronizador de pesos (`_sync_recipe_step_gram_hints`, P1-STEP-GRAM-HINT-STALE) sólo ve menciones con unidad («¾
# taza de avena (80 g)»); las piezas no: «lava y corta ¾ manzana (≈120 g)» con «¾ manzana (≈65 g)» en la lista, «pela ½
# plátano maduro grande (200 g)» con 101 g, «corta ½ pechuga de pollo (porción) (120 g)» con «(≈139 g)». Cuando el paso
# repite EXACTAMENTE el texto de una línea de la lista (misma cantidad, mismo alimento; plural y acentos aparte) y le
# pone otro peso, o el «(porción)» que la lista ya respondió, el paréntesis del paso pasa a ser el de la lista.
# tooltip-anchor: P1-PLAN-LOTE-310
_LINEA_CON_PESO_RE = re.compile(
    r"^\s*(?P<lead>(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s+[^()]+?)\s*"
    r"\(\s*(?P<aprox>[≈~])?\s*(?P<g>\d+(?:[.,]\d+)?)\s*(?P<u>g|ml)\s*\)\s*$", re.IGNORECASE)
_PESO_PASO = r"(?P<cola>\s*(?:\(\s*porcion\s*\))?(?:\s*\(\s*[≈~]?\s*(?P<g>\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?|ml)\s*\))?)"


def _patron_lead(lead: str):
    palabras = _sa(lead.lower()).split()
    if len(palabras) < 2:
        return None
    partes = [re.escape(w[:-1]) + "s?" if (len(w) > 3 and w.endswith("s")) else
              (re.escape(w) + "s?" if len(w) > 3 else re.escape(w)) for w in palabras]
    return re.compile(r"(?<![\w½¼¾⅓⅔.,])" + r"\s+".join(partes) + r"\b" + _PESO_PASO)


def pesos_de_la_lista(meal) -> int:
    """Reescribe en los pasos el «(N g)»/«(porción)» que sigue al texto de una línea de la lista con peso. Nº de
    reescrituras; 0 ante cualquier error. Las notas (⚠/💡/🌱/⚕) no se tocan."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = []
        for ln in meal.get("ingredients") or []:
            m = _LINEA_CON_PESO_RE.match(str(ln))
            if not m:
                continue
            rx = _patron_lead(m.group("lead"))
            if rx is None:
                continue
            g = float(m.group("g").replace(",", "."))
            txt = f"({'≈' if m.group('aprox') else ''}{m.group('g')} {m.group('u').lower()})"
            lineas.append((rx, g, txt))
        if not lineas:
            return 0
        n = 0
        nuevos = []
        for paso in rec:
            if not isinstance(paso, str) or any(e in paso for e in ("⚠", "💡", "🌱", "⚕")):
                nuevos.append(paso)
                continue
            base = _sa(paso.lower())
            if len(base) != len(paso):
                nuevos.append(paso)
                continue
            cambios = []
            for rx, g, txt in lineas:
                for mm in rx.finditer(base):
                    cola = mm.group("cola") or ""
                    if not cola.strip():
                        continue
                    viejo = mm.group("g")
                    porcion = "porcion" in cola
                    if viejo is not None and not porcion and abs(float(viejo.replace(",", ".")) - g) <= max(3.0, 0.1 * g):
                        continue
                    cambios.append((mm.start("cola"), mm.end("cola"), " " + txt))
            if not cambios:
                nuevos.append(paso)
                continue
            s = paso
            for ini, fin, txt in sorted(set(cambios), reverse=True):
                s = s[:ini] + txt + s[fin:]
            nuevos.append(s)
            n += 1
        if n:
            meal["recipe"] = nuevos
            meal.pop("_display", None)      # DELETE-on-write: `_display[locale].recipe` espeja los pasos por índice
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-312 · 2026-09-25] Los decimales de máquina se escriben como en la cocina ──────────────────────────
# «pica 1 tomate y 2.22 cdas de cebolla», «mide 1.25 tazas de lentejas», «15.12g de vainitas», «Sal mínima (0.25 cdta)»:
# 7 de 25 planes de las baterías del 25-sep los enseñaban aun después del sincronizador exacto (el motor escala las
# líneas a decimales y el paso los copia). Cucharadas y tazas van a la fracción de cocina más cercana (¼ ⅓ ½ ⅔ ¾) si está
# a ≤ 0,06; gramos y mililitros, a entero. En los pasos Y en la lista que se ve (`ingredients`), con la misma regla, para
# que sigan diciendo lo mismo; `ingredients_raw` (lo que mide el motor) no se toca. Nunca: trazas (< 1 g/ml, eso es otro
# problema), tres decimales («1.000 ml» es mil), ni lo que no cae cerca de una fracción («0.4 taza»).
# tooltip-anchor: P1-PLAN-LOTE-312
_DECIMAL_COCINA_RE = re.compile(
    r"(?<![\d.,/])(\d+)[.,](\d{1,2})(?![\d])(\s*)(tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|g|gr|ml)\b", re.IGNORECASE)
_FRACCIONES_COCINA = ((0.0, ""), (0.25, "¼"), (1 / 3, "⅓"), (0.5, "½"), (2 / 3, "⅔"), (0.75, "¾"), (1.0, ""))


def _a_cocina(mm) -> str:
    entero = int(mm.group(1))
    frac = float("0." + mm.group(2))
    esp, unidad = mm.group(3), mm.group(4)
    if unidad.lower() in ("g", "gr", "ml"):
        v = entero + frac
        if v < 1.0:
            return mm.group(0)                                   # traza: no se maquilla
        return f"{int(round(v))}{esp}{unidad}"
    val, simb = min(_FRACCIONES_COCINA, key=lambda t: abs(frac - t[0]))
    if abs(frac - val) > 0.06:
        return mm.group(0)
    e = entero + (1 if val == 1.0 else 0)
    if e == 0 and not simb:
        return mm.group(0)
    return f"{e if e else ''}{simb}{esp or ' '}{unidad}"


#: y el número concuerda con la unidad («escurre 2 taza de habichuelas», «pica 3 taza de kale», «1 tazas de cebolla»):
#: más de 1 ⇒ plural; 1 o menos ⇒ singular. Sólo tazas y cucharadas (las unidades que el motor escribe con número).
_CONCORDANCIA_RE = re.compile(
    r"(?<![\d.,/])(\d+(?:[.,]\d+)?[½¼¾⅓⅔]?|[½¼¾⅓⅔])(\s+)(taza|cda|cdta|cucharada|cucharadita)(s?)\b", re.IGNORECASE)
_FRAC_VALOR = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}


def _concuerda(mm) -> str:
    q = mm.group(1)
    try:
        if q in _FRAC_VALOR:
            v = _FRAC_VALOR[q]
        elif q[-1] in _FRAC_VALOR:
            v = float(q[:-1].replace(",", ".")) + _FRAC_VALOR[q[-1]]
        else:
            v = float(q.replace(",", "."))
    except ValueError:
        return mm.group(0)
    plural = "s" if v > 1 else ""
    return f"{q}{mm.group(2)}{mm.group(3)}{plural}"


def decimales_de_cocina(meal) -> int:
    """Nº de textos reescritos (pasos + líneas visibles). 0 ante cualquier error. Las notas (⚠/💡/🌱/⚕) no se tocan."""
    try:
        if not isinstance(meal, dict):
            return 0
        n = 0
        for campo, es_paso in (("recipe", True), ("ingredients", False)):
            lista = meal.get(campo)
            if not isinstance(lista, list):
                continue
            for i, s in enumerate(lista):
                if not isinstance(s, str) or (es_paso and any(e in s for e in ("⚠", "💡", "🌱", "⚕"))):
                    continue
                nuevo = _CONCORDANCIA_RE.sub(_concuerda, _DECIMAL_COCINA_RE.sub(_a_cocina, s))
                if nuevo != s:
                    lista[i] = nuevo
                    n += 1
        if n:
            meal.pop("_display", None)      # DELETE-on-write: `_display[locale]` espeja la lista y los pasos
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-315 · 2026-09-25] Lo que un paso ya mete en la preparación no se sirve «al lado» ─────────────────────
# `_mention_cooked_complement_in_montaje` (P2-CLOSER-MENTION-IN-MONTAJE) añade «Acompaña con X» al Montaje cuando otro
# paso preparó X — pero no distinguía «lo preparó aparte» de «lo metió en la licuadora»: «💪 Agrega queso cottage a la
# licuadora y licúa hasta integrar» + «… Acompaña con queso cottage» (dos destinos para los mismos 40 g). Baterías del
# 25-sep: 6 de 354 comidas, casi todas batidos. tooltip-anchor: P1-PLAN-LOTE-315
_RECIPIENTE = r"(?:licuadora|mezcla|masa|batid[oa]|preparacion|olla|sarten|bowl|vaso|crema|salsa|guiso)"
_INCORPORA = r"(?:agrega|anade|incorpora|mezcla|licua|echa|integra|pon)\w*"


def ya_va_dentro(texto_otros_pasos: str, alimento: str) -> bool:
    """True si algún paso (texto ya en minúscula y sin acentos, o no: se normaliza) incorpora `alimento` a la
    preparación: «agrega el queso cottage a la licuadora», «licúa la guanábana, la leche y el queso cottage»."""
    try:
        t = _sa(str(texto_otros_pasos or "").lower())
        a = re.escape(_sa(str(alimento or "").lower()).strip())
        if not a:
            return False
        if re.search(_INCORPORA + r"\s+(?:(?:el|la|los|las|un|una)\s+|[\d½¼¾⅓⅔]\S*\s+(?:[a-z]+\s+)?de\s+){0,2}" + a
                     + r"\b[^.;]{0,40}?\b(?:a|en|con|dentro de)\s+"
                     r"(?:(?:la|el|los|las|tu|su)\s+)?" + _RECIPIENTE, t):
            return True
        return bool(re.search(r"\b(?:licua|bate|mezcla|integra)\w*\b[^.;]{0,80}\b" + a + r"\b", t))
    except Exception:
        return False


# ── [P1-PLAN-LOTE-316 · 2026-09-25] La misma frase dos veces seguidas en un paso se deja una vez ─────────────────────
# Batería real del 25-sep (nocturno): «Montaje: … sírvelas con la yautía cocida. Acompaña con filete de pescado blanco.
# Acompaña con filete de pescado blanco.» El único escritor de «Acompaña con» (P2-CLOSER-MENTION-IN-MONTAJE) ya se guarda
# de repetir; la segunda copia nace cuando otra pasada reescribe el texto después (una sustitución de proteína vuelve
# idénticas dos frases que no lo eran). Se colapsa al final, frase a frase, sin tocar notas. tooltip-anchor: P1-PLAN-LOTE-316
_FRASE_RE = re.compile(r"[^.!?]+[.!?]+\s*")


def frases_repetidas(meal) -> int:
    """Nº de pasos donde se quitó una frase repetida seguida (comparada sin acentos ni mayúsculas). 0 ante error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        n = 0
        for i, paso in enumerate(rec):
            if not isinstance(paso, str) or any(e in paso for e in ("⚠", "💡", "🌱", "⚕")):
                continue
            frases = _FRASE_RE.findall(paso)
            if len(frases) < 2 or "".join(frases) != paso:
                continue
            quedan = [frases[0]]
            for f in frases[1:]:
                if _sa(f.strip().lower()) == _sa(quedan[-1].strip().lower()):
                    continue
                quedan.append(f)
            if len(quedan) != len(frases):
                rec[i] = "".join(quedan).rstrip() if not paso.endswith(" ") else "".join(quedan)
                n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-319 · 2026-09-25] Una traza no es un ingrediente ─────────────────────────────────────────────────────
# El motor de macros, al cuadrar el día, deja a veces un alimento en MENOS de 1 g/ml: «0.95 ml de leche descremada», «0.07
# ml de leche» («Bate 1 clara de huevo con 0.07 ml de leche»), 1 de cada ~15 planes de las baterías del 25-sep. No aporta
# nada medible y la lista de compras puede comprar un envase entero por él. Sale de la lista (y de `ingredients_raw`, por
# texto), ANTES del contrato final, cuyo paso V5 (`recipe_repair.retirar_sin_lista`, lote 30) retira su mención de los
# pasos. Nunca especias, sal, aceite, ácidos ni esencias (una pizca ES su dosis), ni si el plato quedaría casi vacío.
# tooltip-anchor: P1-PLAN-LOTE-319
_TRAZA_RE = re.compile(r"^\s*(?:0(?:[.,]\d+)?)\s*(?:g|gr|gramos?|ml)\s+(?:de\s+)?(?P<food>.+)$", re.IGNORECASE)
_TRAZA_DOSIS_RE = re.compile(
    r"\b(sal|pimienta|canela|oregano|comino|ajo|cebolla en polvo|pimenton|paprika|curcuma|jengibre|nuez moscada|vainilla|"
    r"esencia|extracto|stevia|edulcorante|polvo de hornear|levadura|bicarbonato|colorante|aceite|vinagre|limon|lima|jugo|"
    r"salsa|mostaza|especias?|hierbas?|perejil|cilantro|laurel|tomillo|romero|clavo|anis|cafe|te|cacao|sazon)\b")


def quitar_trazas(meal) -> int:
    """Nº de líneas-traza quitadas de `ingredients` (y sus iguales de `ingredients_raw`). 0 ante cualquier error."""
    try:
        ings = meal.get("ingredients") if isinstance(meal, dict) else None
        if not isinstance(ings, list) or len(ings) < 3:
            return 0
        fuera = []
        for s in ings:
            m = _TRAZA_RE.match(str(s)) if isinstance(s, str) else None
            if m and not _TRAZA_DOSIS_RE.search(_sa(m.group("food").lower())):
                fuera.append(s)
        if not fuera or len(ings) - len(fuera) < 2:
            return 0
        comidas = {_sa(_TRAZA_RE.match(s).group("food").lower()).strip() for s in fuera}
        meal["ingredients"] = [s for s in ings if s not in fuera]
        raw = meal.get("ingredients_raw")
        if isinstance(raw, list):
            def _es_traza_quitada(r):
                mr = _TRAZA_RE.match(str(r)) if isinstance(r, str) else None
                return bool(mr) and _sa(mr.group("food").lower()).strip() in comidas
            meal["ingredients_raw"] = [r for r in raw if not _es_traza_quitada(r)]
        meal["_trazas_quitadas"] = list(fuera)
        meal.pop("_display", None)
        return len(fuera)
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-308 · 2026-09-25] El sincronizador exacto, también al final ────────────────────────────────────────
# El contrato final de receta es la última palabra de los pasos, pero mide los gramos con la tolerancia del medidor V4
# (±25 %): un cambio TARDÍO de la lista (piso de porción, techo, sustitución) de 50 → 60 g de mango, o de 25 → 30 g de
# aguacate, quedaba en el paso. El sincronizador exacto (`_sync_recipe_step_quantities`, con sus guardas de reparto,
# «cada» y «(en total)») corre ahora como primer paso del contrato en modo `repair`. tooltip-anchor: P1-PLAN-LOTE-308
def sincronizar_exacto(meal) -> int:
    """Nº de menciones re-alineadas; 0 si no hay nada o ante cualquier error (import perezoso: graph_orchestrator
    importa `recipe_contract`, que llama aquí)."""
    try:
        from graph_orchestrator import _sync_recipe_step_quantities as _qs
        return int(_qs(meal) or 0)
    except Exception:
        return 0
