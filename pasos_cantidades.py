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


#: [P1-PLAN-LOTE-353 · 2026-09-26] Cucharadas y cucharaditas van SIEMPRE a la cuadrícula de la cuchara (¼ ½ ¾, la misma
#: del cuantizador `nutrition_db._SPOON_FRACS`; nunca 0: lo mínimo es ¼): «calienta 1.82 cdtas de aceite» (plan
#: bariátrico real, con «1¾ cdtas» en la lista), «pesa 1.56 cdas de ricotta», «0.14 cdta de orégano». La tolerancia de
#: 0,06 del lote 312 es para la taza, donde un décimo sí son 24 ml; en una cucharadita son medio mililitro.
#: tooltip-anchor: P1-PLAN-LOTE-353
_CUCHARA_353_RE = re.compile(r"^(?:cdas?|cdtas?|cucharadas?|cucharaditas?)$", re.IGNORECASE)
_FRACCIONES_CUCHARA = ((0.0, ""), (0.25, "¼"), (0.5, "½"), (0.75, "¾"), (1.0, ""))


def _a_cocina(mm) -> str:
    entero = int(mm.group(1))
    frac = float("0." + mm.group(2))
    esp, unidad = mm.group(3), mm.group(4)
    if unidad.lower() in ("g", "gr", "ml"):
        v = entero + frac
        if v < 1.0:
            return mm.group(0)                                   # traza: no se maquilla
        return f"{int(round(v))}{esp}{unidad}"
    if _CUCHARA_353_RE.match(unidad):                                      # [P1-PLAN-LOTE-353] «1.82 cdtas»
        val, simb = min(_FRACCIONES_CUCHARA, key=lambda t: abs(frac - t[0]))
        if entero == 0 and val == 0.0:
            val, simb = 0.25, "¼"
    else:
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


#: [P1-PLAN-LOTE-326 · 2026-09-25] y el adjetivo concuerda con UNO o una fracción sola: «pela y corta ½ plátano verdes»,
#: «lava ½ hoja grandes de lechuga», «bate 1 huevo enteros» — el reescritor del conteo cambia «2 plátanos» por «½ plátano»
#: y deja el adjetivo en plural (28 pasos en las baterías guardadas; la lista ya decía «½ plátano verde»). Sólo tras un
#: conteo solo (no «1½», que es plural) y un sustantivo en singular; lista cerrada de adjetivos. tooltip-anchor: P1-PLAN-LOTE-326
_ADJ_TRAS_UNO_RE = re.compile(
    r"(?<![\w.,/])((?:1|½|¼|¾|⅓|⅔|un|una|medio|media)\s+[a-záéíóúñü]*[a-rt-záéíóúñü]\s+)"
    r"(verde|grande|madur[oa]|pequeñ[oa]|median[oa]|roj[oa]|blanc[oa]|fresc[oa]|enter[oa]|amarill[oa]|morad[oa])s\b",
    re.IGNORECASE)


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
                nuevo = _ADJ_TRAS_UNO_RE.sub(r"\1\2", nuevo)  # [P1-PLAN-LOTE-326] «½ plátano verdes»
                if nuevo != s:
                    lista[i] = nuevo
                    n += 1
        n += conteos_de_cocina(meal)                                # [P1-PLAN-LOTE-354] «0.27 pepino»
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


#: [P1-PLAN-LOTE-352 · 2026-09-26] «maní tostado SIN SAL» no es una dosis de sal: la palabra «sal» de la negación
#: hacía que la traza («0.01 g de maní tostado sin sal», plan DM2 real) se quedara en la lista y en el paso («mide 0.01 g
#: de maní»). La misma lección del lote 203 en el pulido de líneas. Pero si el NOMBRE del plato nombra ese alimento («Casabe
#: crujiente con queso blanco fresco, maní y huevo») la traza se queda: quitarla deja un plato que promete lo que la lista
#: no trae; el pulido la escribe «1 pizca» y el paso la sigue (lote 358). tooltip-anchor: P1-PLAN-LOTE-352
_SIN_SAL_RE = re.compile(r"\b(?:sin|bajos?\s+en|bajas?\s+en|reducid[oa]s?\s+en)\s+sal\b")


def quitar_trazas(meal) -> int:
    """Nº de líneas-traza quitadas de `ingredients` (y sus iguales de `ingredients_raw`). 0 ante cualquier error."""
    try:
        ings = meal.get("ingredients") if isinstance(meal, dict) else None
        if not isinstance(ings, list) or len(ings) < 3:
            return 0
        fuera = []
        nombre = _sa(str(meal.get("name") or "").lower())
        for s in ings:
            m = _TRAZA_RE.match(str(s)) if isinstance(s, str) else None
            if m and not _TRAZA_DOSIS_RE.search(_SIN_SAL_RE.sub(" ", _sa(m.group("food").lower()))):  # [P1-PLAN-LOTE-352]
                t = _toks(m.group("food"))
                if t and re.search(r"\b" + re.escape(t[0][:5]), nombre):
                    continue                    # [P1-PLAN-LOTE-352] el nombre del plato lo promete: la traza se queda (pizca)
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


# ── [P1-PLAN-LOTE-328..331 · 2026-09-25] El paso dice lo que dice la lista, también en piezas, porciones y pizcas ──────
# Los pases de arriba ven las menciones «N <unidad> de X» contra líneas «N <unidad> de X». Las baterías del 25-sep
# (330 comidas re-medidas con el contrato vigente) enseñaron las formas que ninguno leía. `lo_que_dice_la_lista` las
# corre en orden dentro del contrato final (modo `repair`), después de `pesos_de_la_lista`. Todo fail-closed: ante la
# duda, el texto no se toca.
_NOTAS_PASO = ("⚠", "💡", "🌱", "⚕", "🤰")


def _es_nota(p) -> bool:
    return not isinstance(p, str) or any(e in p for e in _NOTAS_PASO)


# ── 328 · los gramos de un paso siguen a la PIEZA de la lista ────────────────────────────────────────────────────────
# «Mise en place: corta 205 g de pechuga en tiras» con «1 pechuga de pollo (≈158 g)» en la lista; «corta 172 g de pechuga
# de pollo en cubos» con «¾ pechuga de pollo (≈138 g)»: 11 de 330 comidas, casi todas de pollo. El sincronizador sólo lee
# líneas «N g de X» y el lote 310 sólo el paréntesis que sigue al MISMO texto de la línea. El paso pasa a pesar lo que
# dice la pieza. Nunca: lo cocido contra lo crudo (lote 309), un reparto (mitad, resto, cada…), dos pesos distintos para
# la misma pieza, ni un alimento que la lista también trae en gramos (ambiguo). tooltip-anchor: P1-PLAN-LOTE-328
_PIEZA_CON_PESO_RE = re.compile(
    r"^\s*(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s+(?P<cuerpo>[^()\d]+?)\s*"
    r"\(\s*[≈~]\s*(?P<g>\d+(?:[.,]\d+)?)\s*g\s*\)\s*$", re.IGNORECASE)
_LINEA_EN_GRAMOS_RE = re.compile(r"^\s*\d+(?:[.,]\d+)?\s*(?:g|gr|gramos|ml)\s+(?:de\s+)?(?P<cuerpo>.+)$", re.IGNORECASE)
_GRAMOS_EN_PASO_RE = re.compile(
    r"(?<![\w.,/½¼¾⅓⅔])(?P<n>\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+de\s+"
    r"(?P<food>[a-záéíóúñü]+(?:\s+[a-záéíóúñü]+){0,3})", re.IGNORECASE)


def gramos_de_la_pieza(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        piezas, en_gramos = [], []
        for ln in meal.get("ingredients") or []:
            s = str(ln)
            m = _PIEZA_CON_PESO_RE.match(s)
            if m:
                t = _toks(m.group("cuerpo"))
                if t:
                    piezas.append((t, float(m.group("g").replace(",", ".")), m.group("cuerpo")))
                continue
            md = _LINEA_EN_GRAMOS_RE.match(s)
            if md:
                t = _toks(re.sub(r"\(.*?\)", " ", md.group("cuerpo")))
                if t:
                    en_gramos.append(t)
        if not piezas:
            return 0

        def _cual(ft):
            cp = [k for k, p in enumerate(piezas) if p[0][0] == ft[0]]
            cg = [t for t in en_gramos if t[0] == ft[0]]
            if len(cp) + len(cg) > 1:
                if len(ft) < 2:
                    return None
                cp = [k for k in cp if len(piezas[k][0]) > 1 and piezas[k][0][1] == ft[1]]
                cg = [t for t in cg if len(t) > 1 and t[1] == ft[1]]
            return cp[0] if len(cp) == 1 and not cg else None

        hallazgos, vetadas = {}, set()
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            for mm in _GRAMOS_EN_PASO_RE.finditer(p):
                ft = _toks(mm.group("food"))
                k = _cual(ft) if ft else None
                if k is None:
                    continue
                cola = p[mm.end():mm.end() + 40]
                corte = _CORTE_MENCION_RE.search(cola)
                mencion = mm.group(0) + (cola[:corte.start()] if corte else cola)
                cuerpo = piezas[k][2]
                if (_REPARTO_ANTES_RE.search(p[:mm.start()]) or _REPARTO_DESPUES_RE.search(p[mm.end():])
                        or _cocido(mencion) != _cocido(cuerpo) or _forma_huevo(mencion) != _forma_huevo(cuerpo)):
                    vetadas.add(k)
                    continue
                hallazgos.setdefault(k, []).append((i, mm.start("n"), mm.end("n"),
                                                    float(mm.group("n").replace(",", "."))))
        cambios = {}
        for k, lst in hallazgos.items():
            if k in vetadas or len({x[3] for x in lst}) != 1:
                continue
            g = piezas[k][1]
            if abs(lst[0][3] - g) < 0.5:
                continue
            for i, ini, fin, _v in lst:
                cambios.setdefault(i, []).append((ini, fin, str(int(round(g)))))
        if not cambios:
            return 0
        nuevos = list(rec)
        for i, cs in cambios.items():
            s = nuevos[i]
            for ini, fin, txt in sorted(cs, reverse=True):
                s = s[:ini] + txt + s[fin:]
            nuevos[i] = s
        meal["recipe"] = nuevos
        meal.pop("_display", None)      # DELETE-on-write: `_display[locale].recipe` espeja los pasos por índice
        return len(cambios)
    except Exception:
        return 0


# ── 329 · las porciones del paso son las de la lista ─────────────────────────────────────────────────────────────────
# «Mise en place: mide 2 porciones de casabe (15 g)» con «1 porción de casabe (15 g)» en la lista: 3 de 330 comidas (el
# peso se sincronizó y el número no: «porción» no es una unidad del sincronizador). tooltip-anchor: P1-PLAN-LOTE-329
_PORCION_RE = re.compile(
    r"(?<![\w.,/½¼¾⅓⅔])(?P<n>\d+|½|¼|¾)\s+porci(?:ón|on|ones)\s+de\s+(?P<food>[a-záéíóúñü]+)", re.IGNORECASE)
_FRAC_PORCION = {"½": 0.5, "¼": 0.25, "¾": 0.75}


def _valor(q) -> float:
    return _FRAC_PORCION[q] if q in _FRAC_PORCION else float(q)


def porciones_de_la_lista(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lista, amb = {}, set()
        for ln in meal.get("ingredients") or []:
            m = _PORCION_RE.match(str(ln).strip())
            t = _toks(m.group("food")) if m else []
            if not t:
                continue
            if t[0] in lista:
                amb.add(t[0])
            lista[t[0]] = m.group("n")
        for t in amb:
            lista.pop(t, None)
        if not lista:
            return 0
        n = 0
        nuevos = []
        for p in rec:
            if _es_nota(p):
                nuevos.append(p)
                continue

            def _sub(mm, _p=p):
                t = _toks(mm.group("food"))
                q = lista.get(t[0]) if t else None
                if q is None or _valor(q) == _valor(mm.group("n")) or _REPARTO_ANTES_RE.search(_p[:mm.start()]):
                    return mm.group(0)
                return f"{q} {'porción' if _valor(q) <= 1 else 'porciones'} de {mm.group('food')}"
            s = _PORCION_RE.sub(_sub, p)
            n += int(s != p)
            nuevos.append(s)
        if n:
            meal["recipe"] = nuevos
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── 330 · una pizca de la lista no es «½ g» en el paso ───────────────────────────────────────────────────────────────
# «mide … y ½ g de semillas de girasol» con «1 pizca de semillas de girasol sin sal» en la lista; «⅔ g de ajonjolí» con
# «1 pizca de ajonjolí»: el pulido de la lista convierte las migajas en pizcas (lote 181) y el paso se quedaba con el
# gramo de máquina. tooltip-anchor: P1-PLAN-LOTE-330
_MIGAJA_EN_PASO_RE = re.compile(  # [P1-PLAN-LOTE-358] también «mide 0 g de almendras» (el cero a secas)
    r"(?<![\w.,/])(?P<q>0(?:[.,]\d+)?|½|¼|¾|⅓|⅔)\s*(?:g|gr|gramos)\s+de\s+(?P<food>[a-záéíóúñü]+)", re.IGNORECASE)
_PIZCA_LINEA_RE = re.compile(r"^\s*1\s+pizca\s+de\s+(?P<food>.+)$", re.IGNORECASE)


def _primera_358(txt) -> list:
    """[P1-PLAN-LOTE-358] La primera palabra, sin mínimo de letras: `_toks` pide ≥ 4 y la «sal» nunca casaba."""
    w = _sa(str(txt or "").lower()).split()
    return w[:1]


def pizcas_de_la_lista(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        pizcas = set()
        for ln in meal.get("ingredients") or []:
            m = _PIZCA_LINEA_RE.match(str(ln))
            t = _primera_358(m.group("food")) if m else []   # [P1-PLAN-LOTE-358] «sal» tiene 3 letras
            if t:
                pizcas.add(t[0])
        if not pizcas:
            return 0
        n = 0
        nuevos = []
        for p in rec:
            if _es_nota(p):
                nuevos.append(p)
                continue

            def _sub(mm):
                t = _primera_358(mm.group("food"))
                return f"1 pizca de {mm.group('food').lower()}" if t and t[0] in pizcas else mm.group(0)  # [P1-PLAN-LOTE-358] «de Sal»
            s = _MIGAJA_EN_PASO_RE.sub(_sub, p)
            n += int(s != p)
            nuevos.append(s)
        if n:
            meal["recipe"] = nuevos
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── 331 · el mismo alimento dos veces en una enumeración ─────────────────────────────────────────────────────────────
# «Mise en place: … mide 50 g de aguacate, 30 g de aguacate, 5 g de semillas de calabaza» con «50 g de aguacate» en la
# lista (el cerrador añadió la cifra nueva a la enumeración y la vieja se quedó): el mismo alimento, con el MISMO texto,
# dos veces en un paso y una de ellas es la de la lista ⇒ la otra sobra. Sólo en enumeraciones con coma; «semillas de
# girasol» y «semillas de chía» no son el mismo texto. tooltip-anchor: P1-PLAN-LOTE-331
_ITEM_ENUM_RE = re.compile(
    r"(?<![\w.,/])(?P<n>\d+(?:[.,]\d+)?)\s*(?P<u>g|gr|gramos|ml)\s+de\s+(?P<food>[a-záéíóúñü]+(?:\s+de\s+[a-záéíóúñü]+)?)"
    r"(?=\s*(?:,|\by\b|\.|;|$))", re.IGNORECASE)


def mencion_repetida(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lista = {}
        for ln in meal.get("ingredients") or []:
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos|ml)\s+de\s+(.+?)\s*$", str(ln), re.IGNORECASE)
            if m:
                lista[_sa(m.group(2).lower())] = float(m.group(1).replace(",", "."))
        if not lista:
            return 0
        n = 0
        nuevos = []
        for p in rec:
            if _es_nota(p):
                nuevos.append(p)
                continue
            grupos = {}
            for mm in _ITEM_ENUM_RE.finditer(p):
                grupos.setdefault(_sa(mm.group("food").lower()), []).append(mm)
            quitar = []
            for food, mms in grupos.items():
                real = lista.get(food)
                if real is None or len(mms) != 2:
                    continue
                vs = [float(x.group("n").replace(",", ".")) for x in mms]
                buenas = [x for x, v in zip(mms, vs) if abs(v - real) < 0.5]
                if len(buenas) != 1 or abs(vs[0] - vs[1]) < 0.5:
                    continue
                mala = mms[1] if buenas[0] is mms[0] else mms[0]
                despues = p[mala.end():]
                sep = re.match(r"\s*,\s*", despues)
                if sep:
                    quitar.append((mala.start(), mala.end() + sep.end()))
                    continue
                antes = re.search(r",\s*$", p[:mala.start()])
                if antes:
                    quitar.append((antes.start(), mala.end()))
            if not quitar:
                nuevos.append(p)
                continue
            s = p
            for ini, fin in sorted(quitar, reverse=True):
                s = s[:ini] + s[fin:]
            nuevos.append(s)
            n += 1
        if n:
            meal["recipe"] = nuevos
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


def lo_que_dice_la_lista(meal) -> int:
    """Los pases de arriba (y la pista de peso del 332, abajo), en orden. Nº total de pasos reescritos."""
    return (gramos_de_la_pieza(meal) + porciones_de_la_lista(meal) + pizcas_de_la_lista(meal)
            + mencion_repetida(meal) + pistas_de_la_lista(meal) + conteos_de_la_lista(meal)
            + gramos_por_dos_palabras(meal))


# ── [P1-PLAN-LOTE-332 · 2026-09-25] La pista de peso de un paso sigue a la de la lista ───────────────────────────────
# «Mise en place: … ½ cda de mantequilla de maní natural (16 g)» con «½ cda de mantequilla de maní natural sin sal (8 g)»
# en la lista; «pica la cebolla (20 g)» con «½ cda de cebolla picada (5 g)»: el sincronizador cambió la cantidad y la pista
# del paréntesis se quedó con la cifra vieja (32 pasos en el corpus de 308 planes, 3 de 330 comidas en la batería de
# cierre). El lote 310 solo la veía cuando el paso repite EXACTAMENTE el texto de la línea («sin sal» de menos bastaba
# para no verla). Aquí la pista pertenece al alimento de la lista MÁS CERCANO que tiene delante en su cláusula («y» también
# corta: «210 g de jitomate y ½ cebolla (50 g)» es de la cebolla); si esa línea trae peso, el paréntesis pasa a ser el
# suyo. Nunca: una línea sin peso (la pista del paso es entonces la única cifra), una cantidad del paso distinta de la de
# la lista («1 taza … (205 g)» con «½ taza» — eso es del sincronizador), una cláusula que ya pesa el alimento en gramos,
# un reparto (mitad, resto, cada…), cocido contra crudo, ni un alimento con dos líneas. tooltip-anchor: P1-PLAN-LOTE-332
_PISTA_RE = re.compile(r"\(\s*(?P<aprox>[≈~])?\s*(?P<g>\d+(?:[.,]\d+)?)\s*g\s*\)")
_LEAD_DE_LINEA_RE = re.compile(
    r"^\s*(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s*(?:(?:g|gr|gramos|ml|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|"
    r"porci(?:ón|on|ones)|rebanadas?|lonjas?|unidad(?:es)?|pedazos?|dientes?|puñados?|ramitas?|hojas?|latas?|filetes?)"
    r"\.?\s+)?(?:de\s+|del\s+)?", re.IGNORECASE)
_GRAMOS_LIDER_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+de\s+", re.IGNORECASE)
_FRONTERA_PISTA_RE = re.compile(r"[(),;:]|(?<!\d)\.|\.(?!\d)|\by\b|\be\b", re.IGNORECASE)
_CANTIDAD_EN_CLAUSULA_RE = re.compile(r"(?<![\w.,/])(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])(?![\w])")
_GRAMOS_EN_CLAUSULA_RE = re.compile(r"(?<![\w.,/])\d+(?:[.,]\d+)?\s*(?:g|gr|gramos)\s+de\s", re.IGNORECASE)


def _patron_cantidad(lead: str):
    palabras = re.sub(r"\s+del?$", "", _sa(str(lead or "").lower()).strip()).split()
    if not palabras:
        return None
    partes = [re.escape(palabras[0])] + [re.escape(w[:-1] if w.endswith("s") else w) + "s?" for w in palabras[1:]]
    return re.compile(r"(?<![\w.,/])" + r"\s*".join(partes) + r"(?![\w])")


def pistas_de_la_lista(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error. Las notas no se tocan."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas, amb = {}, set()
        for ln in meal.get("ingredients") or []:
            s = str(ln)
            lead = _LEAD_DE_LINEA_RE.match(s)
            cuerpo = re.sub(r"\(.*?\)", " ", s[lead.end():] if lead else s)
            t = _toks(cuerpo)
            if not t:
                continue
            h = _PISTA_RE.search(s)
            gl = _GRAMOS_LIDER_RE.match(s)
            if h:
                peso, aprox = float(h.group("g").replace(",", ".")), bool(h.group("aprox"))
            elif gl:
                peso, aprox = float(gl.group(1).replace(",", ".")), False
            else:
                peso, aprox = None, False
            if t[0] in lineas:
                amb.add(t[0])
            lineas[t[0]] = (peso, aprox, cuerpo, lead.group(0) if lead else "")
        if not any(v[0] is not None for k, v in lineas.items() if k not in amb):
            return 0
        n = 0
        nuevos = []
        for p in rec:
            if _es_nota(p):
                nuevos.append(p)
                continue
            cambios = []
            for h in _PISTA_RE.finditer(p):
                previo = p[:h.start()]
                fronteras = list(_FRONTERA_PISTA_RE.finditer(previo))
                clausula = previo[fronteras[-1].end():] if fronteras else previo
                clave = next((t for t in reversed(_toks(clausula)) if t in lineas), None)
                if clave is None or clave in amb:
                    continue
                peso, aprox, cuerpo, lead = lineas[clave]
                if (peso is None or _GRAMOS_EN_CLAUSULA_RE.search(clausula) or _REPARTO_ANTES_RE.search(clausula)
                        or _REPARTO_DESPUES_RE.search(p[h.end():]) or _cocido(clausula) != _cocido(cuerpo)
                        or _forma_huevo(clausula) != _forma_huevo(cuerpo)):
                    continue
                if _CANTIDAD_EN_CLAUSULA_RE.search(clausula):
                    rx = _patron_cantidad(lead)
                    if rx is None or not rx.search(_sa(clausula.lower())):
                        continue
                viejo = float(h.group("g").replace(",", "."))
                if abs(viejo - peso) <= max(2.0, 0.1 * peso):
                    continue
                cambios.append((h.start(), h.end(), f"({'≈' if aprox else ''}{int(round(peso))} g)"))
            if not cambios:
                nuevos.append(p)
                continue
            s = p
            for ini, fin, txt in sorted(cambios, reverse=True):
                s = s[:ini] + txt + s[fin:]
            nuevos.append(s)
            n += 1
        if n:
            meal["recipe"] = nuevos
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-333 · 2026-09-25] La plantilla del cerrador de proteína, puesta a lo que no es proteína ─────────────
# «Incorpora arroz blanco crudo a la plancha o hervido y sírvelo como proteína del plato» (perfil del dueño, batería de
# cierre), «Cocina cebolla a la plancha o hervida y sírvela como proteína del plato», «Añade tayota…», «Incorpora
# quinoa…»: 22 frases en el corpus de 308 planes, ya en el `pipeline_result` (la IA que corrige un día imita la plantilla
# del cerrador, que ve en su entrada). En las 22 el alimento aparece en OTRO paso del mismo plato: la frase es un eco que
# sobra y se quita. Si la frase es lo único que nombra al alimento, no se toca (quitarla dejaría la línea sin paso).
# Proteína = palabra de proteína o fila del catálogo en «Proteínas»/«Lácteos». tooltip-anchor: P1-PLAN-LOTE-333
_FRASE_CERRADOR_RE = re.compile(
    r"(?:(?<=[.;:!?])\s*|^\s*(?:💪\s*)?)(?P<verbo>Cocina|Incorpora|Agrega|Añade)\s+(?P<obj>[^.;:!?]*?)\s+a\s+la\s+plancha\s+o\s+"
    r"hervid[oa]s?\s+y\s+s[ií]rvel[oa]s?\s+como\s+prote[ií]na\s+del\s+plato\.?", re.IGNORECASE)
_PALABRA_PROTEINA_RE = re.compile(
    r"\b(?:pollo|pechugas?|muslos?|huevos?|claras?|pescados?|filetes?|tilapia|salmon|bacalao|atun|sardinas?|carnes?|res|"
    r"cerdo|chuletas?|pavo|jamon|quesos?|camarones|camaron|langostas?|langostinos?|pulpo|calamar(?:es)?|tofu|tempeh|"
    r"longaniza|salami|chorizo|mero|dorado|chillo|merluza|gandules?|habichuelas?|lentejas?|garbanzos?|frijoles?|soya|"
    r"edamame|seitan|yogu?rt?|cottage|chicharos?|guisantes?|chivo|percebes?|habas?|cordero|conejo|mejillones|almejas?|"
    r"cangrejos?|jaibas?|proteina)\b")
_CATEGORIAS_PROTEINA = ("proteínas", "proteinas", "lácteos", "lacteos")


def plantilla_de_proteina(meal, index=None) -> int:
    """Nº de frases quitadas; 0 ante cualquier error. `index`: el índice culinario del contrato (opcional)."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        quitadas = 0
        nuevos = list(rec)
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            s = p
            for mm in reversed(list(_FRASE_CERRADOR_RE.finditer(p))):
                obj = mm.group("obj")
                if _PALABRA_PROTEINA_RE.search(_sa(obj.lower())):
                    continue
                if index:
                    try:
                        import culinary_coherence as _cc
                        cats = [str((index.get(_cc._norm(nm)) or {}).get("category") or "").lower()
                                for nm in _cc.find_catalog_foods(_sa(obj.lower()), index)]
                    except Exception:
                        cats = ["?"]
                    if any(c in _CATEGORIAS_PROTEINA or c == "?" for c in cats):
                        continue
                t = _toks(obj)
                if not t:
                    continue
                otros = _sa(" ".join([x for j, x in enumerate(nuevos) if j != i and isinstance(x, str)]
                                     + [s[:mm.start()] + " " + s[mm.end():]]).lower())
                if not re.search(r"\b" + re.escape(t[0]), otros):
                    continue                     # la frase es lo único que nombra al alimento: se queda
                s = (s[:mm.start()] + " " + s[mm.end():]).strip()
                quitadas += 1
            if s != p:
                s = re.sub(r"\s{2,}", " ", s)
                nuevos[i] = s if _toks(re.sub(r"^\s*[^:]{0,30}:\s*", "", s)) else None
        if not quitadas:
            return 0
        meal["recipe"] = [x for x in nuevos if x is not None]
        meal.pop("_display", None)
        return quitadas
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-334 · 2026-09-25] El mismo alimento no se sirve dos veces ────────────────────────────────────────────
# «El Toque de Fuego: … Sirve queso cottage al lado para acompañar. Sirve yogurt natural entero al lado para acompañar.» y
# «Montaje: … Acompaña con queso cottage. Acompaña con yogurt natural entero.»: el cerrador de proteína escribe «Sirve X
# al lado» (lote 48), otro pase lo funde en el paso de fuego y el Montaje recibe además «Acompaña con X»
# (P2-CLOSER-MENTION-IN-MONTAJE). 368 de 3.778 comidas del corpus (~10 %) servían así dos veces el mismo alimento. Se
# queda «Acompaña con X» (el Montaje es donde se sirve) y sale la frase «Sirve X al lado para acompañar»; si era un paso
# entero, sale el paso. tooltip-anchor: P1-PLAN-LOTE-334
_SIRVE_AL_LADO_RE = re.compile(
    r"(?:(?<=[.;:!?])\s*|^\s*(?:💪\s*)?)Sirve\s+(?:el\s+|la\s+|los\s+|las\s+)?(?P<x>[^.;:!?]+?)\s+al\s+lado\s+para\s+"
    r"acompañar\.?", re.IGNORECASE)
_ACOMPANA_CON_RE = re.compile(r"\bAcompaña\s+con\s+(?P<xs>[^.;:!?]+)", re.IGNORECASE)
_ARTICULO_RE = re.compile(r"^(?:el|la|los|las|un|una|unos|unas)\s+")


def _servido(txt) -> str:
    return _ARTICULO_RE.sub("", _sa(str(txt or "").lower()).strip()).strip()


def _mismo_servido(x: str, acompana: set) -> bool:
    """[P1-PLAN-LOTE-339 · 2026-09-25] «Sirve Yogurt al lado» con «Acompaña con yogurt natural entero», «Sirve queso
    cottage bajo en sodio al lado» con «Acompaña con queso cottage» (batería real del 25-sep sobre el 331): el mismo
    alimento con el nombre corto en un sitio y el largo en el otro. Mismo sustantivo de cabeza y las palabras de uno
    contenidas en las del otro. tooltip-anchor: P1-PLAN-LOTE-339"""
    if x in acompana:
        return True
    tx = set(_toks(x))
    if not tx:
        return False
    for a in acompana:
        ta = set(_toks(a))
        if ta and _toks(a)[0] == _toks(x)[0] and (ta <= tx or tx <= ta):
            return True
    return False


def servir_una_vez(meal) -> int:
    """Nº de frases «Sirve X al lado para acompañar» quitadas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        acompana = set()
        for p in rec:
            if _es_nota(p):
                continue
            for mm in _ACOMPANA_CON_RE.finditer(p):
                for it in re.split(r",\s*|\s+y\s+|\s+e\s+", mm.group("xs")):
                    x = _servido(it)
                    if x:
                        acompana.add(x)
        if not acompana:
            return 0
        quitadas = 0
        nuevos = []
        for p in rec:
            if _es_nota(p):
                nuevos.append(p)
                continue
            s = p
            for mm in reversed(list(_SIRVE_AL_LADO_RE.finditer(p))):
                if _mismo_servido(_servido(mm.group("x")), acompana):  # [P1-PLAN-LOTE-339] «Yogurt» ↔ «yogurt natural entero»
                    s = (s[:mm.start()] + " " + s[mm.end():]).strip()
                    quitadas += 1
            if s != p:
                s = re.sub(r"\s{2,}", " ", s)
                if not _toks(re.sub(r"^\s*[^:]{0,30}:\s*", "", s)):
                    continue                     # el paso era solo esa frase
            nuevos.append(s)
        if quitadas:
            meal["recipe"] = nuevos
            meal.pop("_display", None)
        return quitadas
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-335 · 2026-09-25] El nombre y los pasos nombran la variedad que trae la lista ──────────────────────
# Baterías del 25-sep (330 comidas re-medidas): «habichuelas blancas» en los pasos con «habichuelas negras» en la lista,
# «pica el ají cubanela» con ají morrón, «tortillas de trigo» con tortilla integral — el usuario compra lo de la lista y
# la receta le nombra otra cosa. Si el plato nombra una fila del catálogo que NO está en su lista y la lista trae
# exactamente UNA hermana (mismo sustantivo de cabeza), la mención pasa a la hermana. Nunca una mención genérica («el
# yogurt griego» con «yogurt griego sin azúcar» en la lista: todas sus palabras están en la hermana), ni una nota del
# sistema, ni un alias del MISMO alimento (el catálogo decide la identidad: «yogur natural entero» es alias de «Yogurt
# griego entero»). tooltip-anchor: P1-PLAN-LOTE-335
def _patron_superficie(superficie_norm: str):
    partes = []
    for t in superficie_norm.split():
        if t in ("yogur", "yogurt"):
            partes.append(r"yogurt?s?")
        elif t in ("de", "del", "con", "sin", "en"):
            partes.append(re.escape(t))
        else:
            base = t[:-2] if (t.endswith("es") and len(t) > 5) else (t[:-1] if (t.endswith("s") and len(t) > 4) else t)
            partes.append(re.escape(base) + r"(?:s|es)?")
    return re.compile(r"(?<![\w])" + r"\s+".join(partes) + r"(?![\w])")


_CONECTORES = ("de", "del", "con", "sin", "en", "y", "al", "a")
_AGUDA = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}


def _plural(frase: str) -> str:
    """«tortilla integral» → «tortillas integrales», «ají morrón» → «ajíes morrones»; lo que sigue a un conector no."""
    out, conector = [], False
    for w in frase.split():
        if conector or w in _CONECTORES or w.endswith("s"):
            conector = conector or w in _CONECTORES
            out.append(w)
        elif w[-1] in "aeoáéó":
            out.append(w + "s")
        elif w[-1] in "iuíú":
            out.append(w + "es")
        elif w.endswith("z"):
            out.append(w[:-1] + "ces")
        else:
            out.append(re.sub(r"[áéíóú](?=[^aeiouáéíóú]*$)", lambda m: _AGUDA[m.group(0)], w) + "es")
    return " ".join(out)


def _singular(tokens) -> set:
    out = set()
    for t in tokens:
        out.add(t[:-2] if (t.endswith("es") and len(t) > 5) else (t[:-1] if (t.endswith("s") and len(t) > 4) else t))
    return out


def variedad_de_la_lista(meal, index=None) -> int:
    """Nº de menciones reescritas; 0 ante cualquier error o sin índice culinario."""
    try:
        if not index or not isinstance(meal, dict):
            return 0
        import culinary_coherence as _cc
        en_lista = set()
        for ln in meal.get("ingredients") or []:
            en_lista.update(_cc.find_catalog_foods(str(ln), index))
        if not en_lista:
            return 0
        rec = meal.get("recipe") if isinstance(meal.get("recipe"), list) else []
        campos = [None] + [i for i, p in enumerate(rec) if isinstance(p, str) and not _es_nota(p)]

        def _texto(i):
            return str(meal.get("name") or "") if i is None else rec[i]

        def _cabeza(n):
            t = _cc._norm(n).split()
            return t[0] if t else ""
        cambios = {}                                   # superficie normalizada -> nombre de la hermana
        for i in campos:
            t = _texto(i)
            blob = _cc._norm(t)
            for ini, fin, nombre in _cc._catalog_food_spans(t, index):
                if nombre in en_lista:
                    continue
                hermanas = [f for f in en_lista if _cabeza(f) == _cabeza(nombre) and f != nombre]
                if len(hermanas) != 1:
                    continue
                sup = blob[ini:fin]
                if _singular(sup.split()) <= _singular(_cc._norm(hermanas[0]).split()):
                    continue                           # mención genérica de la hermana
                cambios[sup] = hermanas[0]
        if not cambios:
            return 0
        n = 0
        for sup, f in sorted(cambios.items(), key=lambda kv: -len(kv[0])):
            rx = _patron_superficie(sup)
            tf = _singular(_cc._norm(f).split()) - set(_CONECTORES)
            nuevo = str(f).lower()
            if sup.split()[0].endswith("s") and not _cc._norm(f).split()[0].endswith("s"):
                nuevo = _plural(nuevo)                 # «las tortillas de trigo» → «las tortillas integrales»
            for i in campos:
                viejo = _texto(i)
                base = _sa(viejo.lower())
                if len(base) != len(viejo):
                    continue
                spans = []
                for m in rx.finditer(base):
                    # «tortilla de trigo integral», «yogurt griego natural sin azúcar»: lo que sigue ya es la hermana
                    sigue = [w for w in re.findall(r"[a-zñ]+", base[m.end():m.end() + 30])[:3] if w not in _CONECTORES][:1]
                    if sigue and _singular(sigue) <= tf:
                        continue
                    spans.append((m.start(), m.end()))
                if not spans:
                    continue
                s = viejo
                for a, b in reversed(spans):
                    rep = nuevo[:1].upper() + nuevo[1:] if s[a:a + 1].isupper() else nuevo
                    s = s[:a] + rep + s[b:]
                if i is None:
                    meal["name"] = s
                else:
                    rec[i] = s
                n += len(spans)
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-337 · 2026-09-25] Las piezas contadas del paso son las de la lista ────────────────────────────────
# «pica 1 diente de ajo» con «3 dientes de ajo» en la lista, «pica ½ ají cubanela» con «1½ ají cubanela», «exprime ½
# limón» con «1 limón»: 33 de 384 comidas de las baterías re-medidas (22 de ajo). El pase de conteos del sincronizador
# solo conoce ocho sustantivos y solo conteos ENTEROS de la lista. Aquí: si la lista trae UNA línea con ese sustantivo
# contado y el paso lo cuenta UNA sola vez (las demás menciones sin número son la misma porción), el número del paso
# pasa a ser el de la lista, con su número gramatical. Dos menciones con número («1 tomate mediano… 1 tomate pequeño
# para decorar») o una marca de reparto (mitad, resto, cada…) no se tocan. tooltip-anchor: P1-PLAN-LOTE-337
_PIEZAS = {
    "diente": ("diente", "dientes"), "aji": ("ají", "ajíes"), "tomate": ("tomate", "tomates"),
    "limon": ("limón", "limones"), "cebolla": ("cebolla", "cebollas"), "pepino": ("pepino", "pepinos"),
    "zanahoria": ("zanahoria", "zanahorias"), "tortilla": ("tortilla", "tortillas"), "arepita": ("arepita", "arepitas"),
    "pechuga": ("pechuga", "pechugas"), "berenjena": ("berenjena", "berenjenas"), "papa": ("papa", "papas"),
}
_PIEZA_NOMBRE_RE = r"(?P<pieza>dientes?|aj[ií](?:es)?|tomates?|lim[oó]n(?:es)?|cebollas?|pepinos?|zanahorias?|tortillas?|arepitas?|pechugas?|berenjenas?|papas?)"
_CUENTA = r"(?P<q>\d+\s*[½¼¾⅓⅔]|\d+(?:[.,]\d+)?|[½¼¾⅓⅔])"
_PIEZA_EN_LISTA_RE = re.compile(r"^\s*" + _CUENTA + r"\s+" + _PIEZA_NOMBRE_RE + r"\b", re.IGNORECASE)
_PIEZA_EN_PASO_RE = re.compile(r"(?<![\w.,/½¼¾⅓⅔])" + _CUENTA + r"\s+" + _PIEZA_NOMBRE_RE + r"\b", re.IGNORECASE)
_ADJ_PIEZA = {"mediano": "medianos", "mediana": "medianas", "pequeño": "pequeños", "pequeña": "pequeñas",
              "grande": "grandes", "verde": "verdes", "maduro": "maduros", "madura": "maduras", "rojo": "rojos",
              "roja": "rojas", "entero": "enteros", "entera": "enteras", "fresco": "frescos", "fresca": "frescas"}
_ADJ_SING = {v: k for k, v in _ADJ_PIEZA.items()}


def _clave_pieza(palabra) -> str:
    w = _sa(str(palabra).lower())
    for k in _PIEZAS:
        if w == k or w == k + "s" or w == k + "es":
            return k
    return ""


def _valor_cuenta(q) -> float:
    q = str(q).replace(" ", "")
    fr = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
    if q in fr:
        return fr[q]
    if q[-1] in fr:
        return float(q[:-1]) + fr[q[-1]]
    return float(q.replace(",", "."))


def conteos_de_la_lista(meal) -> int:
    """Nº de menciones reescritas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lista, amb = {}, set()
        for ln in meal.get("ingredients") or []:
            m = _PIEZA_EN_LISTA_RE.match(str(ln))
            k = _clave_pieza(m.group("pieza")) if m else ""
            if not k:
                continue
            if k in lista:
                amb.add(k)
            lista[k] = m.group("q").replace(" ", "")
        for k in amb:
            lista.pop(k, None)
        if not lista:
            return 0
        menciones = {}
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            for mm in _PIEZA_EN_PASO_RE.finditer(p):
                k = _clave_pieza(mm.group("pieza"))
                if k in lista:
                    menciones.setdefault(k, []).append((i, mm))
        cambios = {}                                       # paso -> [(ini, fin, texto)]
        for k, ms in menciones.items():
            if len(ms) != 1:
                continue                                   # dos menciones con número: un reparto, no se toca
            i, mm = ms[0]
            p = rec[i]
            q_lista = lista[k]
            try:
                if abs(_valor_cuenta(mm.group("q")) - _valor_cuenta(q_lista)) < 1e-6:
                    continue
                plural = _valor_cuenta(q_lista) > 1
            except Exception:
                continue
            if _REPARTO_ANTES_RE.search(p[:mm.start()]) or _REPARTO_DESPUES_RE.search(p[mm.end():]):
                continue
            sing, plur = _PIEZAS[k]
            nombre = plur if plural else sing
            if mm.group("pieza")[:1].isupper():
                nombre = nombre[:1].upper() + nombre[1:]
            ini, fin, texto = mm.start(), mm.end(), f"{q_lista} {nombre}"
            art = re.search(r"\b(los|las|el|la)(\s+)$", p[:mm.start()], re.IGNORECASE)
            if art:                                        # «machaca los 2 dientes» → «machaca el 1 diente»
                a = art.group(1)
                a2 = {"los": "el", "las": "la"}.get(a.lower(), a) if not plural else {"el": "los", "la": "las"}.get(a.lower(), a)
                if a2 != a.lower():
                    a2 = a2[:1].upper() + a2[1:] if a[:1].isupper() else a2
                    ini, texto = art.start(), a2 + art.group(2) + texto
            adj = re.match(r"(\s+)([a-záéíóúñ]+)", p[mm.end():])
            if adj:
                w = adj.group(2)
                w2 = _ADJ_PIEZA.get(w, w) if plural else _ADJ_SING.get(w, w)
                if w2 != w:
                    fin, texto = mm.end() + adj.end(), texto + adj.group(1) + w2
            cambios.setdefault(i, []).append((ini, fin, texto))
        n = 0
        for i, cs in cambios.items():
            s = rec[i]
            for ini, fin, texto in sorted(cs, reverse=True):
                s = s[:ini] + texto + s[fin:]
                n += 1
            rec[i] = s
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-340 · 2026-09-25] La proteína huérfana sale con su frase, no se vuelve «proteína» ─────────────────
# El autocorrector de coherencia (P3-RECIPE-COHERENCE-AUTOFIX, `graph_orchestrator._run_assembly_validations`) ve un paso
# que nombra una proteína sin línea en la lista y la cambia por la proteína REAL del plato; si el plato no trae otra,
# escribía la palabra «proteína»: «Cocina jamón de proteína a la plancha o hervido y sírvelo como proteína del plato» y
# «Acompaña con jamón de proteína» (batería real del 25-sep, perfil del dueño; el parche del revisor había quitado el
# jamón de la lista). Sin reemplazo real, la FRASE que la nombra sale; si esa frase era todo el paso, se conserva la
# conducta de antes (el paso no se queda vacío). tooltip-anchor: P1-PLAN-LOTE-340
_ETIQUETA_PASO_RE = re.compile(r"^\s*([^:.!?]{0,40}:)\s*")
_CORTE_FRASE_RE = re.compile(r"(?<=[.!?;])\s+")


def sin_frases_de(texto, patron, reemplazo="proteína") -> str:
    try:
        s = str(texto or "")
        if not patron.search(s):
            return s
        mlab = _ETIQUETA_PASO_RE.match(s)
        etiqueta = mlab.group(1) if mlab else ""
        cuerpo = s[mlab.end():] if mlab else s
        quedan = [f for f in _CORTE_FRASE_RE.split(cuerpo) if f and not patron.search(f)]
        if quedan and any(_toks(f) for f in quedan):
            return ((etiqueta + " ") if etiqueta else "") + " ".join(quedan)
        # la frase era todo: sale el trozo «y jamón de pavo» / «con jamón de pavo» («Arepitas con queso y jamón de pavo.»)
        trozo = re.compile(r"(?:,\s*|\s+(?:y|e|con|más)\s+)(?:[a-záéíóúñü]+\s+de\s+)?(?:" + patron.pattern + r")",
                           re.IGNORECASE)
        s2 = trozo.sub("", s)
        if s2 != s and _toks(_ETIQUETA_PASO_RE.sub("", s2)):
            return s2
        return patron.sub(reemplazo, s)
    except Exception:
        return str(texto or "")


# ── [P1-PLAN-LOTE-343 · 2026-09-25] Lo cocido del paso es el equivalente de lo SECO de la lista ─────────────────────────
# Batería real sobre el 331 (vegana en México): la lista y el motor cuentan «155 g de frijoles negros secos» (≈530 kcal)
# y el paso dice «Mide 155 g de frijoles negros cocidos» (≈200 kcal): quien sigue la receta come 2,7 veces menos
# legumbre de la planificada. En el corpus de 308 planes, 103 de 137 comidas con legumbre seca en gramos. El reparador
# de cantidades del contrato (C2) copiaba los gramos de la lista sin mirar la base (el sincronizador sí lo mira desde el
# lote 309). Aquí la mención COCIDA de un grano/legumbre cuya línea está en seco pasa a los gramos cocidos equivalentes,
# con el MISMO factor calórico con el que el motor convierte lo cocido a la base de la fila (lote 284): cocidos = secos ×
# kcal_fila / kcal_cocido. «secos (cocidos)» queda «cocidos». Nunca: un reparto (mitad, resto…), dos cifras distintas del
# mismo alimento, ni sin catálogo. tooltip-anchor: P1-PLAN-LOTE-343
_GRAMOS_LINEA_SECA_RE = re.compile(r"^\s*(?P<g>\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+de\s+(?P<nombre>.+)$", re.IGNORECASE)
_SECO_EN_TEXTO_RE = re.compile(r"\b(crud[oa]s?|sec[oa]s?|en seco)\b")
_SECOS_COCIDOS_RE = re.compile(r"\bsec([oa])s?\s*\(\s*(cocid[oa]s?)\s*\)", re.IGNORECASE)


def cocido_de_la_lista(meal, db=None) -> int:
    """Nº de menciones reescritas; 0 ante cualquier error o sin catálogo."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if db is None or not isinstance(rec, list) or not rec:
            return 0
        import cocido_en_catalogo as _cc
        from nutrition_db import _split_qty_unit_name
        secos = {}
        for ln in meal.get("ingredients") or []:
            s = str(ln)
            m = _GRAMOS_LINEA_SECA_RE.match(s)
            if not m or _cc._COCIDO_RX.search(_cc._norm(s)) or _cc._LISTO_RX.search(_cc._norm(s)):
                continue
            _q, _u, nombre = _split_qty_unit_name(s)
            info = db.lookup(nombre)
            fam = _cc.familia(getattr(info, "name", "")) if info else None
            if not fam:
                continue
            kcal_fila = float(getattr(info, "kcal", 0) or 0)
            if kcal_fila <= 0 or kcal_fila / fam[0] < 1.5:
                continue                                   # la fila ya está en cocido
            t = _toks(_SECO_EN_TEXTO_RE.sub(" ", _sa(m.group("nombre").lower())))
            if t:
                secos.setdefault(t[0], []).append(float(m.group("g").replace(",", ".")) * kcal_fila / fam[0])
        secos = {k: v[0] for k, v in secos.items() if len(v) == 1}
        if not secos:
            return 0
        menciones = {}
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            for mm in _GRAMOS_EN_PASO_RE.finditer(p):
                t = _toks(mm.group("food"))
                if not t or t[0] not in secos:
                    continue
                cola = p[mm.end():mm.end() + 30]
                corte = _CORTE_MENCION_RE.search(cola)
                mencion = mm.group(0) + (cola[:corte.start()] if corte else cola)
                if not (_cocido(mencion) or re.match(r"\s*\(\s*cocid[oa]s?\s*\)", cola, re.IGNORECASE)):
                    continue                           # «115 g de frijoles negros secos (cocidos)» también es cocido
                if _REPARTO_ANTES_RE.search(p[:mm.start()]) or _REPARTO_DESPUES_RE.search(p[mm.end():]):
                    menciones[t[0]] = None
                    continue
                if menciones.get(t[0], []) is None:
                    continue
                menciones.setdefault(t[0], []).append((i, mm))
        cambios = {}                                   # paso -> [(ini, fin, texto)]
        for k, ms in menciones.items():
            if not ms or len({float(x[1].group("n").replace(",", ".")) for x in ms}) != 1:
                continue
            cocidos = secos[k]
            actual = float(ms[0][1].group("n").replace(",", "."))
            if abs(actual - cocidos) <= 0.15 * cocidos:
                continue
            nuevo = str(int(round(cocidos / 5.0) * 5))
            for i, mm in ms:
                cambios.setdefault(i, []).append((mm.start("n"), mm.end("n"), nuevo))
        n = 0
        for i, cs in cambios.items():
            p = rec[i]
            for ini, fin, texto in sorted(cs, reverse=True):
                p = p[:ini] + texto + p[fin:]
                n += 1
            rec[i] = _SECOS_COCIDOS_RE.sub(lambda z: z.group(2), p)
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-345 · 2026-09-26] Una sustitución no reescribe las notas-plantilla ────────────────────────────────
# El reescritor de pasos tras una sustitución (`graph_orchestrator._rewrite_recipe_steps_after_subs`: alergias, condición,
# proteína repetida) cambiaba el alimento también dentro de las NOTAS: la de sodio salía «enjuaga los enlatados (yogurt
# griego sin azúcar, granos)» (su «atún» es un EJEMPLO de la plantilla) y la del nutricionista «esta receta usa solo
# pechuga de pollo — NO botes pechuga de pollo: guárdalas tapadas» (batería real sobre el 331: adulto mayor con HTA,
# suplementos + estatina). La nota de sodio se queda como está; una nota 🌱 que nombra el alimento que acaba de salir ya
# no describe el plato y sale. tooltip-anchor: P1-PLAN-LOTE-345
def nota_plantilla(paso) -> bool:
    s = str(paso or "")
    return s.lstrip().startswith("🌱") or "bajas en sodio y enjuaga" in s


def nota_obsoleta(paso, patrones) -> bool:
    s = str(paso or "")
    try:
        return s.lstrip().startswith("🌱") and any(p.search(s) for p in (patrones or []))
    except Exception:
        return False


# ── [P1-PLAN-LOTE-351 · 2026-09-26] Dos quesos en la lista: la mención se ata por sus DOS primeras palabras ──────────
# Plan del dueño (batería real sobre el 331, día 1): «20 g de queso blanco fresco» y «70 g de queso mozzarella» en la
# lista y el paso «ten listos 15 g de queso blanco fresco». El sincronizador identifica cada línea por su PRIMERA palabra
# y, con dos «queso», descarta ambas por ambiguas: el paso se quedaba con la cifra vieja. Aquí, cuando la primera palabra
# es ambigua y las dos primeras señalan UNA sola línea en gramos, el paso toma sus gramos. Mismas guardas que el
# sincronizador: nunca cocido contra crudo ni otra forma del huevo, un reparto, ni dos cifras distintas del mismo
# alimento. tooltip-anchor: P1-PLAN-LOTE-351
_LINEA_GRAMOS_351_RE = re.compile(r"^\s*(?P<g>\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+de\s+(?P<nombre>.+)$", re.IGNORECASE)


def gramos_por_dos_palabras(meal) -> int:
    """Nº de menciones reescritas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        por_primera = {}
        lineas = []
        for ln in meal.get("ingredients") or []:
            s = str(ln)
            t = _toks(re.sub(r"\(.*?\)", " ", s))
            if t:
                por_primera[t[0]] = por_primera.get(t[0], 0) + 1
            m = _LINEA_GRAMOS_351_RE.match(s)
            if not m:
                continue
            tn = _toks(re.sub(r"\(.*?\)", " ", m.group("nombre")))
            if len(tn) >= 2:
                lineas.append(((tn[0], tn[1]), float(m.group("g").replace(",", ".")), m.group("nombre")))
        dos = {}
        for clave, g, nombre in lineas:
            if por_primera.get(clave[0], 0) < 2:
                continue                                   # sin ambigüedad: es del sincronizador
            dos.setdefault(clave, []).append((g, nombre))
        dos = {k: v[0] for k, v in dos.items() if len(v) == 1}
        if not dos:
            return 0
        menciones, vetadas = {}, set()
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            for mm in _GRAMOS_EN_PASO_RE.finditer(p):
                ft = _toks(mm.group("food"))
                if len(ft) < 2 or (ft[0], ft[1]) not in dos:
                    continue
                clave = (ft[0], ft[1])
                cola = p[mm.end():mm.end() + 40]
                corte = _CORTE_MENCION_RE.search(cola)
                mencion = mm.group(0) + (cola[:corte.start()] if corte else cola)
                nombre = dos[clave][1]
                if (_REPARTO_ANTES_RE.search(p[:mm.start()]) or _REPARTO_DESPUES_RE.search(p[mm.end():])
                        or _cocido(mencion) != _cocido(nombre) or _forma_huevo(mencion) != _forma_huevo(nombre)):
                    vetadas.add(clave)
                    continue
                menciones.setdefault(clave, []).append((i, mm))
        cambios = {}
        for clave, ms in menciones.items():
            if clave in vetadas or len({float(x[1].group("n").replace(",", ".")) for x in ms}) != 1:
                continue
            g = dos[clave][0]
            if abs(float(ms[0][1].group("n").replace(",", ".")) - g) < 0.5:
                continue
            txt = str(int(g)) if abs(g - round(g)) < 1e-6 else f"{g:g}"
            for i, mm in ms:
                cambios.setdefault(i, []).append((mm.start("n"), mm.end("n"), txt))
        n = 0
        for i, cs in cambios.items():
            s = rec[i]
            for ini, fin, txt in sorted(cs, reverse=True):
                s = s[:ini] + txt + s[fin:]
                n += 1
            rec[i] = s
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-354 · 2026-09-26] Los conteos con decimales de máquina también se escriben como en la cocina ─────────
# Plan bariátrico real (porciones escaladas ×0,27): «0.27 pepino», «0.27 diente de ajo», «0.27 limón» en la lista y «Pica
# 0.27 diente de ajo, 0.27 de ají cubanela y 0.27 de cebolla… jugo de 0.27 limón» en los pasos. El cuantizador del display
# (`nutrition_db.quantize_ingredient_string`) los deja: su cuadrícula de conteo es (0, ½, 1) y ½ está a un factor 1,85 de
# 0,27, fuera de su guarda. Aquí, sólo la lista visible y los pasos (`ingredients_raw` sigue midiendo): el conteo va a la
# fracción de cocina más cercana (¼ ⅓ ½ ⅔ ¾) si está a ≤ 0,06; en un paso, si la lista ya dice ese alimento con una cantidad
# limpia y cercana (factor 0,6-1,67), manda la de la lista. Sólo sustantivos contables de una lista cerrada: nunca «2.5 cm»,
# «1.5 min», ni huevo/rebanada/tortilla (no se parten). Las notas no se tocan. tooltip-anchor: P1-PLAN-LOTE-354
_CONTABLES_354 = (r"pepinos?|tomates?|cebollas?|dientes?|aj[ií](?:es|s)?|lim[oó]n(?:es)?|limas?|naranjas?|toronjas?|"
                  r"pl[aá]tanos?|guineos?|batatas?|papas?|zanahorias?|aguacates?|mangos?|manzanas?|peras?|chinolas?|"
                  r"pimientos?|berenjenas?|calabac[ií]n(?:es)?|tallos?|ramas?|mazorcas?|remolachas?|nabos?|puerros?|"
                  r"chayotes?|tayotas?|guayabas?|kiwis?")
_CONTEO_DECIMAL_RE = re.compile(
    r"(?<![\d.,/])(?P<e>\d+)[.,](?P<f>\d{1,2})(?!\d)(?P<esp>\s+)(?=(?:de\s+)?(?P<noun>" + _CONTABLES_354 + r")\b)",
    re.IGNORECASE)
_LINEA_CONTEO_RE = re.compile(
    r"^\s*(?P<q>\d+\s*[¼½¾⅓⅔]|\d+(?:[.,]\d+)?|[¼½¾⅓⅔])\s+(?:de\s+)?(?P<noun>" + _CONTABLES_354 + r")\b", re.IGNORECASE)


def _raices_354(palabra) -> set:
    k = _sa(str(palabra or "").lower())
    out = {k}
    if k.endswith("es"):
        out.add(k[:-2])
    if k.endswith("s"):
        out.add(k[:-1])
    return out


def _valor_354(q):
    q = str(q or "").replace(" ", "")
    try:
        if q in _FRAC_VALOR:
            return _FRAC_VALOR[q]
        if q and q[-1] in _FRAC_VALOR:
            return float(q[:-1]) + _FRAC_VALOR[q[-1]]
        return float(q.replace(",", "."))
    except ValueError:
        return None


def _conteo_a_cocina(entero: int, frac: float):
    val, simb = min(_FRACCIONES_COCINA, key=lambda t: abs(frac - t[0]))
    if abs(frac - val) > 0.06:
        return None
    e = entero + (1 if val == 1.0 else 0)
    if e == 0 and not simb:
        return None
    return f"{e if e else ''}{simb}"


def conteos_de_cocina(meal) -> int:
    """Nº de textos reescritos (líneas visibles + pasos). 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        n = 0
        ings = meal.get("ingredients")
        limpias = []                                     # (raíces del sustantivo, valor, texto de la cantidad)
        if isinstance(ings, list):
            for i, s in enumerate(ings):
                if not isinstance(s, str):
                    continue
                inicio = len(s) - len(s.lstrip())
                mm = _CONTEO_DECIMAL_RE.match(s, inicio)
                if mm:
                    c = _conteo_a_cocina(int(mm.group("e")), float("0." + mm.group("f")))
                    if c:
                        s = s[:mm.start()] + c + mm.group("esp") + s[mm.end():]
                        ings[i] = s
                        n += 1
                ml = _LINEA_CONTEO_RE.match(s)
                if ml and not re.search(r"[.,]", ml.group("q")):
                    v = _valor_354(ml.group("q"))
                    if v:
                        limpias.append((_raices_354(ml.group("noun")), v, ml.group("q").replace(" ", "")))
        rec = meal.get("recipe")
        if isinstance(rec, list):
            for i, p in enumerate(rec):
                if _es_nota(p):
                    continue

                def _sub(mm):
                    v_paso = int(mm.group("e")) + float("0." + mm.group("f"))
                    raices = _raices_354(mm.group("noun"))
                    for r, v, txt in limpias:
                        if r & raices and v_paso > 0 and 0.6 <= v / v_paso <= 1.67:
                            return txt + mm.group("esp")
                    c = _conteo_a_cocina(int(mm.group("e")), float("0." + mm.group("f")))
                    return (c + mm.group("esp")) if c else mm.group(0)
                nuevo = _CONTEO_DECIMAL_RE.sub(_sub, p)
                if nuevo != p:
                    rec[i] = nuevo
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-356 · 2026-09-26] Los gramos del paso son los de la PIEZA que cuenta la lista (sin «≈») ──────────────
# Plan renal real: la lista y el motor cuentan «¼ filete de pescado» (37,5 g con el peso del catálogo: el techo renal de
# proteína lo recortó) y el paso dice «mide 90 g de tilapia» — quien sigue la receta come 2,4 veces la proteína que el
# plan calculó. En los planes más recientes del corpus, 15 de 38 traen una pieza contada sin peso («1½ pechugas de pollo»,
# «1 filete de pescado») y un paso en gramos que no es el suyo («300 g de pechuga», «205 g de pescado»). El lote 328 sólo
# lee piezas con «(≈N g)». Aquí manda lo que mide el motor: la línea de `ingredients_raw` del mismo alimento — sus gramos
# si está en gramos (y su base: «cocida» casa con lo cocido del paso), o el peso de catálogo de la pieza si está contada;
# sin línea en `ingredients_raw`, el peso de catálogo de la visible. El paso pasa a esos gramos (a 5 g). Pescado blanco
# genérico acepta la especie en el paso (tilapia, merluza…). Nunca: lo cocido contra lo crudo, un reparto, dos cifras
# distintas, dos piezas o dos líneas del motor de la misma clase, ni sin catálogo. tooltip-anchor: P1-PLAN-LOTE-356
_PIEZA_SIN_PESO_RE = re.compile(
    r"^\s*(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s+(?P<cuerpo>(?:filetes?|pechugas?|muslos?|chuletas?)\b[^()\d≈~]*?)\s*$",
    re.IGNORECASE)
_RAW_EN_GRAMOS_356_RE = re.compile(r"^\s*(?P<g>\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+(?:de\s+)?(?P<cuerpo>.+)$", re.IGNORECASE)
_PESCADO_BLANCO_356 = {"pescado", "tilapia", "merluza", "dorado", "chillo", "corvina", "pargo", "basa"}


def _raiz_356(t: str) -> str:
    return t[:-1] if len(t) > 4 and t.endswith("s") else t


def _peso_del_motor_356(cuerpo_toks: set, visible: str, raw: list, db):
    """(gramos, fila del catálogo, base cocida?) de lo que mide el motor para la pieza visible; None si es ambiguo."""
    cabeza = next(iter(sorted(cuerpo_toks & {"pechuga", "filete", "muslo", "chuleta"})), None)
    candidatas = [r for r in raw if cabeza and cabeza in {_raiz_356(x) for x in _toks(re.sub(r"\(.*?\)", " ", r))}]
    if len(candidatas) > 1:
        return None
    fuente = candidatas[0] if candidatas else visible
    mg = _RAW_EN_GRAMOS_356_RE.match(fuente)
    info = db.macros_from_ingredient_string(fuente)
    fila = _sa(str((info or {}).get("name") or "").lower())
    if mg:
        return float(mg.group("g").replace(",", ".")), fila, _cocido(fuente)
    return float((info or {}).get("grams") or 0), fila, _cocido(fuente)


def pieza_del_catalogo(meal, db=None) -> int:
    """Nº de menciones reescritas; 0 ante cualquier error o sin catálogo."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if db is None or not isinstance(rec, list) or not rec:
            return 0
        raw = [str(x) for x in (meal.get("ingredients_raw") or [])]
        piezas, en_gramos = [], []
        for ln in meal.get("ingredients") or []:
            s = str(ln)
            m = _PIEZA_SIN_PESO_RE.match(s)
            if not m:
                md = _LINEA_EN_GRAMOS_RE.match(s)
                if md:
                    t = _toks(re.sub(r"\(.*?\)", " ", md.group("cuerpo")))
                    if t:
                        en_gramos.append({_raiz_356(x) for x in t})
                continue
            toks = {_raiz_356(x) for x in _toks(m.group("cuerpo"))}
            peso = _peso_del_motor_356(toks, s, raw, db) if toks else None
            if not peso or peso[0] <= 0 or not peso[1]:
                continue
            if "pescado" in peso[1]:
                toks = toks | _PESCADO_BLANCO_356
            piezas.append((toks, peso[0], peso[2]))
        if not piezas:
            return 0

        def _cual(ft):
            cp = [k for k, p in enumerate(piezas) if _raiz_356(ft[0]) in p[0]]
            if len(cp) != 1 or any(_raiz_356(ft[0]) in t for t in en_gramos):
                return None
            return cp[0]

        hallazgos, vetadas = {}, set()
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            for mm in _GRAMOS_EN_PASO_RE.finditer(p):
                ft = _toks(mm.group("food"))
                k = _cual(ft) if ft else None
                if k is None:
                    continue
                cola = p[mm.end():mm.end() + 40]
                corte = _CORTE_MENCION_RE.search(cola)
                mencion = mm.group(0) + (cola[:corte.start()] if corte else cola)
                if (_REPARTO_ANTES_RE.search(p[:mm.start()]) or _REPARTO_DESPUES_RE.search(p[mm.end():])
                        or _cocido(mencion) != piezas[k][2]):
                    vetadas.add(k)
                    continue
                hallazgos.setdefault(k, []).append((i, mm.start("n"), mm.end("n"),
                                                    float(mm.group("n").replace(",", "."))))
        cambios = {}
        for k, lst in hallazgos.items():
            if k in vetadas or len({x[3] for x in lst}) != 1:
                continue
            g = int(round(piezas[k][1] / 5.0) * 5) or int(round(piezas[k][1]))
            if abs(lst[0][3] - g) < 5:
                continue
            for i, ini, fin, _v in lst:
                cambios.setdefault(i, []).append((ini, fin, str(g)))
        if not cambios:
            return 0
        n = 0
        for i, cs in cambios.items():
            s = rec[i]
            for ini, fin, txt in sorted(cs, reverse=True):
                s = s[:ini] + txt + s[fin:]
                n += 1
            rec[i] = s
        meal["recipe"] = rec
        meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-357 · 2026-09-26] Lo listo para comer que la lista compra y ningún paso usa se sirve ────────────────────
# Plan del perfil del dueño (batería sobre el 331): «Pisto criollo… con queso blanco pochado» con «70 g de queso
# mozzarella» en la lista (≈200 kcal y 15 g de proteína contados en el día) y ningún paso que lo nombre — el cerrador lo
# añadió sin su «Acompaña con». Bariátrico sobre el 331: «25 g de pavo molido cocido» en una merienda de mandarina y maní,
# sin paso. En el corpus, 7-8 líneas huérfanas de queso/yogur por cada ~300 planes. Lo que ya se come tal cual (lácteos;
# proteína marcada cocida, de lata o dura) se sirve: el Montaje gana «Acompaña con <alimento>.» (la frase del cerrador; la
# etiqueta de embarazo, que corre después, le añade «pasteurizado»). Nunca lo que hay que cocinar (pollo, pescado o huevo
# crudos), nunca sin Montaje, y con dos quesos en la lista cada uno se reconoce por su segunda palabra.
# tooltip-anchor: P1-PLAN-LOTE-357
_LACTEO_LISTO_RE = re.compile(r"^(?:queso|yogur|yogurt|ricotta|cottage|requeson|mozzarella)\b")
_PROTE_357_RE = re.compile(r"^(?:pollo|pechuga|pavo|res|carne|cerdo|pescado|tilapia|atun|sardinas?|camarones?|huevos?)\b")
_LISTO_357_RE = re.compile(r"\bcocid[oa]s?\b|\ben\s+(?:agua|aceite|lata)\b|\bduros?\b")
_CUERPO_357_RE = re.compile(
    r"^\s*(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s*(?:(?:g|gr|gramos|ml|tazas?|cdas?|cdtas?|lonjas?|rebanadas?|"
    r"porci[oó]n(?:es)?|pedazos?|unidad(?:es)?|potes?)\s+)?(?:de\s+)?(?P<cuerpo>[^()]+?)\s*(?:\(.*)?$", re.IGNORECASE)
_NO_CLAVE_357 = {"fresco", "fresca", "natural", "entero", "entera", "light", "bajo", "baja", "pasteurizado", "pasteurizada",
                 "griego", "rallado", "rallada", "desmenuzado", "desmenuzada", "semidescremado", "descremado"}


def servir_lo_que_sobra(meal) -> int:
    """Nº de alimentos añadidos al Montaje; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        i_mont = next((i for i, s in enumerate(rec) if isinstance(s, str) and s.strip().lower().startswith("montaje")), None)
        if i_mont is None:
            return 0
        texto = _sa(" ".join(str(p) for p in rec if not _es_nota(p)).lower())
        lacteos = []
        for ln in meal.get("ingredients") or []:
            m = _CUERPO_357_RE.match(str(ln))
            if not m:
                continue
            cuerpo = m.group("cuerpo").strip().rstrip(".")
            t = _sa(cuerpo.lower()).split()
            j = " ".join(t)
            if t and (_LACTEO_LISTO_RE.match(j) or (_PROTE_357_RE.match(j) and _LISTO_357_RE.search(j))):
                lacteos.append((cuerpo, t))
        if not lacteos:
            return 0
        n_quesos = sum(1 for _c, t in lacteos if t[0] == "queso")
        faltan = []
        for cuerpo, t in lacteos:
            cab = "yogur" if t[0].startswith("yogur") else t[0]
            if t[0] == "queso" and n_quesos > 1:
                clave = next((w for w in t[1:] if len(w) >= 4 and w not in _NO_CLAVE_357 and w != "de"), None)
                if not clave:
                    continue                                  # ambiguo: no se toca
                patron = r"\b" + re.escape(clave[:6])
            else:
                patron = r"\b" + re.escape(cab[:5])
            if re.search(patron, texto):
                continue
            faltan.append(cuerpo[:1].lower() + cuerpo[1:])
        if not faltan:
            return 0
        mont = str(rec[i_mont]).rstrip()
        if not mont.endswith((".", "!", "?")):
            mont += "."
        rec[i_mont] = f"{mont} Acompaña con {', '.join(faltan[:-1]) + ' y ' + faltan[-1] if len(faltan) > 1 else faltan[0]}."
        meal["recipe"] = rec
        meal.pop("_display", None)
        return len(faltan)
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-359 · 2026-09-26] La pista de gramos no repite la mención ni arrastra calificativos dobles ─────────────
# Replay de la cola real sobre 314 planes: 25 traen «retira del refrigerador 80 g de yogurt griego sin azúcar (80 g)
# natural sin azúcar» (bariátrico real sobre el 331), «mide 25 g de harina de maíz precocida (25 g)», «yogurt griego
# natural sin azúcar (160 g) sin azúcar», «yogurt natural natural sin azúcar». La pista «(N g)» nació al lado de una taza
# («⅓ taza de yogurt (80 g)»); el sincronizador cambió la taza por los gramos de la lista y el nombre por el de la lista,
# y la pista y el calificativo viejo se quedaron detrás. Aquí, en los pasos (no en las notas): (1) tras la pista, los
# calificativos que ya estaban antes de ella se van, con los que los acompañan; (2) la pista que repite los gramos con que
# empieza su misma mención se va; (3) el mismo calificativo dos veces seguidas queda una. tooltip-anchor: P1-PLAN-LOTE-359
# [P1-PLAN-LOTE-400 · 2026-09-26] «yogurt griego sin azúcar pasteurizado (140 g) pasteurizado» (11 en el corpus, perfiles
# de embarazo y lactancia) y «descremado (0-2% de grasa) (135 g) descremado» (batería real sobre el 379): el pasteurizado
# y el paréntesis del porcentaje de grasa entre el calificativo y la pista. tooltip-anchor: P1-PLAN-LOTE-400
_QUAL_359 = r"(?:sin\s+az[uú]car|bajo\s+en\s+sodio|sin\s+sal|natural|enter[oa]|descremad[oa]|light|griego|pasteurizad[oa])"
_PISTA_REPITE_359_RE = re.compile(
    r"(?<![\d.,])(?P<n>\d+(?:[.,]\d+)?)(?P<u>\s*(?:g|gr|gramos)\s+de\s+)(?P<x>(?:(?!\by\b)[^().;:,\d]){2,50}?)"
    r"\s*\(\s*≈?\s*(?P=n)\s*g\s*\)")
_CALIF_TRAS_PISTA_359_RE = re.compile(
    r"\b(?P<antes>(?:" + _QUAL_359 + r"\s*)+(?:\([^()]*%[^()]*\)\s*)?)(?P<pista>\(\s*≈?\s*\d+(?:[.,]\d+)?\s*g\s*\))(?P<despues>(?:\s+" + _QUAL_359 + r")+)",
    re.IGNORECASE)
_CALIF_DOBLE_359_RE = re.compile(r"\b(" + _QUAL_359 + r")\s+\1\b", re.IGNORECASE)


def _calificativos_359(txt: str) -> set:
    return {re.sub(r"\s+", " ", _sa(x.lower())) for x in re.findall(_QUAL_359, str(txt), re.IGNORECASE)}


def pista_sin_eco(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0

        def _tras(mm):
            if _calificativos_359(mm.group("antes")) & _calificativos_359(mm.group("despues")):
                return mm.group("antes") + mm.group("pista")
            return mm.group(0)

        n = 0
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            s = _CALIF_TRAS_PISTA_359_RE.sub(_tras, p)
            s = _PISTA_REPITE_359_RE.sub(lambda mm: mm.group("n") + mm.group("u") + mm.group("x").rstrip(), s)
            s = _CALIF_DOBLE_359_RE.sub(r"\1", s)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-370 · 2026-09-26] La pieza contada de la lista dice cuánto pesa: «¼ filete de pescado (≈55 g)» ─────────
# Plan renal real: la lista visible dice «¼ filete de pescado» y el motor mide 55 g (su línea de `ingredients_raw`); un
# cuarto de un filete de catálogo son 37 g. El humanizador sólo anota el peso de las unidades VAGAS de su tabla (pedazo,
# porción…) y «filete», «pechuga», «muslo» y «chuleta» no lo eran para él, aunque su tamaño varía el doble de una pieza a
# otra: en el corpus de 314 planes, 461 líneas visibles contadas sin peso. Aquí se les anexa «(≈N g)» con lo que mide el
# motor (la misma fuente que el lote 356 usa para los pasos: el raw en gramos, o el peso de catálogo de la pieza contada).
# Sólo display; nunca ante ambigüedad (dos líneas del motor de la misma clase) ni sin catálogo; lo cocido lleva su base
# («(≈270 g cocida)» no: se deja sin anexo). tooltip-anchor: P1-PLAN-LOTE-370
def peso_de_la_pieza(meal, db=None) -> int:
    """Nº de líneas visibles anotadas; 0 ante cualquier error o sin catálogo."""
    try:
        ings = meal.get("ingredients") if isinstance(meal, dict) else None
        if db is None or not isinstance(ings, list) or not ings:
            return 0
        raw = [str(x) for x in (meal.get("ingredients_raw") or [])]
        n = 0
        for i, ln in enumerate(ings):
            s = str(ln)
            m = _PIEZA_SIN_PESO_RE.match(s)
            if not m:
                continue
            toks = {_raiz_356(x) for x in _toks(m.group("cuerpo"))}
            peso = _peso_del_motor_356(toks, s, raw, db) if toks else None
            if not peso or peso[0] <= 0 or not peso[1] or peso[2]:
                continue
            g = int(round(peso[0] / 5.0) * 5) or int(round(peso[0]))
            ings[i] = f"{s.rstrip()} (≈{g} g)"
            n += 1
        if n:
            meal["ingredients"] = ings
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-371 · 2026-09-26] «los 40 g», no «las 40 g»: el artículo concuerda con los gramos ───────────────────
# Replay de la cola real (314 planes): «escurre las 40 g de habichuelas negras cocidas», «ten listas las 135 g de
# habichuelas rojas cocidas», «corta la 20 g de queso blanco fresco», «corta las 350 g de papa» — el sincronizador mete la
# cifra detrás del artículo que el paso tenía para el alimento («escurre las habichuelas»), y «gramos» es masculino. El
# artículo (y un «lista/listas» delante) pasa a masculino plural. Sólo pasos, nunca notas. tooltip-anchor: P1-PLAN-LOTE-371
_ARTICULO_GRAMOS_371_RE = re.compile(
    r"\b(?:(?P<listo>listas?)\s+)?(?P<art>las|la|unas|una)\s+(?=\d+(?:[.,]\d+)?\s*(?:g|gr|gramos|ml)\b)", re.IGNORECASE)


def _articulo_371(mm) -> str:
    art = mm.group("art")
    nuevo = "los" if art.lower() in ("las", "la") else "unos"
    if art[:1].isupper():
        nuevo = nuevo.capitalize()
    listo = mm.group("listo")
    pre = ""
    if listo:
        pre = ("Listos" if listo[:1].isupper() else "listos") + " "
    return pre + nuevo + " "


def articulo_de_los_gramos(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            s = _ARTICULO_GRAMOS_371_RE.sub(_articulo_371, p)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-373 · 2026-09-26] Los nombres del catálogo van en minúscula a media frase; el participio, una vez ─────
# Replay de la cola real (314 planes): «añade el tomate y la pechuga, Sal al gusto y Pimienta negra», «¾ cdta de Aceite de
# oliva, Limón y Orégano dominicano» (15 pasos: el cerrador y las sustituciones copian el nombre de la fila del catálogo,
# con su mayúscula) y «15 g de maní fileteado fileteado» (3 pasos). La lista ya se pule (lote 181, `_mayuscula`); los pasos
# no. Aquí, sólo en pasos y sólo tras «, », «y», «e», «o», «de», «con»: la mayúscula de una palabra que sigue en minúscula
# pasa a minúscula, salvo que la siguiente palabra también empiece por mayúscula (marca o nombre propio: «Corn Flakes»). Y el
# mismo participio o calificativo dos veces seguidas queda una. Nunca las notas. tooltip-anchor: P1-PLAN-LOTE-373
_MAYUS_EN_PASO_373_RE = re.compile(
    r"(?:(?<=,\s)|(?<=\by\s)|(?<=\be\s)|(?<=\bo\s)|(?<=\bde\s)|(?<=\bcon\s))(?P<l>[A-ZÁÉÍÓÚÑ])(?P<r>[a-záéíóúñü]{2,})\b"
    r"(?P<sig>\s+[A-ZÁÉÍÓÚÑ])?")
_PARTICIPIO_DOBLE_373_RE = re.compile(
    r"\b([a-záéíóúñü]{3,}(?:ad[oa]s?|id[oa]s?)|natural(?:es)?|enter[oa]s?|fresc[oa]s?)\s+\1\b", re.IGNORECASE)


#: nombres propios que viven dentro de nombres de alimentos («coles de Bruselas», «mostaza Dijon», «canela de Ceilán»)
_PROPIOS_373 = {"bruselas", "oaxaca", "cotija", "chihuahua", "dijon", "worcestershire", "tabasco", "sosua", "parma", "modena",
                "jerez", "idiazabal", "burgos", "cabrales", "roquefort", "philadelphia", "ceilan", "cassia", "valencia",
                "provenza", "maggi", "goya", "knorr", "kellogg", "nutella", "barilla", "badia", "campos", "manchego",
                "serrano", "iberico", "california", "kalamata", "jalisco", "yucatan", "baviera", "normandia"}


def _minuscula_373(mm) -> str:
    if mm.group("sig"):
        return mm.group(0)                       # «Corn Flakes», un nombre propio: se queda
    if _sa((mm.group("l") + mm.group("r")).lower()) in _PROPIOS_373:
        return mm.group(0)                       # «coles de Bruselas»
    if mm.string[max(0, mm.start() - 9):mm.start()].lower().endswith("toque de "):
        return mm.group(0)                       # el rótulo «El Toque de Fuego»
    return mm.group("l").lower() + mm.group("r")


def pasos_en_minuscula(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            s = _PARTICIPIO_DOBLE_373_RE.sub(r"\1", _MAYUS_EN_PASO_373_RE.sub(_minuscula_373, p))
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-374 · 2026-09-26] La pista «(N g)» de una taza dice lo que mide el motor ─────────────────────────────
# Replay de la cola real (314 planes): 78 de 313 pistas de taza en los pasos no son lo que el motor mide para ese alimento
# (su línea de `ingredients_raw` en gramos): «1¼ tazas de avena en hojuelas (65 g)» con 107 g en el motor (perfil del
# dueño), «⅓ taza de yogurt griego (120 g)» con 79 g, «¼ taza de yogurt (240 g)» con 62 g. La taza del paso ya sigue a la
# lista (el humanizador la sacó de esos gramos); la pista es la cifra vieja del modelo. El lote 332 sólo corrige la pista
# cuando la línea VISIBLE trae peso; la de taza no lo trae. Aquí la pista pasa a los gramos del motor si difieren en más de
# un 10 % y 5 g. Nunca: dos líneas del motor para ese alimento, un reparto, cocido contra crudo, ni una línea del motor que
# no esté en gramos. Las notas no se tocan. tooltip-anchor: P1-PLAN-LOTE-374
_TAZA_CON_PISTA_374_RE = re.compile(
    r"(?<![\w.,/])(?:\d+\s*[½¼¾⅓⅔]|\d+(?:[.,]\d+)?|[½¼¾⅓⅔])\s*tazas?\s+de\s+(?P<food>[a-záéíóúñü]+)(?P<resto>[^()]{0,40}?)"
    r"\(\s*(?P<g>\d+(?:[.,]\d+)?)\s*g\s*\)", re.IGNORECASE)
_RAW_GRAMOS_374_RE = re.compile(r"^\s*(?P<g>\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+(?:de\s+)?(?P<cuerpo>.+)$", re.IGNORECASE)


def pista_de_taza_del_motor(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        raw = [str(x) for x in (meal.get("ingredients_raw") or [])]
        if not raw:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            cambios = []
            for mm in _TAZA_CON_PISTA_374_RE.finditer(p):
                t = _toks(mm.group("food"))
                if not t:
                    continue
                cand = [r for r in raw if t[0] in _toks(re.sub(r"\(.*?\)", " ", r))]
                if len(cand) != 1:
                    continue
                rg = _RAW_GRAMOS_374_RE.match(cand[0])
                if not rg:
                    continue
                mencion = mm.group(0)
                if (_cocido(mencion) != _cocido(cand[0]) or _REPARTO_ANTES_RE.search(p[:mm.start()])
                        or _REPARTO_DESPUES_RE.search(p[mm.end():])):
                    continue
                motor = float(rg.group("g").replace(",", "."))
                viejo = float(mm.group("g").replace(",", "."))
                if motor <= 0 or abs(viejo - motor) <= max(5.0, 0.1 * motor):
                    continue
                cambios.append((mm.start("g"), mm.end("g"), str(int(round(motor)))))
            if cambios:
                s = p
                for ini, fin, txt in sorted(cambios, reverse=True):
                    s = s[:ini] + txt + s[fin:]
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-375 · 2026-09-26] Lo que la lista compra SECO y el paso usa cocido trae su cocción previa ─────────────
# Replay de la cola real (314 planes): 123 comidas con V7c —la legumbre o el grano está SECO (o crudo) en la lista y
# ningún paso lo remoja ni lo hierve—, 46 de ellas en las baterías recientes y casi todas «mayores»: el paso lo da por
# cocido («mide 140 g de garbanzos cocidos», «escurre los frijoles pintos») con los secos en la lista. Lentejas (35),
# arroz blanco (28), garbanzos (25), quinoa, habichuelas, frijoles, gandules, arroz integral, bulgur, cebada. El contrato
# lo detectaba y no lo reparaba. Aquí, tras el «Mise en place», una nota «💡 Cocción previa:» con el remojo y el hervor
# de ese alimento en UNA oración (el detector la lee como cocción: queda idempotente) y, en legumbres, la tanda de varios
# días. Nunca pasta, avena ni soya (su cocción es la del paso) ni un alimento sin plantilla. tooltip-anchor: P1-PLAN-LOTE-375
_COCCION_PREVIA_375 = (
    ("lenteja", "enjuaga las {n} secas y hiérvelas 20-25 min (no necesitan remojo) hasta que estén tiernas, y escúrrelas"),
    ("garbanzo", "remoja los {n} secos 8-12 h y hiérvelos 60-90 min hasta que estén tiernos, y escúrrelos"),
    ("habichuela", "remoja las {n} secas 8-12 h y hiérvelas 60-90 min (los primeros 10 min a fuego fuerte) hasta que estén "
                   "tiernas, y escúrrelas"),
    ("frijol", "remoja los {n} secos 8-12 h y hiérvelos 60-90 min (los primeros 10 min a fuego fuerte) hasta que estén "
               "tiernos, y escúrrelos"),
    ("gandul", "hierve los {n} secos 40-60 min hasta que estén tiernos, y escúrrelos"),
    ("guandul", "hierve los {n} secos 40-60 min hasta que estén tiernos, y escúrrelos"),
    ("haba", "remoja las {n} secas 8-12 h y hiérvelas 60-90 min hasta que estén tiernas, y escúrrelas"),
    ("arroz integral", "enjuaga el {n} crudo y cuécelo en agua 35-45 min hasta que esté tierno"),
    ("arroz", "enjuaga el {n} crudo y cuécelo en agua 15-20 min hasta que esté tierno"),
    ("quinoa", "enjuaga la {n} cruda y cuécela en agua 12-15 min hasta que esté tierna, y escúrrela"),
    ("bulgur", "hidrata el {n} en agua caliente 10-15 min (o cuécelo 10-12 min), y escúrrelo"),
    ("cebada", "cuece la {n} cruda en agua 30-40 min hasta que esté tierna, y escúrrela"),
)
_LEGUMBRE_375 = ("lenteja", "garbanzo", "habichuela", "frijol", "gandul", "guandul", "haba")


def coccion_previa(meal, index=None) -> int:
    """Nº de notas añadidas; 0 ante cualquier error o sin índice."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not index or not isinstance(rec, list) or not rec:
            return 0
        from culinary_coherence import _v7c_seco_sin_coccion
        hallazgos = _v7c_seco_sin_coccion({"day": 0}, meal, index) or []
        notas = []
        for v in hallazgos:
            food = str(v.get("food") or "").strip()
            fn = _sa(food.lower())
            plantilla = next((t for k, t in _COCCION_PREVIA_375 if re.search(r"\b" + re.escape(k), fn)), None)
            if not plantilla:
                continue
            nombre = re.sub(r"\s+(?:sec[oa]s?|crud[oa]s?)\b", "", food.lower()).strip()
            texto = "💡 Cocción previa: " + plantilla.format(n=nombre)
            if any(re.search(r"\b" + re.escape(k), fn) for k in _LEGUMBRE_375):
                texto += " (puedes cocinar la tanda de varios días y guardarla en la nevera hasta 4 días)"
            texto += "."
            if texto not in rec and texto not in notas:
                notas.append(texto)
        if not notas:
            return 0
        i_mise = next((i for i, s in enumerate(rec) if isinstance(s, str) and s.strip().lower().startswith("mise en place")), None)
        pos = (i_mise + 1) if i_mise is not None else 0
        rec[pos:pos] = notas
        meal["recipe"] = rec
        meal.pop("_display", None)
        return len(notas)
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-376 · 2026-09-26] El conteo del paso sigue al de la lista también en piezas, filetes y guineítos ─────
# Replay de la cola real: el contrato acusa 41 V7e («el paso pide 2 y la lista compra 1»), 10 en las baterías recientes:
# «mide 2 piezas de casabe» con «1 pieza de casabe», «ten listos … 1½ piezas de casabe» con 1, «pela y corta los 2
# guineítos verdes» con «1 guineíto verde», «corta 1¾ filetes de pescado» con «1½ filetes». El lote 337 conoce doce
# sustantivos contados sin «de» (diente, tomate, limón…), no las UNIDADES («N piezas DE casabe») ni estos víveres. Aquí,
# con las mismas guardas que el 337 (UNA línea de la lista por clave, UNA mención numerada en los pasos, sin reparto): el
# número del paso pasa a ser el de la lista, con la unidad en su número gramatical. Una unidad con «de» se ata también por
# su alimento («pieza de casabe» no es «pieza de pollo»). Las notas no se tocan. tooltip-anchor: P1-PLAN-LOTE-376
_UNIDADES_376 = {"pieza": ("pieza", "piezas"), "porcion": ("porción", "porciones"), "rebanada": ("rebanada", "rebanadas"),
                 "filete": ("filete", "filetes"), "guineito": ("guineíto", "guineítos"), "guineo": ("guineo", "guineos"),
                 "platano": ("plátano", "plátanos"), "batata": ("batata", "batatas")}
_CON_DE_376 = {"pieza", "porcion", "rebanada", "filete"}
_UNIDAD_376 = r"(?P<u>piezas?|porci[oó]n(?:es)?|rebanadas?|filetes?|guine[ií]tos?|guineos?|pl[aá]tanos?|batatas?)"
_CONTEO_LISTA_376_RE = re.compile(r"^\s*" + _CUENTA + r"\s+" + _UNIDAD_376 + r"\b(?P<resto>.*)$", re.IGNORECASE)
_CONTEO_PASO_376_RE = re.compile(r"(?<![\w.,/½¼¾⅓⅔])" + _CUENTA + r"\s+" + _UNIDAD_376 + r"\b", re.IGNORECASE)
_DE_ALIMENTO_376_RE = re.compile(r"^(?:\s+[a-záéíóúñü]+)?\s+de\s+(?P<food>[a-záéíóúñü]+)", re.IGNORECASE)


def _clave_376(unidad: str, cola: str):
    u = _sa(str(unidad).lower())
    k = next((c for c in _UNIDADES_376 if u in (c, c + "s", c + "es")), "")
    if not k:
        return None
    if k in _CON_DE_376:
        md = _DE_ALIMENTO_376_RE.match(cola or "")
        if not md:
            return None
        return (k, _sa(md.group("food").lower()))
    return (k, "")


def conteos_con_unidad(meal) -> int:
    """Nº de menciones reescritas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lista, amb = {}, set()
        for ln in meal.get("ingredients") or []:
            m = _CONTEO_LISTA_376_RE.match(str(ln))
            clave = _clave_376(m.group("u"), m.group("resto")) if m else None
            if not clave:
                continue
            if clave in lista:
                amb.add(clave)
            lista[clave] = m.group("q").replace(" ", "")
        for c in amb:
            lista.pop(c, None)
        if not lista:
            return 0
        menciones = {}
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            for mm in _CONTEO_PASO_376_RE.finditer(p):
                clave = _clave_376(mm.group("u"), p[mm.end():mm.end() + 60])
                if clave in lista:
                    menciones.setdefault(clave, []).append((i, mm))
        cambios = {}
        for clave, ms in menciones.items():
            if len(ms) != 1:
                continue
            i, mm = ms[0]
            p = rec[i]
            q_lista = lista[clave]
            try:
                if abs(_valor_cuenta(mm.group("q")) - _valor_cuenta(q_lista)) < 1e-6:
                    continue
                plural = _valor_cuenta(q_lista) > 1
            except Exception:
                continue
            if _REPARTO_ANTES_RE.search(p[:mm.start()]) or _REPARTO_DESPUES_RE.search(p[mm.end():]):
                continue
            sing, plur = _UNIDADES_376[clave[0]]
            nombre = plur if plural else sing
            if mm.group("u")[:1].isupper():
                nombre = nombre[:1].upper() + nombre[1:]
            ini, fin, texto = mm.start(), mm.end(), f"{q_lista} {nombre}"
            art = re.search(r"\b(los|las|el|la)(\s+)$", p[:mm.start()], re.IGNORECASE)
            if art:
                a = art.group(1)
                a2 = {"los": "el", "las": "la"}.get(a.lower(), a) if not plural else {"el": "los", "la": "las"}.get(a.lower(), a)
                a2 = a2[:1].upper() + a2[1:] if a[:1].isupper() else a2
                if q_lista == "1":                         # «sobre la pieza de casabe», no «sobre la 1 pieza»
                    ini, texto = art.start(), a2 + art.group(2) + nombre
                elif a2.lower() != a.lower():
                    ini, texto = art.start(), a2 + art.group(2) + texto
            adj = re.match(r"(\s+)([a-záéíóúñ]+)", p[mm.end():])
            if adj:
                w = adj.group(2)
                w2 = _ADJ_PIEZA.get(w, w) if plural else _ADJ_SING.get(w, w)
                if w2 != w:
                    fin, texto = mm.end() + adj.end(), texto + adj.group(1) + w2
            cambios.setdefault(i, []).append((ini, fin, texto))
        n = 0
        for i, cs in cambios.items():
            s = rec[i]
            for ini, fin, texto in sorted(cs, reverse=True):
                s = s[:ini] + texto + s[fin:]
                n += 1
            rec[i] = s
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-377 · 2026-09-26] Lo crudo nunca «ya viene cocido»: el pescado fresco se cocina ─────────────────────
# Batería REAL sobre el 376 (perfil del dueño, día 3, cena): «…dora el queso blanco fresco… Escurre e incorpora filete de
# pescado blanco (ya viene cocido) a la preparación antes de servir» con «¾ filete de pescado (≈88 g)» en la lista —
# pescado CRUDO que la receta manda servir sin cocerlo. El cerrador de proteína puso atún en lata con su frase de
# enlatado; el tope de sodio lo cambió por filete fresco (`swap_canned`) y reescribió el NOMBRE, no la frase. La guarda
# P1-CLOSER-FRESH-COCIDO existe pero su regex arranca en el PRIMER verbo del párrafo («Cocina las arepitas…») y no llega a
# la frase fusionada. Corpus: 2 comidas más. Aquí, en la cola del contrato, la frase «(ya viene cocido)» cuyo alimento
# está en la lista SIN ninguna marca de enlatado o cocido pasa a una cocción con su temperatura segura. Seguridad
# alimentaria: sin línea en la lista para decidirlo, no se toca. tooltip-anchor: P1-PLAN-LOTE-377
_YA_COCIDO_377_RE = re.compile(
    r"(?P<v>Escurre e incorpora|Incorpora)\s+(?P<food>[^.;:()]{3,60}?)\s+\(ya viene cocid[oa]\)\s+"
    r"(?P<post>a la preparaci[oó]n antes de servir|al guiso en los [uú]ltimos minutos)\.?", re.IGNORECASE)
_PRECOCIDO_377_RE = re.compile(r"\blata\b|enlatad|\ben\s+agua\b|\ben\s+aceite\b|sardina|ahumad|cocid|precocid")
_PESCADO_377_RE = re.compile(r"\b(?:pescado|filete|tilapia|merluza|dorado|mero|chillo|corvina|pargo|salmon|bacalao|atun)\b")
_CARNE_377_RE = re.compile(r"\b(?:pollo|pechuga|muslo|pavo|cerdo|res|carne|lomo|chuleta)\b")
_MARISCO_377_RE = re.compile(r"\b(?:camarones?|langostinos?|calamar(?:es)?|mariscos?)\b")
#: cabezas genéricas: «pechuga de pavo» se busca en la lista por «pavo» — la línea «pechuga de pollo cocida» no la hace
#: cocida (así se engañó también el cerrador: pavo CRUDO «ya viene cocido» en el replay, celíaco de producción)
_GENERICO_377 = {"queso", "yogurt", "yogur", "pechuga", "filete", "carne"}


def crudo_no_viene_cocido(meal) -> int:
    """Nº de frases reescritas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = [_sa(str(x).lower()) for x in list(meal.get("ingredients") or []) + list(meal.get("ingredients_raw") or [])]

        def _sub(mm):
            food = mm.group("food").strip()
            fn = _sa(food.lower())
            toks = [t for t in _toks(fn) if t not in ("blanco", "blanca", "fresco", "fresca", "natural", "entero", "entera")]
            if len(toks) > 1:
                toks = [t for t in toks if t not in _GENERICO_377] or toks
            cands = [l for l in lineas if any(t in l for t in toks)]
            if not toks or not cands or any(_PRECOCIDO_377_RE.search(l) for l in cands):
                return mm.group(0)
            primera = fn.split()[0]
            fem = primera.endswith("a") or primera in ("carne",)
            plural = primera.endswith("s") and not primera.endswith("ss")
            art = ("las" if fem else "los") if plural else ("la" if fem else "el")
            lo = ("las" if fem else "los") if plural else ("la" if fem else "lo")
            destino = ("a la preparación antes de servir" if mm.group("post").lower().startswith("a la")
                       else "al guiso en los últimos minutos")
            if _MARISCO_377_RE.search(fn):
                como = "2-3 min por lado, hasta que estén rosados y opacos por dentro"
            elif _PESCADO_377_RE.search(fn):
                como = "a la plancha 3-4 min por lado, hasta que se desmenuce fácilmente (63 °C al centro)"
            elif _CARNE_377_RE.search(fn):
                como = "a la plancha 5-7 min por lado, hasta que no quede rosado por dentro (74 °C al centro)"
            else:
                como = "por completo, hasta que esté bien cocido por dentro"
            return f"Cocina {art} {food} {como}, y agréga{lo} {destino}."

        n = 0
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            s = _YA_COCIDO_377_RE.sub(_sub, p)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-379 · 2026-09-26] La licuadora va antes del fuego, y ni la carne ni el huevo van a ella ────────────────
# Batería REAL sobre el 376 (perfil del dueño, día 1): «licúa la avena, el yogurt…; cocina los panqueques en sartén… 2-3
# minutos por lado… Agrega queso cottage a la licuadora y licúa hasta integrar» — el cerrador de proteína ve «licúa» en los
# pasos y escribe su frase de batido, pero la masa YA se cocinó. Corpus de 315 planes: 5 masas así (panqueques, tortitas,
# una tarta horneada) y dos disparates peores: «Agrega pechuga de pavo a la licuadora» en unas arepitas y «Agrega huevo a
# la licuadora» en un bowl FRÍO (huevo crudo licuado). En la cola del contrato: (a) carne, pescado o huevo crudos nunca van
# a la licuadora — si otro paso ya los cocina la frase sobra; si no, pasa a cocción con su punto seguro y «sírvelo al
# lado»; lo enlatado o cocido, al lado; (b) un lácteo sobre una MASA ya cocinada (panqueque, tortita, arepita, tarta…) va
# al lado, o sobra si el plato ya lo sirve aparte. Un batido, un bowl frío o un yogur licuado conservan su frase.
# tooltip-anchor: P1-PLAN-LOTE-379
_LICUADORA_379_RE = re.compile(r"(?:💪\s*)?Agrega (?P<food>[^.;:()]{2,60}?) a la licuadora y licúa hasta integrar\.",
                               re.IGNORECASE)
_MASA_379_RE = re.compile(r"\b(?:panqueques?|pancakes?|tortitas?|arepitas?|crepes?|crepas?|waffles?|tartas?|bizcochos?|"
                          r"muffins?|magdalenas?|quiches?|budin|pudin)\b")
_FUEGO_379_RE = re.compile(r"sarten|hornea|\bhorno\b|airfryer|a fuego|plancha|\bcocina\b|cocinal|vierte la masa")
_CRUDO_379_RE = re.compile(r"\b(?:huevos?|claras?|pollo|pechuga|pavo|res|cerdo|carne|pescado|filete|tilapia|merluza|"
                           r"salmon|atun|sardinas?|camarones?|chuleta|lomo)\b")
_COCCION_379_RE = re.compile(r"\b(?:cocin\w*|cuec\w*|cuece|hierv\w*|plancha|hornea\w*|saltea\w*|dora\w*|frie|sofrie\w*|"
                             r"revuelv\w*)\b")
_APARTE_379_RE = re.compile(r"\b(?:acompana con|al lado|por encima|sirve con)\b")
_ADJ_379 = {"griego", "griega", "entero", "entera", "natural", "fresco", "fresca", "blanco", "blanca", "pasteurizado",
            "descremado", "light", "bajo", "grasa", "azucar"}
_CABECERA_379_RE = re.compile(r"^\s*[^:.]{1,40}:\s*$")


def _frase_coccion_379(food: str) -> str:
    fn = _sa(food.lower())
    primera = fn.split()[0]
    fem = primera.endswith("a") or primera in ("carne",)
    plural = primera.endswith("s") and not primera.endswith("ss")
    art = ("las" if fem else "los") if plural else ("la" if fem else "el")
    lo = ("las" if fem else "los") if plural else ("la" if fem else "lo")
    if re.search(r"\bclaras?\b", fn):
        como = "en la sartén a fuego medio hasta que estén firmes y opacas"
    elif re.search(r"\bhuevos?\b", fn):
        como = "en la sartén a fuego medio hasta que la clara y la yema estén firmes"
    elif _MARISCO_377_RE.search(fn):
        como = "2-3 min por lado, hasta que estén rosados y opacos por dentro"
    elif _PESCADO_377_RE.search(fn):
        como = "a la plancha 3-4 min por lado, hasta que se desmenuce fácilmente (63 °C al centro)"
    elif _CARNE_377_RE.search(fn):
        como = "a la plancha 5-7 min por lado, hasta que no quede rosado por dentro (74 °C al centro)"
    else:
        como = "por completo, hasta que esté bien cocido por dentro"
    return f"Cocina {art} {food} {como} y sírve{lo} al lado."


def licuadora_a_tiempo(meal) -> int:
    """Nº de frases del cerrador corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        nombre = _sa(str(meal.get("name") or "").lower())
        lineas = [_sa(str(x).lower()) for x in list(meal.get("ingredients") or []) + list(meal.get("ingredients_raw") or [])]
        n = 0
        i = 0
        while i < len(rec):
            p = rec[i]
            mm = _LICUADORA_379_RE.search(p) if isinstance(p, str) else None
            if not mm:
                i += 1
                continue
            food = mm.group("food").strip()
            fn = _sa(food.lower())
            toks = [t for t in _toks(fn) if t not in _ADJ_379]
            if len(toks) > 1:                                    # «queso cottage» se busca por «cottage», no por «queso»
                toks = [t for t in toks if t not in _GENERICO_377] or toks
            if not toks:
                i += 1
                continue
            resto = [_sa(str(q).lower()) if j != i else _sa((p[:mm.start()] + " " + p[mm.end():]).lower())
                     for j, q in enumerate(rec)]
            clausulas = [c for c in re.split(r"[.;:](?!\d)", " . ".join(resto))    # [P1-PLAN-LOTE-52] sin partir decimales
                         if any(re.search(r"\b" + re.escape(t), c) for t in toks)]
            aparte = any(_APARTE_379_RE.search(c) for c in clausulas)
            if _CRUDO_379_RE.search(fn):
                precocido = _PRECOCIDO_377_RE.search(fn) or any(
                    _PRECOCIDO_377_RE.search(l) for l in lineas if any(t in l for t in toks))
                if precocido:
                    nuevo = "" if aparte else f"Sirve {food} al lado para acompañar."
                elif any(_COCCION_379_RE.search(c) for c in clausulas):
                    nuevo = ""
                else:
                    nuevo = _frase_coccion_379(food)
            else:
                antes = " ".join(_sa(str(q).lower()) for q in rec[:i]) + " " + _sa(p[:mm.start()].lower())
                ult = antes.rfind("licu")                  # el fuego cuenta DESPUÉS de la última licuada (el batido
                if not (_MASA_379_RE.search(nombre)        # que se licúa tras cocinar las arepitas sigue en su vaso)
                        and _FUEGO_379_RE.search(antes[ult:] if ult >= 0 else antes)):
                    i += 1
                    continue
                nuevo = "" if aparte else f"Sirve {food} al lado para acompañar."
            s = re.sub(r"\s{2,}", " ", p[:mm.start()] + nuevo + p[mm.end():]).strip()
            n += 1
            if not s or _CABECERA_379_RE.match(s) or s in ("💪",):
                del rec[i]
                continue
            rec[i] = s
            i += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-390 · 2026-09-26] Las claras duras se hierven dentro de su huevo ──────────────────────────────────────
# Plan de la batería real (adulto mayor con HTA): «hierve 3 huevos y 3 claras de huevo 10-11 minutos, enfríalos y
# pélalos» — una clara suelta no se hierve ni se pela. El LLM escribió «hierve 6 huevos… pélalos» y la regla 2 del
# contrato (`egg_forms_step_sync`, lote 24: los pasos siguen a la forma de la lista) convirtió los enteros de las claras en
# «claras de huevo». Corpus de 315 planes: 10 comidas («hierve 6 claras de huevo 8-10 min, pélalos», «cuece el huevo
# entero y 4 claras en agua hirviendo…; enfría, pela y reserva»). El paso conserva la cantidad de la lista (el sync la
# vigila) y dice cómo: las claras se hierven dentro de su huevo entero y la yema se quita al pelar. Nunca en una oración
# que bate, mezcla, revuelve, cuaja, licúa o separa. tooltip-anchor: P1-PLAN-LOTE-390
_CLARAS_HERVIDAS_390_RE = re.compile(
    r"(?P<cl>\b(?:\d+(?:[.,]\d+)?|[½¼¾⅓⅔]|\d+\s*[½¼¾])\s+(?:g\s+de\s+)?(?P<n>claras?)(?:\s+de\s+huevos?)?\b)"
    r"(?P<par>\s*\([^)]*\))?", re.IGNORECASE)
_HIERVE_390_RE = re.compile(r"\bhierv\w*|\bhi[eé]rvel\w*|\bhirviendo\b|\bsancoch\w*|\bherv(?:ir|id[oa]s?)\b",  # 405: «a hervir»
                            re.IGNORECASE)
_PELA_390_RE = re.compile(r"\bp[eé]l(?:al[oa]s?|arl[oa]s?)\b|\bpela(?=\s*(?:y\b|,|;|\.|$))|\bc[aá]scara\b|\bduros?\b",
                          re.IGNORECASE)
_OTRO_VERBO_390_RE = re.compile(r"\b(?:bate|batir|batid[oa]s?|mezcla|revuelve|agrega|añade|incorpora|vierte|licúa|licua|"
                                r"cuaja|cuájal\w*|separa|separar)\b", re.IGNORECASE)


def claras_en_su_huevo(meal) -> int:
    """Nº de oraciones corregidas (una por plato); 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        if any(isinstance(p, str) and "dentro de su huevo entero" in p for p in rec):
            return 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p):
                continue
            partes = re.split(r"(?<=\.)\s+", p)
            for k, frase in enumerate(partes):
                mm = _CLARAS_HERVIDAS_390_RE.search(frase)
                if not mm or not _HIERVE_390_RE.search(frase) or _OTRO_VERBO_390_RE.search(frase):
                    continue
                # «hierve 1 huevo y 3 claras de huevo 10 minutos» (se pelan en el Montaje): el verbo rige a las claras
                directo = re.search(r"\b(?:hierv[ea]|cuece)\s+(?:(?:\d+|[½¼¾])\s+huevos?\s+(?:enteros?\s+)?y\s+)?$",
                                    frase[:mm.start()], re.IGNORECASE)
                if not directo and not _PELA_390_RE.search(frase[mm.end():]):
                    continue
                uno = mm.group("n").lower() == "clara"
                dentro = f" ({'hiérvela' if uno else 'hiérvelas'} dentro de su huevo entero, con cáscara)"
                par = mm.group("par") or ""
                if par and re.search(r"c[aá]scara", par, re.IGNORECASE):
                    nueva = frase[:mm.end("cl")] + dentro + frase[mm.end():]         # «(se pesan sin cáscara)» se sustituye
                else:
                    nueva = frase[:mm.end()] + dentro + frase[mm.end():]
                nueva = nueva.rstrip()
                if nueva.endswith((".", ";")):
                    nueva = nueva[:-1]
                # sin «el/los huevo(s)»: con sólo claras en la lista, la regla 2 los reescribiría a «la(s) clara(s)»
                cola = ("quita la yema del huevo pelado: usa solo la clara" if uno
                        else "quita la yema de cada huevo pelado: usa solo la clara")
                partes[k] = f"{nueva}; {cola}."
                rec[i] = " ".join(partes)
                meal["recipe"] = rec
                meal.pop("_display", None)
                return 1
        return 0
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-391 · 2026-09-26] Lo que se desgrana es la granada ─────────────────────────────────────────────────────
# Plan de la batería real (adulto mayor con HTA): «desgrana 45 g de guineo», «desgrana 65 g de piña» y, en el corpus,
# «¼ taza de guineo desgranada» o «desgrana guineo y desecha la cáscara blanca»: la sustitución que saca la GRANADA del
# plato (interacciones del perfil) cambia el alimento y deja su verbo y su adjetivo — la cáscara blanca es la de la
# granada. Corpus de 315 planes: ~25 menciones, casi todas en perfiles con HTA, estatina o IMAO. Una fruta que no se
# desgrana se pela y se corta; la uva, el maíz, los guandules y la granada sí se desgranan y no se tocan.
# tooltip-anchor: P1-PLAN-LOTE-391
_FRUTA_391 = (r"guineos?(?:\s+maduros?)?|bananas?|pl[aá]tanos?(?:\s+maduros?)?|piñas?|lechosas?|papayas?|mangos?|"
              r"mel[oó]n(?:es)?|sand[ií]as?|peras?|manzanas?|chinolas?|toronjas?|naranjas?|mandarinas?")
_DESGRANA_391_RE = re.compile(
    r"\b(?P<v>[Dd]esgrana)\s+(?P<q>(?:(?:\d+(?:[.,]\d+)?|[½¼¾⅓⅔]|\d+\s*[½¼¾])\s*(?:(?:g|gr|gramos|tazas?)\s+de\s+)?|"
    r"(?:suficiente|el|la|los|las)\s+)?)(?P<f>" + _FRUTA_391 + r")\b", re.IGNORECASE)
_DESGRANADO_391_RE = re.compile(r"\b(?P<f>" + _FRUTA_391 + r")\s+desgranad[oa]s?\b", re.IGNORECASE)
_CASCARA_BLANCA_391_RE = re.compile(r"\s+y\s+desecha\s+la\s+c[aá]scara\s+blanca\b", re.IGNORECASE)


def _forma_391(fruta: str) -> str:
    return "en ruedas" if re.match(r"(?:guineo|banana|pl[aá]tano)", fruta, re.IGNORECASE) else "en cubos"


def lo_que_se_desgrana(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p):
                continue
            s = _DESGRANA_391_RE.sub(lambda m: ("Pela y corta " if m.group("v")[0].isupper() else "pela y corta ")
                                     + (m.group("q") or "") + m.group("f"), p)
            s = _DESGRANADO_391_RE.sub(lambda m: f"{m.group('f')} {_forma_391(m.group('f'))}", s)
            if s != p:
                s = _CASCARA_BLANCA_391_RE.sub("", s)          # la cáscara blanca era la de la granada
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-392 · 2026-09-26] El pollo y el pavo se cocinan a 74 °C ───────────────────────────────────────────────
# Batería REAL sobre el 379 (adulto mayor con HTA, día 1): «Marina pechuga de pollo…; cocina el filete 4 minutos por
# lado, hasta 63 °C en el centro» — el cambio de proteína repetida (`_protein_autofix_applied: pescado->pollo`) cambia el
# alimento y deja el punto del PESCADO. 63 °C es seguro para el pescado; el ave necesita 74 °C (USDA/FSIS, también
# molida). Corpus de 315 planes: 18 comidas con pollo o pavo «a 63 °C», 11 por ese cambio y 7 escritas así por el
# modelo — cinco en perfiles de EMBARAZO —, incluida la nota ⚠️ reescrita: «cocina pechuga de pollo hasta que… se separe
# en lascas (63 °C en el centro)». En la cola: en una frase del ave (sin pescado ni huevo) la temperatura interna por
# debajo de 74 °C sube a 74 °C y la señal del pescado («se separe en lascas») pasa a «sin partes rosadas». La del huevo
# (≥71 °C) y las del horno (≥150 °C) no se tocan. tooltip-anchor: P1-PLAN-LOTE-392
_AVE_392_RE = re.compile(r"\b(?:pollo|pechugas?|pavo|muslos?|contramuslos?|gallina)\b")
_NO_CARNE_392_RE = re.compile(r"\b(?:caldo|consom\w*|cubitos?|sazon\w*|polvo)\b")
_PEZ_392_RE = re.compile(r"\b(?:pescados?|tilapia|merluza|salmon|atun|sardinas?|bacalao|camarones?|mero|chillo|dorado|"
                         r"corvina|pargo|mariscos?|langostinos?|calamar(?:es)?|lascas? de pescado)\b")
_HUEVO_392_RE = re.compile(r"\b(?:huevos?|claras?|yemas?)\b")
_TEMP_392_RE = re.compile(r"(?<![\d.,])(?P<t>[5-7]\d)(?P<sep>\s*)°\s*C")   # «160 °C» del horno no es «60 °C»
_LASCAS_392_RE = re.compile(r"\s+y\s+se\s+separe\s+en\s+(?:lascas|l[aá]minas)\b")


def ave_a_74(meal) -> int:
    """Nº de frases corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = [_sa(str(x).lower()) for x in (meal.get("ingredients") or [])]
        if not any(_AVE_392_RE.search(l) and not _NO_CARNE_392_RE.search(l) for l in lineas):
            return 0
        pez_en_lista = any(_PEZ_392_RE.search(l) for l in lineas)
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or "°" not in p and "separe en" not in p:
                continue
            ave_en_paso = bool(_AVE_392_RE.search(_sa(p.lower())))
            trozos = re.split(r"((?<=[.;])\s+)", p)
            cambio = False
            for k in range(0, len(trozos), 2):
                c = trozos[k]
                sc = _sa(c.lower())
                if _HUEVO_392_RE.search(sc) or _PEZ_392_RE.search(sc):
                    continue
                if not (_AVE_392_RE.search(sc) or (ave_en_paso and not pez_en_lista)):
                    continue
                nuevo = _TEMP_392_RE.sub(lambda m: (f"74{m.group('sep')}°C" if int(m.group("t")) < 74 else m.group(0)), c)
                if nuevo != c or _AVE_392_RE.search(sc):
                    nuevo = _LASCAS_392_RE.sub(", sin partes rosadas", nuevo)
                if nuevo != c:
                    trozos[k] = nuevo
                    cambio = True
            if cambio:
                rec[i] = "".join(trozos)
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-393 · 2026-09-26] «Escurre 355 g de garbanzos cocidos» con los secos en la lista trae su cocción ───────
# Batería REAL sobre el 379 (estatina, día 1): la lista compra «1 taza de garbanzos secos» y los pasos dicen «escurre 355 g
# de garbanzos cocidos… incorpora los garbanzos… y cocina 10 min». El lote 375 añade la «💡 Cocción previa» cuando el
# detector V7c acusa, pero V7c da por cocidos los garbanzos que el paso «cocina 10 min» — y 10 min no cuecen un garbanzo
# seco (60-90 min tras remojo). Aquí, sin V7c: si la lista compra el alimento seco/crudo, un paso lo usa «cocido» y ningún
# paso lo remoja ni lo cuece de verdad (≥20 min, «según el paquete», «hasta que esté tierno»), sale la misma nota del 375.
# tooltip-anchor: P1-PLAN-LOTE-393
_SECO_393_RE = re.compile(r"\b(?P<f>garbanzos?|lentejas?|habichuelas?|frijol(?:es)?|gandules|guandules|habas?|"
                          r"arroz(?:\s+integral)?|quinoa|bulgur|cebada)\b(?P<resto>[^|]*)")
_REAL_393_RE = re.compile(r"\bremoj\w*|seg[uú]n\s+(?:las\s+(?:instrucciones|indicaciones)\s+del\s+|el\s+)?(?:paquete|empaque)|"
                          r"hasta\s+que\s+(?:est[eé]n?|queden?)\s+tiern")
_MIN_393_RE = re.compile(r"(\d{1,3})\s*(?:-\s*(\d{1,3})\s*)?min")
_COCCION_393_RE = re.compile(r"\b(?:hierv\w*|cuec\w*|cuece|cocin\w*|sancoch\w*)")


def seco_usado_cocido(meal) -> int:
    """Nº de notas añadidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        pasos = [_sa(str(p).lower()) for p in rec if isinstance(p, str) and not _es_nota(p)]
        notas_txt = " ".join(_sa(str(p).lower()) for p in rec if isinstance(p, str))
        notas = []
        for linea in (meal.get("ingredients") or []):
            ln = _sa(str(linea).lower())
            mm = _SECO_393_RE.search(ln)
            if not mm or not re.search(r"\b(?:sec[oa]s?|crud[oa]s?)\b", mm.group("resto")):
                continue
            food = mm.group("f")
            raiz = re.sub(r"(?:es|s)$", "", food.split()[0]) if food.split()[0] not in ("arroz", "bulgur") else food.split()[0]
            if "coccion previa" in notas_txt and raiz in notas_txt[notas_txt.find("coccion previa"):]:
                continue
            usa_cocido = any(re.search(r"\b" + re.escape(raiz) + r"\w*\s+(?:\w+\s+){0,2}cocid[oa]s?\b", p) for p in pasos)
            if not usa_cocido:
                continue
            real = False
            for p in pasos:
                for cl in re.split(r"[.;](?!\d)", p):                  # [P1-PLAN-LOTE-52] «0.5 taza» no parte la frase
                    if not re.search(r"\b" + re.escape(raiz), cl):
                        continue
                    if _REAL_393_RE.search(cl):
                        real = True
                    elif _COCCION_393_RE.search(cl):
                        tiempos = [max(int(t.group(1)), int(t.group(2) or 0)) for t in _MIN_393_RE.finditer(cl)]
                        if tiempos and max(tiempos) >= 20:
                            real = True
            if real:
                continue
            plantilla = next((t for k, t in _COCCION_PREVIA_375 if re.search(r"\b" + re.escape(k), food)), None)
            if not plantilla:
                continue
            nombre = re.sub(r"\s+(?:sec[oa]s?|crud[oa]s?)\b", "", str(linea).lower())
            nombre = re.sub(r"^\s*[\d½¼¾⅓⅔.,/]+\s*(?:tazas?|g|gr|gramos|cdas?|cucharadas?)?\s*(?:de\s+)?", "", nombre).strip()
            texto = "💡 Cocción previa: " + plantilla.format(n=nombre or food)
            if any(k in food for k in _LEGUMBRE_375):
                texto += " (puedes cocinar la tanda de varios días y guardarla en la nevera hasta 4 días)"
            texto += "."
            if texto not in rec and texto not in notas:
                notas.append(texto)
        if not notas:
            return 0
        i_mise = next((i for i, s in enumerate(rec) if isinstance(s, str) and s.strip().lower().startswith("mise en place")), None)
        pos = (i_mise + 1) if i_mise is not None else 0
        rec[pos:pos] = notas
        meal["recipe"] = rec
        meal.pop("_display", None)
        return len(notas)
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-394 · 2026-09-26] Un hervor no lo mide la plancha de otra frase ───────────────────────────────────────
# Batería REAL sobre el 379 (estatina, día 3): «cocina la cebada en agua según las instrucciones del paquete, hasta que
# esté tierna, aproximadamente 3-4 min por lado a fuego medio-alto». El recorte de tiempos implausibles
# (`_clamp_recipe_time_temp_outliers`) elige la técnica con el PASO entero: la otra frase dice «sartén», el techo de la
# plancha es 30 min y los 35-40 min de la cebada pasan al «3-4 min por lado» de la plancha. Corpus: 7 hervores así —
# «cocina el bulgur en agua hirviendo 1-2 min de licuado a velocidad altautos» (el recorte partía «minutos» y dejaba
# «utos»; su regex se corrige en el orquestador). En la cola: en la frase que hierve un alimento conocido, la plantilla de
# otra técnica pasa al tiempo de hervor de ese alimento, y el «utos» huérfano sale. tooltip-anchor: P1-PLAN-LOTE-394
_HERVOR_394 = (("arroz integral", "35-45 min"), ("arroz", "15-20 min"), ("cebada", "30-40 min"), ("quinoa", "12-15 min"),
               ("bulgur", "10-12 min"), ("pasta", "8-10 min"), ("fideo", "8-10 min"), ("espagueti", "8-10 min"),
               ("lenteja", "20-25 min"), ("garbanzo", "60-90 min"), ("habichuela", "60-90 min"), ("frijol", "60-90 min"),
               ("yuca", "20-25 min"), ("platano", "20-25 min"), ("guineo", "15-20 min"), ("papa", "15-20 min"),
               ("batata", "15-20 min"), ("mapuey", "20-25 min"), ("yautia", "15-20 min"), ("name", "20-25 min"),
               ("auyama", "10-15 min"), ("zanahoria", "10-12 min"))
_HIERVE_394_RE = re.compile(r"\ben agua\b|\bhierv\w*|\bherv\w*|\bcuec\w*|\bcuece\b|\bsancoch\w*")
_AJENA_394_RE = re.compile(r"(?P<pre>(?:unos|aproximadamente|durante)\s+)?\d+\s*-\s*\d+\s+min\s+(?:por\s+lado\s+a\s+fuego\s+"
                           r"medio(?:-alto)?|de\s+licuado\s+a\s+velocidad\s+alta|hasta\s+dorar)(?:utos\b)?")
_UTOS_394_RE = re.compile(r"(velocidad alta|fuego medio-alto|fuego medio|hasta dorar|al vapor|en agua hirviendo|a 180 °C)utos\b")


def hervor_con_su_tiempo(meal) -> int:
    """Nº de frases corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p):
                continue
            trozos = re.split(r"((?<=[.;])\s+)", p)
            for k in range(0, len(trozos), 2):
                c = trozos[k]
                nuevo = _UTOS_394_RE.sub(r"\1", c)
                sc = _sa(nuevo.lower())
                if _HIERVE_394_RE.search(sc) and _AJENA_394_RE.search(nuevo):
                    tiempo = next((t for k2, t in _HERVOR_394 if re.search(r"\b" + k2, sc)), None)
                    if tiempo:
                        nuevo = _AJENA_394_RE.sub(lambda m: (m.group("pre") or "") + tiempo, nuevo, count=1)
                if nuevo != c:
                    trozos[k] = nuevo
                    n += 1
            if n:
                rec[i] = "".join(trozos)
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-395 · 2026-09-26] El yogur no se corta en cubos ─────────────────────────────────────────────────────────
# Batería REAL sobre el 379 (adulto mayor con HTA, día 2): «corta 1 taza de yogurt natural sin azúcar (231 g) bajo en
# sodio en cubos… sirve los bollitos con los cubos de yogurt» — el ajuste de sodio cambió el queso del plato por yogur y
# dejó la forma del queso (corpus: «yogurt … en láminas» dos veces más). Una forma de corte sobre el yogur sale: «corta»
# pasa a «mide» y «los cubos de yogurt» a «el yogurt». tooltip-anchor: P1-PLAN-LOTE-395
_FORMA_395 = r"(?:cubos|cubitos|l[aá]minas|lonjas|rodajas|tiras|dados|trocitos)"
_CORTA_YOG_395_RE = re.compile(r"\b(?P<v>[Cc]orta|[Rr]ebana|[Dd]esmenuza|[Rr]alla|[Tt]rocea)\s+(?P<q>[^.;,]*?\byogu?rt?\w*[^.;,]*?)"
                               r"\s+en\s+" + _FORMA_395 + r"\b")
_LOS_CUBOS_395_RE = re.compile(r"\b(?:los|las)\s+" + _FORMA_395 + r"\s+de\s+(?P<y>yogu?rt?\w*)", re.IGNORECASE)
_YOG_EN_FORMA_395_RE = re.compile(
    r"(?P<y>\byogu?rt?\w*(?:\s+(?:natural|griega?o?|sin|az[uú]car|descremad[oa]|enter[oa]|bajo|en|sodio|light|"
    r"\(\d+\s*g\)))*)\s+en\s+" + _FORMA_395 + r"\b", re.IGNORECASE)


def yogur_sin_forma(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p) or not re.search(r"yogu?rt?", p, re.IGNORECASE):
                continue
            s = _CORTA_YOG_395_RE.sub(
                lambda m: m.group(0) if " y " in m.group("q") or re.search(r"\bqueso\b", m.group("q"), re.IGNORECASE)
                else ("Mide " if m.group("v")[0].isupper() else "mide ") + m.group("q"), p)
            s = _LOS_CUBOS_395_RE.sub(lambda m: "el " + m.group("y"), s)
            def _sin_forma(m, s_=s):
                clausula = s_[max(s_.rfind(".", 0, m.start()), s_.rfind(";", 0, m.start()), s_.rfind(",", 0, m.start())) + 1:m.end()]
                if re.search(r"\bqueso\b|helad|congel", clausula, re.IGNORECASE):   # «corta el queso y el yogurt helado»
                    return m.group(0)
                return m.group("y")
            s = _YOG_EN_FORMA_395_RE.sub(_sin_forma, s)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-396 · 2026-09-26] El queso y el cilantro no se hierven «hasta que ablanden» ────────────────────────────
# Batería REAL sobre el 379 (adulto mayor con HTA, día 3): «Incorpora cilantro picado en agua hasta que ablanden e
# incorpóralo al plato». La frase es la del cerrador para una LEGUMBRE (`_closer_protein_step_text`: «Cocina {x} en agua
# hasta que ablanden e incorpóralo al plato»); una sustitución posterior cambia el alimento y el verbo y deja el hervor.
# Corpus de 315 planes: 15 frases así — «Incorpora queso blanco fresco en agua hasta que ablanden» (11), cilantro, aceite
# de oliva, «Añade agua en agua». Lo que no se hierve deja el hervor: si otro paso ya lo usa, la frase sobra; si no, queda
# «Incorpora {x} al plato.». Legumbres, granos y verduras conservan la suya. tooltip-anchor: P1-PLAN-LOTE-396
_ABLANDEN_396_RE = re.compile(
    r"(?P<v>Cocina|Incorpora|Añade|Agrega)\s+(?P<obj>[^.;:]{2,50}?)\s+en\s+agua\s+hasta\s+que\s+ablanden\s+e\s+"
    r"incorp[oó]ral[oa]s?\s+al\s+plato(?:\s*\(~[^)]*\))?\.?", re.IGNORECASE)
_NO_SE_HIERVE_396_RE = re.compile(r"\b(?:quesos?|cilantro|perejil|aceite|agua|yogu?rt?\w*|sal|pimienta|oregano|comino|canela|"
                                  r"mantequilla|aguacate|limon|jugo|vinagre|miel|cottage|ricotta|mozzarella)\b")


def no_se_hierve(meal) -> int:
    """Nº de frases corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i in range(len(rec)):
            p = rec[i]
            if not isinstance(p, str):
                continue
            mm = _ABLANDEN_396_RE.search(p)
            if not mm:
                continue
            obj = mm.group("obj").strip()
            on = _sa(obj.lower())
            if not _NO_SE_HIERVE_396_RE.search(on):
                continue
            toks = [t for t in _toks(on) if t not in ("blanco", "blanca", "fresco", "fresca", "picado", "picada")] or _toks(on)
            resto = " ".join(_sa(str(q).lower()) for j, q in enumerate(rec) if j != i and isinstance(q, str))
            resto += " " + _sa((p[:mm.start()] + " " + p[mm.end():]).lower())
            if (toks and any(re.search(r"\b" + re.escape(t), resto) for t in toks)) or on == "agua":
                nuevo = ""
            else:
                nuevo = f"Incorpora {obj[:1].lower() + obj[1:]} al plato."
            s = re.sub(r"\s{2,}", " ", (p[:mm.start()] + nuevo + p[mm.end():])).strip()
            rec[i] = s
            n += 1
        if n:
            meal["recipe"] = [x for x in rec if not (isinstance(x, str) and re.fullmatch(r"\s*[^:.]{1,40}:\s*", x))
                              and x != ""]
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-397 · 2026-09-26] «El filete de 195 g» pesa lo que dice la lista ──────────────────────────────────────
# Batería REAL sobre el 379 (adulto mayor con HTA): «seca el filete de pescado blanco de 195 g» con «1 filete de pescado
# (≈160 g)» en la lista, y «seca pechuga de pollo de 150 g» con «½ pechuga de pollo (≈100 g)». Los lotes 310/370 siguen
# la pista «(N g)» de la lista; la forma «el filete … DE N g» no la sigue nadie (corpus: 16 de 22 discrepan >15 %). Con UNA
# línea de la lista para esa pieza y su peso, el paso pasa a ese peso (por pieza si el paso cuenta varias o dice «cada»).
# tooltip-anchor: P1-PLAN-LOTE-397
_PIEZA_DE_G_397_RE = re.compile(
    r"(?:(?P<c>\d+(?:[.,]\d+)?|[½¼¾⅓⅔]|\d+\s*[½¼¾])\s+)?\b(?P<k>filetes?|pechugas?)\b(?P<mid>[^.;()]{0,40}?)\bde\s+"
    r"(?P<g>\d+(?:[.,]\d+)?)\s*g\b(?P<cada>\s+cada\s+un[oa]|\s+c/u)?", re.IGNORECASE)
_PROT_397_RE = re.compile(r"\b(pollo|pavo|res|cerdo|pescado|tilapia|salmon|merluza|mero|dorado|chillo|atun|corvina|pargo)\b")
_G_LINEA_397_RE = re.compile(r"\(\s*≈?\s*(\d+(?:[.,]\d+)?)\s*g\s*\)|^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\b")


def _num_397(s) -> float:
    s = str(s).strip()
    fr = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
    tot = 0.0
    for ch, v in fr.items():
        if ch in s:
            tot += v
            s = s.replace(ch, "")
    s = s.strip()
    return tot + (float(s.replace(",", ".")) if s else 0.0)


def peso_de_la_pieza_en_el_paso(meal) -> int:
    """Nº de pesos corregidos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = [str(x) for x in (meal.get("ingredients") or [])]
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p):
                continue

            def _sub(m):
                nonlocal n
                k = _sa(m.group("k").lower()).rstrip("s")
                cands = [l for l in lineas if re.search(r"\b" + k + r"s?\b", _sa(l.lower()))]
                if len(cands) != 1:
                    return m.group(0)
                linea_n = _sa(cands[0].lower())
                for prot in _PROT_397_RE.findall(_sa(m.group("mid").lower())):   # «pechuga de pavo» no es la de pollo
                    if prot not in linea_n and not (prot == "pescado" and _PEZ_392_RE.search(linea_n)):
                        return m.group(0)
                mg = _G_LINEA_397_RE.search(cands[0])
                if not mg:
                    return m.group(0)
                total = float((mg.group(1) or mg.group(2)).replace(",", "."))
                c = _num_397(m.group("c")) if m.group("c") else 1.0
                por_pieza = bool(m.group("cada")) or c > 1
                objetivo = total / c if (por_pieza and c > 0) else total
                g = float(m.group("g").replace(",", "."))
                if objetivo <= 0 or abs(g - objetivo) / objetivo <= 0.10:
                    return m.group(0)
                n += 1
                nuevo_g = f"{round(objetivo):d}"
                return m.group(0)[:m.start("g") - m.start()] + nuevo_g + m.group(0)[m.end("g") - m.start():]

            s = _PIEZA_DE_G_397_RE.sub(_sub, p)
            if s != p:
                rec[i] = s
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-398 · 2026-09-26] Tras el verbo, el alimento va en minúscula ───────────────────────────────────────────
# Batería REAL sobre el 379 (adulto mayor con HTA): «Añade Filete de pescado blanco al guiso», «Cocina Filete de pescado
# blanco a la plancha», «Incorpora también Queso blanco», «añade Sal al gusto» — el cerrador y las sustituciones copian el
# nombre de la fila del catálogo con su mayúscula. El lote 373 lo baja tras «, », «y», «de», «con»; no tras el verbo que
# lo gobierna. Corpus de 317 planes: 117 («Pechuga» 34, «Filete» 17, «Tilapia» 12, «Plátano» 10…). Mismas excepciones del
# 373: nombre propio, marca de dos palabras con mayúscula, «Toque de Fuego». Nunca las notas. tooltip-anchor: P1-PLAN-LOTE-398
_VERBO_MAYUS_398_RE = re.compile(
    r"\b(?P<v>(?i:añade|agrega|incorpora|cocina|sirve|mezcla|coloca|pon|usa|corta|pica|mide|saltea|hierve|tuesta|marina|"
    r"sazona|sella|también|acompaña con|reparte|calienta|espolvorea|rocía|unta|ralla|lava|pela|desmenuza|licúa|bate|"
    r"vierte))\s+(?P<l>[A-ZÁÉÍÓÚÑ])(?P<r>[a-záéíóúñü]{2,})\b(?P<sig>\s+[A-ZÁÉÍÓÚÑ])?")


def _minuscula_398(mm) -> str:
    if mm.group("sig") or _sa((mm.group("l") + mm.group("r")).lower()) in _PROPIOS_373:
        return mm.group(0)
    return mm.group(0)[:mm.start("l") - mm.start()] + mm.group("l").lower() + mm.group(0)[mm.start("r") - mm.start():]


def minuscula_tras_verbo(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if _es_nota(p):
                continue
            s = _VERBO_MAYUS_398_RE.sub(_minuscula_398, p)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-399 · 2026-09-26] «1 rebanadas», «1 tortas pequeñas»: con uno, en singular ─────────────────────────────
# Batería REAL sobre el 379 (estatina): «mide ½ aguacate y 1 rebanadas de pan integral», «1 ajíes morrones» en la lista;
# en el corpus de 317 planes, 84 pasos con «1 rebanadas» y 23 líneas de lista («1 tortas pequeñas de casabe» 13,
# «1 dátiles sin hueso» 6, «1 ajíes morrones» 3): la unidad del catálogo viene en plural y el conteo 1 no la cambia. Con
# «1» exacto (no «1½», no «11», no «0,1»), el sustantivo y sus adjetivos conocidos pasan a singular, en pasos y en la
# lista. Cantidades, alimentos y macros intactos. tooltip-anchor: P1-PLAN-LOTE-399
_SING_399 = {"rebanadas": "rebanada", "tortas": "torta", "dátiles": "dátil", "datiles": "dátil", "ajíes": "ají", "ajies": "ají",
             "rábanos": "rábano", "rabanos": "rábano", "pedazos": "pedazo", "piezas": "pieza", "porciones": "porción",
             "tazas": "taza", "cucharadas": "cucharada", "cucharaditas": "cucharadita", "unidades": "unidad", "filetes": "filete",
             "pechugas": "pechuga", "huevos": "huevo", "claras": "clara", "tomates": "tomate", "limones": "limón",
             "guineos": "guineo", "plátanos": "plátano", "lonjas": "lonja", "hojas": "hoja", "ramitas": "ramita",
             "dientes": "diente", "tallos": "tallo", "sobres": "sobre", "latas": "lata", "vasos": "vaso"}
_ADJ_SING_399 = {"pequeñas": "pequeña", "pequeños": "pequeño", "medianas": "mediana", "medianos": "mediano",
                 "grandes": "grande", "morrones": "morrón", "maduros": "maduro", "maduras": "madura", "verdes": "verde",
                 "integrales": "integral", "finas": "fina", "finos": "fino", "gruesas": "gruesa", "gruesos": "grueso",
                 "enteros": "entero", "enteras": "entera", "frescos": "fresco", "frescas": "fresca", "rojos": "rojo",
                 "rojas": "roja", "cubanelas": "cubanela", "dulces": "dulce", "picados": "picado", "picadas": "picada",
                 "pelados": "pelado", "peladas": "pelada", "cortados": "cortado", "cortadas": "cortada",
                 "rallados": "rallado", "ralladas": "rallada", "tostados": "tostado", "tostadas": "tostada"}
_UNO_PLURAL_399_RE = re.compile(
    r"(?<![\d.,/½¼¾⅓⅔])\b1\s+(?P<n>" + "|".join(sorted(_SING_399, key=len, reverse=True)) + r")\b"
    r"(?P<adj>(?:\s+(?:" + "|".join(sorted(_ADJ_SING_399, key=len, reverse=True)) + r")\b)*)")


def _singular_399(mm) -> str:
    n = mm.group("n")
    sing = _SING_399.get(n) or _SING_399.get(n.lower(), n)
    adj = re.sub(r"\S+", lambda a: _ADJ_SING_399.get(a.group(0), a.group(0)), mm.group("adj") or "")
    return f"1 {sing}{adj}"


def uno_en_singular(meal) -> int:
    """Nº de textos corregidos (pasos + líneas de la lista); 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        n = 0
        for campo in ("recipe", "ingredients"):
            xs = meal.get(campo)
            if not isinstance(xs, list):
                continue
            for i, x in enumerate(xs):
                if not isinstance(x, str) or (campo == "recipe" and _es_nota(x)):
                    continue
                s = _UNO_PLURAL_399_RE.sub(_singular_399, x)
                if s != x:
                    xs[i] = s
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-401 · 2026-09-26] La frase del guiso concuerda ─────────────────────────────────────────────────────────
# Batería REAL sobre el 379 (adulto mayor con HTA, día 2): «Agrega yautía al guiso y cocínalos a fuego medio 12-15 minutos,
# hasta que esté cocidos por dentro; Incorpóralos con cuidado para no deshacer el resto». La plantilla del cerrador para
# guisos (`_closer_protein_step_text`, stewy) concuerda el verbo con el alimento pero deja «esté» en singular y abre con
# mayúscula tras el punto y coma. Corpus de 317 planes: «esté cocidos/as» 18, «; Incorpóralo…» 54. tooltip-anchor:
# P1-PLAN-LOTE-401
_ESTE_PLURAL_401_RE = re.compile(r"\bhasta que esté (cocid[oa]s)\b")
_PC_MAYUS_401_RE = re.compile(r"; (Incorpóral|Sírvel|Agrégal|Añádel|Mézclal)(\w*)")


def guiso_concuerda(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p):
                continue
            s = _ESTE_PLURAL_401_RE.sub(r"hasta que estén \1", p)
            s = _PC_MAYUS_401_RE.sub(lambda m: "; " + m.group(1)[0].lower() + m.group(1)[1:] + m.group(2), s)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-402 · 2026-09-26] El pescado del guiso no se cuece 15 minutos ──────────────────────────────────────────
# La misma batería: «Añade Filete de pescado blanco al guiso y cocínalo a fuego medio 12-15 minutos, hasta que esté cocido
# por dentro» — la plantilla del cerrador para guisos da a toda proteína el tiempo del pollo; 12-15 min deshacen el
# pescado (y el propio paso pide «con cuidado para no deshacer el resto»). Corpus: 13. Pescado: 5-7 min, hasta que se
# desmenuce fácilmente (63 °C al centro); camarones y mariscos: 2-3 min, rosados y opacos. tooltip-anchor: P1-PLAN-LOTE-402
_GUISO_PEZ_402_RE = re.compile(
    r"(?P<pre>\b(?:Añade|Agrega|Incorpora)\s+(?P<food>[^.;]{2,50}?)\s+al\s+guiso\s+y\s+cocínal(?P<cl>[oa]s?)\s+a\s+fuego\s+medio\s+)"
    r"12-15\s+minutos,\s+hasta\s+que\s+est(?:é|én)\s+cocid[oa]s?\s+por\s+dentro", re.IGNORECASE)


def pescado_del_guiso(meal) -> int:
    """Nº de frases corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0

        def _sub(m):
            nonlocal n
            fn = _sa(m.group("food").lower())
            if _CARNE_377_RE.search(fn):
                return m.group(0)                       # «filete de pollo», «filete de res»: el tiempo del ave/la carne se queda
            if _MARISCO_377_RE.search(fn):
                n += 1
                return m.group("pre") + "2-3 minutos, hasta que estén rosados y opacos"
            if _PESCADO_377_RE.search(fn) or _PEZ_392_RE.search(fn):
                n += 1
                return m.group("pre") + "5-7 minutos, hasta que se desmenuce fácilmente (63 °C al centro)"
            return m.group(0)

        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p):
                continue
            s = _GUISO_PEZ_402_RE.sub(_sub, p)
            if s != p:
                rec[i] = s
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-403 · 2026-09-26] «tuesta las 1 rebanada» → «tuesta la rebanada» ──────────────────────────────────────
# Batería REAL sobre el 399 (perfil del dueño, día 1): «tuesta las 1 rebanada de pan integral» — el 399 dejó la unidad en
# singular pero el artículo plural y el número quedaron (corpus: 33, 26 de «las 1 rebanada»). Con artículo y una sola
# pieza, como en el 376: sin número y con el artículo en singular (género por la terminación del sustantivo).
# tooltip-anchor: P1-PLAN-LOTE-403
_ART_UNO_403_RE = re.compile(r"\b(?P<a>[Ll]os|[Ll]as|[Uu]nos|[Uu]nas)\s+1\s+(?P<n>[a-záéíóúñ]+)\b")


def _art_uno_403(m) -> str:
    n = m.group("n")
    fem = n.endswith(("a", "ción", "sión", "dad")) or n in ("cdta", "cda")
    a = m.group("a")
    indef = a.lower().startswith("un")
    art = ("una" if fem else "un") if indef else ("la" if fem else "el")
    if a[0].isupper():
        art = art.capitalize()
    return f"{art} {n}"


def articulo_de_uno(meal) -> int:
    """Nº de textos corregidos (pasos + lista); 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        n = 0
        for campo in ("recipe", "ingredients"):
            xs = meal.get(campo)
            if not isinstance(xs, list):
                continue
            for i, x in enumerate(xs):
                if not isinstance(x, str) or (campo == "recipe" and _es_nota(x)):
                    continue
                s = _ART_UNO_403_RE.sub(_art_uno_403, x)
                if s != x:
                    xs[i] = s
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-404 · 2026-09-26] «1 de cebolla» → «1 cebolla» ────────────────────────────────────────────────────────
# La misma batería: la lista trae «1 de cebolla roja», «1 de cebolla» (corpus: 26 líneas, todas cebolla): el humanizador
# escribe «½ de cebolla» para la fracción y, cuando la cantidad redondea a 1, deja el «de». Con «1» exacto, sin «de».
# tooltip-anchor: P1-PLAN-LOTE-404
_UNO_DE_404_RE = re.compile(r"(?<![\d.,/½¼¾⅓⅔])\b1\s+de\s+(?!cada\b|los\b|las\b|el\b|la\b)(?=[a-záéíóúñ])")


def uno_sin_de(meal) -> int:
    """Nº de textos corregidos (pasos + lista); 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        n = 0
        for campo in ("recipe", "ingredients"):
            xs = meal.get(campo)
            if not isinstance(xs, list):
                continue
            for i, x in enumerate(xs):
                if not isinstance(x, str) or (campo == "recipe" and _es_nota(x)):
                    continue
                s = _UNO_DE_404_RE.sub("1 ", x)
                if s != x:
                    xs[i] = s
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-405 · 2026-09-26] Las claras de la lista también se cocinan ───────────────────────────────────────────
# Batería REAL sobre el 399 (embarazo, día 2): «3 huevos» y «3 claras de huevo» en la lista y el paso «hierve el huevo en
# agua durante 10-12 min»: las claras que el tope de yemas añadió (lote 235) no las cocina ningún paso. Corpus de 317
# planes: 119 platos de huevo duro así; 12 «pon 3 huevos y 2 claras de huevo a hervir… y pela» que el 390 no veía («a
# hervir»); y 17 con sólo claras en la lista y el cerrador «Cocina huevo a la plancha o hervido y sírvelo como proteína»
# (una clara suelta no se hierve). (1) El hervor del 390 reconoce «hervir»; (2) la frase que hierve los huevos suma, sin
# cifras que la regla 2 reescriba, «un huevo por cada clara… quítales la yema»; (3) con sólo claras, el cerrador las
# cuaja en la sartén. Nunca si un paso ya cocina las claras. tooltip-anchor: P1-PLAN-LOTE-405
_CLARA_COCINADA_405_RE = re.compile(
    r"(?:\bbate\w*|\bcuaj\w*|\bhierv\w*|\bherv(?:ir|id[oa]s?)\b|\bcocin\w*|\brevuelv\w*|\bvierte\b|\blicu\w*|\bmezcl\w*|"
    r"\bincorpor\w*|\bagreg\w*|\banad\w*)[^.;]{0,40}\bclaras?\b|\bclaras?\b[^.;]{0,20}(?:\bcuaj\w*|\bfirmes\b|\brevuelt\w*|"
    r"\bbatid\w*)")
_HUEVO_DURO_405_RE = re.compile(r"(?:\bhierv\w*|\bherv(?:ir|id[oa]s?)\b|agua hirviendo|\bcuece\b)[^.]*\bhuevos?\b|"
                                r"\bhuevos?\b[^.]*(?:\bhierv\w*|\bherv(?:ir|id[oa]s?)\b|agua hirviendo)")
_CERRADOR_HUEVO_405_RE = re.compile(r"Cocina huevo a la plancha o hervido y sírvelo como proteína del plato\.")


def claras_de_la_lista(meal) -> int:
    """Nº de frases corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lista = " | ".join(_sa(str(x).lower()) for x in (meal.get("ingredients") or []))
        if not re.search(r"\bclaras? de huevos?\b", lista):
            return 0
        enteros = bool(re.search(r"\bhuevos?\b", re.sub(r"(?:claras?|yemas?) de huevos?", " ", lista)))
        pasos = [(i, p) for i, p in enumerate(rec) if isinstance(p, str) and not _es_nota(p)]
        texto = " ".join(_sa(p.lower()) for _, p in pasos)
        if "dentro de su huevo entero" in texto or "un huevo por cada clara" in texto:
            return 0
        if not enteros:
            for i, p in pasos:
                s = _CERRADOR_HUEVO_405_RE.sub("Cuaja las claras de huevo en la sartén, revueltas, hasta que estén firmes y "
                                               "opacas, y sírvelas como proteína del plato.", p)
                if s != p:
                    rec[i] = s
                    meal["recipe"] = rec
                    meal.pop("_display", None)
                    return 1
            return 0
        if _CLARA_COCINADA_405_RE.search(texto):
            return 0
        for i, p in pasos:
            partes = re.split(r"(?<=\.)\s+", p)
            for k, frase in enumerate(partes):
                fn = _sa(frase.lower())
                if "clara" in fn or not _HUEVO_DURO_405_RE.search(fn):
                    continue
                cuerpo = frase.rstrip()
                punto = cuerpo.endswith(".")
                if punto:
                    cuerpo = cuerpo[:-1]
                partes[k] = (cuerpo + "; para las claras de huevo de la lista, hierve también con cáscara un huevo por cada "
                             "clara y, al pelarlos, quítales la yema" + ("." if punto else ""))
                rec[i] = " ".join(partes)
                meal["recipe"] = rec
                meal.pop("_display", None)
                return 1
        return 0
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-406 · 2026-09-26] La masa de harina de maíz lleva su agua ─────────────────────────────────────────────
# Batería REAL sobre el 399 (perfil del dueño, día 3): «mezcla 130 g de harina de maíz precocida con ½ taza de agua» —
# 0,9 ml por gramo, cuando la masa pide ~2,3 (la mediana del corpus es 2,4, la del paquete 1¼ taza por taza). El ajuste
# de calorías sube la harina y el agua, que no suma nada, se queda: «100 g de harina + 15 ml de agua» no hace masa. Corpus
# de 319 planes: 9 de 54 masas por debajo de 1,2 ml/g, casi todas del perfil del dueño (ganar músculo). Como el 311 con la
# avena: el agua que falta no suma calorías ni se compra. Con UNA línea de harina en gramos y UNA cantidad de agua (lista
# o paso) por debajo de 1,5 ml/g, el agua pasa a 2,3 ml/g (en decenas), en la lista, en el motor y en el paso.
# tooltip-anchor: P1-PLAN-LOTE-406
_HARINA_406_RE = re.compile(r"^\s*(?P<g>\d+(?:[.,]\d+)?)\s*g\s+de\s+harina\s+de\s+ma[ií]z\s+precocida", re.IGNORECASE)
_AGUA_406_RE = re.compile(r"(?P<q>(?:\d+\s*[½¼¾⅓⅔]|\d+(?:[.,]\d+)?|[½¼¾⅓⅔]))\s*(?P<u>ml|tazas?|cdas?|cucharadas?)\s+de\s+agua\b",
                          re.IGNORECASE)


_MEZCLA_406 = r"\b(?:mezcla\w*|amasa\w*|une|combina\w*|hidrata\w*|forma\w*)\b"


def pasos_406(rec) -> list:
    return [(i, p) for i, p in enumerate(rec or []) if isinstance(p, str) and not _es_nota(p)]


def _ml_406(q: str, u: str) -> float:
    return _num_397(q) * (240.0 if u.lower().startswith("taza") else 15.0 if u.lower().startswith(("cda", "cuchar")) else 1.0)


def masa_con_su_agua(meal) -> int:
    """Nº de textos corregidos; 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        ings = meal.get("ingredients") if isinstance(meal.get("ingredients"), list) else []
        harinas = [float(m.group("g").replace(",", ".")) for m in (_HARINA_406_RE.match(str(x)) for x in ings) if m]
        if len(harinas) != 1 or harinas[0] <= 0:
            return 0
        g = harinas[0]
        rec = meal.get("recipe") if isinstance(meal.get("recipe"), list) else []
        aguas_lista = [(i, m) for i, x in enumerate(ings) for m in [_AGUA_406_RE.match(str(x).strip())] if m]
        aguas_paso = [(i, m) for i, p in enumerate(rec) if isinstance(p, str) and not _es_nota(p)
                      for m in _AGUA_406_RE.finditer(p)]
        if len(aguas_lista) > 1 or (not aguas_lista and len({m.group(0) for _, m in aguas_paso}) != 1):
            return 0
        ref = aguas_lista[0][1] if aguas_lista else aguas_paso[0][1]
        ml = _ml_406(ref.group("q"), ref.group("u"))
        if ml <= 0 or ml / g >= 1.5:
            return 0
        nuevo_ml = int(round(g * 2.3 / 10.0) * 10)
        viejo_txt = ref.group(0)
        # sólo el agua que la receta usa EN la masa: «mezcla la harina… con ½ taza de agua»; la del Mise («mide… 2 cdas de
        # agua») o la de otro alimento («la auyama con la cucharada de agua») no es la de la masa (replay del 406)
        masa = any(re.search(_MEZCLA_406, cl, re.IGNORECASE) and re.search(r"\bharina\b", cl, re.IGNORECASE) and viejo_txt in cl
                   for _, p in pasos_406(rec) for cl in re.split(r"[.;](?!\d)", p))
        if not masa:
            return 0
        nuevo_txt = f"{nuevo_ml} ml de agua"
        n = 0
        if aguas_lista:
            i, m = aguas_lista[0]
            ings[i] = str(ings[i]).replace(m.group(0), nuevo_txt, 1)
            n += 1
            motor = meal.get("ingredients_raw")
            if isinstance(motor, list):               # la línea del motor se resuelve por su TEXTO (el agua con esos ml), nunca
                hecho, nuevo_motor = False, []        # por índice paralelo a la lista (P1-RAW-INDEX-INVENTORY)
                for x in motor:
                    mr = _AGUA_406_RE.match(str(x).strip())
                    if not hecho and mr and abs(_ml_406(mr.group("q"), mr.group("u")) - ml) < 1.0:
                        nuevo_motor.append(str(x).replace(mr.group(0), nuevo_txt, 1))
                        hecho = True
                    else:
                        nuevo_motor.append(x)
                if hecho:
                    meal["ingredients_raw"] = nuevo_motor
        for i, p in enumerate(rec):
            if isinstance(p, str) and not _es_nota(p) and viejo_txt in p:
                rec[i] = p.replace(viejo_txt, nuevo_txt)
                n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0
