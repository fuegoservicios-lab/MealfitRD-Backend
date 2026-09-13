# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-29 · 2026-09-12] CUL-P1-04 · el reparador: adapta la receta a lo que el plato ya declara, sin desmontarla.

`dish_structure` (C5, lote 27) ACUSA tres defectos de estructura como V9 — la crema que promete espesor sin sólidos ni
proceso, el wrap cuyo relleno no cabe, la tortilla que suelta agua porque nadie preparó el vegetal — y hasta hoy nadie los
reparaba: el contrato final (C2/C3) alinea CANTIDADES y FORMAS, no la técnica ni el montaje. Este módulo repara los tres
con la única regla que hace la reparación segura para la nutrición: **no toca la lista de ingredientes ni una cifra de
compra; sólo el texto de los pasos**. Lo que sobra se sirve APARTE con técnica o acompañamiento deliberado (la cláusula
del backlog), no se recorta ni se estira:

  · `tortilla_vegetales_crudos` → se inserta, antes del paso que vierte o cuaja el huevo, el salteado y escurrido de los
    vegetales de agua «con un chorrito de agua» (así lo hacen 12 de 14 recetas curadas; sin aceite añadido, no cambia
    ningún macro);
  · `wrap_desproporcionado` → montaje: se rellena la tortilla con lo que cierra (el máximo curado, 3,8 × el pan) y el
    resto del relleno se sirve al lado como ensalada — misma compra, mismos macros, un plato que se puede comer;
  · `crema_sin_espesante` → ajuste de textura: la crema usa sólo el líquido que la deja espesa (2 ml por gramo de sólido,
    el doble de holgura que la mediana curada) y el resto se sirve como bebida al lado — misma leche comprada, misma
    nutrición, dos componentes honestos en vez de una promesa.

`dish_structure` reconoce lo servido aparte y deja de acusar: la reparación es idempotente. Corre como paso (4) de
`recipe_contract.reconcile_meal`, detrás de cantidades y formas, bajo el mismo knob `MEALFIT_RECIPE_FINAL_CONTRACT`
(`repair` aplica, `shadow` anota, `off` nada), con su propia cuenta en la telemetría (`estructura`).

Además, `degradar_paso` sirve al camino sin LLM (`cron_tasks._degrade_offending_steps`): al degradar un paso que ofendía
(V1/V2) a «Sirve el X», los OTROS alimentos que ese paso nombraba se conservan en la frase (antes quedaban comprados y
sin ningún paso que los usara: V3) y, si alguno es una legumbre, grano o pasta que se compra seca, se antepone su
cocción genérica (antes aparecía «declarado seco y ningún paso lo hierve»: V7c). Medido con el benchmark de superficies
sobre el corpus fijo antes de escribirlo: 1 hallazgo nuevo del postfix con el adaptador que imita al cron.

Puro; nunca lanza. tooltip-anchor: P1-PLAN-LOTE-29-RECIPE-REPAIR
"""
from __future__ import annotations

import math
import re
from typing import Optional

SECCIONES = ("Mise en place", "El Toque de Fuego", "Montaje")
#: g de relleno que cierra por g de pan: el máximo de la biblioteca curada (3,8), no un redondo
WRAP_CIERRA_POR_G_DE_PAN = 3.8
#: ml de líquido por g de sólido con los que una crema espesa de verdad (mediana curada 0,90 g/ml ⇒ ~1,1 ml/g; se da el doble)
CREMA_ML_POR_G_SOLIDO = 2.0
CREMA_MIN_ML = 30.0
#: por debajo de esto no hay nada que servir aparte: la reparación se descarta y se dice
CREMA_MIN_RESTO_ML = 50.0

_HUEVO_COCINA_RE = re.compile(r"\b(vierte|anade|agrega|incorpora|echa|cuaja|revuelve|mezcla|pon|vuelca)\b[^.]{0,50}?\b(huevos?|claras?)\b|"
                              r"\b(huevos?|claras?)\b[^.]{0,40}?\b(en la sarten|a la sarten|al sarten|cuaj\w+|revuelv\w+)")
_HUEVO_RE = re.compile(r"\b(huevos?|claras?|yemas?)\b")
_CLARAS_RE = re.compile(r"\bclaras?\b")


def _norm(s) -> str:
    from dish_structure import _norm as n
    return n(s)


def _seccion(paso: str) -> tuple:
    """`(prefijo, cuerpo)`: «El Toque de Fuego: Hornea…» → («El Toque de Fuego: », «Hornea…»); sin sección, («», paso)."""
    s = str(paso or "")
    cab = s.split(":", 1)[0].strip()
    if ":" in s and cab in SECCIONES:
        return f"{cab}: ", s.split(":", 1)[1].strip()
    return "", s


def _lista(nombres: list) -> str:
    n = [str(x) for x in nombres if str(x).strip()]
    if not n:
        return ""
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " y " + n[-1]


def _indice_paso_huevo(pasos: list) -> int:
    """El paso que VIERTE o cuaja el huevo; si ninguno lo dice, el último que lo nombra; -1 si ninguno."""
    ult = -1
    for i, p in enumerate(pasos):
        n = _norm(p)
        if _HUEVO_COCINA_RE.search(n):
            return i
        if _HUEVO_RE.search(n):
            ult = i
    return ult


# ─────────────────────────────────────────────────────────────────────────────────────────────
# las tres reparaciones de estructura
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _rep_tortilla(meal: dict, comp: dict) -> Optional[dict]:
    veg = list(comp.get("vegetales_agua") or [])
    if not veg:
        return None
    pasos = [str(p) for p in (meal.get("recipe") or [])]
    i = _indice_paso_huevo(pasos)
    ings = _norm(" . ".join(str(x) for x in (meal.get("ingredients") or [])))
    sin_formas = re.sub(r"\b(?:claras?|yemas?) de huevos?\b", " ", ings)      # «claras de huevo» no es un huevo entero
    huevo_txt = "las claras" if _CLARAS_RE.search(ings) and not re.search(r"\bhuevos?\b", sin_formas) else "el huevo"
    prefijo = _seccion(pasos[i])[0] if i >= 0 else ""
    if prefijo == "Mise en place: ":
        prefijo = ""
    nuevo = (f"{prefijo}Saltea {_lista(veg)} 3-4 min en la sartén con un chorrito de agua hasta que suelten el agua y "
             f"escúrrelos bien antes de añadir {huevo_txt}: así la tortilla no suelta agua.")
    pos = i if i >= 0 else len(pasos)
    pasos.insert(pos, nuevo)
    meal["recipe"] = pasos
    return {"tipo": "tortilla_vegetales_crudos", "paso": pos, "texto": nuevo}


def _rep_wrap(meal: dict, comp: dict) -> Optional[dict]:
    sop = comp.get("soporte")
    if not sop or not sop[1]:
        return None
    pan_nombre, pan_g = sop
    relleno = float(comp.get("solidos_g") or 0) - float(pan_g)
    cierra = math.floor(pan_g * WRAP_CIERRA_POR_G_DE_PAN / 10) * 10
    resto = relleno - cierra
    if cierra <= 0 or resto < 20:
        return None
    pasos = [str(p) for p in (meal.get("recipe") or [])]
    nuevo = (f"Montaje: rellena la {pan_nombre} con lo que cierra (unos {cierra:g} g del relleno) y sirve el resto del "
             f"relleno (~{resto:g} g) al lado, como ensalada: misma compra, un wrap que se puede cerrar.")
    pasos.append(nuevo)
    meal["recipe"] = pasos
    return {"tipo": "wrap_desproporcionado", "paso": len(pasos) - 1, "texto": nuevo}


def _rep_crema(meal: dict, comp: dict) -> Optional[dict]:
    from dish_structure import _linea, _LIQUIDO_RE
    liquido = None
    for ing in (meal.get("ingredients") or []):
        nm, g, ml = _linea(ing)
        if nm and _LIQUIDO_RE.search(nm) and (ml or g):
            liquido = nm
            break
    total_ml = float(comp.get("liquidos_ml") or 0)
    solidos = float(comp.get("solidos_g") or 0)
    if not liquido or total_ml <= 0:
        return None
    usar = max(CREMA_MIN_ML, math.ceil(solidos * CREMA_ML_POR_G_SOLIDO / 10) * 10)
    resto = total_ml - usar
    if resto < CREMA_MIN_RESTO_ML:
        return None
    pasos = [str(p) for p in (meal.get("recipe") or [])]
    # las dos cifras nombran el líquido: V7d suma por alimento y así ve los ml completos en los pasos
    nuevo = (f"Ajuste de textura: para que espese de verdad usa solo {usar:g} ml de {liquido} en la crema y sirve los "
             f"{resto:g} ml de {liquido} restantes como bebida al lado.")
    pasos.append(nuevo)
    meal["recipe"] = pasos
    return {"tipo": "crema_sin_espesante", "paso": len(pasos) - 1, "texto": nuevo}


_REPARADORES = {"tortilla_vegetales_crudos": _rep_tortilla, "wrap_desproporcionado": _rep_wrap, "crema_sin_espesante": _rep_crema}


def reparar_estructura(meal: dict) -> dict:
    """Repara en `meal["recipe"]` lo que `dish_structure.relaciones` acusa. Devuelve
    `{"aplicado": [tipos], "descartado": [tipos], "cambios": [{"tipo", "paso", "texto"}]}`. La lista de ingredientes no
    se toca jamás. Idempotente: tras reparar, `relaciones` deja de acusar y una segunda pasada no cambia nada."""
    out = {"aplicado": [], "descartado": [], "cambios": []}
    try:
        if not isinstance(meal, dict) or not isinstance(meal.get("recipe"), list):
            return out
        from dish_structure import componentes, familia, relaciones
        fam = familia(meal)
        comp = componentes(meal)
        for r in relaciones(meal, comp, fam):
            fn = _REPARADORES.get(r.get("tipo"))
            cambio = fn(meal, comp) if fn else None
            if cambio:
                out["aplicado"].append(r["tipo"])
                out["cambios"].append(cambio)
            else:
                out["descartado"].append(r.get("tipo"))
    except Exception:
        return out
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# el camino sin LLM: degradar un paso sin dejar huérfanos
# ─────────────────────────────────────────────────────────────────────────────────────────────

def degradar_paso(paso: str, food: str, index: dict, prefijo: str = "") -> list:
    """Sustituye un paso que ofende (V1/V2 sobre `food`) por pasos SIN verbo de cocción para ese alimento, conservando los
    demás alimentos que nombraba: «Sirve el X con Y y Z.», y si alguno de ellos se compra SECO (legumbre, grano, pasta),
    antes «Cocina Y según su envase…». Sin otros alimentos devuelve exactamente lo de siempre: `[«{prefijo}Sirve el X.»]`."""
    try:
        from culinary_coherence import find_catalog_foods
        otros = [f for f in find_catalog_foods(str(paso or ""), index) if f != food]
    except Exception:
        otros = []
    if not otros:
        return [f"{prefijo}Sirve el {food}."]
    cocinables = [f for f in otros if _secable(f)]
    pasos = []
    if cocinables:
        pasos.append(f"{prefijo}Cocina {_lista(cocinables)} según su envase hasta que estén tiernos "
                     f"(la legumbre seca 45-60 min; la pasta o el arroz 10-20 min).")
    pasos.append(f"{prefijo}Sirve el {food} con {_lista(otros)}.")
    return pasos


def _secable(food: str) -> bool:
    try:
        from culinary_coherence import _V7_SECABLES_RE
        return bool(_V7_SECABLES_RE.search(_norm(food)))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-30 · 2026-09-13] la lista tiene la última palabra también cuando PIERDE un alimento
# ─────────────────────────────────────────────────────────────────────────────────────────────
# Medido en el benchmark en modo real (3 planes recién generados, 33 comidas): el piso de porciones servibles
# (`_floor_subservible_portions`, GAP-05) borró «15 g de Granola» de un smoothie bowl sin cabida calórica y el paso siguió
# diciendo «corona con la granola» — un V5 (el paso usa lo que la lista no trae) nacido en la propia cadena de
# persistencia, no en el LLM. Ningún cerrador que quita una línea toca los pasos, así que la reparación va en la cola del
# contrato, para TODOS: se retira la MENCIÓN del alimento del paso (el ítem de una enumeración, el complemento «con X» o,
# si el paso no decía otra cosa, la frase entera), se comprueba con el propio detector V5 y, si la mención sobrevive, se
# deshace y se declara. Nunca toca la lista. tooltip-anchor: P1-PLAN-LOTE-30-SIN-LISTA

_ART = r"(?:(?:el|la|los|las|un|una|unos|unas|del|al)\s+)?"
_CANT = (r"(?:[\d½⅓¼¾⅔⅛][\d.,/½⅓¼¾⅔⅛]*\s*(?:g|gr|gramos?|kg|ml|l|cdas?|cdtas?|cucharad\w*|tazas?|unid\w*|piezas?|rebanadas?|"
         r"hojas?|dientes?|pizcas?|pu[nñ]ad\w*|ramitas?)?\s*(?:de\s+)?)?")
_MOD = r"(?:\s+(?:reservad|tostad|picad|rallad|fresc|restant|troce|cortad|desmenuzad|cocid|crud|madur|natural|enter|integral)\w*)?"
_TRAS = r"(?=\s*(?:[.,;:)]|$|\s+\d|\s+(?:y|e|o|hasta|para|en|con|sobre|por|durante|mientras|al|a)\b))"
_PREP = r"(?:con|de|sobre|junto\s+a|junto\s+con|acompa[nñ]ad[oa]s?\s+de|encima\s+de|m[aá]s)"
#: lo que sigue a «ITEM y » para que cuente como enumeración: otro ítem con artículo o cantidad, no un verbo
_SIGUIENTE_ITEM = r"(?:[\d½⅓¼¾⅔⅛]|(?:el|la|los|las|un|una|unos|unas|del|al)\s)"
_ACENTOS = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]", "n": "[nñ]"}


def _rx_alimento(nombre: str) -> str:
    """Regex del alimento tolerante a acentos, mayúsculas y plural, palabra a palabra: «jamon» casa «Jamón» y «jamones»;
    «yogur de coco» casa «Yogur de Coco» entero (quitar sólo «yogur» dejaría «de coco, 65 ml de leche» colgando)."""
    partes = [p for p in re.split(r"\s+", str(nombre or "").lower().strip()) if p]
    return r"\s+".join("".join(_ACENTOS.get(c, re.escape(c)) for c in p) + r"(?:s|es)?\b" for p in partes)


def _cabeza(nombre: str) -> str:
    """La palabra que V5 usa para localizar la mención (`_v5_mas_especifico`): la primera de ≥4 letras."""
    return next((w for w in re.split(r"[^a-z0-9]+", str(nombre or "").lower()) if len(w) >= 4), "")


def _item(cabeza: str) -> str:
    return rf"{_CANT}{_ART}\b{_rx_alimento(cabeza)}{_MOD}"


def _frases(texto: str) -> list:
    return [f for f in re.split(r"(?<=[.;])\s+", texto) if f.strip()]


def quitar_mencion(paso: str, alimento: str) -> str:
    """Quita del paso la mención del alimento (nombre normalizado, p. ej. «granola» o «yogur de coco»; si el nombre entero
    no aparece, se busca su cabeza). Devuelve el paso reescrito, el mismo paso si no lo nombraba, o «» si sin ese alimento
    no queda nada que decir. Orden: ítem de una enumeración («X, ITEM y Y» / «ITEM y Y» / «X y ITEM») → complemento
    («corona con ITEM») → la frase entera que lo nombra."""
    prefijo, cuerpo = _seccion(paso)
    fl = re.IGNORECASE
    nombre = next((n for n in (str(alimento or "").strip().lower(), _cabeza(alimento)) if n and cuerpo and re.search(rf"\b{_rx_alimento(n)}", cuerpo, fl)), "")
    if not nombre:
        return paso
    al = _rx_alimento(nombre)
    it = _item(nombre)
    nuevo = re.sub(rf",\s*{it}(?=\s*(?:,|\s+(?:y|e)\s+))", "", cuerpo, count=1, flags=fl)          # «X, ITEM, Y» / «X, ITEM y Y»
    if nuevo == cuerpo:
        # «ITEM y Y» / «ITEM, Y»: sólo si lo que sigue es otro ítem (artículo o cantidad), no un verbo («…el jamón y mezcla bien»)
        nuevo = re.sub(rf"\b{it}\s*(?:,|\s+(?:y|e))\s+(?={_SIGUIENTE_ITEM})", "", cuerpo, count=1, flags=fl)
    if nuevo == cuerpo:
        m = re.search(rf"\s+(?:y|e)\s+{it}{_TRAS}", cuerpo, fl)                                     # «X y ITEM»
        if m:
            antes, despues = cuerpo[:m.start()], cuerpo[m.end():]
            ini = max(antes.rfind(". "), antes.rfind(": "), antes.rfind("; "), -1) + 1
            clausula = antes[ini:]
            if ", " in clausula and not re.search(r"\s(?:y|e)\s", clausula):                        # «A, B, C» → «A, B y C»
                i = clausula.rfind(", ")
                clausula = clausula[:i] + " y " + clausula[i + 2:]
            nuevo = antes[:ini] + clausula + despues
    if nuevo == cuerpo:
        nuevo = re.sub(rf"\s+{_PREP}\s+{it}{_TRAS}", "", cuerpo, count=1, flags=fl)                 # «corona con ITEM»
    if nuevo == cuerpo or re.search(rf"\b{al}", nuevo, fl):                                          # cae la frase que lo nombra
        nuevo = " ".join(f for f in _frases(cuerpo) if not re.search(rf"\b{al}", f, fl))
    nuevo = re.sub(r"\s+", " ", nuevo).replace(" ,", ",").replace(" .", ".").replace(",.", ".").replace(" ;", ";").strip()
    nuevo = " ".join(f for f in _frases(nuevo) if len(re.findall(r"\w+", f)) >= 2).strip()          # «Corona.» no dice nada
    if not nuevo:
        return ""
    if cuerpo[:1].isupper():                                                                          # respeta la grafía del paso
        nuevo = nuevo[0].upper() + nuevo[1:]
    return f"{prefijo}{nuevo}"


def _hallazgos(meal: dict, index: dict) -> set:
    """Los hallazgos de capa 1 que dependen del TEXTO de los pasos y sólo necesitan el índice (V1, V3, V5, V7a/b/c/e): el
    espejo con el que la retirada comprueba que no abrió otro hallazgo. `{(check, alimento)}`; fail-open por check."""
    out = set()
    try:
        import culinary_coherence as cc
        dia = {"day": 0}
        for fn in ("_v1_verbo_alimento", "_v3_huerfanos", "_v5_paso_usa_lo_que_no_esta", "_v7a_lista_compra_de_mas",
                   "_v7b_duplicado_incompatible", "_v7c_seco_sin_coccion", "_v7e_paso_pide_mas_piezas"):
            try:
                for v in (getattr(cc, fn)(dia, meal, index) or []):
                    out.add((v.get("check"), v.get("food")))
            except Exception:
                continue
    except Exception:
        pass
    return out


def retirar_sin_lista(meal: dict, index: dict) -> dict:
    """Retira de `meal["recipe"]` los alimentos que V5 acusa (el paso los usa, la lista no los trae) y lo verifica con el
    mismo detector; si la mención sobrevive o la receta se quedaría vacía, deshace ese alimento y lo declara. Devuelve
    `{"aplicado": [alimentos], "descartado": [alimentos], "cambios": [{"tipo", "food", "paso", "antes", "despues"}]}`.
    La lista de ingredientes no se toca jamás."""
    out = {"aplicado": [], "descartado": [], "cambios": []}
    try:
        if not isinstance(meal, dict) or not isinstance(meal.get("recipe"), list) or not meal["recipe"] or not index:
            return out
        from culinary_coherence import _v5_paso_usa_lo_que_no_esta
        dia = {"day": 0}
        foods = []
        for v in _v5_paso_usa_lo_que_no_esta(dia, meal, index):
            f = v.get("food")
            if f and f not in foods:
                foods.append(f)
        v5s = {("V5", f) for f in foods}
        base = _hallazgos(meal, index) - v5s
        for food in foods:
            if not _cabeza(food):
                out["descartado"].append(food)
                continue
            antes = [str(p) for p in meal["recipe"]]
            nuevos, cambios = [], []
            for i, p in enumerate(antes):
                q = quitar_mencion(p, food)
                if q != p:
                    cambios.append({"tipo": "sin_lista", "food": food, "paso": i, "antes": p, "despues": q})
                if q:
                    nuevos.append(q)
            meal["recipe"] = nuevos
            despues = _hallazgos(meal, index)
            # se deshace si la mención sobrevive, si la receta se vació o si la retirada ABRIÓ otro hallazgo (un V3 porque la
            # frase que cayó era la única que nombraba a otro alimento — medido sobre el corpus fijo antes de escribir esto)
            if ("V5", food) in despues or (despues - base - v5s) or not nuevos or not cambios:
                meal["recipe"] = antes
                out["descartado"].append(food)
            else:
                out["aplicado"].append(food)
                out["cambios"].extend(cambios)
    except Exception:
        return out
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-31 · 2026-09-13] la misma mención, tres veces seguidas: el LLM que se repite
# ─────────────────────────────────────────────────────────────────────────────────────────────
# Medido en el bench real: una cena DM2 recién generada decía «separa 3 huevos y 6 claras de huevo y 6 claras de huevo y 6
# claras» y «casca 3 huevos y 6 claras de huevo y 6 claras de huevo sobre el guiso». No lo escribió la cadena: el sello del
# contrato registra 6 piezas recortadas (el LLM las había escrito con OTRO número, seis veces) y el reescritor de la forma del
# huevo es idempotente (probado ×3). Es el modelo repitiéndose — y V7e, sumando el paso, leía 18 claras contra 6. Aquí se
# deja UNA: la misma mención numérica (alimento, familia y cantidad) repetida en cadena, unida sólo por «y»/«e»/coma. Se
# comprueba con los detectores de capa 1 y se deshace si abre un hallazgo. La lista no se toca. tooltip-anchor: P1-PLAN-LOTE-31-REPETICION

_CONECTOR_RE = re.compile(r"^(?:\s*,\s*|\s+(?:y|e)\s+)$", re.IGNORECASE)


def colapsar_repeticiones(meal: dict, index: dict) -> dict:
    """«separa 3 huevos y 6 claras de huevo y 6 claras de huevo y 6 claras.» → «separa 3 huevos y 6 claras de huevo.»
    Devuelve `{"aplicado": [alimentos], "descartado": [alimentos], "cambios": [{"tipo", "food", "paso", "antes", "despues"}]}`.
    «6 claras de huevo reservando 6 claras» no es una cadena (hay un verbo en medio) y no se toca."""
    out = {"aplicado": [], "descartado": [], "cambios": []}
    try:
        if not isinstance(meal, dict) or not isinstance(meal.get("recipe"), list) or not meal["recipe"] or not index:
            return out
        from recipe_contract import _es_nota, _menciones_paso
        antes = [str(p) for p in meal["recipe"]]
        nuevos, cambios, foods = list(antes), [], []
        for i, paso in enumerate(antes):
            if _es_nota(paso):
                continue
            ms = sorted((m for m in _menciones_paso(paso, index) if m.get("food_fin") is not None), key=lambda x: x["ini"])
            cortes, ultimo = [], None
            for m in ms:
                if (ultimo is not None and (m["food"], m["familia"], m["valor"]) == (ultimo["food"], ultimo["familia"], ultimo["valor"])
                        and _CONECTOR_RE.match(paso[ultimo["food_fin"]:m["ini"]])):
                    cortes.append((ultimo["food_fin"], m["food_fin"]))                  # cae « y 6 claras de huevo»
                    if m["food"] not in foods:
                        foods.append(m["food"])
                ultimo = m
            if not cortes:
                continue
            nuevo = paso
            for a, b in sorted(cortes, reverse=True):
                nuevo = nuevo[:a] + nuevo[b:]
            nuevo = re.sub(r"\s+([.,;:])", r"\1", nuevo)
            if nuevo != paso and nuevo.strip():
                nuevos[i] = nuevo
                cambios.append({"tipo": "repeticion", "food": ", ".join(sorted({f for f in foods})), "paso": i, "antes": paso, "despues": nuevo})
        if not cambios:
            return out
        base = _hallazgos(meal, index)
        meal["recipe"] = nuevos
        if _hallazgos(meal, index) - base:                                              # abrió otro hallazgo: no era una reparación
            meal["recipe"] = antes
            out["descartado"] = list(foods)
        else:
            out["aplicado"] = list(foods)
            out["cambios"] = cambios
    except Exception:
        return out
    return out

