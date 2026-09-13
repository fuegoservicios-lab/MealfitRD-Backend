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
