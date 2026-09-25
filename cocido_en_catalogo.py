# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-284 · 2026-09-25] «¾ taza de arroz blanco cocido» son ~150 kcal, no 491.

Las filas de granos y legumbres del catálogo están en CRUDO/SECO (arroz blanco 358,6 kcal/100 g; habichuelas rojas
344,7). El resolvedor de macros (`nutrition_db.IngredientNutritionDB.grams_from_ingredient_string`) pela el estado
(«cocido») al buscar el nombre y multiplica los gramos de la línea por la fila seca: una línea en estado COCIDO se
contaba ~×3. `P1-COOKED-GRAIN-DRY` (jul) lo arregló reescribiendo el TEXTO de las líneas «N g de X cocido» ancladas al
final — en assemble y en la cola de los bloques, DESPUÉS del solver — y dejó fuera las de tazas («⅓ taza de arroz
blanco cocido»), las que siguen («… cocido y refrigerado», «… cocidas sin sal», «… (86 g)») y las de lata. Baterías del
25-sep: 75 de 581 líneas de granos/legumbres (13 %, en 57 planes). En el plan del dueño (ganar músculo), «¾ taza de
arroz blanco cocido y refrigerado» contaba 491 kcal: el almuerzo decía 832 y aportaba ~490 — el superávit no existía,
y el solver, que dimensiona la ración con ese número, la dejaba corta.

Aquí el resolvedor devuelve los gramos en la BASE DE LA FILA: si la línea dice cocido/hervido/sancochado (o, para una
legumbre, de lata/escurrida, o se mide en latas) y la fila es un grano o legumbre en seco, los gramos cocidos —del hint
«(N g)», de la cantidad en gramos, o de las tazas × la densidad COCIDA— × kcal_cocido / kcal_fila; una lata, su
equivalente seco (el factor del lote 282). Las kcal de referencia son las MISMAS de `graph_orchestrator.
_COOKED_GRAIN_REF_KCAL` (test de paridad): una línea reescrita después a «crudo» da el mismo número que antes. La
línea que ya dice crudo/seco no se toca, ni la fila que ya esté en cocido (el ratio cae bajo 1,5 y se apaga solo).
Knob `MEALFIT_COOKED_LINE_CATALOG_BASIS` (on). Puro; nunca lanza. tooltip-anchor: P1-PLAN-LOTE-284-COCIDO-EN-BASE"""
from __future__ import annotations

import re
import unicodedata

#: (tokens de la FILA, kcal/100 g cocido — mismas de `_COOKED_GRAIN_REF_KCAL` —, g por taza COCIDA, ¿legumbre?)
FAMILIAS = (
    (("arroz",), 130.0, 158.0, False),
    (("quinoa",), 130.0, 185.0, False),
    (("pasta", "espagueti", "espaguetis", "coditos", "fideos", "macarrones", "macarron"), 158.0, 140.0, False),
    (("lenteja", "lentejas"), 127.0, 198.0, True),
    (("garbanzo", "garbanzos"), 127.0, 164.0, True),
    (("habichuela", "habichuelas", "frijol", "frijoles", "gandul", "gandules", "guandul", "guandules"), 127.0, 172.0, True),
    # sin normalizador de texto (solo el resolvedor): USDA cocido — bulgur 83, cebada perlada 123, habas 110
    (("bulgur",), 83.0, 182.0, False),
    (("cebada",), 123.0, 157.0, False),
    (("haba", "habas"), 110.0, 170.0, True),
    # soya texturizada hidratada: el mismo 0,35× que usa la lista (327 × 0,35); sin densidad cocida fiable → solo gramos
    (("texturizada",), 114.0, None, False),
)
_COCIDO_RX = re.compile(r"\b(cocid[oa]s?|hervid[oa]s?|sancochad[oa]s?)\b")
_LISTO_RX = re.compile(r"\b(de lata|en lata|enlatad[oa]s?|escurrid[oa]s?)\b")
_CRUDO_RX = re.compile(r"\b(crud[oa]s?|sec[oa]s?|en seco|peso seco)\b")
_PAREN_RX = re.compile(r"\([^)]*\)")
_UNIDAD_MASA = {"g", "gr", "grs", "gramo", "gramos", "kg", "kilo", "kilos", "oz", "onza", "onzas", "lb", "lbs",
                "libra", "libras"}
_UNIDAD_VOLUMEN = {"taza", "tazas", "cda", "cdas", "cucharada", "cucharadas", "cdta", "cdtas", "cucharadita",
                   "cucharaditas", "ml", "mililitro", "mililitros", "l", "litro", "litros"}
_UNIDAD_LATA = {"lata", "latas"}


def activo() -> bool:
    try:
        from knobs import _env_bool
        return bool(_env_bool("MEALFIT_COOKED_LINE_CATALOG_BASIS", True))
    except Exception:
        return True


def _norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s or "").lower())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


def familia(nombre_fila):
    """`(kcal_cocido, g_por_taza_cocida, es_legumbre)` para la fila, o None."""
    toks = set(re.findall(r"[a-z]+", _norm(nombre_fila)))
    for tokens, kcal, taza, leg in FAMILIAS:
        if toks & set(tokens):
            return kcal, taza, leg
    return None


def en_base_de_la_fila(linea, gramos, db):
    """`gramos` leídos de `linea` (lo que el resolvedor calculó literal) → gramos en la base de la fila del catálogo."""
    if gramos is None:
        return None
    try:
        if not activo():
            return gramos
        texto = _norm(linea)
        fuera = _PAREN_RX.sub(" ", texto)
        cocido = bool(_COCIDO_RX.search(fuera))
        listo = bool(_LISTO_RX.search(fuera))
        if not (cocido or listo or re.search(r"\blatas?\b", fuera)):
            return gramos
        if _CRUDO_RX.search(fuera):
            return gramos                                   # «secas», «en crudo»: la línea ya habla en la base
        from nutrition_db import _GRAM_ONLY_HINT_RE, _split_qty_unit_name
        qty, unidad, nombre = _split_qty_unit_name(linea)
        info = db.lookup(nombre)
        fam = familia(getattr(info, "name", "")) if info else None
        if not fam:
            return gramos
        kcal_cocido, taza_cocida, es_legumbre = fam
        kcal_fila = float(getattr(info, "kcal", 0) or 0)
        if kcal_fila <= 0 or kcal_fila / kcal_cocido < 1.5:
            return gramos                                   # la fila ya está en cocido
        u = _norm(unidad)
        if u in _UNIDAD_LATA and not _GRAM_ONLY_HINT_RE.search(str(linea)):
            if not es_legumbre:
                return gramos
            return round(float(gramos) * __import__("envase_legumbre").factor_listo_a_seco(), 2)
        if not (cocido or (listo and es_legumbre)):
            return gramos
        hint = _GRAM_ONLY_HINT_RE.search(str(linea))
        if hint and _CRUDO_RX.search(_norm(hint.group(0))):
            return gramos                                   # «(5 g en seco)»: el paréntesis ya habla en la base
        if hint or u in _UNIDAD_MASA:
            cocidos = float(gramos)
        elif u in _UNIDAD_VOLUMEN:
            dens = float(getattr(info, "density_g_per_cup", 0) or 0)
            if dens <= 0 or not taza_cocida:
                return gramos
            cocidos = float(gramos) / dens * taza_cocida    # las mismas tazas, medidas cocidas
        else:
            return gramos
        return round(cocidos * kcal_cocido / kcal_fila, 2)
    except Exception:
        return gramos
