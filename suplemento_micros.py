# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1112 · 2026-11-19] Una toma de un pote con tabla de vitaminas suma esas vitaminas a «Micros de hoy».

Caso real del dueño (19-nov): «DESIGNS FOR SPORT MULTI + PHYTO (4 cápsulas)» quedó en el diario con 0 kcal y
`nutrition = null`, aunque su pote de la Alacena tenía la tabla completa de internet (1500 mcg RAE de vitamina A, 400 mg
de C, 50 mcg de D… por porción de 4 cápsulas). `log_consumed_meal` solo leía las MACROS del pote
(`suplementos.macros_de_porciones`); las vitaminas nunca llegaban a la fila, y el contador del día (que lee
`nutrition.source == 'food_label'` o los ingredientes) sumaba cero.

`etiqueta_del_pote` traduce `serving_label.supplement_facts` a la etiqueta del contador (`food_label_nutrients.FoodLabel`)
con las MISMAS reglas estrictas que ya tiene el contador: cantidad absoluta impresa (nunca % VD), tabla legible y por
porción, la unidad exacta del indicador (sin convertir UI; el folato en «mcg» cuenta 1:1 como DFE, lote 1113) y, en minerales, nunca la masa del
compuesto. Lo que no cumple se queda fuera (mejor «sin dato» que un número inventado). tooltip-anchor: P1-PLAN-LOTE-1112
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Optional

# clave del contador → (patrón del nombre impreso, unidad exigida). El orden importa: la primera que encaja gana.
_MAPA = (
    ("vit_b12", "b12_mcg", r"\bb-?\s?12\b|cobalamin", "mcg"),
    ("vit_b6", "vit_b6_mg", r"\bb-?\s?6\b|pyridox|piridox", "mg"),
    ("vit_a", "vit_a_mcg", r"\bvitamin[ae]?\s+a\b", "mcg rae"),
    ("vit_c", "vit_c_mg", r"\bvitamin[ae]?\s+c\b|ascorb", "mg"),
    ("vit_d", "vit_d_mcg", r"\bvitamin[ae]?\s+d\d?\b|colecalciferol|cholecalciferol", "mcg"),
    # solo la vitamina E de la tabla (alfa-tocoferol): «Vitamin E Isomers» (tocotrienoles) o «mixed tocopherols» no lo son
    ("vit_e", "vit_e_mg", r"^vitamin[ae]?\s+e(\s*\(.*\))?$|^(d-?|dl-?)?alpha-?tocopher|^(d-?|dl-?)?alfa-?tocofer", "mg"),
    ("vit_k", "vit_k_mcg", r"\bvitamin[ae]?\s+k\d?\b|phyllo|filoquin|menaquin", "mcg"),
    ("folato", "folate_mcg", r"folat|folic|folico", "mcg dfe"),
    ("calcio", "calcium_mg", r"\bcalci(um|o)\b", "mg"),
    ("hierro", "iron_mg", r"\biron\b|\bhierro\b", "mg"),
    ("magnesio", "magnesium_mg", r"\bmagnesi(um|o)\b", "mg"),
    ("zinc", "zinc_mg", r"\bzin[ck]\b", "mg"),
    ("selenio", "selenium_mcg", r"\bseleni(um|o)\b", "mcg"),
    ("potasio", "potassium_mg", r"\bpotassium\b|\bpotasio\b", "mg"),
    ("sodio", "sodium_mg", r"\bsodium\b|\bsodio\b", "mg"),
    ("yodo", "iodine_mcg", r"\biodine\b|\byodo\b", "mcg"),
    ("fibra", "fiber_g", r"\bfib(er|re|ra)\b", "g"),
)
_MAPA_RX = tuple((clave, re.compile(p), u) for _, clave, p, u in _MAPA)
_MINERALES = {"calcium_mg", "iron_mg", "magnesium_mg", "zinc_mg", "selenium_mcg", "potassium_mg", "sodium_mg",
              "iodine_mcg"}
# la unidad que el contador declara para cada clave (`food_label_nutrients.FoodLabelNutrient.unit`)
_UNIDAD_SALIDA = {"g": "g", "mg": "mg", "mcg": "mcg", "mcg rae": "mcg RAE", "mcg dfe": "mcg DFE"}


def _plano(s) -> str:
    s = unicodedata.normalize("NFD", str(s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _unidad(u) -> str:
    u = _plano(u).strip().replace("µ", "mc").replace("μ", "mc")
    return "mcg" + u[2:] if u.startswith("ug") else u


def etiqueta_del_pote(serving_label) -> Optional[dict]:
    """La etiqueta del contador (`FoodLabel`, por UNA porción) desde la tabla de vitaminas del pote, o None."""
    from supplement_labels import normalize
    if isinstance(serving_label, str):
        try:
            serving_label = json.loads(serving_label)
        except ValueError:
            return None
    if not isinstance(serving_label, dict):
        return None
    facts = normalize(serving_label.get("supplement_facts"))
    if not facts or not facts.get("table_readable") or facts.get("amount_basis") != "per_serving":
        return None
    vistos, nutrientes = set(), []
    for n in facts["nutrients"]:
        if n.get("amount") is None:
            continue
        nombre = _plano(n.get("name")).strip()
        for clave, rx, unidad in _MAPA_RX:
            if not rx.search(nombre):
                continue
            if clave == "vit_e_mg" and re.search(r"tocotrien|mixed|mezcla|isomer", _plano(n.get("form"))):
                break
            # minerales: solo la cantidad del mineral, nunca la masa de la sal (citrato, óxido…)
            if clave in _MINERALES and n.get("basis") == "compound":
                break
            u = _unidad(n.get("unit"))
            # [P1-PLAN-LOTE-1113] decisión del dueño: el folato en «mcg» a secas cuenta 1:1 como DFE. Con folato o
            # metilfolato es lo correcto (las tablas nuevas ya dan el DFE: «680 mcg DFE (408 mcg 5-MTHF)»); con ácido
            # fólico se queda corto (1 mcg = 1,7 DFE), nunca se pasa.
            if clave == "folate_mcg" and u == "mcg":
                u = unidad
            if clave in vistos or u != unidad:
                break
            vistos.add(clave)
            nutrientes.append({"key": clave, "amount": float(n["amount"]), "unit": _UNIDAD_SALIDA[unidad]})
            break
    if not nutrientes:
        return None
    return {"serving_text": (facts.get("serving_text") or "1 porción")[:200], "nutrients": nutrientes}
