# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-592 · 2026-09-27] Cuando los topes atan la proteína del día, entra una proteína NUEVA, no más de la topada.

Baterías reales de esta tarde (16 perfiles, 48 días): 6 días se entregaron bajo su banda (proteína 0,73-0,88; kcal
0,80-0,91). El patrón es uno: un día sin carne ni pescado —huevos, claras, edamame, cottage—, el cerrador sube lo que el
plato ya tiene, y los topes de realismo (huevos del día, 300 g de edamame, pan) lo recortan. `protein_floor_last_word`
lo deja escrito a propósito («el bump propone, el cap dispone»), pero su bump sólo re-escala lo que YA está: con todo
topado, el hueco no tiene salida. Ejemplo (texto libre, maní/sésamo, día 3): 1.885 kcal y 129 g de 2.350/176; el cierre
final lo llevaba a 179 g con 5 huevos, 9 claras y 460 g de edamame… y los topes lo devolvían a 129.

Aquí, después del bump y del tope y ANTES de medir, en un día que sigue bajo el piso: una proteína magra que el día no
usa (pechuga de pollo o de pavo, pescado blanco, res magra…: las candidatas del cerrador del repo, filtradas por alergias,
rechazos, religión, dieta, país, tiempo de cocina y compra única), en la comida principal con más hueco, hasta 150 g y
sin pasar el 107 % de las kcal del día. Nunca huevo, lácteo ni legumbre (lo que los topes ya recortaron), nunca en un
plato dulce o licuado, nunca con techo renal. Una sola vez por día: no hay bucle con los topes (150 g de pechuga no
dispara ninguno). Knob `MEALFIT_PROTEIN_NEW_WHEN_CAPPED` (True). tooltip-anchor: P1-PLAN-LOTE-592
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata

logger = logging.getLogger(__name__)

MAX_ADD_G = 150
MIN_ADD_G = 40
TECHO_KCAL = 1.07
_FUERA = re.compile(r"\b(huevos?|claras?|yemas?|queso|ricotta|cottage|requeson|leche|yogur\w*|edamame|soya|soja|tofu|"
                    r"tempeh|seitan|lentejas?|garbanzos?|habichuelas?|frijol\w*|guisantes?|gandul\w*|jamon|salami|"
                    r"salchich\w*|chorizo|bacalao|arenque|sardinas?|atun|higado|mariscos?|camarones?)\b")
_PREFERIDAS = ("pechuga de pollo", "pollo", "pechuga de pavo", "pavo", "pescado", "tilapia", "res", "cerdo")
_EMBARAZO = re.compile(r"embaraz|lactan", re.IGNORECASE)
_PESCADO = re.compile(r"\b(pescado|tilapia|mero|dorado|chillo|salmon|merluza|corvina)\b")


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PROTEIN_NEW_WHEN_CAPPED", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _condiciones(form_data) -> str:
    fd = form_data or {}
    partes = list(fd.get("medicalConditions") or []) + [str(fd.get("otherConditions") or "")]
    return _sa(" ".join(str(p) for p in partes))


def _candidatas(form_data, db, go, day) -> list:
    import constants
    restr = constants.alergias_y_rechazos(form_data)
    cands = go._safe_high_density_proteins(restr, db, min_protein=18.0, diet=form_data.get("dietType"),
                                           country=constants.country_for_form_data(form_data))
    cands = __import__("proteina_lista").filtrar_listas(cands, form_data)
    cands = __import__("compra_unica").candidatos_del_dia(cands, day, form_data)
    try:                                           # con la Nevera exigida por el revisor, sólo lo que la Nevera tiene
        ne = __import__("nevera_exigida")
        if ne.activo() and ne.lista(form_data) is not None:
            cands = [c for c in cands if ne.admite(getattr(c[2], "name", None) if len(c) > 2 else c[1], form_data)]
    except Exception:                                                          # noqa: BLE001
        pass
    embarazo = bool(_EMBARAZO.search(_condiciones(form_data)))
    texto_dia = " ".join(_sa(x) for m in (day.get("meals") or []) if isinstance(m, dict)
                         for x in (m.get("ingredients") or []) if isinstance(x, str))
    usadas = set()
    for m in (day.get("meals") or []):
        if isinstance(m, dict):
            try:
                usadas |= set(go._protein_gate_labels_in_meal(m))       # las etiquetas del gate de proteína repetida
            except Exception:                                                  # noqa: BLE001
                pass
    out = []
    for c in cands or []:
        info = c[2] if len(c) > 2 else None
        nombre = _sa(getattr(info, "name", None) or c[1])
        if _FUERA.search(nombre) or (embarazo and _PESCADO.search(nombre)):
            continue
        try:
            if set(go._protein_gate_labels_in_text(nombre)) & usadas:
                continue                               # «muslo de pollo» en un día con pechuga: el revisor lo rechaza
        except Exception:                                                      # noqa: BLE001
            pass
        cabeza = next((w for w in re.findall(r"[a-z]+", nombre) if len(w) >= 4 and w not in ("pechuga", "filete", "carne")),
                      nombre)
        if re.search(r"\b" + re.escape(cabeza), texto_dia):
            continue                                   # el día ya la usa: no se repite la misma proteína
        out.append(c)
    out.sort(key=lambda c: next((i for i, p in enumerate(_PREFERIDAS) if p in _sa(getattr(c[2], "name", c[1])
                                                                                  if len(c) > 2 else c[1])),
                                len(_PREFERIDAS)))
    return out


def _comida(day, go):
    """La comida principal (almuerzo o cena) con menos proteína; nunca un plato dulce o licuado."""
    sa = getattr(go, "strip_accents", None) or _sa
    mejores = []
    for m in (day.get("meals") or []):
        if not isinstance(m, dict):
            continue
        franja = _sa(m.get("meal"))
        if not re.search(r"almuerzo|cena|comida", franja):
            continue
        nombre = _sa(m.get("name"))
        if any(b in nombre for b in go._NO_COOK_BLENDED):
            continue
        try:
            if go._is_sweet_meal(m, sa):
                continue
        except Exception:                                                      # noqa: BLE001
            pass
        mejores.append((go._meal_macro_num(m.get("protein")), id(m), m))
    mejores.sort(key=lambda x: (x[0], x[1]))
    return mejores[0][2] if mejores else None


def cerrar(plan_data, form_data, db=None) -> list:
    """`["día 3: +150 g de pechuga de pollo en «…»"]`; [] si no aplica o ante cualquier error."""
    hechos = []
    try:
        if not activo() or not isinstance(plan_data, dict) or not form_data:
            return hechos
        if __import__("recorte_renal").techo_renal(plan_data) > 0:
            return hechos
        import graph_orchestrator as go
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
        tgt_p = go._meal_macro_num((plan_data.get("macros") or {}).get("protein"))
        tgt_k = go._meal_macro_num(plan_data.get("calories"))
        if tgt_p <= 0 or tgt_k <= 0:
            return hechos
        piso = tgt_p * 0.90
        max_add = MAX_ADD_G
        try:
            if any(t in _condiciones(form_data) for t in ("bariatr",)):
                max_add = min(max_add, int(go.BARIATRIC_PROTEIN_PORTION_CAP_G))
        except Exception:                                                      # noqa: BLE001
            pass
        for n, day in enumerate(plan_data.get("days") or [], 1):
            if not isinstance(day, dict):
                continue
            meals = [m for m in (day.get("meals") or []) if isinstance(m, dict)]
            p_dia = sum(go._meal_macro_num(m.get("protein")) for m in meals)
            k_dia = sum(go._meal_macro_num(m.get("cals") or m.get("calories")) for m in meals)
            if p_dia >= piso:
                continue
            meal = _comida(day, go)
            cands = _candidatas(form_data, db, go, day) if meal is not None else []
            if not cands:
                continue
            info = cands[0][2] if len(cands[0]) > 2 else None
            nombre = str(getattr(info, "name", None) or cands[0][1])
            p100 = float(getattr(info, "protein", 0) or 0)
            k100 = float(getattr(info, "kcal", 0) or 0)
            if p100 <= 0 or k100 <= 0:
                continue
            g = min(max_add, math.ceil((piso - p_dia) / p100 * 100 / 10.0) * 10)
            g = min(g, int((TECHO_KCAL * tgt_k - k_dia) / k100 * 100 / 10) * 10)
            if g < MIN_ADD_G:
                continue
            linea = f"{g} g de {nombre[:1].lower() + nombre[1:]}"
            meal.setdefault("ingredients", []).append(linea)
            if isinstance(meal.get("ingredients_raw"), list):
                meal["ingredients_raw"].append(linea)
            try:
                go._append_closer_protein_step(meal, nombre[:1].lower() + nombre[1:], no_cook=False)
            except Exception:                                                  # noqa: BLE001
                pass
            try:
                go._reflect_added_protein_in_name(meal, nombre, getattr(go, "strip_accents", None) or _sa)
            except Exception:                                                  # noqa: BLE001
                pass
            go._truth_up_meal_macros_from_strings(meal, db)
            meal["_proteina_nueva_592"] = f"+{linea}"
            meal.pop("_display", None)
            hechos.append(f"día {n}: +{linea} en «{str(meal.get('name'))[:40]}»")
        if hechos:
            try:
                go.refresh_delivered_macros(plan_data)
            except Exception:                                                  # noqa: BLE001
                pass
            logger.info(f"🍗 [P1-PLAN-LOTE-592] proteína nueva donde los topes ataban: {'; '.join(hechos)}")
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-592] no-op: {type(e).__name__}: {e}")
    return hechos


__all__ = ["cerrar"]
