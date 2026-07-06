"""[P1-DISPLAY-RESTORE-FROM-RAW + P2-CLOSER-FAMILY-ANNOT + P2-LEGUME-NO-LATA-DEFAULT
· 2026-07-06] Batch "implementa" del review #12 (plan cd4ae3c3).

1. P1 CRÓNICO mitigado: pases históricos QUITAN líneas del display sin tocar raw
   (mozzarella del wrap con Mise/Montaje que la usan; 'Sal al gusto' del autofix
   de sodio). En la frontera, toda fila raw-huérfana cuyo stem SÍ se usa en los
   pasos no-nota vuelve al display (prettificada, cap 3/meal). El WARN
   [P1-RAW-MISALIGN] sigue vivo para cazar a los droppers de raíz.
2. Closer/repair family-aware con ANOTACIÓN: "½ taza de queso ricotta (105g)"
   existente + closer eligiendo queso → se ESCALA esa línea (anotación como
   base) en vez de añadir "75 g de queso" genérica que ningún paso usa.
3. Legumbres sin LATA en defaults: el aggregator convierte cocido→seco (yield
   0.35×) porque el SKU es seco; una lata (cocida) hereda la necesidad seca y
   sub-compra ~3× ("Habichuelas rojas: 1 lata 15 Oz" para ~10 porciones).
4. "20 g de Maní fileteadas" → "20 g de maní fileteado" (unidades de peso en la
   rama de concordancia, invariantes: jamás "20 gs").
"""
import pytest

import graph_orchestrator as go
import shopping_calculator as sc


# ───────────── 1. restore display desde raw-huérfanos ─────────────

def _wrap_meal():
    return {"days": [{"day": 1, "meals": [{
        "name": "Wrap de Tortilla con Revoltillo",
        "ingredients": ["4 tortillas integrales", "1 huevo", "1½ pepinos"],
        "ingredients_raw": ["4 tortilla integral", "1 huevos", "15 g de queso mozzarella",
                            "1.5 pepino", "Sal al gusto"],
        "recipe": [
            "Mise en place: bate el huevo con una pizca de sal. Corta la mozzarella en tiras.",
            "Montaje: extiende la tortilla, añade la mozzarella y el pepino.",
        ],
    }]}]}


def test_raw_orphans_used_in_steps_restored():
    plan = _wrap_meal()
    n = go._restore_display_from_raw_orphans(plan["days"])
    ings = plan["days"][0]["meals"][0]["ingredients"]
    blob = " ".join(str(x).lower() for x in ings)
    assert n >= 2, f"mozzarella y sal (usadas en pasos) deben volver al display: {ings}"
    assert "mozzarella" in blob, "la mozzarella que Mise/Montaje usan ya no es invisible"
    assert "sal" in blob, "la sal de la pizca del Mise vuelve (stem de 3 chars, bounded)"


def test_raw_orphan_not_in_steps_left_alone():
    plan = _wrap_meal()
    meal = plan["days"][0]["meals"][0]
    meal["ingredients_raw"].append("10 g de alcaparras")  # nadie las usa en pasos
    go._restore_display_from_raw_orphans(plan["days"])
    assert not any("alcaparra" in str(x).lower() for x in meal["ingredients"]), (
        "raw-only que la receta NO usa → no se restaura (conservador)"
    )


def test_no_restore_when_aligned():
    plan = {"days": [{"day": 1, "meals": [{
        "name": "X", "ingredients": ["1 huevo"], "ingredients_raw": ["1 huevo"],
        "recipe": ["Montaje: sirve el huevo."],
    }]}]}
    assert go._restore_display_from_raw_orphans(plan["days"]) == 0


# ───────────── 2. closer family-aware con anotación ─────────────

def test_closer_scales_annotated_family_line():
    meal = {
        "name": "Tostadas de Ricotta",
        "ingredients": ["3 rebanadas de pan integral", "½ taza de queso ricotta (105g)"],
        "ingredients_raw": ["3 rebanadas de pan integral", "½ taza de queso ricotta (105g)"],
        "recipe": ["Montaje: unta la ricotta."],
    }
    ok = go._scale_congruent_protein_line(meal, "Queso blanco", 75.0, None)
    assert ok is True, "línea anotada de la MISMA familia (queso) → se escala, no se apila"
    assert len(meal["ingredients"]) == 2, f"cero segunda línea genérica: {meal['ingredients']}"
    _q = next(s for s in meal["ingredients"] if "ricotta" in str(s).lower())
    assert "105g" not in str(_q).replace(" ", ""), f"la anotación creció con el cierre: {_q}"


# ───────────── 3. legumbres sin lata en defaults ─────────────

def test_legume_lata_dropped_from_defaults():
    defaults = {"habichuelas rojas": [
        {"grams": 425.0, "price": 50.0, "label": "15 Oz · Genérico", "unit": "lata"},
        {"grams": 453.6, "price": 65.0, "label": "1 Lb seca · Genérico", "unit": "funda"},
    ]}
    got = sc._resolve_brand_default("Habichuelas rojas", defaults)
    assert got is not None and all(p["unit"] != "lata" for p in got), (
        "lata (cocida) hereda la necesidad SECA del yield 0.35× → sub-compra 3×; fuera"
    )


def test_legume_all_lata_falls_back_to_master():
    defaults = {"habichuelas rojas": [
        {"grams": 425.0, "price": 50.0, "label": "15 Oz · Genérico", "unit": "lata"},
    ]}
    assert sc._resolve_brand_default("Habichuelas rojas", defaults) is None, (
        "solo latas → sin default (el master seco sigue siendo la base coherente)"
    )


def test_non_legume_lata_unaffected():
    defaults = {"atun en agua": [
        {"grams": 170.0, "price": 60.0, "label": "170 gr · Genérico", "unit": "lata"},
    ]}
    assert sc._resolve_brand_default("Atún en agua", defaults), "el atún EN LATA es su forma normal"


# ───────────── 4. gramática maní + unidades invariantes ─────────────

def test_mani_fileteado_grammar():
    from humanize_ingredients import _prettify_quantity_display as p
    assert p("20 g de Maní fileteadas") == "20 g de maní fileteado"
    assert p("110 g de queso") == "110 g de queso", "unidades de peso invariantes (jamás '110 gs')"
