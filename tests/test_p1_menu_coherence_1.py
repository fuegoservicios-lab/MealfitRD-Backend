"""[P1-MENU-COHERENCE-1 · 2026-07-29] Coherencia de menú pedida por el owner sobre el plan
vivo 73db1e79 ("comer lechosa en el almuerzo es raro — quedaría mejor de merienda"):

1. [FRUTA-AGUA×CARNE] "Brochetas de Chuleta de Cerdo … y Ensalada de LECHOSA" no disparaba
   el clash dulce↔salado: la decisión original excluía proteínas ("pollo con piña aceptable").
   Refinamiento: las frutas DE AGUA (lechosa/papaya/melón/sandía/mamey/zapote) junto a un
   plato de carne/pescado SÍ son clash; mango/piña/guayaba conservan la exclusión original.
2. [SAL-BOLT] "Añade Sal al guiso… Incorpóralo con cuidado" ×3 en el revoltillo del jueves:
   (a) "Sal al gusto" resolvía macros TODO-cero y `0>=0` la hacía "proteína dominante" →
   piso de proteína FÍSICA ≥3 g en el sweep de parity; (b) "Sal" (3 chars) no producía
   tokens ≥4 → el dedup jamás se activaba y cada re-corrida del chain añadía otra copia →
   fallback al nombre completo.
3. [BIGFRUIT-FRAC] "½ Piña mediana (100g)" escapaba DOS veces (P mayúscula + lead ½) y ni
   ¼ es honesto (100/1500 = 6.7%) → lead de GRAMOS. "½ lechosa mediana (395g)" (±12%) y
   "1 lechosa (650g)" (≥60%) intactos.
4. [DISPLAY] "1 pedazos de yautía" → "1 pedazo…"; "2 lonjas de pan" → "2 rebanadas de pan"
   (lonja es unidad de queso/embutido); "1½ cilantro fresco" (fracción unicode pegada
   evadía `\\d+`, clase P1-CITRUS-UNICODE-FRAC) → "1½ ramitas de cilantro fresco".

tooltip-anchor: P1-MENU-COHERENCE-1
"""
from __future__ import annotations

import re

import graph_orchestrator as go


# ─────────────────────── 1. fruta de agua × carne ───────────────────────

def test_water_fruit_beside_meat_is_clash():
    for nm in (
        "Brochetas de Chuleta de Cerdo a la Parrilla con Yautía Asada y Ensalada de Lechosa",
        "Pollo Salteado al Wok con Guineo y Lechosa sobre Cama de Espinacas",
        "Mero a la Plancha con Ensalada de Melón",
    ):
        assert go._meal_has_sweet_savory_clash({"name": nm}) is True, nm


def test_original_tropical_pairings_still_accepted():
    for nm in (
        "Cerdo Agridulce con Piña",
        "Pollo con Guayaba al Horno",
        "Ensalada de Lechosa con Yogurt",
        "Lechosa Fresca en Cubos",
        "Mango con Queso Fresco",
    ):
        assert go._meal_has_sweet_savory_clash({"name": nm}) is False, nm


def test_arroz_mango_clash_unchanged():
    assert go._meal_has_sweet_savory_clash({"name": "Arroz con Mango"}) is True
    assert go._meal_has_sweet_savory_clash({"name": "Revoltillo con Mango"}) is True


# ─────────────────────── 2. sal-bolt del sweep de parity ───────────────────────

class _ParityDB:
    def macros_from_ingredient_string(self, s):
        sl = str(s).lower()
        if "sal" in sl and "al gusto" in sl:
            return {"name": "Sal", "protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}
        if "pollo" in sl:
            return {"name": "Pechuga de pollo", "protein": 30.0, "carbs": 0.0, "fats": 3.0, "kcal": 150.0}
        if "res" in sl:
            return {"name": "Res", "protein": 25.0, "carbs": 0.0, "fats": 8.0, "kcal": 180.0}
        return None


def test_sal_never_gets_fabricated_step():
    plan = {"days": [{"meals": [{
        "name": "Revoltillo de Queso Blanco con Arepitas",
        "ingredients": ["Sal al gusto", "Pimienta negra al gusto"],
        "recipe": ["Mise en place: bate el huevo.", "Montaje: sirve."]}]}]}
    assert go.ensure_protein_step_parity(plan, db=_ParityDB()) == 0
    steps = plan["days"][0]["meals"][0]["recipe"]
    assert not any("Sal al guiso" in str(s) for s in steps)


def test_parity_still_adds_step_for_real_protein():
    plan = {"days": [{"meals": [{
        "name": "Bowl de Vegetales",
        "ingredients": ["150 g de pechuga de pollo"],
        "recipe": ["Mise en place: lava.", "Montaje: sirve."]}]}]}
    assert go.ensure_protein_step_parity(plan, db=_ParityDB()) == 1


def test_short_name_dedup_via_full_name_fallback():
    """"Res" (3 chars, cero tokens ≥4) re-añadía una copia por corrida — el fallback al
    nombre completo corta la oscilación con STEP-INTEGRATE."""
    plan = {"days": [{"meals": [{
        "name": "Guiso Criollo",
        "ingredients": ["150 g de res en trozos"],
        "recipe": ["Mise en place: pica.", "Añade la res al guiso y cocina 20 minutos."]}]}]}
    assert go.ensure_protein_step_parity(plan, db=_ParityDB()) == 0, \
        "la receta YA usa la res — nada que añadir en ninguna re-corrida"


# ─────────────────────── 3. bigfruit fraccionario ───────────────────────

def test_bigfruit_fraction_lead_to_grams():
    days = [{"meals": [{"ingredients": ["½ Piña mediana (100g)", "trozo de queso"]}]}]
    assert go._bigfruit_count_fraction_honesty(days) == 1
    assert days[0]["meals"][0]["ingredients"][0] == "100 g de pina"


def test_bigfruit_honest_leads_untouched():
    days = [{"meals": [{"ingredients": [
        "½ lechosa mediana (395g)",      # ±12% de ½·700 → honesto
        "1 lechosa mediana (650 g)",     # ≥60% de la unidad → contrato original
    ]}]}]
    assert go._bigfruit_count_fraction_honesty(days) == 0


def test_bigfruit_count_one_to_fraction_unchanged_behavior():
    days = [{"meals": [{"ingredients": ["1 lechosa mediana (202 g)"]}]}]
    assert go._bigfruit_count_fraction_honesty(days) == 1
    assert days[0]["meals"][0]["ingredients"][0].startswith("¼ de lechosa mediana")


# ─────────────────────── 4. display: pedazos / lonjas / 1½ cilantro ───────────────────────

def test_display_pedazos_and_lonjas_de_pan():
    days = [{"meals": [{"name": "Mero con Yautía",
                        "ingredients": ["1 pedazos de yautía (≈250 g)",
                                        "2 lonjas de pan integral"],
                        "recipe": ["Sirve."]}]}]
    go._polish_finalize_display(days)
    ings = days[0]["meals"][0]["ingredients"]
    assert ings[0].startswith("1 pedazo de yautía")
    assert ings[1] == "2 rebanadas de pan integral"


def test_herb_unicode_fraction_lead():
    days = [{"meals": [{"name": "Conejo Guisado",
                        "ingredients": ["1½ cilantro fresco para decorar", "½ cilantro fresco"],
                        "recipe": []}]}]
    assert go._herb_count_and_gender_polish(days) >= 1
    ings = days[0]["meals"][0]["ingredients"]
    assert ings[0].startswith("1½ ramitas de cilantro")
    assert ings[1].startswith("½ ramita de cilantro")
