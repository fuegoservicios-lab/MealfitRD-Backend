"""[P1-SWAP-PROSE-HONEST · 2026-07-29] Evidencia: plan vivo
deefa5f0-51c6-40ba-9579-c9fc660cb4c4, día 1 desayuno ("Batido Cremoso de Auyama y Yogurt Griego
con Arándanos"). El swap huevo→yogur de `_substitute_blended_raw_egg` (P1-MENU-COHERENCE-2)
producía prosa rota de dos formas independientes:

1. "el yogur griego de yogur griego" — la idioma "la clara de huevo" se partía entre dos patrones
   secuenciales del chain `_egg_step_subs` (uno consumía "la clara", el otro "huevo" por separado).
2. "Vierte el en un vaso" — `_run_assembly_validations` detectaba 'huevo' como huérfano SOLO
   porque la nota de seguridad ⚠ (que menciona "huevo" a propósito) entraba al blob de detección;
   el rewrite resultante corría `_reparar_concordancia_huevo` sobre TODOS los pasos —incluido el
   Montaje, que no mencionaba huevo/clara en absoluto— y `_ORPHAN_EGG_PARTICIPIO_RX` borraba
   "batido" (el SUSTANTIVO del plato, no un participio de huevo) sin mirar contexto.

Además `desc` nunca se sincronizaba con el swap (seguía vendiendo "claras de proteína"/"huevo").

tooltip-anchor: P1-SWAP-PROSE-HONEST
"""
from __future__ import annotations

import graph_orchestrator as go


# ─────────────── Defect 1: "X de X" stutter en el swap huevo→yogur ───────────────

def _evidence_meal():
    """Reconstrucción del meal vivo (día 1 desayuno, plan deefa5f0-51c6-40ba-9579-c9fc660cb4c4)
    ANTES del fix: ingredients con clara de huevo (idioma singular, sin 'pasteurizada' — el caso
    que rompía), recipe con el Toque de Fuego + Montaje ("Vierte el batido en un vaso...")."""
    return {
        "name": "Batido Cremoso de Auyama y Claras de Huevo con Arándanos",
        "desc": (
            "Un batido fresco y espeso que combina la dulzura natural de la auyama con el poder "
            "proteico de las claras de proteína, realzado por un toque de canela y vainilla."
        ),
        "ingredients": [
            "1 taza de auyama cocida (puré, 240g)",
            "1 clara de huevo",
            "¾ taza de leche descremada (180 ml)",
            "¼ taza de arándanos frescos (30g)",
            "½ cdta de canela en polvo",
        ],
        "ingredients_raw": [
            "1 taza de auyama cocida (puré, 240g)",
            "1 clara de huevo",
            "¾ taza de leche descremada (180 ml)",
            "¼ taza de arándanos frescos (30g)",
            "½ cdta de canela en polvo",
        ],
        "recipe": [
            "Mise en place: pesa la auyama, mide la leche y los arándanos.",
            "El Toque de Fuego: En una licuadora, combina el puré de auyama, la clara de huevo, "
            "la leche descremada, los arándanos, la canela, la vainilla, la miel y una pizca de sal.",
            "Montaje: Vierte el batido en un vaso grande o bowl. Decora con unos arándanos "
            "enteros por encima y sirve frío.",
        ],
        "protein": 15, "carbs": 30, "fats": 5, "cals": 220,
    }


def test_defect1_stutter_is_impossible():
    """La frase "el yogur griego de yogur griego" (idioma "la clara de huevo" partido entre dos
    patrones secuenciales) nunca sobrevive al swap."""
    meal = _evidence_meal()
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    blob = " ".join(str(s) for s in meal["recipe"]).lower()
    assert "yogur griego de yogur griego" not in blob
    assert "de yogur griego" not in blob  # ninguna variante de "X de yogur griego"
    assert "clara" not in blob and "huevo" not in blob


def test_defect1_montaje_step_untouched_noun_preserved():
    """El Montaje ("Vierte el batido en un vaso...") no menciona huevo/clara → debe quedar
    BYTE-IDÉNTICO tras el swap (ninguna substitución debería tocarlo)."""
    meal = _evidence_meal()
    _montaje_before = meal["recipe"][2]
    go._substitute_blended_raw_egg(meal, None)
    assert meal["recipe"][2] == _montaje_before, (
        f"un paso que no mencionaba huevo/clara cambió: {meal['recipe'][2]!r}"
    )
    assert "vierte el batido en un vaso" in meal["recipe"][2].lower()


def test_defect1_idempotent():
    """Aplicar el swap una segunda vez (ya sin términos de huevo en `ingredients`) no cambia nada."""
    meal = _evidence_meal()
    go._substitute_blended_raw_egg(meal, None)
    _rec_after_first = list(meal["recipe"])
    _ing_after_first = list(meal["ingredients"])
    _name_after_first = meal["name"]
    changed_again = go._substitute_blended_raw_egg(meal, None)
    assert changed_again is False
    assert meal["recipe"] == _rec_after_first
    assert meal["ingredients"] == _ing_after_first
    assert meal["name"] == _name_after_first


def test_defect1_already_correct_sentence_left_alone():
    """Un paso YA correcto (sin huevo/clara) sobrevive byte-idéntico al swap, aunque OTRO paso
    del mismo meal sí dispare la sustitución (`changed=True` a nivel de meal)."""
    meal = _evidence_meal()
    meal["recipe"].append("Nota extra: Sirve el plato frío con una ramita de menta.")
    _extra_before = meal["recipe"][-1]
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    assert meal["recipe"][-1] == _extra_before


# ─────────────── Defect 2: "Vierte el en un vaso" (participio huérfano sobre-borra) ───────────────

def test_defect2_orphan_rewrite_never_deletes_dish_noun():
    """Unit-level: `_reparar_concordancia_huevo` NUNCA debe borrar "batido" cuando funciona como
    sustantivo del plato (precedido por artículo), solo cuando es participio adjetival."""
    paso_montaje = "Montaje: Vierte el batido en un vaso grande o bowl. Decora y sirve."
    out = go._reparar_concordancia_huevo(paso_montaje, ["huevo"], go.strip_accents)
    assert out == paso_montaje, f"'batido' (sustantivo) fue borrado: {out!r}"
    assert "vierte el en un vaso" not in out.lower()

    # Contraparte: SÍ debe borrar el participio adjetival huérfano (caso P1-EGG-PROSE-ORPHAN).
    paso_adj = "Vierte yogur griego entero batidas y cocina 2-3 min hasta que cuajen."
    out2 = go._reparar_concordancia_huevo(paso_adj, ["huevo"], go.strip_accents)
    assert "batidas" not in out2.lower()
    assert "cuajen" not in out2.lower()


def test_defect2_gate_requires_egg_orphan():
    """Sin huevo/clara en `orphan_keys`, el paso se conserva intacto (aunque diga 'batido')."""
    paso = "Montaje: Vierte el batido en un vaso."
    out = go._reparar_concordancia_huevo(paso, ["pollo"], go.strip_accents)
    assert out == paso


def test_defect2_dangling_article_safety_net():
    """Si (hipotéticamente) el borrado del participio dejara un artículo colgando que el
    original NO tenía, la red de seguridad revierte al paso original."""
    # Fabricamos un paso donde el participio SÍ está en posición adjetival ambigua para forzar
    # el escenario: confiar en que el propio regex+lookbehind ya lo evita es el objetivo del
    # test anterior; aquí probamos la red de seguridad directamente contra el regex de
    # artículo colgando.
    assert go._DANGLING_ARTICLE_RX.search("Vierte el en un vaso grande.")
    assert not go._DANGLING_ARTICLE_RX.search("Vierte el batido en un vaso grande.")
    assert not go._DANGLING_ARTICLE_RX.search("el de la casa")  # elipsis válida, no debe matchear


def test_defect2_end_to_end_note_excluded_from_orphan_blob():
    """End-to-end: un meal YA corregido por `_substitute_blended_raw_egg` (ingredients/recipe/name
    en yogur griego) + la nota ⚠ (que menciona 'huevo' a propósito) NO debe disparar un orphan-key
    falso en `_run_assembly_validations` — y por lo tanto NUNCA llega a
    `_reparar_concordancia_huevo`, así que "Vierte el batido..." sobrevive intacto."""
    meal = _evidence_meal()
    go._substitute_blended_raw_egg(meal, None)
    # Simula lo que `_apply_food_safety_fixes` hace después: anexar la nota ⚠ (contiene "huevo").
    meal["recipe"] = list(meal["recipe"]) + [go._FOOD_SAFETY_NOTE_BLENDED_SUBBED]
    montaje_before = meal["recipe"][2]

    plan = {"days": [{"day": 1, "meals": [meal]}]}
    go._run_assembly_validations(plan, skeleton={}, affected_days_set=set())

    assert meal["recipe"][2] == montaje_before, (
        f"la nota de seguridad disparó un rewrite espurio sobre el Montaje: {meal['recipe'][2]!r}"
    )
    assert "vierte el en un vaso" not in " ".join(meal["recipe"]).lower()


# ─────────────── Defect 3: `desc` nunca se sincronizaba ───────────────

def test_desc_synced_never_names_absent_ingredient():
    meal = _evidence_meal()
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    desc_low = meal["desc"].lower()
    assert "clara" not in desc_low and "huevo" not in desc_low
    assert "yogur griego" in desc_low
    ing_blob = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert "huevo" not in ing_blob and "clara" not in ing_blob


def test_desc_untouched_when_it_never_mentioned_egg():
    """`desc` que ya no menciona huevo/clara queda byte-idéntico (nada que sincronizar)."""
    meal = _evidence_meal()
    meal["desc"] = "Un batido fresco con auyama, arándanos y un toque de canela."
    _desc_before = meal["desc"]
    go._substitute_blended_raw_egg(meal, None)
    assert meal["desc"] == _desc_before


# ─────────────── Badge: `_dish_quality_reason` companion ───────────────

class _FakeInfo:
    def __init__(self, protein=25.0, carbs=0.0, fats=2.0):
        self.protein, self.carbs, self.fats = protein, carbs, fats


class _FakeDB:
    def lookup(self, bare):
        return _FakeInfo()


def test_portion_estimate_reason_set_on_quantity_backstop():
    meal = {
        "ingredients": ["Salmón"],
        "ingredients_raw": ["Salmón"],
        "recipe": ["Mise en place: prepara todo.", "El Toque de Fuego: cocina 10 min.",
                   "Montaje: sirve caliente."],
    }
    fixed = go._ensure_ingredient_quantities(meal, _FakeDB())
    assert fixed >= 1
    assert meal.get("_dish_quality_degraded") is True
    assert meal.get("_dish_quality_reason") == "portion_estimate"


def test_recipe_missing_reason_set_on_nonempty_recipe_backstop():
    meal = {"name": "Pollo Guisado", "ingredients": ["Pollo", "Arroz"], "recipe": []}
    filled = go._ensure_nonempty_recipe(meal)
    assert filled is True
    assert meal.get("_dish_quality_reason") == "recipe_missing"


def test_ingredients_missing_reason_set_on_nonempty_ingredients_backstop():
    meal = {"name": "Pollo Guisado con Yuca", "ingredients": []}
    filled = go._ensure_nonempty_ingredients(meal)
    assert filled is True
    assert meal.get("_dish_quality_reason") == "ingredients_missing"


def test_setdefault_does_not_override_more_specific_reason():
    """Si un backstop más severo (ingredientes ausentes) YA marcó la razón, un backstop de
    portion-estimate posterior sobre el MISMO meal no debe pisarla."""
    meal = {"name": "Pollo Guisado con Yuca", "ingredients": []}
    go._ensure_nonempty_ingredients(meal)
    assert meal["_dish_quality_reason"] == "ingredients_missing"
    go._ensure_ingredient_quantities(meal, _FakeDB())
    assert meal["_dish_quality_reason"] == "ingredients_missing", (
        "portion_estimate no debe pisar una razón más severa ya presente"
    )


# ═══════════════ Ronda adversarial P1-SWAP-PROSE-HONEST-2 (2026-07-29) ═══════════════
#
# Finding [important] (badge): en el ORDEN REAL de producción (persist boundary
# graph_orchestrator.py ~L23000/~L23140 y finalizador de update ~L23680/~L23750),
# `_ensure_ingredient_quantities` (severidad baja, 'portion_estimate') corre ANTES que
# `_ensure_nonempty_recipe` (severidad alta, 'recipe_missing'). `setdefault` puro deja ganar
# al primero, no al más severo → un plato con receta enteramente boilerplate se etiqueta
# "Cantidad estimada" en vez de "Receta básica", ocultando el caso que sí justificaría
# regenerar. Fix: `_set_dish_quality_reason` con tabla de severidad — ver arriba en
# graph_orchestrator.py. Los tests de esta sección reproducen el ORDEN DE PRODUCCIÓN
# exacto (portion_estimate primero) que el test `test_setdefault_does_not_override_more_specific_reason`
# de arriba NO cubría (ese fija el orden INVERSO).
#
# Finding [important] (prosa): el swap huevo→yogur (`_egg_step_subs`) reemplaza el
# SUSTANTIVO ("huevo(s)"/"clara(s)") por "(el) yogur griego" (masc. singular) pero deja
# colgando el adjetivo/participio que en es-DO SIEMPRE concordaba en género/número con el
# sustantivo original ("huevos ENTEROS", "claras BATIDAS", "Huevos REVUELTOS", "claras...
# completamente COCIDAS") — produciendo prosa que ningún hablante nativo escribiría. Medido
# contra 212 comidas reales de producción: 16/81 (≈20%) de los swaps disparados quedaban con
# esta discordancia. `_reparar_concordancia_huevo`/`_ORPHAN_EGG_PARTICIPIO_RX` NUNCA la
# atrapaban porque solo se disparan cuando `_run_assembly_validations` detecta 'huevo'/'clara'
# como huérfano en el texto — y para cuando ese chequeo corre, `_egg_step_subs` YA borró todo
# rastro del sustantivo original. Fix: `_fix_egg_swap_dangling_adjectives`, anclado al literal
# "yogur griego" que la propia sustitución inserta (no un scan global del paso).


def test_p2_production_order_recipe_missing_beats_portion_estimate():
    """Reproduce el ORDEN REAL de producción (persist boundary y finalizador de update):
    `_ensure_ingredient_quantities` corre PRIMERO (marca 'portion_estimate'), después
    `_ensure_nonempty_recipe` (marca 'recipe_missing'). El backstop MÁS SEVERO debe ganar el
    badge, no el que corrió primero — pre-fix este test fallaba con 'portion_estimate'."""
    meal = {
        "name": "Pollo Guisado",
        "ingredients": ["Pollo"],          # sin cantidad líder → dispara qty-presence guard
        "ingredients_raw": ["Pollo"],
        "recipe": [],                       # vacío → dispara el backstop de receta
    }
    fixed_qty = go._ensure_ingredient_quantities(meal, _FakeDB())
    assert fixed_qty >= 1
    assert meal["_dish_quality_reason"] == "portion_estimate"  # gana el primero en correr (aún)

    filled_recipe = go._ensure_nonempty_recipe(meal)
    assert filled_recipe is True
    assert meal["_dish_quality_reason"] == "recipe_missing", (
        "el backstop de receta ausente (más severo) debe desplazar a portion_estimate "
        "aunque haya corrido DESPUÉS — mealAdvisories.js decide el label solo por este campo"
    )


def test_p2_ingredients_missing_beats_portion_estimate_regardless_of_order():
    """Empate de severidad entre ingredients_missing/recipe_missing vs portion_estimate: sin
    importar cuál de los dos backstops de severidad 3 corrió, portion_estimate (severidad 1)
    nunca lo desplaza, corra antes o después."""
    meal = {"name": "Pollo Guisado con Yuca", "ingredients": []}
    go._ensure_nonempty_ingredients(meal)
    assert meal["_dish_quality_reason"] == "ingredients_missing"
    # portion_estimate llega DESPUÉS esta vez (orden real de ambos finalizers)
    meal["ingredients"] = ["Pollo"]  # ahora sí hay ingredientes, pero sin cantidad
    go._ensure_ingredient_quantities(meal, _FakeDB())
    assert meal["_dish_quality_reason"] == "ingredients_missing"


def test_p2_severity_table_tie_preserves_first_writer():
    """Empate real de severidad (ingredients_missing vs recipe_missing, ambos 3): el helper
    preserva al PRIMERO que corrió — no invierte el comportamiento pre-existente donde no hay
    evidencia de un caso vivo que pida lo contrario."""
    meal = {}
    go._set_dish_quality_reason(meal, "ingredients_missing")
    go._set_dish_quality_reason(meal, "recipe_missing")
    assert meal["_dish_quality_reason"] == "ingredients_missing"


def test_p2_severity_helper_never_lowers_priority():
    """Un reason desconocido/vacío (severidad 0 por default) nunca desplaza uno ya asignado."""
    meal = {"_dish_quality_reason": "recipe_missing"}
    go._set_dish_quality_reason(meal, "portion_estimate")
    assert meal["_dish_quality_reason"] == "recipe_missing"
    go._set_dish_quality_reason(meal, "")
    assert meal["_dish_quality_reason"] == "recipe_missing"


# ─────────────── Prosa: concordancia de género/número tras el swap ───────────────

def _meal_egg_step(step_text, name="Batido de prueba", desc=None):
    meal = {
        "name": name,
        "ingredients": ["2 huevos crudos"],
        "ingredients_raw": ["2 huevos crudos"],
        "recipe": [step_text],
    }
    if desc is not None:
        meal["desc"] = desc
    return meal


def test_p1_dangling_adjective_enteros_singularized():
    """'Bate los huevos ENTEROS con la clara...' → 'enteros' (masc. plural) queda colgando
    tras 'el yogur griego' (masc. singular) — DEBE concordar a singular, no quedar en plural."""
    meal = _meal_egg_step("Bate los huevos enteros con la clara, una pizca de sal y pimienta.")
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    step = meal["recipe"][0].lower()
    assert "enteros" not in step, f"'enteros' (plural) sobrevivió sin concordar: {step!r}"
    assert "griego entero" in step, f"esperaba concordancia a singular 'entero': {step!r}"


def test_p1_dangling_adjective_batidos_dropped():
    """'Vierte los huevos BATIDOS...' — 'batidos' describe un MÉTODO DE COCCIÓN DEL HUEVO que
    no aplica al yogur ('nadie bate el yogur para que cuaje') → se borra, no se concuerda."""
    meal = _meal_egg_step("Vierte los huevos batidos y cocina por 2-3 minutos.",
                          name="Tortilla de Huevo Express")
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    step = meal["recipe"][0].lower()
    assert "batidos" not in step and "batido" not in step, f"participio huérfano sobrevivió: {step!r}"
    assert step == "vierte el yogur griego y cocina por 2-3 minutos."


def test_p1_dangling_adjective_capitalization_preserved_at_sentence_start():
    """'Huevos revueltos cremosos...' a INICIO de oración (tras un punto) — la mayúscula debe
    sobrevivir al swap (bare pattern, antes se perdía: 'yogur griego revueltos...' en minúscula
    a media frase) Y 'revueltos'/'cremosos' deben concordar/borrarse correctamente."""
    meal = _meal_egg_step(
        "Mise en place: prepara todo.",
        name="Revoltillo de Huevos con Plátano Maduro",
        desc="...con energía. Huevos revueltos cremosos con plátano maduro y un toque de canela.",
    )
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    desc = meal["desc"]
    assert "Yogur griego cremoso" in desc, f"mayúscula perdida o concordancia rota: {desc!r}"
    assert "revueltos" not in desc.lower() and "revuelto" not in desc.lower()
    assert "yogur griego revueltos" not in desc.lower()


def test_p1_dangling_adverb_plus_participle_both_dropped():
    """'...las claras completamente COCIDAS' — el adverbio 'completamente' media entre el
    sustantivo huérfano y el participio; ambos deben desaparecer juntos (un adverbio sin nada
    que modificar quedaría tan huérfano como el propio participio)."""
    meal = _meal_egg_step(
        "Hornea hasta que la avena horneada esté firme y las claras completamente cocidas.",
        name="Avena horneada con clara",
    )
    meal["ingredients"] = ["2 claras de huevo"]
    meal["ingredients_raw"] = ["2 claras de huevo"]
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    step = meal["recipe"][0].lower()
    assert "cocidas" not in step and "cocida" not in step
    assert "completamente" not in step
    assert step == "hornea hasta que la avena horneada esté firme y el yogur griego."


def test_p1_dangling_adjective_unrelated_tail_word_untouched():
    """Si la palabra que sigue al swap NO es un participio/adverbio conocido ('con'), el resto
    de la oración se preserva verbatim — la función no debe sobre-alcanzar."""
    meal = _meal_egg_step("Bate los huevos con sal y pimienta al gusto.", name="Huevos con vegetales")
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    assert meal["recipe"][0] == "Bate el yogur griego con sal y pimienta al gusto."


def test_p1_dangling_adjective_no_tail_untouched():
    """Sin participio colgando ('Vierte los huevos en el sartén...'), el swap no introduce
    ninguna palabra nueva ni corta contenido de más."""
    meal = _meal_egg_step("Vierte los huevos en el sartén y cocina a fuego medio.")
    changed = go._substitute_blended_raw_egg(meal, None)
    assert changed is True
    assert meal["recipe"][0] == "Vierte el yogur griego en el sartén y cocina a fuego medio."
