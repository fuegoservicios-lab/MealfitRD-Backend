"""[P1-RECIPE-STEP-SANITIZE · 2026-07-11] Pasos de receta coherentes tras swaps de proteína
+ techo global de bolt del closer.

Tres absurdos medidos en vivo (12 screenshots del owner, plan renovado 2026-07-11):
1. "Cocina yogurt griego entero a la plancha o hervido y sírvelo como proteína del plato" —
   el closer bolteó HUEVO (plantilla de cocción correcta entonces); el egg-swap posterior
   reescribió tokens a ciegas y dejó la plantilla sobre un lácteo blando.
2. "⚠️ Seguridad alimentaria: cocina yogurt griego entero por completo (≥71°C, yogurt griego
   entero firmes...)" — nota de seguridad de HUEVO heredada por el mismo swap.
3. "Batir 6 pechuga de pollo de pechuga de pollo con pechuga de pollo entero" — tartamudeo:
   los tokens multi-palabra (claras + huevo) se reemplazan secuencialmente y la idempotencia
   por spans no cubre spans creados en la MISMA pasada por otro patrón.
+ "275g de atún" pegado por el closer (macro-perfecto, plato irreal) → CLOSER_BOLT_MAX_ADD_G.

tooltip-anchor: P1-RECIPE-STEP-SANITIZE
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

_EGG_TOKENS = ["huevo", "huevos", "clara", "claras", "yema", "yemas"]


def _swap(recipe_steps, new_name):
    from graph_orchestrator import _rewrite_recipe_steps_after_subs
    meal = {"name": "Bowl de Avena", "ingredients": [], "recipe": list(recipe_steps)}
    _rewrite_recipe_steps_after_subs(meal, [(_EGG_TOKENS, new_name)])
    return meal["recipe"]


def test_cook_template_rederived_for_soft_dairy():
    steps = _swap(
        ["💪 Cocina claras de huevo a la plancha o hervido y sírvelo como proteína del plato."],
        "yogurt griego entero",
    )
    blob = " ".join(steps).lower()
    assert "a la plancha" not in blob, (
        "plantilla de cocción sobre lácteo blando (vivo: 'Cocina yogurt griego entero a la "
        "plancha') debe re-derivarse del SSOT _closer_protein_step_text"
    )
    assert "incorpora yogurt griego entero" in blob


def test_cook_template_rederived_for_precooked():
    steps = _swap(
        ["Cocina claras de huevo a la plancha o hervido y sírvelo como proteína del plato."],
        "atún en lata",
    )
    blob = " ".join(steps).lower()
    assert "a la plancha" not in blob
    assert "escurre e incorpora" in blob, "pre-cocido → wording 'Escurre e incorpora' del SSOT"


def test_egg_safety_note_dropped_for_soft_dairy():
    steps = _swap(
        ["⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes, "
         "sin partes líquidas) antes de servir; evita el huevo crudo o poco cocido."],
        "yogurt griego entero",
    )
    blob = " ".join(steps).lower()
    assert "71" not in blob and "seguridad" not in blob, (
        "nota de seguridad de huevo heredada por un reemplazo no-cocción debe ELIMINARSE "
        "(vivo: 'yogurt griego entero ≥71°C firmes')"
    )


def test_egg_safety_note_rewritten_for_meat():
    steps = _swap(
        ["⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes, "
         "sin partes líquidas) antes de servir; evita el huevo crudo o poco cocido."],
        "pechuga de pollo",
    )
    blob = " ".join(steps).lower()
    assert "yema" not in blob and "71" not in blob
    assert "cocina pechuga de pollo por completo" in blob, (
        "reemplazo cocinable conserva UNA nota de seguridad, con wording genérico coherente"
    )


def test_batter_safety_note_preserved():
    steps = _swap(
        ["⚠️ Seguridad alimentaria: cocina la preparación con huevo POR COMPLETO — la masa no "
         "debe quedar líquida ni cruda en el centro (≥71°C); evita probar la mezcla cruda."],
        "yogurt griego entero",
    )
    blob = " ".join(steps).lower()
    assert "masa" in blob and "seguridad" in blob, (
        "la nota de MASA se conserva: la masa sigue cocinándose aunque el huevo se sustituya"
    )


def test_stutter_collapse_multi_token():
    steps = _swap(
        ["Batir 6 claras de huevo con huevo entero hasta espumar."],
        "pechuga de pollo",
    )
    blob = " ".join(steps).lower()
    assert "pechuga de pollo de pechuga" not in blob and "con pechuga de pollo entero" not in blob, (
        "tartamudeo vivo: 'Batir 6 pechuga de pollo de pechuga de pollo con pechuga de "
        "pollo entero' — el colapso X (de|con|y) X debe dejar UNA mención"
    )
    assert blob.count("pechuga de pollo") == 1


def test_adjacent_duplicate_word_residue():
    steps = _swap(
        ["Batir 6 claras de huevo con huevo entero hasta espumar."],
        "yogurt griego entero",
    )
    blob = " ".join(steps).lower()
    assert "entero entero" not in blob, (
        "residuo cuando la última palabra del reemplazo coincide con el modificador de cola"
    )


def test_legit_prose_untouched():
    steps = _swap(
        ["Vierte la mezcla poco a poco y cocina a fuego medio.",
         "Mezcla el huevo con canela y canela molida aparte."],
        "yogurt griego entero",
    )
    assert steps[0] == "Vierte la mezcla poco a poco y cocina a fuego medio."
    assert "canela y canela molida" in steps[1], "conectores legítimos entre palabras distintas intactos"


def test_bolt_cap_knob_and_applied():
    import graph_orchestrator as go
    assert go.CLOSER_BOLT_MAX_ADD_G == 180
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("min(grams, max_add_g, int(CLOSER_BOLT_MAX_ADD_G))") >= 2, (
        "el techo de bolt debe aplicarse en AMBOS puntos de _close_protein_gap_for_meal "
        "(pre-snack-cap y el floor cocinable de FASE A) — vivo: 275g de atún"
    )


def test_protagonist_floor_decline_is_logged():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "piso PROTAGONISTA DECLINADO" in src, (
        "la declinación por headroom debe ser visible en logs (vivo: 25g de pavo en "
        "'Locrio de Pavo' escapó el piso sin rastro)"
    )


def test_marker_anchored_in_source():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("P1-RECIPE-STEP-SANITIZE") >= 4
