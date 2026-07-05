"""[P1-RECIPE-CONTRACT-NOCOOK · 2026-07-05] Exención SIN-COCCIÓN en el lint del contrato de pasos.

Causa raíz (screenshot del owner, plan vivo): el lint `_recipe_step_contract_issues` exigía
"El Toque de Fuego" + tiempo/temperatura a TODOS los platos, pero un plato honesto sin cocción
("Mango Fresco con Maní Tostado y Queso Cottage") no puede cumplirlo — y el backstop
`_inject_recipe_time_temp_defaults` solo AUMENTA un paso de fuego existente, jamás lo crea →
chip amarillo "Receta con pasos incompletos — regenera para detalle" en recetas perfectas
(falso positivo estructural; el flag se setea per-meal en el persist boundary para TODO plan).

Fix en dos capas:
  1. Lint: plato sin paso de fuego emitido Y sin NINGUNA señal de cocción (verbos/técnicas en
     pasos no-nota + nombre, `_meal_is_no_cook`) → contrato de 2 pilares (Mise en place →
     Montaje). Cualquier señal de fuego → contrato completo (sesgo conservador).
  2. Prompt §3: el day-gen aprende la variante ("OMITE 'El Toque de Fuego'" en platos fríos).
El gate del review (RECIPE_CONTRACT_GATE) usa el MISMO lint vía contract_ratio → consistencia
automática. Knob: MEALFIT_RECIPE_CONTRACT_NOCOOK (default ON).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GO = (_REPO_ROOT / "backend" / "graph_orchestrator.py").read_text(encoding="utf-8")
_DG = (_REPO_ROOT / "backend" / "prompts" / "day_generator.py").read_text(encoding="utf-8")


# ───────────────────────── parser-based ─────────────────────────

def test_knob_defined_default_on():
    m = re.search(r'RECIPE_CONTRACT_NOCOOK_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_RECIPE_CONTRACT_NOCOOK"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"


def test_prompt_section3_teaches_nocook_variant():
    assert "P1-RECIPE-CONTRACT-NOCOOK" in _DG
    i = _DG.index("EXCEPCIÓN SIN COCCIÓN")
    win = _DG[i:i + 600]
    assert "OMITE \"El Toque de Fuego\"" in win or "OMITE 'El Toque de Fuego'" in win.replace('"', "'")
    assert "JAMÁS inventes un tiempo de fuego falso" in win


# ───────────────────────── funcional (lint) ─────────────────────────

def _issues(meal):
    import graph_orchestrator as go
    return go._recipe_step_contract_issues(meal)


def _fresh_snack(recipe=None):
    return {
        "name": "Mango Fresco con Maní Tostado y Queso Cottage",
        "ingredients": ["150g de mango", "15g de maní", "80g de queso cottage"],
        "recipe": recipe if recipe is not None else [
            "Mise en place: pela y corta el mango en cubos; mide el maní y el cottage.",
            "Montaje: sirve el mango en un bowl, corona con el cottage y espolvorea el maní.",
        ],
    }


def test_nocook_meal_with_two_pillars_passes():
    """El caso del screenshot: merienda fría perfecta con Mise en place → Montaje → SIN chip."""
    assert _issues(_fresh_snack()) == []


def test_nocook_still_requires_mise_and_montaje():
    meal = _fresh_snack(recipe=["Montaje: sirve todo junto."])
    issues = _issues(meal)
    assert any("Mise" in i for i in issues), "la exención es SOLO del fuego — Mise en place sigue obligatorio"
    meal2 = _fresh_snack(recipe=["Mise en place: corta el mango."])
    assert any("Montaje" in i for i in issues + _issues(meal2))


def test_nocook_order_still_enforced():
    meal = _fresh_snack(recipe=[
        "Montaje: sirve.",
        "Mise en place: corta el mango.",
    ])
    assert any("orden" in i for i in _issues(meal))


def test_cooked_meal_missing_fuego_still_flagged():
    """Señal de cocción en pasos → el contrato completo aplica aunque falte el prefijo de fuego."""
    meal = {
        "name": "Pollo Guisado",
        "recipe": [
            "Mise en place: corta el pollo.",
            "Cocina el pollo en el caldero hasta dorar.",
            "Montaje: sirve con arroz.",
        ],
    }
    assert any("Toque de Fuego" in i for i in _issues(meal))


def test_cooked_name_signal_flags_even_with_cold_steps():
    """El nombre delata cocción (revoltillo) → sin exención aunque los pasos sean vagos."""
    meal = {
        "name": "Revoltillo Criollo",
        "recipe": ["Mise en place: bate los huevos.", "Montaje: sirve."],
    }
    assert any("Toque de Fuego" in i for i in _issues(meal))


def test_emitted_fuego_step_keeps_time_check():
    """Si el LLM SÍ emitió el paso de fuego, el check de tiempo/temperatura sigue vivo."""
    meal = _fresh_snack(recipe=[
        "Mise en place: corta el mango.",
        "El Toque de Fuego: mezcla suavemente.",
        "Montaje: sirve.",
    ])
    assert any("tiempo" in i for i in _issues(meal))


def test_guard_notes_do_not_count_as_cook_signal():
    """La nota 💪 del closer ('Cocina X a la plancha…') es inyección de guard, no técnica del plato."""
    meal = _fresh_snack()
    meal["recipe"] = [
        meal["recipe"][0],
        "💪 Nota del Nutricionista AI: añade 40g de pechuga de pollo, cocínala a la plancha.",
        meal["recipe"][1],
    ]
    assert _issues(meal) == []


def test_guisantes_not_a_cook_signal():
    """'guisantes' (arvejas) no es un guiso — lookahead negativo `guis(?!ante)`."""
    meal = {
        "name": "Ensalada Fría de Guisantes",
        "recipe": [
            "Mise en place: escurre los guisantes y corta el tomate.",
            "Montaje: mezcla todo con el aderezo y sirve frío.",
        ],
    }
    assert _issues(meal) == []


def test_knob_off_restores_previous_behavior(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "RECIPE_CONTRACT_NOCOOK_ENABLED", False)
    assert any("Toque de Fuego" in i for i in go._recipe_step_contract_issues(_fresh_snack()))
