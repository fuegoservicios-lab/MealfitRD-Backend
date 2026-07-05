"""[P1-NOCOOK-TDF-STRIP · 2026-07-05] (screenshots del plan vivo 23c958bb) El paso absurdo
"EL TOQUE DE FUEGO: No requiere cocción (~10–12 min a fuego medio)" — medido en 2 de 6 recetas.

Cadena del bug: el LLM emite el pilar de fuego como PLACEHOLDER en platos fríos ("No requiere
cocción") y el backstop timetemp (`_inject_recipe_time_temp_defaults`) le appendeaba un tiempo de
fuego FALSO porque solo chequea "falta tiempo" (`_CONTRACT_TIME_RE`), no el contenido del paso.

Fix (knob MEALFIT_RECIPE_NOCOOK_TDF_STRIP, ON): al encontrar un TdF placeholder,
  (a) JAMÁS inyectarle tiempo;
  (b) si el resto del plato tampoco tiene señales de cocción (`_meal_is_no_cook`) → ELIMINAR el
      paso (el contrato de 2 pilares P1-RECIPE-CONTRACT-NOCOOK ya permite su ausencia);
  (c) si el plato SÍ cocina en otros pasos → placeholder intacto (contradicción del LLM: la caza
      el lint, no la maquillamos con un tiempo inventado).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_default_on():
    m = re.search(r'RECIPE_NOCOOK_TDF_STRIP_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_RECIPE_NOCOOK_TDF_STRIP"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"


def test_strip_runs_before_time_check():
    """El strip corre ANTES del check de tiempo — el placeholder con tiempo falso YA inyectado
    ("No requiere cocción (~10-12 min)") también se limpia en re-finalizaciones."""
    i = _GO.index("def _inject_recipe_time_temp_defaults")
    body = _GO[i:i + 3500]
    i_strip = body.index("_NOCOOK_TDF_PLACEHOLDER_RE.search")
    i_time = body.index("_CONTRACT_TIME_RE.search(step)")
    assert i_strip < i_time


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


def _batido(tdf_step):
    return {
        "name": "Batido Tropical de Melón con Linaza y Yogurt",
        "ingredients": ["2 taza de melón (300g)", "¾ taza de yogurt griego (178g)"],
        "recipe": [
            "Mise en place: corta el melón en cubos y mide el yogurt.",
            tdf_step,
            "Montaje: coloca todo en una licuadora, licúa 30 segundos y sirve en un vaso alto.",
        ],
    }


def test_placeholder_stripped_on_nocook_meal(go):
    meal = _batido("El Toque de Fuego: No requiere cocción.")
    assert go._inject_recipe_time_temp_defaults(meal) is True
    assert not any("toque de fuego" in str(s).lower() for s in meal["recipe"]), \
        "el placeholder de fuego se ELIMINA en un plato genuinamente frío"
    assert meal.get("_nocook_tdf_stripped") is True
    assert go._recipe_step_contract_issues(meal) == [], "post-strip el contrato de 2 pilares pasa limpio"


def test_placeholder_with_injected_fake_time_also_stripped(go):
    """El caso del screenshot: el tiempo falso ya fue appendeado en una corrida anterior."""
    meal = _batido("El Toque de Fuego: No requiere cocción (~10-12 min a fuego medio).")
    assert go._inject_recipe_time_temp_defaults(meal) is True
    assert not any("fuego medio" in str(s).lower() for s in meal["recipe"])


def test_no_fake_time_injected_when_meal_actually_cooks(go):
    """Placeholder + fuego real en otros pasos → NO se inyecta tiempo NI se elimina (contradicción
    honesta que el lint reporta)."""
    meal = _batido("El Toque de Fuego: No requiere cocción.")
    meal["recipe"][0] = "Mise en place: hierve el huevo 10 min y corta el melón."
    assert go._inject_recipe_time_temp_defaults(meal) is False
    _tdf = next(s for s in meal["recipe"] if str(s).lower().startswith("el toque de fuego"))
    assert "(~" not in _tdf, "jamás inyectar un tiempo de fuego falso sobre 'no requiere cocción'"


def test_real_cooked_meal_still_gets_time_injected(go):
    """Regresión del backstop: TdF real sin tiempo → default por técnica inyectado (sin cambios)."""
    meal = {
        "name": "Pollo al Horno",
        "recipe": ["Mise en place: sazona el pollo.",
                   "El Toque de Fuego: hornea el pollo hasta dorar.",
                   "Montaje: sirve."],
    }
    assert go._inject_recipe_time_temp_defaults(meal) is True
    _tdf = next(s for s in meal["recipe"] if str(s).lower().startswith("el toque de fuego"))
    assert "(~" in _tdf


def test_knob_off_restores_previous_behavior(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_NOCOOK_TDF_STRIP_ENABLED", False)
    meal = _batido("El Toque de Fuego: No requiere cocción.")
    assert go._inject_recipe_time_temp_defaults(meal) is True
    _tdf = next(s for s in meal["recipe"] if str(s).lower().startswith("el toque de fuego"))
    assert "(~" in _tdf, "con el knob OFF vuelve el comportamiento previo (inyección)"
