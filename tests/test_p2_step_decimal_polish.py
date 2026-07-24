"""[P2-STEP-DECIMAL-POLISH · 2026-07-24] Los pasos de receta no citan decimales crudos.

Defecto en vivo (plan a060108b, revisión de recetas del owner). Textos REALES de la DB:

    "El Toque de Fuego: Calienta 0.25 cda de aceite de oliva en un sartén antiadherente…"
    "Mise en place: Mide ½ taza de avena, 0.33 taza de leche, ¼ cdta de canela y la sal."
                              ↑ fracción bonita        ↑ decimal crudo, misma frase

Por qué no lo cubría nada:
    `_polish_finalize_display` (P1-FINALIZE-COUNTABLE-POLISH) recorre `ingredients` y
    `ingredients_raw` — NUNCA los pasos. Y los decimales en `ingredients_raw` son
    DELIBERADOS (fuente de macros y lista de compras). El error no es que existan: es que
    el texto del paso los copie tal cual.

Seguridad: los pasos son display puro (no alimentan macros, lista ni medidores), y la
sustitución exige una UNIDAD de cocina detrás — así no toca temperaturas ("71.5°C"),
tiempos ("1.5 horas") ni tamaños ("cubos de 1.5 cm"), que legítimamente llevan decimal.
"""
from __future__ import annotations

import graph_orchestrator as g


def _days(steps):
    return [{"meals": [{"name": "Plato", "recipe": list(steps)}]}]


def _steps_of(days):
    return days[0]["meals"][0]["recipe"]


# ---------------------------------------------------------------------------
# 1. Los casos reales
# ---------------------------------------------------------------------------
def test_reescribe_el_caso_reportado():
    days = _days(["El Toque de Fuego: Calienta 0.25 cda de aceite de oliva en un sartén."])
    n = g._polish_recipe_step_decimals(days)
    assert n == 1
    assert "¼ cda de aceite" in _steps_of(days)[0]
    assert "0.25" not in _steps_of(days)[0]


def test_no_mezcla_fraccion_y_decimal_en_la_misma_frase():
    days = _days(["Mise en place: Mide ½ taza de avena, 0.33 taza de leche, ¼ cdta de canela."])
    g._polish_recipe_step_decimals(days)
    out = _steps_of(days)[0]
    assert "0.33" not in out
    assert "½ taza de avena" in out and "¼ cdta de canela" in out  # lo ya bonito no se toca


def test_varias_unidades_y_valores():
    days = _days([
        "Cocina la batata en una sartén con 0.75 cdta de aceite.",
        "En un sartén, calienta 1.5 cdta de aceite de oliva a fuego medio.",
        "Agrega 0.5 taza de leche y 2.25 cdas de avena.",
    ])
    n = g._polish_recipe_step_decimals(days)
    assert n == 3
    txt = " | ".join(_steps_of(days))
    assert "¾ cdta" in txt and "1½ cdta" in txt and "½ taza" in txt and "2¼ cdas" in txt


# ---------------------------------------------------------------------------
# 2. Lo que NO se puede tocar
# ---------------------------------------------------------------------------
def test_no_toca_temperaturas_tiempos_ni_tamanos():
    originales = [
        "Seguridad alimentaria: cocina el huevo a 71.5°C hasta que cuaje.",
        "Hornea por 1.5 horas hasta que esté tierno.",
        "Corta la yuca en cubos de 1.5 cm.",
        "Cocina por 2.5 minutos por lado.",
    ]
    days = _days(originales)
    n = g._polish_recipe_step_decimals(days)
    assert n == 0, f"no debe tocar magnitudes que no son cantidades de cocina: {_steps_of(days)}"
    assert _steps_of(days) == originales


def test_no_toca_numeros_pegados_ni_rangos():
    originales = ["Cocina 15-20 min a fuego bajo.", "Usa la versión 2.0 del molde."]
    days = _days(originales)
    assert g._polish_recipe_step_decimals(days) == 0
    assert _steps_of(days) == originales


# ---------------------------------------------------------------------------
# 3. Contrato del pase
# ---------------------------------------------------------------------------
def test_idempotente():
    days = _days(["Calienta 0.25 cda de aceite."])
    assert g._polish_recipe_step_decimals(days) == 1
    assert g._polish_recipe_step_decimals(days) == 0  # segunda pasada: nada que hacer


def test_knob_permite_rollback(monkeypatch):
    monkeypatch.setattr(g, "STEP_DECIMAL_POLISH_ENABLED", False)
    days = _days(["Calienta 0.25 cda de aceite."])
    assert g._polish_recipe_step_decimals(days) == 0
    assert "0.25" in _steps_of(days)[0]


def test_tolera_entradas_raras():
    for bad in (None, [], [{}], [{"meals": None}], [{"meals": [{"recipe": "no es lista"}]}]):
        assert g._polish_recipe_step_decimals(bad) == 0


def test_cableado_al_pase_de_finalize():
    """Si no se invoca desde el finalize, el fix no llega al usuario."""
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8", errors="replace")
    assert "_polish_recipe_step_decimals(days)" in src
    assert 'parts.append(f"step_decimals=' in src
    assert "[P2-STEP-DECIMAL-POLISH · 2026-07-24]" in src
