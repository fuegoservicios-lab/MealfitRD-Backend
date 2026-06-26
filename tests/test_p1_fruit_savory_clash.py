"""[P1-FRUIT-SAVORY-CLASH · 2026-06-26] (auditoría gap #5) Detector determinista del pareo intra-plato
chocante: fruta dulce dominante + base salada en el MISMO plato (caso del owner: "mango con arroz").

Antes esto dependía 100% del prompt advisory (day_generator 15f), sin red determinista. Ahora
`build_variety_report` cuenta `sweet_savory_clash` y `_variety_repeat_gate_issues` lo promueve a un
rechazo de retry (con la misma degradación a advisory en intento final que el resto de gates de variedad,
así que NO produce cero-plan). Este test ancla la detección + la ausencia de falsos positivos comunes.
"""
from __future__ import annotations

import graph_orchestrator as go


def _plan(*meal_names):
    return {"days": [{"day": 1, "meals": [{"name": n, "ingredients": []} for n in meal_names]}]}


def test_clash_mango_arroz():
    """El caso exacto del owner: 'mango con arroz'."""
    assert go.build_variety_report(_plan("Arroz blanco con mango"))["sweet_savory_clash"] >= 1


def test_clash_revoltillo_mango_and_coliflor_mango():
    """Los dos ejemplos nombrados en el prompt."""
    assert go.build_variety_report(_plan("Revoltillo de huevos con mango"))["sweet_savory_clash"] >= 1
    assert go.build_variety_report(_plan("Coliflor salteada y mango"))["sweet_savory_clash"] >= 1
    assert go.build_variety_report(_plan("Espagueti con lechosa"))["sweet_savory_clash"] >= 1


def test_no_clash_on_compatible_pairings():
    """Pareos compatibles o aceptables NO deben dispararse (cero falsos positivos)."""
    for name in (
        "Yogur griego con mango",          # fruta + lácteo dulce = OK
        "Avena con mango y nueces",         # fruta + avena/nueces = OK
        "Pollo a la plancha con piña",      # fruta + PROTEÍNA (tropical) = aceptable, no es base salada
        "Cerdo guisado con guayaba",        # idem
        "Ensalada verde con manzana",       # manzana excluida (aceptable en ensalada)
        "Batido de lechosa",                # fruta sola
        "Tostones de plátano con pollo",    # plátano salado, sin fruta dulce dominante
    ):
        vr = go.build_variety_report(_plan(name))
        assert vr["sweet_savory_clash"] == 0, f"FALSO POSITIVO en: {name!r}"


def test_no_false_positive_word_boundary():
    """'espina' NO debe matchear 'pina' (word-boundary) → sin fruta dulce real, sin clash."""
    vr = go.build_variety_report(_plan("Sardina con espina y arroz"))
    assert vr["sweet_savory_clash"] == 0


def test_gate_promotes_clash_when_on(monkeypatch):
    monkeypatch.setattr(go, "VARIETY_GATE_FRUIT_CLASH", True)
    issues = go._variety_repeat_gate_issues({"sweet_savory_clash": 1})
    assert any("PAREO CHOCANTE" in i for i in issues), "el clash debe promoverse a rechazo de retry"


def test_gate_silent_when_off(monkeypatch):
    monkeypatch.setattr(go, "VARIETY_GATE_FRUIT_CLASH", False)
    issues = go._variety_repeat_gate_issues({"sweet_savory_clash": 3})
    assert not any("PAREO CHOCANTE" in i for i in issues), "knob OFF → no rechaza por clash"


def test_knob_default_is_true():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    m = re.search(
        r'VARIETY_GATE_FRUIT_CLASH\s*=\s*_env_bool\(\s*["\']MEALFIT_VARIETY_GATE_FRUIT_CLASH["\']\s*,\s*(True|False)\s*\)',
        src,
    )
    assert m is not None and m.group(1) == "True", "el gate de pareo chocante debe venir ON por default"
    assert "P1-FRUIT-SAVORY-CLASH" in src, "falta el tooltip-anchor P1-FRUIT-SAVORY-CLASH"
