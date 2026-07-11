"""[P1-AUTOFIX-IMPOTENCE-TELEMETRY · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10): `_protein_repeat_
autofix` detectó proteína repetida same-day pero `_replace_meal_egg_lines` devolvió None (sin candidato
válido) → SILENCIO total. 4 minutos después el reviewer rechazó el plan sin que ningún log conectara la
causa. Este batch añade un log estructurado `[P1-AUTOFIX-IMPOTENCE]` en los 5 puntos donde el autofix
detecta pero no puede corregir (ladder agotado, label sin escalera, sin target seguro en contexto
dulce/ligero, huevo sin candidato ×2) — NO cambia el resultado (el gate/backstop sigue decidiendo), solo
hace observable la impotencia.
"""
import logging
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO_SRC = f.read()


def test_marker_and_helper_present():
    assert "P1-AUTOFIX-IMPOTENCE" in _GO_SRC
    assert "def _log_autofix_impotent(" in _GO_SRC


def test_all_five_impotence_points_call_the_logger():
    """Ancla estructural: los 5 puntos donde `_protein_repeat_autofix` detecta-pero-no-puede deben
    invocar `_log_autofix_impotent` (huevo×2, sin-escalera, sweet/light sin target, ladder agotado)."""
    i = _GO_SRC.index("def _protein_repeat_autofix")
    j = _GO_SRC.index("\ndef ", i + 1)  # fin de la función (siguiente def a nivel módulo)
    window = _GO_SRC[i:j]
    n_calls = window.count("_log_autofix_impotent(")
    assert n_calls >= 5, f"esperaba >=5 llamadas a _log_autofix_impotent en el cuerpo, hay {n_calls}"
    for reason in ("egg_no_candidate", "no_ladder_for_label",
                  "no_safe_target_sweet_or_light", "ladder_exhausted"):
        assert reason in window, f"falta el reason={reason!r} en el cuerpo de la función"


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _meal(slot, name, ings, steps=None):
    return {"meal": slot, "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": list(steps or ["Cocina."])}


def test_no_ladder_label_logs_impotence(go, caplog, monkeypatch):
    # [P2-PROTEIN-LADDER-GAPS · 2026-07-11] atún YA tiene escalera (caso vivo corr=c0a950c6:
    # no_ladder_for_label → rechazo de plan completo). Para probar el LOG de impotencia
    # simulamos un label sin escalera quitándosela al atún vía monkeypatch — el path
    # no_ladder_for_label sigue vivo como backstop para labels futuros.
    _ladder_sin_atun = {k: v for k, v in go._PROTEIN_REPEAT_SWAP_LADDER.items() if k != "atun"}
    monkeypatch.setattr(go, "_PROTEIN_REPEAT_SWAP_LADDER", _ladder_sin_atun)
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Ensalada de Atún", ["150 g de atún en agua", "1 tomate"]),
        _meal("Cena", "Wrap de Atún", ["120 g de atún en agua", "1 tortilla integral"]),
    ]}]
    with caplog.at_level(logging.INFO, logger="graph_orchestrator"):
        go._protein_repeat_autofix(days, {})
    assert "P1-AUTOFIX-IMPOTENCE" in caplog.text
    assert "no_ladder_for_label" in caplog.text
    assert "atun" in caplog.text.lower()


def test_ladder_exhausted_logs_impotence(go, caplog):
    # pescado repetido; su escalera es (pollo, pavo) — AMBOS ya presentes en el día (day_labels los
    # incluye desde el escaneo) → ladder agotado, break silencioso antes de este fix. mainGoal=
    # gain_muscle desactiva el fallback no-gated (queso/legumbre) que si no, resolvería el swap.
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Pollo al Horno", ["150 g de pechuga de pollo", "100 g de arroz"]),
        _meal("Merienda", "Pavo Ahumado", ["80 g de pechuga de pavo"]),
        _meal("Almuerzo", "Pescado a la Plancha", ["150 g de filete de pescado blanco", "100 g de arroz"]),
        _meal("Cena", "Tilapia Guisada", ["150 g de tilapia", "100 g de yuca"]),
    ]}]
    with caplog.at_level(logging.INFO, logger="graph_orchestrator"):
        go._protein_repeat_autofix(days, {"mainGoal": "gain_muscle"})
    assert "P1-AUTOFIX-IMPOTENCE" in caplog.text
    assert "ladder_exhausted" in caplog.text
    assert "pescado" in caplog.text.lower()


def test_impotence_never_changes_outcome(go, caplog, monkeypatch):
    """El log es puramente observacional: el meal NO se toca cuando no hay target seguro.
    [P2-PROTEIN-LADDER-GAPS · 2026-07-11] atún YA tiene escalera → se simula el label
    sin escalera vía monkeypatch (el contrato observacional del log no cambia)."""
    _ladder_sin_atun = {k: v for k, v in go._PROTEIN_REPEAT_SWAP_LADDER.items() if k != "atun"}
    monkeypatch.setattr(go, "_PROTEIN_REPEAT_SWAP_LADDER", _ladder_sin_atun)
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Ensalada de Atún", ["150 g de atún en agua"]),
        _meal("Cena", "Wrap de Atún", ["120 g de atún en agua"]),
    ]}]
    before = [dict(m) for m in days[0]["meals"]]
    with caplog.at_level(logging.INFO, logger="graph_orchestrator"):
        fixed = go._protein_repeat_autofix(days, {})
    assert fixed == 0
    assert [m["ingredients"] for m in days[0]["meals"]] == [m["ingredients"] for m in before]
