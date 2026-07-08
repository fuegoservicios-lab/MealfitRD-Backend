"""[P1-CLOSER-STEP-INTEGRATE · 2026-07-08] Fusión del paso 💪 del closer en el Toque de Fuego.

Review en vivo: los pasos 💪 del closer ("💪 Cocina camarones… / Incorpora queso…") leen como bolt-on
(paso APARTE). Ahora, en platos COCINADOS (con 'El Toque de Fuego'), el 💪 se fusiona dentro del TdF →
la receta lee integrada (3 pilares limpios). Los no-cook puros (sin TdF) conservan el 💪 antes del Montaje.
Pasada SEPARADA (`_integrate_complement_steps`) — NO toca `_append_closer_protein_step` (SSOT del closer).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_BOLT = "\U0001f4aa"

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_knob_and_callsites():
    assert "P1-CLOSER-STEP-INTEGRATE" in _GO
    assert 'CLOSER_STEP_INTEGRATE_ENABLED = _env_bool("MEALFIT_CLOSER_STEP_INTEGRATE", True)' in _GO
    assert "def _integrate_complement_steps(days)" in _GO
    # corre en assemble (_apply_macro_engine) Y en el finalize (chunks/persist)
    assert _GO.count("_integrate_complement_steps(days)") >= 2


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


def _cooked():
    return {"name": "Revuelto con Camarones", "recipe": [
        "Mise en place: lava la lechuga.",
        "El Toque de Fuego: calienta el aceite y cocina el huevo con el queso.",
        _BOLT + " Cocina camarones a la plancha o hervido y sírvelo como proteína del plato.",
        "Montaje: sirve el revuelto con la yuca."]}


def test_cooked_merges_into_toque_de_fuego(go):
    m = _cooked()
    n = go._integrate_complement_steps([{"meals": [m]}])
    assert n == 1
    assert sum(_BOLT in s for s in m["recipe"]) == 0, "no debe quedar paso 💪 bolt-on"
    assert len(m["recipe"]) == 3, "4 pasos → 3 (Mise/TdF/Montaje)"
    tdf = next(s for s in m["recipe"] if s.startswith("El Toque"))
    assert "camarones a la plancha" in tdf, "el complemento se fusionó en el TdF"
    assert tdf.count("El Toque de Fuego") == 1


def test_nocook_keeps_bolt_step(go):
    """Plato no-cook puro (sin TdF) conserva el 💪 antes del Montaje (ya coherente)."""
    m = {"name": "Yogurt con Guineo", "recipe": [
        "Mise en place: pela el guineo.",
        _BOLT + " Incorpora queso a la preparación y mézclalo antes de servir.",
        "Montaje: sirve el yogurt con guineo."]}
    n = go._integrate_complement_steps([{"meals": [m]}])
    assert n == 0
    assert sum(_BOLT in s for s in m["recipe"]) == 1, "sin TdF → el 💪 se mantiene"


def test_idempotent(go):
    m = _cooked()
    go._integrate_complement_steps([{"meals": [m]}])
    assert go._integrate_complement_steps([{"meals": [m]}]) == 0, "2da corrida no-op (ya no hay 💪)"


def test_multiple_bolt_steps_all_merge(go):
    m = {"name": "Plato doble", "recipe": [
        "Mise en place: prepara.",
        "El Toque de Fuego: cocina la base.",
        _BOLT + " Cocina pollo a la plancha o hervido y sírvelo como proteína del plato.",
        _BOLT + " Incorpora queso a la preparación y mézclalo antes de servir.",
        "Montaje: sirve."]}
    n = go._integrate_complement_steps([{"meals": [m]}])
    assert n == 2 and sum(_BOLT in s for s in m["recipe"]) == 0


def test_knob_off_leaves_bolt(go, monkeypatch):
    monkeypatch.setattr(go, "CLOSER_STEP_INTEGRATE_ENABLED", False)
    m = _cooked()
    assert go._integrate_complement_steps([{"meals": [m]}]) == 0
    assert sum(_BOLT in s for s in m["recipe"]) == 1, "knob OFF → 💪 intacto"


def test_notes_not_touched(go):
    """Las notas 🌱/💡/⚠ NO son pasos 💪 del closer → intactas."""
    m = {"name": "x", "recipe": [
        "Mise en place: prepara.",
        "El Toque de Fuego: cocina.",
        "🌱 Nota del Nutricionista AI: espolvorea semillas.",
        "💡 Ajustamos las porciones.",
        "Montaje: sirve."]}
    go._integrate_complement_steps([{"meals": [m]}])
    assert any("🌱" in s for s in m["recipe"]) and any("💡" in s for s in m["recipe"])
    assert len(m["recipe"]) == 5, "sin 💪 → nada que fusionar; notas intactas"
