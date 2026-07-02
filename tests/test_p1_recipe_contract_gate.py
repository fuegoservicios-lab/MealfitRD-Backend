"""[P1-RECIPE-CONTRACT-GATE · 2026-07-02] (batch P1-OBJECTIVE-V4-BATCH)

"Recetas 100% production-ready": el contrato de pasos (Mise en place → El Toque de Fuego con
tiempo/temperatura → Montaje) vivía solo en prompt + telemetría — una receta sin tiempos se
entregaba sin rescate. Dos capas:
  (a) BACKSTOP determinista ON: `_inject_recipe_time_temp_defaults` añade un tiempo/temp default
      POR TÉCNICA al Toque de Fuego cuando falta (texto puro, idempotente). Corre en assemble
      (form-gen), en el finalizer SSOT de updates y en /recipe/expand EN ORIGEN.
  (b) GATE de retry por contract_ratio OFF: flip con datos de la serie (playbook
      P1-DISH-QUALITY-GATE-ON).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PL_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _meal(fuego_step, name="Pollo al Horno con Batata"):
    return {"name": name, "ingredients": ["200g de pollo"],
            "recipe": ["Mise en place: pesa y seca el pollo.", fuego_step, "Montaje: sirve caliente."]}


# ════════════════════════════════════════════════════════════════════════════
# Backstop determinista
# ════════════════════════════════════════════════════════════════════════════
def test_injects_technique_default_when_missing(go):
    meal = _meal("El Toque de Fuego: hornea el pollo hasta dorar.")
    assert go._inject_recipe_time_temp_defaults(meal) is True
    fuego = meal["recipe"][1]
    assert "180 °C" in fuego, f"técnica horno debía inyectar temp: {fuego}"
    assert meal.get("_recipe_timetemp_injected") is True
    # Post-inyección el lint del contrato YA no acusa tiempo/temperatura.
    issues = go._recipe_step_contract_issues(meal)
    assert not any("tiempo/temperatura" in i for i in issues), issues


def test_plancha_default_and_fallback(go):
    m1 = _meal("El Toque de Fuego: cocina a la plancha por ambos lados.", name="Pechuga a la Plancha")
    assert go._inject_recipe_time_temp_defaults(m1) is True
    assert "min por lado" in m1["recipe"][1]
    m2 = _meal("El Toque de Fuego: prepara la proteína.", name="Plato Criollo")
    assert go._inject_recipe_time_temp_defaults(m2) is True
    assert "fuego medio" in m2["recipe"][1], "sin técnica detectable → fallback genérico"


def test_idempotent_and_respects_existing_time(go):
    meal = _meal("El Toque de Fuego: sella la pechuga 4 min por lado a fuego alto.")
    assert go._inject_recipe_time_temp_defaults(meal) is False, "ya trae tiempo → no-op"
    injected = _meal("El Toque de Fuego: hornea el pollo.")
    go._inject_recipe_time_temp_defaults(injected)
    assert go._inject_recipe_time_temp_defaults(injected) is False, "segunda pasada = no-op"


def test_knob_off_is_noop(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_TIMETEMP_BACKSTOP_ENABLED", False)
    meal = _meal("El Toque de Fuego: hornea el pollo.")
    assert go._inject_recipe_time_temp_defaults(meal) is False
    assert "_recipe_timetemp_injected" not in meal


# ════════════════════════════════════════════════════════════════════════════
# Cableado en las superficies + defaults
# ════════════════════════════════════════════════════════════════════════════
def test_defaults(go):
    assert go.RECIPE_TIMETEMP_BACKSTOP_ENABLED is True, "backstop determinista nace ON"
    assert go.RECIPE_CONTRACT_GATE_ENABLED is False, "gate nace OFF (flip con serie contract_ratio)"
    assert go.RECIPE_CONTRACT_GATE_RATIO == pytest.approx(0.5)


def test_wired_in_assemble_finalizer_and_expand():
    # form-gen (assemble, pre-engine)
    i_asm = _GO_SRC.index("if RECIPE_TIMETEMP_BACKSTOP_ENABLED:")
    assert "_inject_recipe_time_temp_defaults" in _GO_SRC[i_asm:i_asm + 1200]
    # finalizer SSOT de updates (swap / chat-modify / recalc / swap-persist)
    i_fin = _GO_SRC.index("def finalize_single_meal_recipe_coherence(")
    fin_body = _GO_SRC[i_fin:_GO_SRC.index("\ndef ", i_fin + 50)]
    assert "_inject_recipe_time_temp_defaults" in fin_body
    # /recipe/expand EN ORIGEN (persist + guests)
    assert "_inject_recipe_time_temp_defaults" in _PL_SRC


def test_review_gate_wired_with_final_advisory():
    i = _GO_SRC.index("if RECIPE_CONTRACT_GATE_ENABLED:")
    window = _GO_SRC[i:i + 3000]
    assert "contract_ratio" in window
    assert "_recipe_contract_advisory_final" in window
    assert 'severity = _severity_max(severity, "high")' in window
