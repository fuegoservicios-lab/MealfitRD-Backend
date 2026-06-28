"""[P1-RENEWAL-PANTRY-AWARE · 2026-06-28] Renovación pantry-aware — Fase 0+1.

Objetivo del owner: al "Renovar plan" → variedad (intacta) + REUSAR los DURADEROS
sobrantes de la nevera como SUGERENCIA. La tensión histórica (band-0.0, incidente
d4bc3af5) fue que el reuso como GATE colapsaba la variedad y degradaba planes.

Contrato verificado aquí:
  - default OFF: build_pantry_context(variety) sigue devolviendo "" (nevera ignorada).
  - knob ON + `renewal_pantry_aware`: emite los DURADEROS como sugerencia advisory,
    NUNCA 'OBLIGATORIO', filtra perecederos (esos van a la lista de faltantes), y
    respeta el cap.
  - review_plan_node NO se tocó: sigue saltando la validación para variety, así que
    el reuso pantry-aware jamás es un gate (apoyado en el skip existente).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants  # noqa: E402
from prompts.plan_generator import build_pantry_context, _build_durable_advisory_hint  # noqa: E402

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _form(**kw):
    base = {"update_reason": "variety", "current_pantry_ingredients": ["arroz", "aceite"]}
    base.update(kw)
    return base


def test_default_knob_is_off():
    # Contrato del default: la renovación NO es pantry-aware salvo flip explícito.
    assert constants.RENEWAL_PANTRY_AWARE_ENABLED is False
    assert constants.PANTRY_COMPLETION_LIST_ENABLED is False
    assert constants.RENEWAL_DURABLE_HINT_MAX_ITEMS == 8


def test_default_off_variety_returns_empty():
    # Aun con el flag y duraderos, con el knob OFF (default) variety ignora la nevera.
    out = build_pantry_context(_form(renewal_pantry_aware=True,
                                     durable_pantry_ingredients=["arroz", "aceite de oliva"]))
    assert out == ""


def test_on_without_flag_returns_empty(monkeypatch):
    monkeypatch.setattr(constants, "RENEWAL_PANTRY_AWARE_ENABLED", True)
    # Knob ON pero sin `renewal_pantry_aware` → variety puro (sigue vacío).
    out = build_pantry_context(_form(durable_pantry_ingredients=["arroz"]))
    assert out == ""


def test_on_with_flag_emits_durables_advisory(monkeypatch):
    monkeypatch.setattr(constants, "RENEWAL_PANTRY_AWARE_ENABLED", True)
    out = build_pantry_context(_form(
        renewal_pantry_aware=True,
        durable_pantry_ingredients=["arroz", "aceite de oliva", "lentejas"],
    ))
    assert out != ""
    assert "arroz" in out and "lentejas" in out
    # NUNCA la palabra que colapsaba la variedad (reuso forzado):
    assert "OBLIGATORIO" not in out
    # Tono advisory + prioridad de variedad:
    assert "no obligatorio" in out.lower()
    assert "VARIEDAD" in out.upper()


def test_works_without_current_pantry(monkeypatch):
    # Los duraderos van en campo SEPARADO; el hint debe emitirse aunque
    # current_pantry_ingredients esté vacío (se evalúa antes del guard).
    monkeypatch.setattr(constants, "RENEWAL_PANTRY_AWARE_ENABLED", True)
    out = build_pantry_context({
        "update_reason": "variety",
        "renewal_pantry_aware": True,
        "durable_pantry_ingredients": ["arroz", "avena"],
    })
    assert "arroz" in out and "avena" in out


def test_perishables_filtered_from_hint(monkeypatch):
    monkeypatch.setattr(constants, "RENEWAL_PANTRY_AWARE_ENABLED", True)
    # 'pollo' y 'leche' son perecederos → NO van al hint de reuso (van a faltantes).
    out = build_pantry_context(_form(
        renewal_pantry_aware=True,
        durable_pantry_ingredients=["arroz", "pollo", "leche"],
    ))
    assert "arroz" in out
    assert "pollo" not in out.lower()
    assert "leche" not in out.lower()


def test_durable_hint_respects_cap(monkeypatch):
    monkeypatch.setattr(constants, "RENEWAL_DURABLE_HINT_MAX_ITEMS", 2)
    durables = ["arroz", "aceite", "lentejas", "avena", "sal"]
    out = _build_durable_advisory_hint({"durable_pantry_ingredients": durables})
    assert "arroz" in out and "aceite" in out
    assert "lentejas" not in out and "avena" not in out


def test_empty_durables_returns_empty():
    assert _build_durable_advisory_hint({"durable_pantry_ingredients": []}) == ""
    assert _build_durable_advisory_hint({}) == ""
    # Solo perecederos → tras filtrar queda vacío.
    assert _build_durable_advisory_hint({"durable_pantry_ingredients": ["pollo", "leche"]}) == ""


def test_review_plan_node_variety_skip_untouched():
    # CONTRATO: NO tocamos review_plan_node — sigue saltando la validación de
    # despensa para update_reason=='variety'. El reuso pantry-aware se apoya en
    # este skip existente, por eso jamás es un gate. Tooltip-anchor: renombrar
    # rompe el test antes de cambiar producción.
    src = open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8").read()
    assert 'is_variety_regen = form_data.get("update_reason") == "variety"' in src
    assert "P1-VARIETY-IGNORE-PANTRY" in src


def test_markers_and_knobs_present():
    pg = open(os.path.join(_BACKEND, "prompts", "plan_generator.py"), encoding="utf-8").read()
    cst = open(os.path.join(_BACKEND, "constants.py"), encoding="utf-8").read()
    assert "P1-RENEWAL-PANTRY-AWARE" in pg
    assert "P1-RENEWAL-PANTRY-AWARE" in cst
    assert 'MEALFIT_RENEWAL_PANTRY_AWARE_ENABLED' in cst
    assert 'MEALFIT_RENEWAL_DURABLE_HINT_MAX_ITEMS' in cst
    assert 'MEALFIT_PANTRY_COMPLETION_LIST_ENABLED' in cst
