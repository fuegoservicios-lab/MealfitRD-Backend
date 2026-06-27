"""[P2-UPDATE-MICRO-STEER · 2026-06-27] (audit G2) Lleva a las superficies de UPDATE (swap S3 /
chat-modify) los pisos de micronutrientes (Mg/Fe/Ca/fibra/K) que S1 ya inyecta. Cierra el gap: el
usuario SANO sin condición NO recibía guía de densidad al cambiar platos (S1 sí; updates solo re-medían).

ADVISORY (prompt), nunca gate. El caller NO la inyecta en el path pantry-strict (el pantry manda y la
presión de micros subiría fallos de convergencia) → ése era el riesgo que la auditoría citó para
recomendar default OFF; con el skip pantry-aware el default ON es seguro.

Cubre:
  - graph_orchestrator.build_update_micronutrient_directive (SSOT, reusa build_micronutrient_targets_directive)
  - knob OFF → '' ; default ON
  - parser-anchored: swap inyecta cuando NO hay pantry; chat-modify con skip pantry-aware
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go

_GO = Path(go.__file__).resolve()
_BACKEND = _GO.parent


def test_sane_user_gets_micro_floors():
    """El gap central: usuario sano sin condición → directiva con pisos de micros (hierro/fibra/magnesio)."""
    d = go.build_update_micronutrient_directive({"gender": "male", "age": 30, "goal": "maintenance"})
    assert d
    low = d.lower()
    assert "micronutrientes" in low
    assert "hierro" in low and "fibra" in low and "magnesio" in low


def test_failsafe_on_empty_form():
    # fail-safe: form vacío → defaults (no crashea); jamás rompe el update
    assert isinstance(go.build_update_micronutrient_directive({}), str)
    assert go.build_update_micronutrient_directive(None) == ""


def test_knob_off_returns_empty(monkeypatch):
    monkeypatch.setattr(go, "UPDATE_MICRO_STEER_ENABLED", False)
    assert go.build_update_micronutrient_directive({"gender": "male", "age": 30}) == ""


def test_knob_default_on():
    assert go.UPDATE_MICRO_STEER_ENABLED is True


# ──────────────────────────── parser-anchored ────────────────────────────

def _src(p):
    return Path(p).read_text(encoding="utf-8")


def test_anchor_present_in_surfaces():
    for f in ("graph_orchestrator.py", "agent.py", "tools.py"):
        assert "P2-UPDATE-MICRO-STEER" in _src(_BACKEND / f), f"falta anchor en {f}"


def test_swap_injects_when_not_pantry():
    src = _src(_BACKEND / "agent.py")
    assert "build_update_micronutrient_directive" in src
    # el skip pantry-aware: inyecta solo cuando no hay nevera detectada
    assert "if not clean_ingredients:" in src


def test_chat_modify_injects_with_pantry_skip():
    src = _src(_BACKEND / "tools.py")
    assert "build_update_micronutrient_directive" in src
    assert "if not (clean_ingredients and not allow_pantry_expansion):" in src
