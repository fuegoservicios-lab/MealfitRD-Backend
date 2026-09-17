# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-78 · 2026-09-17] El techo del day-gen (170 s) y el hedge (120 s) se midieron con un modelo que no razona.
Con el proveedor alterno (lote 74) el day-gen razona ~9.000 tokens antes de contestar (medido: 11.752 tokens y 57 s a
solas, frente a 2.752 y 15 s sin razonar) y en el 3.er bench real de la noche un día de 5 comidas pasó de 170 s → día de
contingencia matemática. Con ese proveedor y los knobs en su default: hedge 150 s, techo 240 s."""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_con_el_proveedor_alterno_y_knobs_en_default_se_ensancha(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("MEALFIT_HEDGE_AFTER_BASE_S", raising=False)
    monkeypatch.delenv("MEALFIT_HARD_CEILING_S", raising=False)
    assert go._daygen_hedge_ceiling_for_provider(120.0, 170.0) == (150.0, 240.0)
    assert go._daygen_hedge_ceiling_for_provider(200.0, 300.0) == (200.0, 300.0)      # nunca reduce


def test_un_knob_puesto_a_mano_sigue_mandando(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setenv("MEALFIT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("MEALFIT_HARD_CEILING_S", "200")
    monkeypatch.delenv("MEALFIT_HEDGE_AFTER_BASE_S", raising=False)
    assert go._daygen_hedge_ceiling_for_provider(120.0, 200.0) == (150.0, 200.0)


def test_con_zai_no_cambia_nada(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.delenv("MEALFIT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MEALFIT_HEDGE_AFTER_BASE_S", raising=False)
    monkeypatch.delenv("MEALFIT_HARD_CEILING_S", raising=False)
    assert go._daygen_hedge_ceiling_for_provider(120.0, 170.0) == (120.0, 170.0)
    # (los knobs del módulo NO se comprueban: el .env local ya fija MEALFIT_HEDGE_AFTER_BASE_S=150 y el gate lo vio)


def test_el_nodo_lo_usa_y_el_marcador():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("async def generate_days_parallel_node(")
    assert "HEDGE_AFTER_BASE, HARD_CEILING = _daygen_hedge_ceiling_for_provider(HEDGE_AFTER_BASE_S, HARD_CEILING_S)" in src[i:i + 60000]
    assert 'P1-PLAN-LOTE-78 · 2026-09-17' in (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-78" in (_BACKEND / "docs" / "llm_tier_routing.md").read_text(encoding="utf-8")
