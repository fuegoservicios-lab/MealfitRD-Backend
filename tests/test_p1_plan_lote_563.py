# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-563 · 2026-09-27] El presupuesto también al cambiar platos.

Auditoría del formulario: `swap_meal` no leía el presupuesto y `_enrich_clinical_from_profile` no lo hidrataba; con la
Nevera apagada o vacía (lote 550) el swap elige del catálogo entero, así que un «presupuesto ajustado» podía recibir
salmón o camarones al cambiar un plato.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_el_swap_hidrata_el_presupuesto(monkeypatch):
    import db
    import routers.plans as rp
    hp = {"budget": "low", "groceryDuration": "weekly"}
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": hp})
    data = {"user_id": "u-563"}
    rp._enrich_clinical_from_profile(data, "u-563")
    assert data.get("budget") == "low" and data.get("groceryDuration") == "weekly"
    from prompts.plan_generator import build_budget_context
    assert "AJUSTADO" in build_budget_context(data)


def test_el_prompt_del_swap_lo_lleva():
    agent = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "from prompts.plan_generator import build_budget_context as _bbc563" in agent
    plans = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert re.search(r'"budget": data\.get\("budget"\), "budgetAmount": data\.get\("budgetAmount"\)', plans)
