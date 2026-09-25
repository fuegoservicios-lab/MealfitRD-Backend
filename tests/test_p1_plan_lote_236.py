# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-236 · 2026-09-25] La promoción del regen quirúrgico re-valida alérgenos, rechazos y mercurio.

Auditoría del 25-sep: `_surgical_promote_blocked_reason` solo miraba piso de proteína y dieta. Un alérgeno que metiera el
corrector LLM se promovía a «approved»; la re-revisión lo rechazaba CRÍTICO, pero el swap al mejor intento restauraba
`review_passed=True` y el fallback crítico no corría.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _plan(nombre, ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Desayuno", "name": nombre, "ingredients": ings,
                                           "protein": 60, "carbs": 40, "fats": 10, "cals": 500}]}]}


def _sin_piso(monkeypatch):
    monkeypatch.setattr(go, "_protein_floor_shortfall", lambda *a, **k: [])


def test_alergeno_bloquea_la_promocion(monkeypatch):
    _sin_piso(monkeypatch)
    r = go._surgical_promote_blocked_reason(_plan("Tostada con mantequilla de maní", ["1 cda de mantequilla de maní",
                                                                                     "1 rebanada de pan"]),
                                            {"allergies": ["Mani"], "dietType": "balanced"})
    assert r and r.startswith("alérgeno"), r


def test_rechazo_bloquea_la_promocion(monkeypatch):
    _sin_piso(monkeypatch)
    r = go._surgical_promote_blocked_reason(_plan("Bowl de atún", ["120 g de atún en agua", "½ taza de arroz"]),
                                            {"dislikes": ["Pescado"]})
    assert r and r.startswith("rechazo"), r


def test_mercurio_en_embarazo_bloquea_la_promocion(monkeypatch):
    _sin_piso(monkeypatch)
    r = go._surgical_promote_blocked_reason(_plan("Pez espada a la plancha", ["150 g de pez espada"]),
                                            {"medicalConditions": ["Embarazo"], "gender": "female"})
    assert r and r.startswith("mercurio"), r


def test_limpio_se_promueve(monkeypatch):
    _sin_piso(monkeypatch)
    assert go._surgical_promote_blocked_reason(_plan("Pollo con arroz", ["150 g de pechuga de pollo"]),
                                               {"allergies": ["Mani"], "dislikes": ["Pescado"]}) is None


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 236
