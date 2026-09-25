# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-237 · 2026-09-25] El plan de emergencia pasa cada plantilla por el escáner SSOT.

Auditoría del 25-sep: el fallback (la red tras un rechazo crítico, a menudo por alérgeno) filtraba con 13 clases propias;
sus plantillas «neutrales» llevan aguacate y semillas → «Arroz con Vegetales y Aguacate» a un alérgico al aguacate.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from fallback_pools import _FALLBACK_MEAL_POOLS  # noqa: E402


def _texto(tmpl):
    name, _t, _d, ings = tmpl
    return (name + " " + " ".join(ings)).lower()


def _dias(slot, form_data, n=10):
    pool = _FALLBACK_MEAL_POOLS[slot]
    tokens = go._fallback_restricted_tokens(form_data)
    return [go._select_safe_fallback_meal(pool, tokens, day_number=d, form_data=form_data) for d in range(1, n + 1)]


def test_alergia_al_aguacate_no_recibe_aguacate():
    fd = {"allergies": ["Aguacate"]}
    for slot in _FALLBACK_MEAL_POOLS:
        for tmpl in _dias(slot, fd):
            assert "aguacate" not in _texto(tmpl), (slot, tmpl[0])


def test_alergia_tecleada_a_mano_tambien_cuenta():
    fd = {"allergies": [], "otherAllergies": "aguacate"}
    for tmpl in _dias("Almuerzo", fd):
        assert "aguacate" not in _texto(tmpl), tmpl[0]


def test_rechazo_tambien_cuenta():
    fd = {"dislikes": ["Aguacate"]}
    for tmpl in _dias("Cena", fd):
        assert "aguacate" not in _texto(tmpl), tmpl[0]


def test_sin_restricciones_rota_igual_que_antes():
    pool = _FALLBACK_MEAL_POOLS["Almuerzo"]
    for d in range(1, 8):
        assert (go._select_safe_fallback_meal(pool, frozenset(), day_number=d, form_data={})
                == go._select_safe_fallback_meal(pool, frozenset(), day_number=d))


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 237
