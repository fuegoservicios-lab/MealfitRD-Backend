# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-218 · 2026-09-24] Sin congelador, la proteína fresca no pasa del día 3 — tampoco en la primera semana.

La «semana de frescos» (días 1-7 sin exigencia de durabilidad) valía también para el pollo y el pescado, que en la nevera
aguantan 3 días según la propia tabla de `pantry_durability`. Batería real del 24-sep (30 días, sin congelador): la
compra del día 1 traía 3 lb de pechuga y 1,4 lb de tilapia para cocinar hasta el día 7. Con congelador (limitado o
completo) nada cambia: la proteína se congela.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pantry_durability as pd  # noqa: E402
import horizon as hz  # noqa: E402

SIN = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"}}
LIMITADO = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "limited"}}


def test_sin_congelador_la_exigencia_empieza_el_dia_4():
    assert pd.single_trip_requirements(SIN, 2) is None
    req = pd.single_trip_requirements(SIN, 3)
    assert req == {"need_days": 4, "allow_frozen": False, "freezer_mode": "none", "freeze_window_days": 0}
    # el pollo del día 4 ya no cabe; la lechuga y el huevo sí (su propio plazo)
    assert pd.ingredient_issue_beyond_horizon("150 g de pechuga de pollo", 3, False) == "protein_beyond_freeze_window"
    assert pd.ingredient_issue_beyond_horizon("2 tazas de lechuga", 6, False) is None
    assert pd.ingredient_issue_beyond_horizon("2 huevos", 6, False) is None


def test_con_congelador_la_semana_de_frescos_sigue_igual():
    assert pd.single_trip_requirements(LIMITADO, 3) is None and pd.single_trip_requirements(LIMITADO, 6) is None
    assert pd.single_trip_requirements(LIMITADO, 7)["allow_frozen"] is True


def test_el_knob_devuelve_la_semana_entera(monkeypatch):
    monkeypatch.setenv("MEALFIT_SINGLE_TRIP_NO_FREEZER_FREE_DAYS", "7")
    assert pd.single_trip_requirements(SIN, 6) is None and pd.single_trip_requirements(SIN, 7) is not None


def test_el_prompt_lo_dice():
    txt = " ".join(hz.single_trip_prompt_lines(SIN, {"days": []}))
    assert "SOLO en los primeros 3 días" in txt and "Sin congelador" in txt
