"""[P1-COUNTRY-STAMP-NO-FALLBACK-WRITE · 2026-08-23]

El país que ``country_for_plan`` deduce del perfil sirve para LEER este recálculo,
pero no es un hecho histórico del plan. La operación ejecutable de persistencia debe
conservar esa diferencia: sin sello no crea ``_country`` ni reescribe el régimen.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _country_system_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def test_country_for_plan_expone_si_leyo_hecho_o_fallback():
    from constants import country_for_plan

    assert country_for_plan({"_country": "ES"}, {"country": "DO"}, return_source=True) == (
        "ES",
        "plan",
    )
    assert country_for_plan({}, {"country": "DO"}, return_source=True) == (
        "DO",
        "profile",
    )


def test_recalc_legacy_no_convierte_fallback_en_sello_ni_borra_regimen(monkeypatch):
    import constants

    events = []
    monkeypatch.setattr(
        constants,
        "emit_country_plan_regime_changed_best_effort",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    plan = {"_pricing_mode": "beta_no_prices", "days": [{"day": 1}]}

    country, pricing_mode, source = constants.apply_recalc_plan_regime(
        plan,
        {"country": "DO"},
        plan_id="legacy-beta",
        emit_observability=True,
    )

    assert (country, pricing_mode, source) == ("DO", "beta_no_prices", "profile")
    assert "_country" not in plan
    assert plan["_pricing_mode"] == "beta_no_prices"
    assert events == []


def test_fallback_sigue_releyendo_el_perfil_despues_del_recalc():
    from constants import apply_recalc_plan_regime

    plan = {"_pricing_mode": "beta_no_prices"}
    apply_recalc_plan_regime(plan, {"country": "DO"})

    country, pricing_mode, source = apply_recalc_plan_regime(plan, {"country": "ES"})

    assert (country, pricing_mode, source) == ("ES", "beta_no_prices", "profile")
    assert "_country" not in plan


def test_recalc_de_plan_sellado_sanea_regimen_y_deja_evento(monkeypatch):
    import constants

    events = []
    monkeypatch.setattr(
        constants,
        "emit_country_plan_regime_changed_best_effort",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    plan = {"_country": "DO", "_pricing_mode": "beta_no_prices"}

    country, pricing_mode, source = constants.apply_recalc_plan_regime(
        plan,
        {"country": "ES"},
        plan_id="stamped-do",
        emit_observability=True,
    )

    assert (country, pricing_mode, source) == ("DO", None, "plan")
    assert plan["_country"] == "DO"
    assert "_pricing_mode" not in plan
    assert len(events) == 1
    assert events[0][1]["pricing_mode_removed"] is True


def test_recalc_de_plan_beta_sellado_repara_regimen_ausente():
    from constants import apply_recalc_plan_regime

    plan = {"_country": "ES"}
    result = apply_recalc_plan_regime(plan, {"country": "DO"})

    assert result == ("ES", "beta_no_prices", "plan")
    assert plan == {"_country": "ES", "_pricing_mode": "beta_no_prices"}


def test_router_usa_el_helper_y_marker_movil_existe():
    router = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")

    assert "apply_recalc_plan_regime" in router
    assert "P1-COUNTRY-STAMP-NO-FALLBACK-WRITE" in app
