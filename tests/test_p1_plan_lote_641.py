# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-641 · 2026-09-27] Apagar el sistema de países no borra el sello del plan (G42).

`MEALFIT_COUNTRY_SYSTEM=false` es la palanca de emergencia del flip. Con ella bajada, `country_for_form_data` dice 'DO'
para todo —también para el sello de un plan español—, y el recálculo de la lista (`apply_recalc_plan_regime`) lo
tomaba como hecho: escribía `_country='DO'` y borraba `_pricing_mode`. Al volver a subir la palanca, ese plan ya era
dominicano para siempre: el rollback no era reversible. Con la palanca bajada, el recálculo LEE (manda RD) pero no
escribe nada.

tooltip-anchor: P1-PLAN-LOTE-641
"""
import pytest


@pytest.fixture
def plan_es():
    return {"_country": "ES", "_pricing_mode": "beta_no_prices", "days": []}


def test_con_la_palanca_bajada_el_sello_sobrevive(monkeypatch, plan_es):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    from constants import apply_recalc_plan_regime
    antes = dict(plan_es)
    country, _modo, source = apply_recalc_plan_regime(plan_es, {"country": "ES"})
    assert country == "DO"                  # el motor se comporta como RD mientras la palanca está bajada…
    assert plan_es == antes                 # …pero el artefacto no se toca
    assert source == "plan"


def test_con_la_palanca_subida_sigue_saneando(monkeypatch, plan_es):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    from constants import apply_recalc_plan_regime
    country, modo, source = apply_recalc_plan_regime(plan_es, {"country": "DO"})
    assert (country, source) == ("ES", "plan")
    assert plan_es["_country"] == "ES" and plan_es["_pricing_mode"] == modo


def test_bajar_y_subir_la_palanca_devuelve_el_mismo_plan(monkeypatch, plan_es):
    from constants import apply_recalc_plan_regime
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    apply_recalc_plan_regime(plan_es, {"country": "ES"})
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    country, _m, _s = apply_recalc_plan_regime(plan_es, {"country": "ES"})
    assert country == "ES" and plan_es["_country"] == "ES"
