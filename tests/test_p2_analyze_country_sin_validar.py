"""G44: /analyze no persiste country crudo ni oculta una constraint rota."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException


BACKEND_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("country", ["DO", "ES", "US", "MX", "PR", "CO", None])
def test_helper_acepta_solo_valores_persistibles(country) -> None:
    from constants import assert_supported_country

    assert assert_supported_country(country) == country


@pytest.mark.parametrize("country", ["España", "es", " ES ", "", 7, {}])
def test_helper_rechaza_sin_coercer_a_do(country) -> None:
    from constants import UnsupportedCountryError, assert_supported_country

    with pytest.raises(UnsupportedCountryError):
        assert_supported_country(country)


def test_hidratacion_invalida_propaga_400_fuera_del_fail_open(monkeypatch) -> None:
    import db
    from routers.plans import _hydrate_country_from_profile_for_submit

    monkeypatch.setattr(
        db,
        "get_user_profile",
        lambda _uid: {"health_profile": {"country": "España"}},
    )
    with pytest.raises(HTTPException) as caught:
        _hydrate_country_from_profile_for_submit({"update_reason": "renew"}, "user-1")
    assert caught.value.status_code == 400


def test_payload_invalido_devuelve_400() -> None:
    from routers.plans import _assert_supported_country_for_request

    with pytest.raises(HTTPException) as caught:
        _assert_supported_country_for_request("España")
    assert caught.value.status_code == 400
    assert "Permitidos" in str(caught.value.detail)


def test_constraint_violation_emite_alerta_per_user(monkeypatch) -> None:
    import db_core
    from routers.plans import _emit_country_profile_constraint_alert_best_effort

    writes = []
    monkeypatch.setattr(
        db_core,
        "execute_sql_write",
        lambda sql, params=None, *args, **kwargs: writes.append((sql, params)),
    )
    error = RuntimeError("violates user_profiles_country_supported")
    _emit_country_profile_constraint_alert_best_effort("user-1", "España", error)

    sql, params = writes[0]
    assert "INSERT INTO system_alerts" in sql
    assert params[0] == "country_profile_constraint_violation:user-1"
    assert json.loads(params[3])["country"] == "España"


def test_las_tres_puertas_de_analyze_validan_antes_del_merge() -> None:
    source = (BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
    assert source.count('_assert_supported_country_for_request(data.get("country"))') >= 2
    postprocess = source[source.index("def _postprocess_pipeline_result"):source.index("def _hydrate_country_from_profile_for_submit")]
    assert '_assert_supported_country_for_request(hp_data.get("country"))' in postprocess
    assert postprocess.index('_assert_supported_country_for_request(hp_data.get("country"))') < postprocess.index("update_user_health_profile_atomic")


def test_alerta_tiene_fila_de_resolucion() -> None:
    doc = (BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "`country_profile_constraint_violation:<user_id>`" in doc


def test_pfix_marker_cierra_g44() -> None:
    app = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    assert "P2-ANALYZE-COUNTRY-SIN-VALIDAR · 2026-08-23" in app
