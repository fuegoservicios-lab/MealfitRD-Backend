"""G38: los dos cambios de régimen de país dejan alertas operables."""

from __future__ import annotations

import json
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_primer_plan_beta_emite_una_alerta_insert_only(monkeypatch) -> None:
    import db_core
    from constants import stamp_plan_country

    writes = []
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: writes.append((sql, params)))

    stamp_plan_country({}, {"country": "ES"}, emit_observability=True)

    assert len(writes) == 1
    sql, params = writes[0]
    assert "INSERT INTO system_alerts" in sql
    assert "ON CONFLICT (alert_key) DO NOTHING" in sql
    assert params[0] == "country_beta_first_plan:ES"
    assert json.loads(params[3])["country"] == "ES"


def test_plan_dominicano_no_emite_alerta_de_primer_beta(monkeypatch) -> None:
    import db_core
    from constants import stamp_plan_country

    writes = []
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: writes.append((sql, params)))

    stamp_plan_country({}, {"country": "DO"}, emit_observability=True)
    assert writes == []


def test_sobrescribir_pais_distinto_emite_cambio_per_plan(monkeypatch) -> None:
    import db_core
    from constants import stamp_plan_country

    writes = []
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: writes.append((sql, params)))

    stamp_plan_country(
        {"_country": "ES", "_pricing_mode": "beta_no_prices"},
        {"country": "DO", "plan_id": "plan-123"},
        emit_observability=True,
    )

    assert len(writes) == 1
    sql, params = writes[0]
    assert params[0] == "country_plan_regime_changed:plan-123"
    assert "resolved_at = NULL" in sql
    metadata = json.loads(params[3])
    assert metadata["previous_country"] == "ES"
    assert metadata["country"] == "DO"


def test_pricing_mode_removed_emite_metadata_diagnostica(monkeypatch) -> None:
    import db_core
    from constants import emit_country_plan_regime_changed_best_effort

    writes = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: writes.append((sql, params)))

    emit_country_plan_regime_changed_best_effort(
        "plan-456",
        previous_country=None,
        country="DO",
        previous_pricing_mode="beta_no_prices",
        pricing_mode=None,
        pricing_mode_removed=True,
    )

    assert writes[0][1][0] == "country_plan_regime_changed:plan-456"
    assert json.loads(writes[0][1][3])["pricing_mode_removed"] is True


def test_recalculo_llama_alerta_antes_de_persistir_borrado() -> None:
    router = (BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
    constants = (BACKEND_ROOT / "constants.py").read_text(encoding="utf-8")
    assert "apply_recalc_plan_regime" in router
    assert "_pricing_mode_removed = _had_pricing_mode and not _pricing_mode" in constants
    assert "emit_country_plan_regime_changed_best_effort" in constants


def test_stamp_sigue_puro_por_default(monkeypatch) -> None:
    import db_core
    from constants import stamp_plan_country

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(
        db_core,
        "execute_sql_write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("I/O inesperado")),
    )
    plan = {}
    stamp_plan_country(plan, {"country": "ES"})
    assert plan == {"_country": "ES"}


def test_assemble_activa_observabilidad_del_sello() -> None:
    source = (BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "stamp_plan_country(result, form_data, emit_observability=True)" in source


def test_las_dos_alertas_estan_en_la_tabla_de_resolucion() -> None:
    doc = (BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md").read_text(
        encoding="utf-8"
    )
    assert "`country_beta_first_plan:<country>`" in doc
    assert "`country_plan_regime_changed:<plan_id>`" in doc


def test_pfix_marker_cierra_g38() -> None:
    constants = (BACKEND_ROOT / "constants.py").read_text(encoding="utf-8")
    assert "P2-COUNTRY-OBSERVABILIDAD-CERO" in constants
