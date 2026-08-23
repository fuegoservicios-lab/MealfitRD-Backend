"""G40: el flip backend de país es verificable desde fuera y en telemetría."""

from __future__ import annotations

import json
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_health_version_expone_nombre_y_valor_vivo(monkeypatch) -> None:
    import app

    monkeypatch.setattr(app, "connection_pool", None)
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert app.health_version()["MEALFIT_COUNTRY_SYSTEM"] is True

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    assert app.health_version()["MEALFIT_COUNTRY_SYSTEM"] is False


def test_health_no_usa_snapshot_de_import() -> None:
    source = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    start = source.index("def health_version")
    end = source.index("\n@app.", start)
    body = source[start:end]
    assert '"MEALFIT_COUNTRY_SYSTEM": country_system_enabled' in body
    normalized = " ".join(body.split())
    assert '_env_bool_country_health( "MEALFIT_COUNTRY_SYSTEM", False )' in normalized
    assert "from constants import COUNTRY_SYSTEM_ENABLED" not in body
    assert '"MEALFIT_COUNTRY_SYSTEM": COUNTRY_SYSTEM_ENABLED' not in body


def test_slot_drift_lleva_el_estado_vivo_del_knob(monkeypatch) -> None:
    import db_core
    import graph_orchestrator as go

    writes = []
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(
        db_core,
        "execute_sql_write",
        lambda sql, params=None, *args, **kwargs: writes.append((sql, params)),
    )

    go._emit_slot_drift_metric_best_effort(
        {"score": 0.2},
        {"days": [{"day": 1}]},
        {"country": "ES"},
    )

    metadata = json.loads(writes[0][1][-1])
    assert metadata["country"] == "ES"
    assert metadata["MEALFIT_COUNTRY_SYSTEM"] is True


def test_slot_drift_distingue_knob_apagado_de_usuario_do(monkeypatch) -> None:
    import db_core
    import graph_orchestrator as go

    metadata_rows = []
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    monkeypatch.setattr(
        db_core,
        "execute_sql_write",
        lambda _sql, params=None, *args, **kwargs: metadata_rows.append(json.loads(params[-1])),
    )

    go._emit_slot_drift_metric_best_effort(
        {"score": 0.2},
        {"days": [{"day": 1}]},
        {"country": "ES"},
    )

    assert metadata_rows[0]["country"] == "DO"
    assert metadata_rows[0]["MEALFIT_COUNTRY_SYSTEM"] is False


def test_pfix_marker_cierra_g40() -> None:
    app_source = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    assert "P2-FLIP-BACKEND-INVERIFICABLE · 2026-08-23" in app_source
