"""G39: el panel de calidad y coste se puede segmentar por país."""

from __future__ import annotations

from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class _Request:
    headers = {"authorization": "Bearer test"}


def _idx(score: int) -> dict:
    return {
        "score": score,
        "componentes": {"variedad": score, "coherencia": score, "nutricion": score},
        "defectos": {},
        "dias": 7,
        "degradado": False,
    }


def test_panel_expone_plan_y_resumen_por_pais(monkeypatch) -> None:
    import routers.system as system

    rows = [
        {
            "id": "es-ok",
            "fecha": "2026-08-20T00:00:00+00:00",
            "idx": _idx(80),
            "pais": "ES",
            "es_pre_indice": False,
            "usd": 2.0,
            "calls": 2,
            "modelos": ["m1"],
        },
        {
            "id": "es-missing",
            "fecha": "2026-08-21T00:00:00+00:00",
            "idx": None,
            "pais": "ES",
            "es_pre_indice": False,
            "usd": 0,
            "calls": 0,
            "modelos": None,
        },
        {
            "id": "do-ok",
            "fecha": "2026-08-19T00:00:00+00:00",
            "idx": _idx(90),
            "pais": "DO",
            "es_pre_indice": False,
            "usd": 1.0,
            "calls": 1,
            "modelos": ["m1"],
        },
        {
            "id": "legacy",
            "fecha": "2026-07-20T00:00:00+00:00",
            "idx": None,
            "pais": "(pre-sistema)",
            "es_pre_indice": True,
            "usd": 0,
            "calls": 0,
            "modelos": None,
        },
    ]
    monkeypatch.setattr(system, "_verify_admin_token", lambda _token: None)
    monkeypatch.setattr(system, "_check_admin_rate_limit", lambda _request: None)
    monkeypatch.setattr(system, "execute_sql_query", lambda *_args, **_kwargs: rows)

    payload = system.admin_plan_quality(_Request(), days=90, limit=40)

    assert {p["pais"] for p in payload["planes"]} == {"ES", "DO"}
    resumen = payload["resumen"]
    assert resumen["sin_indice"] == 1
    assert resumen["sin_indice_reciente"] == 1
    assert resumen["por_pais"]["ES"] == {
        "con_indice": 1,
        "sin_indice": 0,
        "sin_indice_reciente": 1,
        "score_medio": 80.0,
        "usd_medio_por_plan": 2.0,
    }
    assert resumen["por_pais"]["(pre-sistema)"]["sin_indice"] == 1
    assert resumen["por_pais"]["DO"]["score_medio"] == 90.0


def test_sql_resuelve_pais_por_sello_perfil_o_pre_sistema() -> None:
    source = (BACKEND_ROOT / "routers" / "system.py").read_text(encoding="utf-8")
    start = source.index("def admin_plan_quality")
    body = source[start:source.index("\n@router.", start)] if "\n@router." in source[start:] else source[start:]
    assert "LEFT JOIN user_profiles up ON up.id = m.user_id" in body
    assert "m.plan_data->>'_country'" in body
    assert "up.health_profile->>'country'" in body
    assert "'(pre-sistema)'" in body
    assert "TIMESTAMPTZ '2026-07-31 00:00:00+00'" in body


def test_pfix_marker_cierra_g39() -> None:
    app_source = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    assert "P2-PANEL-CALIDAD-SIN-PAIS · 2026-08-23" in app_source
