"""[P1-COUNTRY-PLAN-VS-PERFIL-EN-BLOQUES · 2026-08-23]

Un plan existente conserva el país con que nació: el perfil vivo puede nutrir
otros campos, pero no puede convertir bloques, swaps o ediciones en un híbrido.
"""
from __future__ import annotations

import ast
from pathlib import Path


_BACKEND = Path(__file__).resolve().parents[1]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no existe {name} en {path.name}")


def _directly_calls_imported(fn: ast.FunctionDef, imported_name: str) -> bool:
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == imported_name
    }
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in aliases
        for node in ast.walk(fn)
    )


def test_merge_del_worker_actualiza_perfil_sin_pisar_pais_del_snapshot():
    from cron_tasks import _merge_chunk_live_profile

    snapshot = {
        "country": "ES",
        "goal": "lose_fat",
        "_days_offset": 7,
        "session_id": "s-original",
    }
    live = {
        "country": "DO",
        "goal": "build_muscle",
        "weight": 82,
        "session_id": "s-ajena",
        "_private": "omitido",
    }

    merged = _merge_chunk_live_profile(snapshot, live)

    assert merged is snapshot
    assert merged["country"] == "ES"
    assert merged["goal"] == "build_muscle"
    assert merged["weight"] == 82
    assert merged["session_id"] == "s-original"
    assert "_private" not in merged


def test_el_ssot_del_plan_prefiere_es_sobre_perfil_do():
    from constants import country_for_plan

    assert country_for_plan({"_country": "ES"}, {"country": "DO"}) == "ES"


def test_generacion_persist_y_chat_modify_llaman_directo_al_ssot_del_plan():
    surfaces = (
        (_BACKEND / "routers" / "plans.py", "api_swap_meal"),
        (_BACKEND / "routers" / "plans.py", "api_swap_meal_persist"),
        (_BACKEND / "tools.py", "execute_modify_single_meal"),
    )
    for path, name in surfaces:
        fn = _function(path, name)
        assert _directly_calls_imported(fn, "country_for_plan"), (
            f"{name} debe resolver el país desde el sello del plan, no sólo desde el perfil"
        )


def test_swap_carga_solo_el_sello_del_plan_y_respeta_ownership(monkeypatch):
    import db_core
    from routers.plans import _load_swap_plan_country_stub

    seen = {}

    def fake_query(sql, params, fetch_one=False):
        seen.update(sql=sql, params=params, fetch_one=fetch_one)
        return {"plan_country": "ES"}

    monkeypatch.setattr(db_core, "execute_sql_query", fake_query)
    assert _load_swap_plan_country_stub("u-1", "p-1") == {"_country": "ES"}
    assert "plan_data->>'_country'" in seen["sql"]
    assert "id = %s" in seen["sql"] and "user_id = %s" in seen["sql"]
    assert seen["params"] == ("p-1", "u-1")
    assert seen["fetch_one"] is True


def test_marker_movil_del_gap():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX = "P' in app and " · 2026-" in app
    assert "P1-COUNTRY-PLAN-VS-PERFIL-EN-BLOQUES" in cron
