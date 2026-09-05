"""[P1-ARQ25-F5-SHOPPING-PROJECTION · 2026-09-04] Fase 5, rebanada 2: consumidor `shopping_projection`
(las listas 7/15/30 como proyección, con el mismo agregador del recálculo) + `GET /{plan_id}/projections`
(estado none/pending/ready/failed/stale + read model). Sin DB: agregador y carga del plan simulados.
"""
from pathlib import Path

import pytest

import plan_jobs as pj

_BACKEND = Path(__file__).resolve().parents[1]
_PLAN = "e45e649c-231d-493a-adbf-af8aa8b73ce8"
_USER = "f47126cb-e137-4003-9db3-cbec22b02d59"
_WINDOWS = [
    {"kind": "main", "start_day": 0, "end_day": 30, "days": 30, "cycle_days": 30, "fresh_only": False},
    {"kind": "fresh_topup", "start_day": 7, "end_day": 14, "days": 7, "cycle_days": 7, "fresh_only": True},
]


def _fake_sc(monkeypatch, calls):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan", lambda uid, pd, is_new_plan=False: (["inv"], ["cons"]))
    monkeypatch.setattr(sc, "active_trip_window_days", lambda pd: [0, 1, 2])
    monkeypatch.setattr(sc, "cycle_qty_multiplier", lambda dur: {"weekly": 1.0, "biweekly": 2.0, "monthly": 4.0}[dur])

    def fake_delta(user_id, plan_data, is_new_plan=False, structured=False, multiplier=1.0, *, inventory_override=None,
                   consumed_override=None, cycle_days=None, window_days=None):
        calls.append({"multiplier": multiplier, "cycle_days": cycle_days, "inv": inventory_override, "window_days": window_days})
        return [
            {"name": "Pollo", "display_string": "2 lb de Pollo", "category": "Proteínas", "market_qty": 2, "market_unit": "lb",
             "base_qty": 907, "base_unit": "g", "is_perishable": True, "estimated_cost": 300.0},
            {"name": "Arroz", "display_string": "1 funda de Arroz", "category": "Granos", "market_qty": 1, "market_unit": "funda",
             "base_qty": 2000, "base_unit": "g", "is_perishable": False, "estimated_cost": 120.0},
        ]

    monkeypatch.setattr(sc, "get_shopping_list_delta", fake_delta)


def test_a_una_lista_por_ventana_con_el_multiplicador_del_recalculo(monkeypatch):
    calls = []
    _fake_sc(monkeypatch, calls)
    pd = {"calc_household_multiplier": 1.5, "days": []}
    out = pj.build_shopping_projection(pd, _USER, _WINDOWS, revision=9, policy_hash="abc")
    assert out["schema_version"] == 1 and out["revision"] == 9 and out["policy_hash"] == "abc"
    assert [w["kind"] for w in out["windows"]] == ["main", "fresh_topup"]
    main, fresh = out["windows"]
    assert calls[0]["multiplier"] == pytest.approx(1.5 * 4.0), "ciclo de 30 días: hogar × multiplicador mensual"
    assert calls[0]["cycle_days"] == 30 and calls[0]["inv"] == ["inv"] and calls[0]["window_days"] == [0, 1, 2]
    assert calls[1]["multiplier"] == pytest.approx(1.5), "top-up de frescos: solo hogar"
    assert main["item_count"] == 2 and main["cost_rd"] == pytest.approx(420.0)
    assert fresh["item_count"] == 1 and fresh["items"][0]["name"] == "Pollo", "fresh_only filtra perecederos"
    assert main["items"][1]["cost_rd"] == 120.0 and "estimated_cost" not in main["items"][1]


def test_b_consumidor_stale_reencola_para_la_revision_vigente(monkeypatch):
    monkeypatch.setattr(pj, "_load_plan_for_projection", lambda plan_id, user_id: ({"days": []}, 12))
    seen = {}
    monkeypatch.setattr(pj, "enqueue_plan_job", lambda jt, pid, uid, **kw: seen.update(kw, job_type=jt) or "new")
    job = {"id": "old", "plan_id": _PLAN, "user_id": _USER, "plan_revision": 3, "dedup_key": f"shopping_projection:{_PLAN}:3:f3d08411ebee",
           "payload": {"windows": _WINDOWS, "policy_hash": "abc"}}
    status, code, result = pj._consume_shopping_projection(job)
    assert (status, code) == ("stale", "revision_changed")
    assert seen["dedup_key"] == f"shopping_projection:{_PLAN}:12:f3d08411ebee" and seen["plan_revision"] == 12
    assert seen["payload"]["windows"] == _WINDOWS and seen["payload"]["requeued_from"] == "old"


def test_c_consumidor_done_guarda_la_proyeccion_en_el_resultado(monkeypatch):
    monkeypatch.setattr(pj, "_load_plan_for_projection", lambda plan_id, user_id: ({"days": []}, 3))
    monkeypatch.setattr(pj, "build_shopping_projection", lambda pd, uid, w, **kw: {"windows": [{"kind": "main"}], "revision": kw["revision"]})
    job = {"id": "j", "plan_id": _PLAN, "user_id": _USER, "plan_revision": 3, "payload": {"windows": _WINDOWS}}
    status, code, result = pj._consume_shopping_projection(job)
    assert (status, code) == ("done", None) and result["projection"]["revision"] == 3
    # sin ventanas → payload inválido → dead (no reintentar lo que nunca funcionará)
    assert pj._consume_shopping_projection({"id": "k", "plan_id": _PLAN, "user_id": _USER, "payload": {}})[0] == "dead"


def test_d_registrado_y_habilitado_por_defecto(monkeypatch):
    monkeypatch.delenv("MEALFIT_PLAN_JOBS_SHOPPING_PROJECTION", raising=False)
    assert pj.CONSUMERS["shopping_projection"] is pj._consume_shopping_projection
    assert "shopping_projection" in pj.enabled_consumers()


def test_e_estado_ui_por_revision():
    ready = {"id": "a", "status": "done", "plan_revision": 5, "payload": {"result": {"projection": {"windows": [1]}}}}
    old = {"id": "b", "status": "done", "plan_revision": 4, "payload": {"result": {"projection": {"windows": [1]}}}}
    assert pj.classify_projection_jobs(5, [ready, old])["status"] == "ready"
    assert pj.classify_projection_jobs(6, [old])["status"] == "stale"
    assert pj.classify_projection_jobs(6, [{"id": "c", "status": "pending", "plan_revision": 6}, old])["status"] == "pending"
    f = pj.classify_projection_jobs(6, [{"id": "d", "status": "failed", "plan_revision": 6, "attempts": 2, "error_code": "x"}])
    assert f["status"] == "failed" and f["retrying"] is True
    assert pj.classify_projection_jobs(6, [{"id": "e", "status": "dead", "plan_revision": 6}])["retrying"] is False
    assert pj.classify_projection_jobs(1, [])["status"] == "none"


def test_f_endpoint_exento_con_ownership_y_documentado():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert '@router.get("/{plan_id}/projections")' in src
    assert "_PROJECTIONS_LIMITER = RateLimiter(max_calls=30, period_seconds=60)" in src
    i = src.index("def _projection_snapshot(")
    body = src[i:i + 1600]
    assert body.count("AND user_id = %s") == 2, "plan y jobs, ambos con ownership"
    assert "job_type = 'shopping_projection'" in body
    claude = (_BACKEND / "CLAUDE.md").read_text(encoding="utf-8")
    assert "/{plan_id}/projections" in claude and "P1-ARQ25-F5-SHOPPING-PROJECTION" in claude
