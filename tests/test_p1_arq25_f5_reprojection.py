"""[P1-ARQ25-F5-REPROJECTION · 2026-09-05] La proyección de compras se re-encola en el commit de recálculo
(cubre Nevera/restock/consumo), swap, regeneración de día y relleno de bloques. La huella de la lista
(`list_fingerprint`) evita una proyección de ~1 min por cada visita al Dashboard: el recálculo bumpea
`revision` siempre, pero si la lista no cambió la proyección sería idéntica.
"""
from pathlib import Path

import pytest

import plan_jobs as pj

_BACKEND = Path(__file__).resolve().parents[1]
_PLAN = "e45e649c-231d-493a-adbf-af8aa8b73ce8"
_USER = "f47126cb-e137-4003-9db3-cbec22b02d59"


def _pd(qty=907):
    return {
        "calc_household_multiplier": 1.5, "total_days_requested": 30, "days": [{}] * 3, "_archived_days": [],
        "aggregated_shopping_list_weekly": [
            {"name": "Pollo", "base_qty": qty, "base_unit": "g", "market_qty": 2, "market_unit": "lb"},
            {"name": "Arroz", "base_qty": 2000, "base_unit": "g", "market_qty": 1, "market_unit": "funda"},
        ],
    }


def test_a_huella_estable_e_insensible_al_orden_pero_sensible_a_la_cantidad():
    a = pj.shopping_list_fingerprint(_pd())
    b = pj.shopping_list_fingerprint({**_pd(), "aggregated_shopping_list_weekly": list(reversed(_pd()["aggregated_shopping_list_weekly"]))})
    assert a == b and len(a) == 16
    assert pj.shopping_list_fingerprint(_pd(qty=500)) != a
    assert pj.shopping_list_fingerprint({**_pd(), "calc_household_multiplier": 2.0}) != a, "el hogar cambia la proyección"
    assert pj.shopping_list_fingerprint({}) == pj.shopping_list_fingerprint({"aggregated_shopping_list_weekly": []})


def _arm(monkeypatch, *, last_fp=None, rev=7):
    import horizon
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_ENABLED", "1")
    monkeypatch.setattr(horizon, "shopping_projection_jobs_enabled", lambda: True)
    monkeypatch.setattr(horizon, "effective_policy_for_plan", lambda pd, form_data=None: {"policy_hash": "h1", "shopping": {"main_cycle_days": 30, "fresh_topup_days": 7}})
    monkeypatch.setattr(pj, "current_plan_revision", lambda plan_id: rev)
    monkeypatch.setattr(pj, "_load_plan_for_projection", lambda plan_id, user_id: (_pd(), rev))
    import db
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: ({"status": "done", "fp": last_fp} if last_fp else None))
    seen = {}
    monkeypatch.setattr(pj, "enqueue_plan_job", lambda jt, pid, uid, **kw: seen.update(kw, job_type=jt) or "job-1")
    return seen


def test_b_misma_huella_que_el_ultimo_job_no_encola(monkeypatch):
    seen = _arm(monkeypatch, last_fp=pj.shopping_list_fingerprint(_pd()))
    assert pj.enqueue_shopping_reprojection(_PLAN, _USER, reason="recalculate", plan_data=_pd()) is None
    assert not seen


def test_c_lista_distinta_encola_para_la_revision_vigente_con_ventanas(monkeypatch):
    seen = _arm(monkeypatch, last_fp="otra-huella", rev=9)
    jid = pj.enqueue_shopping_reprojection(_PLAN, _USER, reason="swap", plan_data=_pd())
    assert jid == "job-1"
    fp = pj.shopping_list_fingerprint(_pd())
    assert seen["dedup_key"] == f"shopping_projection:{_PLAN}:9:{fp[:12]}"
    assert seen["plan_revision"] == 9 and seen["job_type"] == "shopping_projection"
    p = seen["payload"]
    assert p["list_fingerprint"] == fp and p["reason"] == "swap" and p["policy_hash"] == "h1" and p["total_days"] == 30
    assert [w["kind"] for w in p["windows"]][:2] == ["main", "fresh_topup"], "las ventanas de la política (principal + top-ups)"


def test_d_sin_plan_data_lo_carga_y_sin_politica_no_encola(monkeypatch):
    seen = _arm(monkeypatch, last_fp=None)
    assert pj.enqueue_shopping_reprojection(_PLAN, _USER, reason="chunk_fill") == "job-1", "carga el plan por id (T1 no lo tiene a mano)"
    import horizon
    monkeypatch.setattr(horizon, "effective_policy_for_plan", lambda pd, form_data=None: None)
    seen.clear()
    assert pj.enqueue_shopping_reprojection(_PLAN, _USER, reason="chunk_fill", plan_data=_pd()) is None, "plan pre-F3: sin ventanas"
    assert not seen


def test_e_knob_apagado_y_guests_no_encolan(monkeypatch):
    monkeypatch.delenv("MEALFIT_PLAN_JOBS_ENABLED", raising=False)
    assert pj.enqueue_shopping_reprojection(_PLAN, _USER, reason="recalculate", plan_data=_pd()) is None
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_ENABLED", "1")
    assert pj.enqueue_shopping_reprojection(_PLAN, "guest", reason="recalculate", plan_data=_pd()) is None


def test_f_cableado_en_los_cuatro_commits():
    plans = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    for reason, pd in (("recalculate", "merged_plan_data"), ("swap", "result"), ("regenerate_day", "result")):
        assert f'reason="{reason}", plan_data={pd})' in plans, reason
    # cada hook va DESPUÉS del persist atómico (la revisión ya subió) y del 404 de ownership
    i = plans.index('reason="recalculate"')
    assert plans.rfind("update_plan_data_atomic(", 0, i) > plans.rfind("def api_recalculate_shopping_list", 0, i)
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert '_f5_reproj(meal_plan_id, user_id, reason="chunk_fill")' in cron
    assert cron.index('reason="chunk_fill"') < cron.index("from plan_display_i18n import schedule_plan_display_enrichment as _p1_i18n_schedule")
