# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-811 · 2026-09-29] La renovación (`rolling_refill`) conserva la PlanPolicy y el relleno de 7 días por
`/shift-plan` vuelve a existir.

Tres defectos del mismo camino:
  1. `/shift-plan` detectaba el «gap huérfano» de un plan de 7 días (0 bloques vivos, ventana incompleta) y SOLO lo
     escribía en el log: el `elif` de esa rama se quedaba con el caso y la rama que encola nunca corría (commit 29889c97).
  2. El snapshot de TODA renovación (cron y HTTP) era `{**health_profile}`: la política efectiva del plan
     (`plan_data._plan_policy.effective`) no viajaba, así que el relleno se generaba sin anclas, sin banda de
     recurrencia, sin compra única y sin el bloque 📐 — en silencio. Además el HTTP no ponía las marcas de continuación
     que el cron sí pone y que el gate temporal consume.
  3. El WARNING «gap huérfano» del cron salía en cada tick (84 en tres semanas) aunque no encolara nada.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Response

_BACKEND = Path(__file__).resolve().parents[1]

_EFF = {
    "policy_hash": "hash-del-plan-811",
    "recurrence": {"global_mode": "routine"},
    "food_anchors": [{"name": "Huevo", "min_per_7d": 5, "max_per_7d": 7, "slots": ["breakfast"]}],
    "budget": {"mode": "hard", "tier": "low", "status": "ok"},
}
_HP = {"age": 30, "gender": "female", "mainGoal": "lose_fat", "tzOffset": 0}


@pytest.fixture(autouse=True)
def _politica_activa(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "enforce")
    monkeypatch.delenv("MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL", raising=False)
    monkeypatch.delenv("MEALFIT_REFILL_CARRIES_POLICY", raising=False)


def _ayer_utc() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()


def _plan(total=7, n_dias=3, status="complete", ancla=None, politica=True):
    pd = {
        "grocery_start_date": ancla or _ayer_utc(),
        "generation_status": status,
        "total_days_requested": total,
        "days": [{"day": i, "meals": [{"name": f"Plato {i}"}]} for i in range(1, n_dias + 1)],
    }
    if politica:
        pd["_plan_policy"] = {"requested": {}, "effective": dict(_EFF), "relaxations": []}
    return pd


def _cursor_despachador(plan_data, vivos, *, conflicto=None):
    """Cada consulta recibe SU fila (despacho por SQL, no por posición)."""
    ultimo = {"q": ""}

    def _exec(sql, *a, **k):
        ultimo["q"] = " ".join(str(sql).split())

    def _uno():
        q = ultimo["q"]
        if "plan_mode" in q:
            return {"plan_mode": "plan", "plan_mode_changed_at": None}
        if "SELECT id FROM meal_plans" in q:
            return {"id": "plan-811"}
        if "health_profile" in q:
            return {"health_profile": dict(_HP)}
        if "plan_data" in q:
            return {"plan_data": plan_data}
        if "en_vuelo" in q:
            return {"en_vuelo": 0}
        if "COUNT(*) AS cnt" in q:
            return {"cnt": vivos}
        if "max_week" in q:
            return {"max_week": 2}
        if "chunk_kind" in q:
            return conflicto
        return {}

    cur = MagicMock()
    cur.execute.side_effect = _exec
    cur.fetchone.side_effect = _uno
    cur.fetchall.return_value = []
    return cur


def _pool_con(cur):
    pool = MagicMock()
    conn = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    conn.transaction.return_value.__enter__.return_value = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return pool


def _shift_http(plan_data, vivos):
    from routers.plans import api_shift_plan
    cur = _cursor_despachador(plan_data, vivos)
    with patch("cron_tasks._enqueue_plan_chunk") as enq, patch("db_core.connection_pool", _pool_con(cur)), \
            patch("routers.plans.update_user_health_profile_atomic", create=True):
        r = api_shift_plan(Response(), {"user_id": "user-811", "tzOffset": 0}, verified_user_id="user-811")
    return r, enq


def _shift_cron(plan_data, vivos, inventario=None):
    import cron_tasks
    cur = _cursor_despachador(plan_data, vivos)
    with patch("cron_tasks._enqueue_plan_chunk") as enq, patch("db_core.connection_pool", _pool_con(cur)), \
            patch("db_inventory.get_user_inventory_net", return_value=inventario):
        r = cron_tasks._background_shift_plan_for_user("user-811", 0)
    return r, enq


def _snapshot(enq, i=0):
    args, kwargs = enq.call_args_list[i]
    return args[5]


# ─────────────── 1. la decisión, pura y compartida ───────────────
def test_decidir_gap_huerfano_de_7_dias_encola():
    import relleno_rolling as rr
    assert rr.decidir(7, 2, 6, 3, 0, "complete") == (True, "gap_huerfano_7d")


@pytest.mark.parametrize("args,motivo", [
    ((7, 2, 6, 3, 1, "complete"), "bloques_vivos"),
    ((7, 2, 6, 3, 0, "partial"), "plan_en_generacion"),
    ((7, 2, 6, 3, 0, "generating_next"), "plan_en_generacion"),
    ((7, 3, 6, 3, 0, "complete"), "ventana_completa"),
    ((7, 0, 0, 0, 0, "complete"), "sin_dias_restantes"),
])
def test_decidir_no_encola(args, motivo):
    import relleno_rolling as rr
    assert rr.decidir(*args) == (False, motivo)


def test_decidir_con_el_gap_de_7_dias_apagado_no_encola_y_los_planes_largos_siguen():
    import relleno_rolling as rr
    assert rr.decidir(7, 2, 6, 3, 0, "complete", gap_7d=False) == (False, "gap_7d_apagado")
    assert rr.decidir(15, 2, 14, 3, 0, "complete", gap_7d=False) == (True, "ventana_incompleta")


# ─────────────── 2. el snapshot: marcas de continuación + política del plan ───────────────
def test_snapshot_lleva_marcas_y_politica_del_plan():
    import relleno_rolling as rr
    hp = dict(_HP, _plan_policy_effective={"policy_hash": "del-cliente"}, _blueprint_slice={"x": 1})
    s = rr.snapshot_relleno(hp=hp, user_id="user-811", chunk_count=4, ancla_iso="2026-09-29",
                            plan_data=_plan(), previous_meals=["A"], triggered_by="t")
    fd = s["form_data"]
    assert fd["_is_continuation"] is True and fd["_continuation_anchor_iso"] == "2026-09-29"
    assert fd["_plan_start_date"] == "2026-09-29" and fd["totalDays"] == 4 and fd["user_id"] == "user-811"
    assert fd["_plan_policy_effective"]["policy_hash"] == "hash-del-plan-811", "manda la política del PLAN"
    assert fd["_policy_enforced"] is True
    assert "_blueprint_slice" not in fd, "sin mapeo fiable día-del-ciclo ↔ offset del relleno: sin rebanada"
    assert s["_is_rolling_refill"] is True and s["_triggered_by"] == "t" and s["previous_meals"] == ["A"]
    assert "_is_weekly_renewal" not in s


def test_con_el_knob_de_politica_apagado_el_snapshot_es_el_de_antes(monkeypatch):
    import relleno_rolling as rr
    monkeypatch.setenv("MEALFIT_REFILL_CARRIES_POLICY", "false")
    s = rr.snapshot_relleno(hp=dict(_HP), user_id="u", chunk_count=3, ancla_iso="a", plan_data=_plan(),
                            previous_meals=[], continuacion=False)
    assert set(s["form_data"]) == set(_HP) | {"user_id", "totalDays", "_plan_start_date"}
    assert set(s) == {"form_data", "taste_profile", "memory_context", "previous_meals", "totalDays", "_is_rolling_refill"}


def test_con_la_politica_del_motor_apagada_no_viaja_nada(monkeypatch):
    import relleno_rolling as rr
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "off")
    s = rr.snapshot_relleno(hp=dict(_HP), user_id="u", chunk_count=3, ancla_iso="a", plan_data=_plan(),
                            previous_meals=[])
    assert "_plan_policy_effective" not in s["form_data"] and "_policy_enforced" not in s["form_data"]


# ─────────────── 3. /shift-plan: el gap huérfano de 7 días se rellena ───────────────
def test_http_plan_7d_ancla_ayer_sin_bloques_vivos_encola_uno_con_continuacion_y_politica():
    r, enq = _shift_http(_plan(), vivos=0)
    assert r["success"] is True
    assert enq.call_count == 1, "antes: 0 (el elif del «gap huérfano» sólo escribía en el log)"
    args, kwargs = enq.call_args_list[0]
    assert args[3:5] == (2, 4) and kwargs.get("chunk_kind") == "rolling_refill"
    fd = args[5]["form_data"]
    assert fd["_is_continuation"] is True
    assert fd["_continuation_anchor_iso"] == fd["_plan_start_date"]
    assert fd["_plan_policy_effective"]["policy_hash"] == "hash-del-plan-811"
    assert fd["_policy_enforced"] is True


def test_http_con_bloques_vivos_no_encola():
    _, enq = _shift_http(_plan(), vivos=1)
    enq.assert_not_called()


def test_http_con_el_knob_apagado_vuelve_la_conducta_vieja(monkeypatch):
    monkeypatch.setenv("MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL", "false")
    _, enq = _shift_http(_plan(), vivos=0)
    enq.assert_not_called()


def test_http_partial_no_encola():
    """Documentado: un plan de 7 días en `partial` con 0 bloques vivos no lo rellena NADIE (ni HTTP ni cron).
    Ese hueco queda fuera de este lote: `partial` significa «hay generación en curso» para las dos ramas."""
    _, enq = _shift_http(_plan(status="partial"), vivos=0)
    enq.assert_not_called()
    r, enq = _shift_cron(_plan(status="partial"), vivos=0)
    assert r is False
    enq.assert_not_called()


def test_http_plan_15d_catchup_tambien_lleva_la_politica_y_las_marcas():
    _, enq = _shift_http(_plan(total=15, n_dias=2), vivos=0)
    assert enq.call_count >= 1
    fd = _snapshot(enq)["form_data"]
    assert fd["_is_continuation"] is True and fd["_plan_policy_effective"]["policy_hash"] == "hash-del-plan-811"


# ─────────────── 4. el cron: mismo relleno, la política viaja, el WARNING sólo si encola ───────────────
def test_cron_relleno_7d_lleva_la_politica_y_las_marcas():
    r, enq = _shift_cron(_plan(), vivos=0)
    assert r is True and enq.call_count == 1
    fd = _snapshot(enq)["form_data"]
    assert fd["_is_continuation"] is True and fd["_continuation_anchor_iso"] == fd["_plan_start_date"]
    assert fd["_plan_policy_effective"]["policy_hash"] == "hash-del-plan-811"
    assert _snapshot(enq)["_triggered_by"] == "background_cron_p0_2"


def test_cron_renovacion_semanal_lleva_la_politica():
    from constants import CHUNK_MIN_FRESH_PANTRY_ITEMS
    ancla = (datetime.now(timezone.utc) - timedelta(days=15)).date().isoformat()
    inv = [f"Alimento {i}" for i in range(CHUNK_MIN_FRESH_PANTRY_ITEMS + 2)]
    r, enq = _shift_cron(_plan(total=15, n_dias=2, ancla=ancla), vivos=0, inventario=inv)
    assert r is True and enq.call_count >= 1
    s = _snapshot(enq)
    assert s["_is_weekly_renewal"] is True and s["form_data"]["_is_continuation"] is True
    assert s["form_data"]["_plan_policy_effective"]["policy_hash"] == "hash-del-plan-811"
    assert s["form_data"]["current_pantry_ingredients"] == inv


def test_cron_sin_encolar_no_hay_warning_de_gap(caplog):
    """5 días visibles, ancla ayer ⇒ 4 tras el shift ≥ ventana 3: no hay relleno y NO debe salir el WARNING."""
    with caplog.at_level(logging.DEBUG):
        _, enq = _shift_cron(_plan(n_dias=5), vivos=0)
    enq.assert_not_called()
    avisos = [r for r in caplog.records if r.levelno >= logging.WARNING and "gap" in r.getMessage().lower()]
    assert avisos == [], [r.getMessage() for r in avisos]


def test_cron_cuando_encola_si_avisa(caplog):
    with caplog.at_level(logging.WARNING):
        _, enq = _shift_cron(_plan(), vivos=0)
    assert enq.call_count == 1
    assert any("gap huérfano" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


# ─────────────── 5. el worker recalcula el enforce también sin rebanada ───────────────
def test_el_worker_recalcula_el_enforce_con_la_politica_sin_rebanada():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index("tooltip-anchor: P1-PLAN-LOTE-811-ENFORCE")
    bloque = src[i:i + 600]
    assert 'form_data.get("_plan_policy_effective")' in bloque and 'form_data["_policy_enforced"]' in bloque


# ─────────────── 6. la alerta de `partial` sin días dice lo que mide ───────────────
def test_alerta_stranded_aclara_la_edad_y_lleva_frozen_at():
    import cron_tasks
    filas = [{"plan_id": "dea00a2f", "user_id": "u", "gen_status": "partial", "age_hours": 182.6,
              "frozen_at": "2026-09-17T12:40:00+00:00"}]
    with patch.object(cron_tasks, "execute_sql_query", return_value=filas), \
            patch.object(cron_tasks, "execute_sql_write") as w:
        cron_tasks._alert_stranded_partial_plans()
    insert = [c for c in w.call_args_list if "INSERT INTO system_alerts" in str(c.args[0])][0]
    msg, meta = insert.args[1][2], json.loads(insert.args[1][3])
    assert "edad del plan" in msg
    assert meta["_frozen_at"] == "2026-09-17T12:40:00+00:00"


def test_la_alerta_no_excluye_los_congelados():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index("def _alert_stranded_partial_plans(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert "_frozen_at" in cuerpo
    assert "NOT (plan_data ? '_frozen_at')" not in cuerpo and "? '_frozen_at')" not in cuerpo.split("WHERE", 1)[1].split("ORDER BY")[0]


def test_marker_y_knobs():
    src = (_BACKEND / "relleno_rolling.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-811" in src
    assert "MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL" in src and "MEALFIT_REFILL_CARRIES_POLICY" in src
    import relleno_rolling as rr
    from knobs import get_knobs_registry_snapshot
    rr.gap_7d_http_activo()
    rr.lleva_politica()
    snap = get_knobs_registry_snapshot()
    nombres = set(snap) if isinstance(snap, dict) else {k.get("name") for k in snap}
    assert {"MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL", "MEALFIT_REFILL_CARRIES_POLICY"} <= nombres
