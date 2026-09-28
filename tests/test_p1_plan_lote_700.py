# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-700 · 2026-09-28] Al descongelar un plan, sus pestañas dicen el día de la semana correcto.

`_resume_frozen_plan` adelanta las anclas los días que el plan estuvo congelado (`_shift_plan_dates_for_freeze`) pero
no tocaba `days[].day_name` ni `days[].date`: 92328ff7 enseñaba «Jueves 17 … Miércoles 23» con el ancla en el 22, y
el Dashboard pinta las pestañas con `day_name`. El shift normal no lo arregla: tras descongelar no hay días que
archivar, así que no persiste nada. Ahora el descongelado reetiqueta cada día desde la fecha LOCAL de la nueva ancla
(`constants.fecha_local_del_ancla`), con `update_plan_data_atomic` (FOR UPDATE, invariante I7).

tooltip-anchor: P1-PLAN-LOTE-700
"""
import cron_tasks


def test_reetiqueta_los_dias_desde_la_nueva_ancla():
    pd = {"grocery_start_date": "2026-09-22T15:47:33.496595+00:00",
          "days": [{"day": 1, "day_name": "Jueves", "date": "2026-09-17"},
                   {"day": 2, "day_name": "Viernes", "date": "2026-09-18"}]}
    cron_tasks._reetiquetar_dias_tras_descongelar(pd, 240)
    # 2026-09-22 15:47 UTC = 11:47 en RD, un martes
    assert [d["day_name"] for d in pd["days"]] == ["Martes", "Miércoles"]
    assert [d["date"] for d in pd["days"]] == ["2026-09-22", "2026-09-23"]
    assert [d["day"] for d in pd["days"]] == [1, 2]


def test_sin_ancla_no_toca_nada():
    pd = {"days": [{"day": 1, "day_name": "Jueves"}]}
    cron_tasks._reetiquetar_dias_tras_descongelar(pd, 240)
    assert pd["days"][0]["day_name"] == "Jueves"


def test_el_descongelado_lo_aplica_con_el_mutador_atomico(monkeypatch):
    import db_plans
    aplicados = []

    def atomico(plan_id, mutator, lock_timeout_ms=None, *, user_id=None):
        pd = {"grocery_start_date": "2026-09-22", "days": [{"day": 1, "day_name": "Jueves"}]}
        mutator(pd)
        aplicados.append((plan_id, user_id, pd["days"][0]["day_name"]))
        return pd

    monkeypatch.setattr(db_plans, "update_plan_data_atomic", atomico)
    monkeypatch.setattr(cron_tasks, "_shift_plan_dates_for_freeze", lambda *a, **k: 0)
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda *a, **k: None)
    monkeypatch.setattr(cron_tasks, "_get_user_tz_minutes_optional", lambda uid: 240)
    try:
        cron_tasks._resume_frozen_plan("plan-1", "user-1", "2026-09-20T12:00:00+00:00")
    except Exception:
        pass   # lo que viene después (reanudar chunks, push) no es de este test
    assert aplicados and aplicados[0][:2] == ("plan-1", "user-1")
    assert aplicados[0][2] == "Martes"   # 2026-09-22 fue martes
