# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-653 · 2026-09-27] Un plan congelado no avanza, y al descongelar sus fechas siguen siendo ISO.

1. `/shift-plan` y el cron de refill no miraban `_frozen_at`: con la Nevera vacía el plan se congela (P1-PLAN-FREEZE),
   pero el shift seguía archivando días, y al descongelar `_shift_plan_dates_for_freeze` adelantaba además las anclas
   los días congelados: esos días se descontaban DOS veces. En producción dea00a2f se congeló el 17-sep a las 12:40 y
   los shifts de las 15:20 lo dejaron en 0 días. Ahora las dos ramas salen antes de tocar nada (el endpoint responde
   200 con `reason_code: plan_frozen`; el Dashboard ignora un shift sin éxito).
2. El descongelado escribía las anclas con `(ts + interval)::text` → «2026-09-22 15:47:33.496595+00», un formato que
   WebKit (la app de iOS) no garantiza parsear con `new Date()`. Ahora ISO con `T` y `+00:00`, y una fecha sola sigue
   siendo fecha sola (verificado con SELECT en Neon: «2026-09-22 15:47:33.496595+00» + 3 → «2026-09-25T15:47:33.496595+00:00»).

tooltip-anchor: P1-PLAN-LOTE-653
"""
import inspect


def _cuerpo(mod, nombre):
    return inspect.getsource(getattr(mod, nombre))


def test_el_endpoint_sale_antes_de_tocar_un_plan_congelado():
    from routers import plans
    src = _cuerpo(plans, "api_shift_plan")
    i_carga = src.index('plan_data = plan_record.get("plan_data", {})')
    i_congelado = src.index('plan_data.get("_frozen_at")')
    i_rebase = src.index("_p1cor_pre(cursor, plan_id, len(days))")
    assert i_carga < i_congelado < i_rebase
    assert '"reason_code": "plan_frozen"' in src


def test_el_cron_tampoco_avanza_un_plan_congelado():
    import cron_tasks
    src = _cuerpo(cron_tasks, "_background_shift_plan_for_user")
    i_carga = src.index('plan_data = plan_record.get("plan_data", {})')
    i_congelado = src.index('plan_data.get("_frozen_at")')
    assert i_carga < i_congelado < src.index('days = plan_data.get("days", [])', i_carga) + 400


def test_el_descongelado_escribe_iso_y_respeta_la_fecha_sola():
    import cron_tasks
    src = _cuerpo(cron_tasks, "_shift_plan_dates_for_freeze")
    assert "::timestamptz) + make_interval(days => %s))::text" not in src
    assert "to_char(" in src
    assert "'^[0-9]{4}-[0-9]{2}-[0-9]{2}$'" in src


def test_el_descongelado_manda_los_parametros_en_orden(monkeypatch):
    import cron_tasks
    escritos = []
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda q, p=None, **k: escritos.append((q, p)))
    cron_tasks._shift_plan_dates_for_freeze("plan-1", "user-1", 3)
    ancla = [(q, p) for q, p in escritos if "UPDATE meal_plans" in q]
    assert len(ancla) == 4
    q, p = ancla[0]
    assert q.count("%s") == len(p)
    assert """'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'""" in q          # el SQL que de verdad sale
    assert p[0] == "{_plan_start_date}" and p[-4:-2] == ("plan-1", "user-1")
