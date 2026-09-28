# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-659 · 2026-09-28] `chunk_lag_excessive` no cuenta como lag el tiempo que el plan estuvo congelado.

La alerta mide `effective_lag_seconds_at_pickup` (cuánto tardó el scheduler en recoger un chunk vencido). Un plan
CONGELADO (Nevera vacía, P1-PLAN-FREEZE) tiene sus chunks pausados a propósito: al descongelar, el chunk sale con todo
ese tiempo como «lag». dea00a2f sumó 41 825 s (697 min) que eran su congelado. Verificado con SELECT en Neon sobre 30
días: la consulta vieja devolvía esa fila; la nueva, ninguna.

tooltip-anchor: P1-PLAN-LOTE-659
"""
import inspect


def test_la_alerta_excluye_el_tiempo_congelado():
    import cron_tasks
    src = inspect.getsource(cron_tasks._alert_chunk_lag_excessive)
    assert "LEFT JOIN meal_plans mp_fz ON mp_fz.id = q.meal_plan_id" in src
    assert "(mp_fz.plan_data->>'_last_unfrozen_at')::timestamptz" in src
    assert "q.updated_at - make_interval(secs => q.effective_lag_seconds_at_pickup)" in src
