"""[P2-CHUNK-OVERDUE-SIGNAL] Predicado SSOT + payload de /chunk-status + cron + coach."""
import re
from datetime import date, timedelta
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat_history_context as chc


def _plan(days_dates, total):
    return {"total_days_requested": total,
            "days": [{"date": d, "meals": []} for d in days_dates]}


def test_overdue_cuando_hoy_supera_el_ultimo_dia_y_nada_corre():
    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15)
    overdue, since = chc.compute_chunk_overdue(plan, in_flight_count=0, today=date(2026, 8, 4))
    assert overdue is True and since == "2026-08-04"


def test_no_overdue_si_algo_esta_corriendo():
    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15)
    assert chc.compute_chunk_overdue(plan, in_flight_count=1, today=date(2026, 8, 4)) == (False, None)


def test_no_overdue_si_hoy_es_el_ultimo_dia_generado():
    plan = _plan(["2026-08-02", "2026-08-03", "2026-08-04"], 15)
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 4)) == (False, None)


def test_no_overdue_si_el_plan_ya_entrego_todos_los_dias():
    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 3)
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 9)) == (False, None)


def test_fail_open_plan_legacy_sin_dates():
    plan = {"total_days_requested": 15,
            "days": [{"day_name": "Lunes", "meals": []}, {"day_name": "Martes", "meals": []}]}
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 9)) == (False, None)


def test_fail_open_ante_basura():
    assert chc.compute_chunk_overdue({"days": "no-una-lista"}, 0, today=date(2026, 8, 9)) == (False, None)


# ---------------------------------------------------------------------------
# Payload de /chunk-status — mismo patrón (TestClient + dependency_overrides +
# patch("db_core.execute_sql_query", side_effect=...) por-query) que
# tests/test_p2_hist_new_3_chunk_days_range_payload.py y
# tests/test_p1_audit_hist_6_tier_breakdown.py (hermanos reales del endpoint
# que sí invocan `api_chunk_status` vía TestClient; test_p0_dash_chip_honesty
# es whitebox/source-inspection puro y no monta un fixture de request).
# ---------------------------------------------------------------------------

_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_CHUNK_A = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _fake_chunk_status_queries(plan_data, upcoming_rows, counters=None):
    """Dispatcher por-query: cada SELECT del endpoint tiene una firma
    textual distinta que permite enrutar sin tocar orden de invocación."""
    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"user_id": _USER_A, "plan_data": plan_data}
        if "id::text AS chunk_id" in query and "ORDER BY execute_after" in query:
            # [P2-CHUNK-OVERDUE-SIGNAL] query nueva de upcoming_chunks.
            # [Ronda 3 · FIX 2] El dispatcher enrutaba por la lista literal de
            # estados (`status IN ('pending', 'processing')`), así que añadir
            # 'stale' a la query de producción lo dejaba sin matchear y el test
            # media contra un `None`. Ahora enruta por la FORMA de la consulta
            # (proyección + orden), que es lo que la distingue de sus hermanas:
            # el ETA no proyecta chunk_id y las pausadas ordenan por semana.
            return upcoming_rows
        if "SELECT execute_after FROM plan_chunk_queue" in query:
            return None  # next_chunk_eta (fetch_one), sin next chunk aquí
        # [Ronda 4] El orden de estas dos ramas IMPORTA y estaba al revés: la
        # query de counters contiene `COUNT(*) FILTER (WHERE status =
        # 'pending_user_action')`, o sea la subcadena por la que se enrutaba
        # `paused_rows`. Resultado: los counters devolvían `[]` ⇒ `counters_row`
        # quedaba `{}` ⇒ TODOS los contadores del endpoint se leían como 0 en
        # cada test de este fichero, y un test sobre `pending_user_action_count`
        # no podía fallar aunque el endpoint lo ignorase. Se evalúa primero la
        # rama específica (agregado) y luego la de la lista.
        if "COUNT(*) FILTER" in query:
            return counters or {
                "in_flight_count": 0,
                "pending_user_action_count": 0,
                "failed_count": 0,
                "completed_count": 3,
            }
        if "status = 'pending_user_action'" in query:
            return []  # paused_rows
        if "FROM user_profiles" in query:
            return None
        if "quality_tier, COUNT" in query:
            return []  # tier_breakdown
        return None
    return _fake


def _plan_3_dias_viejo():
    return {
        "total_days_requested": 15,
        "days": [
            {"date": "2026-08-01", "meals": []},
            {"date": "2026-08-02", "meals": []},
            {"date": "2026-08-03", "meals": []},
        ],
    }


def _upcoming_rows_1_pending():
    return [{
        "chunk_id": _CHUNK_A,
        "week_number": 2,
        "days_offset": 3,
        "days_count": 4,
        "status": "pending",
        "execute_after": None,
    }]


def test_chunk_status_expone_upcoming_y_overdue(monkeypatch):
    """Plan de 3 días con dates viejas, total 15, cola con 1 chunk pending
    → upcoming_chunks con {chunk_id, week_number, days_offset, days_count,
    status, execute_after} → overdue True (in_flight_count del
    counters_row = 0)."""
    # rd_today() fijo para determinismo — el predicado usa `last_day + 1`
    # para `overdue_since`, no `today`, pero fijamos igual para que el
    # test no dependa del reloj de la máquina que lo corre.
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    plan_data = _plan_3_dias_viejo()
    upcoming_rows = _upcoming_rows_1_pending()

    client = _build_test_client()
    from auth import get_verified_user_id
    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch(
        "db_core.execute_sql_query",
        side_effect=_fake_chunk_status_queries(plan_data, upcoming_rows),
    ):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overdue"] is True
    assert body["overdue_since"] == "2026-08-04"
    assert body["upcoming_chunks"] == upcoming_rows


def test_chunk_status_knob_off_no_manda_los_campos(monkeypatch):
    monkeypatch.setenv("MEALFIT_UPCOMING_DAYS_UI", "false")
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    plan_data = _plan_3_dias_viejo()
    upcoming_rows = _upcoming_rows_1_pending()

    client = _build_test_client()
    from auth import get_verified_user_id
    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch(
        "db_core.execute_sql_query",
        side_effect=_fake_chunk_status_queries(plan_data, upcoming_rows),
    ):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert "upcoming_chunks" not in body
    assert "overdue" not in body
    assert "overdue_since" not in body


# ---------------------------------------------------------------------------
# Cron horario `_chunk_overdue_alert_job` (Task 2) — mismo patrón de
# monkeypatch por-función sobre `ct.execute_sql_query` / `ct.execute_sql_write`
# (side_effect posicional consumido con un iterator) que
# tests/test_p3_b_coherence_block_metrics_cron.py, el cron hermano más simple.
# ---------------------------------------------------------------------------

_CRON_PLAN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_CRON_USER_ID = "11111111-1111-1111-1111-111111111111"


def _cron_overdue_plan_row(plan_id=_CRON_PLAN_ID, user_id=_CRON_USER_ID):
    """Plan con 3 días de dates viejas y total=15: overdue si hoy > 2026-08-03
    y la cola in-flight está vacía."""
    return {
        "plan_id": plan_id,
        "user_id": user_id,
        "plan_data": {
            "total_days_requested": 15,
            "days": [
                {"date": "2026-08-01", "meals": []},
                {"date": "2026-08-02", "meals": []},
                {"date": "2026-08-03", "meals": []},
            ],
        },
    }


def _cron_complete_plan_row(plan_id=_CRON_PLAN_ID, user_id=_CRON_USER_ID):
    """Mismo plan pero ya completo (total_days_requested == len(days)) →
    compute_chunk_overdue debe devolver (False, None)."""
    days = [{"date": f"2026-08-0{i + 1}", "meals": []} for i in range(3)]
    return {"plan_id": plan_id, "user_id": user_id,
            "plan_data": {"total_days_requested": 3, "days": days}}


def _upserts(captured):
    return [c for c in captured if "INSERT INTO system_alerts" in c["sql"]]


def _sweeps(captured):
    """El UPDATE único de resolución (Ronda 3): resuelve TODA alerta
    `chunk_overdue` abierta que no esté en la lista de exclusión."""
    return [c for c in captured
            if "UPDATE system_alerts" in c["sql"] and "alert_type = 'chunk_overdue'" in c["sql"]]


def test_cron_overdue_upsertea_y_autoresuelve(monkeypatch):
    """Corrida 1: 1 plan overdue (dates viejas, total 15, cola vacía) → upsert
    con alert_key='chunk_overdue:<id>'. Corrida 2 con el plan ya completo
    (total == len(days)) → resolved_at (auto-resuelve, modelo Auto implicit:
    el job re-emite mientras la condición exista).

    [Ronda 3 · FIX 1] La resolución dejó de ser un UPDATE POR PLAN dentro del
    loop y pasó a ser un único UPDATE de barrido al final (ver
    `test_cron_resuelve_alertas_huerfanas_...`): con el acotamiento al plan
    vigente, un plan que deja de ser el vigente SALE de la población barrida y
    un `else` por-plan jamás volvería a verlo. Este test conserva sus dos
    aserciones originales (se emite / se resuelve la alerta de ESE plan), leídas
    ahora sobre la forma nueva."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    captured = []

    def _fake_write(sql, params=None, **kwargs):
        captured.append({"sql": sql, "params": params})
        return None

    monkeypatch.setattr(ct, "execute_sql_write", _fake_write)

    # --- Corrida 1: plan overdue, cola in-flight vacía (count=0) ---
    def _fake_query_overdue(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [_cron_overdue_plan_row()]
        if "FROM plan_chunk_queue" in sql:
            return {"c": 0}
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query_overdue)
    ct._chunk_overdue_alert_job()

    assert len(_upserts(captured)) == 1, captured
    upsert = _upserts(captured)[0]
    assert "ON CONFLICT" in upsert["sql"]
    assert upsert["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"
    # El barrido de resolución NO puede cerrar la alerta que acabamos de emitir.
    assert len(_sweeps(captured)) == 1, captured
    assert f"chunk_overdue:{_CRON_PLAN_ID}" in _sweeps(captured)[0]["params"][0]

    # --- Corrida 2: mismo plan ahora completo → resolve ---
    captured.clear()

    def _fake_query_complete(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [_cron_complete_plan_row()]
        if "FROM plan_chunk_queue" in sql:
            return {"c": 0}
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query_complete)
    ct._chunk_overdue_alert_job()

    assert _upserts(captured) == [], captured
    assert len(_sweeps(captured)) == 1, captured
    resolve = _sweeps(captured)[0]
    assert "resolved_at = NOW()" in resolve["sql"]
    # Lista de exclusión VACÍA ⇒ el barrido resuelve toda alerta abierta,
    # incluida la de este plan (que ya no cumple la condición).
    assert resolve["params"][0] == [], resolve


def test_cron_overdue_fail_open_por_plan(monkeypatch):
    """El primer plan lanza excepción en su COUNT in-flight; el segundo es
    overdue. El job NO debe abortar: el segundo plan recibe su alerta igual
    (fail-open POR PLAN, no fail-open global).

    [Ronda 3 · FIX 1] Y el plan que NO se pudo evaluar debe quedar EXCLUIDO del
    barrido de resolución: "no pude calcularlo" no es "ya no pasa". Cerrar su
    alerta por una excepción transitoria del COUNT sería inventar un veredicto
    que nadie midió."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    _plan_boom_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    plan_boom = _cron_overdue_plan_row(plan_id=_plan_boom_id)
    plan_ok = _cron_overdue_plan_row(plan_id=_CRON_PLAN_ID)

    captured = []

    def _fake_write(sql, params=None, **kwargs):
        captured.append({"sql": sql, "params": params})
        return None

    _count_results = iter([RuntimeError("boom: COUNT falló"), {"c": 0}])

    def _fake_query(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [plan_boom, plan_ok]
        if "FROM plan_chunk_queue" in sql:
            nxt = next(_count_results)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query)
    monkeypatch.setattr(ct, "execute_sql_write", _fake_write)

    ct._chunk_overdue_alert_job()  # no debe lanzar

    assert len(_upserts(captured)) == 1, captured
    assert _upserts(captured)[0]["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"

    assert len(_sweeps(captured)) == 1, captured
    excluidos = _sweeps(captured)[0]["params"][0]
    assert f"chunk_overdue:{_plan_boom_id}" in excluidos, excluidos
    assert f"chunk_overdue:{_CRON_PLAN_ID}" in excluidos, excluidos


def test_alert_key_documentada_en_la_tabla():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "chunk_overdue:" in doc


# ---------------------------------------------------------------------------
# Índice del coach (Task 3) — `build_pending_plan_days_lines` declara los
# días PENDIENTE/ATRASADO para que el chat no invente un menú que aún no
# existe. Mismo helper `_plan` de arriba, mismo predicado SSOT por debajo.
# ---------------------------------------------------------------------------


def test_coach_declara_dias_pendientes():
    plan = _plan(["2026-08-02", "2026-08-03", "2026-08-04"], 7)
    lines = chc.build_pending_plan_days_lines(plan, today=date(2026, 8, 4), in_flight_count=1)
    assert any("PENDIENTE" in l for l in lines) and not any("ATRASADO" in l for l in lines)


def test_coach_declara_atrasado_cuando_el_predicado_lo_dice():
    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15)
    lines = chc.build_pending_plan_days_lines(plan, today=date(2026, 8, 5), in_flight_count=0)
    assert any("ATRASADO" in l for l in lines)


def test_coach_legacy_sin_dates_no_declara_nada():
    plan = {"total_days_requested": 15, "days": [{"day_name": "Lunes"}]}
    assert chc.build_pending_plan_days_lines(plan, date(2026, 8, 9), 0) == []


def test_coach_cap_de_lineas():
    plan = _plan(["2026-08-02", "2026-08-03", "2026-08-04"], 30)
    lines = chc.build_pending_plan_days_lines(plan, date(2026, 8, 4), 1)
    assert len(lines) <= 4 and "más pendientes" in lines[-1]


# ---------------------------------------------------------------------------
# Ronda 1 de arreglos (revisor) — ALTO 1: `days` es ventana rolling; contar/
# numerar solo contra `days` (sin sumar `_archived_days`) sobrecuenta
# pendientes y desnumera en cualquier plan que ya rotó. Caso EXACTO medido
# por el revisor: 10 archivados + 5 vivos + total 20 ⇒ 5 pendientes reales,
# numerados desde el día 16 (no 15 "pendientes" numerados desde el 4).
# ---------------------------------------------------------------------------


def test_archivados_cuentan_para_el_pendiente_y_la_numeracion():
    archived_dates = [
        "2026-07-24", "2026-07-25", "2026-07-26", "2026-07-27", "2026-07-28",
        "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02",
    ]  # 10 archivados
    live_dates = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]  # 5 vivos
    plan = {
        "total_days_requested": 20,
        "_archived_days": [{"date": d, "meals": []} for d in archived_dates],
        "days": [{"date": d, "meals": []} for d in live_dates],
    }
    # in_flight_count=1 fuerza overdue=False: aísla el fix de conteo/numeración
    # del cálculo de ATRASADO, que ya cubren los otros tests de esta sección.
    lines = chc.build_pending_plan_days_lines(plan, today=date(2026, 8, 7), in_flight_count=1)

    assert len(lines) == 4, lines  # cap 3 días + 1 resumen (5 pendientes > cap 3)
    assert "día 16" in lines[0], lines
    assert "día 17" in lines[1], lines
    assert "día 18" in lines[2], lines
    assert "y 2 día(s) más pendientes" in lines[3], lines
    # Regresión explícita contra la numeración VIEJA (buggy): `len(days)+k` con
    # k=1..3 daría "día 6"/"día 7"/"día 8" — NUNCA deben aparecer.
    assert not any("día 6" in l or "día 7" in l or "día 8" in l for l in lines[:3])


# ---------------------------------------------------------------------------
# Ronda 1 de arreglos (revisor) — MEDIO 3: el test de legacy-sin-dates pasaba
# por ACCIDENTE — quitar el guard `if last is None: return []` no lo rompía
# porque el try/except externo atrapa el TypeError de `last + timedelta(...)`
# igual. Este test ancla el comportamiento REAL del guard: para un plan
# legacy, `compute_chunk_overdue` NUNCA llega a invocarse.
# ---------------------------------------------------------------------------


def test_coach_legacy_sin_dates_no_llega_a_computar_overdue():
    plan = {"total_days_requested": 15, "days": [{"day_name": "Lunes"}]}
    with patch("chat_history_context.compute_chunk_overdue") as mock_overdue:
        result = chc.build_pending_plan_days_lines(plan, date(2026, 8, 9), 0)
    assert result == []
    mock_overdue.assert_not_called()


# ---------------------------------------------------------------------------
# Ronda 1 de arreglos (revisor) — ALTO 2: `agent._build_pending_days_lines_block`
# debe filtrar el COUNT de `plan_chunk_queue` por `meal_plan_id`, NUNCA por
# `user_id` — un COUNT por user_id sumaría chunks de OTRO plan del mismo
# usuario (p.ej. el plan viejo cancelándose en segundo plano tras /restore) y
# escondería un ATRASADO real detrás de un in_flight_count ajeno. Sin
# `plan_id` en el callsite: saltar el COUNT por completo (nunca contar por
# user_id como aproximación) y no declarar ATRASADO por falta de certeza.
# ---------------------------------------------------------------------------


def _plan_atrasado_con_pendientes():
    """total=15, 3 días vivos con dates viejas (última 2026-08-03) — con
    today=2026-08-10 y sin nada in-flight para ESE plan, el día 4 (04/08)
    está ATRASADO."""
    return _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15)


def test_agent_count_filtra_por_meal_plan_id_no_por_user_id(monkeypatch):
    import agent

    plan = _plan_atrasado_con_pendientes()
    calls = []

    def _fake_execute(query, params=None, **kwargs):
        calls.append({"query": query, "params": params})
        return {"c": 0}  # cola vacía PARA ESTE PLAN → el día 4 sale ATRASADO

    monkeypatch.setattr("db.execute_sql_query", _fake_execute)

    out = agent._build_pending_days_lines_block(
        "user-1", plan, date(2026, 8, 10), plan_id="plan-abc-123",
    )

    assert len(calls) == 1, calls
    assert "meal_plan_id = %s" in calls[0]["query"], calls
    assert "WHERE user_id" not in calls[0]["query"], calls
    assert calls[0]["params"] == ("plan-abc-123",)
    assert "ATRASADO" in out, out


def test_agent_sin_plan_id_salta_el_count_y_no_declara_atrasado(monkeypatch):
    import agent

    plan = _plan_atrasado_con_pendientes()

    def _boom(*a, **kw):
        raise AssertionError("no debería tocar la DB sin plan_id")

    monkeypatch.setattr("db.execute_sql_query", _boom)

    out = agent._build_pending_days_lines_block(
        "user-1", plan, date(2026, 8, 10), plan_id=None,
    )

    assert "ATRASADO" not in out, out
    assert "PENDIENTE" in out, out


# ---------------------------------------------------------------------------
# Ronda 2 de arreglos (revisor) — el MISMO bug de Ronda 1/ALTO 1, pero en
# `compute_chunk_overdue` mismo (la Ronda 1 solo lo arregló en
# `build_pending_plan_days_lines`, que LLAMA al predicado pero no cambia su
# guard interno). `days` es ventana rolling: el guard `total <= len(days)`
# necesita `len(_archived_days) + len(days)`, igual que `resolve_day_dates`
# (líneas 121-165) y que `build_pending_plan_days_lines` ya aplica. Reproducido
# por el revisor EJECUTANDO la función: plan de 20 días YA COMPLETO (15
# archivados + 5 vivos) daba (True, '2026-08-13') en vez de (False, None) —
# todo plan terminado cuyo usuario pasó el último día queda ATRASADO PARA
# SIEMPRE (el cron horario emitiría `chunk_overdue:<plan_id>` indefinidamente
# sobre un plan ya completo).
# ---------------------------------------------------------------------------


def test_archivados_cuentan_plan_completo_no_es_overdue():
    """Caso EXACTO reproducido por el revisor: 15 archivados + 5 vivos,
    total=20 (plan ya completo: 15+5=20), hoy 4 días después del último día
    vivo. Pre-fix: (True, '2026-08-13') — falso positivo permanente.
    Post-fix: (False, None)."""
    archived_dates = [
        "2026-07-24", "2026-07-25", "2026-07-26", "2026-07-27", "2026-07-28",
        "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02",
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07",
    ]  # 15 archivados
    live_dates = ["2026-08-08", "2026-08-09", "2026-08-10", "2026-08-11", "2026-08-12"]  # 5 vivos
    plan = {
        "total_days_requested": 20,
        "_archived_days": [{"date": d, "meals": []} for d in archived_dates],
        "days": [{"date": d, "meals": []} for d in live_dates],
    }
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 16)) == (False, None)


def test_archivados_no_esconden_un_overdue_legitimo():
    """El fix NO debe matar la detección legítima: 10 archivados + 3 vivos,
    total=20 (13 generados, plan AÚN NO completo — 7 días pendientes reales),
    hoy 3 días después del último día vivo, sin nada in-flight ⇒ sigue
    ATRASADO. Nota de honestidad (mismo patrón que el comentario MEDIO-3 de
    arriba): con estos números concretos el guard `total <= n_generated`
    (13) sigue siendo False tanto pre-fix (`total <= len(days)`=3) como
    post-fix — total=20 es mucho mayor que ambos denominadores, así que este
    test PASA contra el código pre-fix también. Su valor es como regression
    guard de que el fix (sumar archivados) no sobre-actúa y apaga la
    detección legítima de ATRASADO cuando el plan de verdad no ha terminado."""
    archived_dates = [
        "2026-07-24", "2026-07-25", "2026-07-26", "2026-07-27", "2026-07-28",
        "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02",
    ]  # 10 archivados
    live_dates = ["2026-08-03", "2026-08-04", "2026-08-05"]  # 3 vivos
    plan = {
        "total_days_requested": 20,
        "_archived_days": [{"date": d, "meals": []} for d in archived_dates],
        "days": [{"date": d, "meals": []} for d in live_dates],
    }
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 8)) == (True, "2026-08-06")


def test_archivados_ausente_o_invalido_no_rompe_ni_cambia_comportamiento():
    """`_archived_days` ausente / no-lista / con basura ⇒ tratado como vacío
    (mismo patrón `[d for d in (... or []) if isinstance(d, dict)]` que
    `resolve_day_dates`/`build_pending_plan_days_lines`), sin crashear."""
    base_days = ["2026-08-01", "2026-08-02", "2026-08-03"]

    # Ausente: comportamiento idéntico a los 6 tests originales del predicado
    # (ninguno tenía `_archived_days` — deben seguir intactos).
    plan_ausente = _plan(base_days, 15)
    assert chc.compute_chunk_overdue(plan_ausente, 0, today=date(2026, 8, 4)) == (True, "2026-08-04")

    # No-lista (string): filtrado a vacío, no crashea.
    plan_string = {**_plan(base_days, 15), "_archived_days": "corrupted"}
    assert chc.compute_chunk_overdue(plan_string, 0, today=date(2026, 8, 4)) == (True, "2026-08-04")

    # No-lista (dict): filtrado a vacío, no crashea.
    plan_dict = {**_plan(base_days, 15), "_archived_days": {"oops": "no es lista"}}
    assert chc.compute_chunk_overdue(plan_dict, 0, today=date(2026, 8, 4)) == (True, "2026-08-04")

    # Lista con basura: solo cuentan los elementos dict válidos (2 de 5) —
    # total=8 hace que el conteo SIN filtrar (5 archivados + 3 vivos = 8)
    # dispararía el guard "ya completo" incorrectamente (8<=8 → False,None);
    # filtrando correctamente (2 archivados válidos + 3 vivos = 5) el plan
    # sigue incompleto (8 > 5) y el overdue real se preserva.
    plan_basura = {
        **_plan(base_days, 8),
        "_archived_days": ["nope", 123, None, {"date": "2026-07-31", "meals": []},
                            {"date": "2026-07-30", "meals": []}],
    }
    assert chc.compute_chunk_overdue(plan_basura, 0, today=date(2026, 8, 4)) == (True, "2026-08-04")


# ---------------------------------------------------------------------------
# Ronda 3 · FIX 1 — el cron alertaba sobre la POBLACIÓN equivocada.
#
# Medido contra Neon producción con el predicado real (2026-08-04): 19 de 23
# planes en estado activo alertarían el día 1 del despliegue, y NINGUNO es un
# fallo. Cadena causal: al insertar un plan nuevo, `db_plans.py:1526` cancela
# TODOS los chunks vivos del usuario y el cron GAP-11 purga las filas
# `cancelled` a las 48h. El plan SUPERADO se queda para siempre con
# `generation_status='complete_partial'`, N días de M y CERO filas en la cola
# ⇒ `in_flight_count == 0` ⇒ el predicado lo declara ATRASADO. Regenerar el
# plan es una acción NORMAL: el ruido es el caso común, no un borde. De los 23
# planes activos, 20 ni siquiera eran el plan vigente de su usuario.
#
# La UI y el coach nunca sufrieron esto porque ambos operan sobre el plan
# vigente (`get_latest_meal_plan_with_id`, ORDER BY created_at DESC LIMIT 1).
# Solo el cron divergía en población.
# ---------------------------------------------------------------------------


def _capturar_sql_de_planes(monkeypatch, rows=None):
    """Corre el cron y devuelve el SQL REAL emitido contra `meal_plans`.

    Se recoge lo EJECUTADO, no el texto del fuente: un test que lee el archivo
    seguiría verde si el job dejara de usar esa query."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))
    monkeypatch.setattr(ct, "execute_sql_write", lambda *a, **k: None)

    visto = {}

    def _fake_query(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            visto["sql"] = sql
            return rows if rows is not None else []
        if "FROM plan_chunk_queue" in sql:
            return {"c": 0}
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query)
    ct._chunk_overdue_alert_job()
    assert "sql" in visto, "el cron no consultó meal_plans"
    return visto["sql"]


def _span_de_la_subconsulta_distinct_on(sql):
    """(interior, exterior) del paréntesis que envuelve al `DISTINCT ON`.

    Escanea hacia atrás desde `DISTINCT ON` buscando el `(` sin cerrar que lo
    contiene, y hacia adelante balanceando paréntesis hasta su `)`. Si el
    `DISTINCT ON` NO está dentro de ninguna subconsulta (la versión ingenua:
    un solo nivel con el filtro de estado en su WHERE) el helper falla — que es
    exactamente el caso que hay que rechazar."""
    i = sql.upper().find("DISTINCT ON")
    assert i != -1, f"sin DISTINCT ON:\n{sql}"
    depth, open_idx = 0, None
    for j in range(i - 1, -1, -1):
        if sql[j] == ")":
            depth += 1
        elif sql[j] == "(":
            if depth == 0:
                open_idx = j
                break
            depth -= 1
    assert open_idx is not None, (
        "El DISTINCT ON no está dentro de una subconsulta: el filtro de estado "
        f"solo puede ir en la consulta EXTERNA.\n{sql}")
    depth, close_idx = 0, None
    for j in range(open_idx, len(sql)):
        if sql[j] == "(":
            depth += 1
        elif sql[j] == ")":
            depth -= 1
            if depth == 0:
                close_idx = j
                break
    assert close_idx is not None, f"subconsulta sin cerrar:\n{sql}"
    return sql[open_idx:close_idx + 1], sql[close_idx + 1:]


def test_cron_elige_el_plan_vigente_de_cada_usuario(monkeypatch):
    """El barrido usa el MISMO criterio que `get_latest_meal_plan_with_id`
    (el plan que ven la UI y el coach): el más reciente por `created_at` de
    cada usuario."""
    sql = _capturar_sql_de_planes(monkeypatch)
    normal = " ".join(sql.split())
    assert "DISTINCT ON (user_id)" in normal, normal
    assert "ORDER BY user_id, created_at DESC" in normal, normal


def test_cron_filtra_el_estado_DESPUES_de_elegir_el_vigente(monkeypatch):
    """LA TRAMPA. `DISTINCT ON` debe elegir el plan más reciente del usuario
    entre TODOS sus planes, y SOLO DESPUÉS filtrar por `generation_status`.

    Caso concreto que esto cierra (y que es el bug medido): usuario con un
    `complete_partial` VIEJO (plan superado, cola purgada) y un `complete`
    NUEVO (su plan vigente, ya entregado). Filtrando por estado PRIMERO, el
    `DISTINCT ON` solo ve el viejo — es el único candidato — y lo devuelve
    como si fuera el plan del usuario ⇒ alerta ATRASADO sobre un plan que
    nadie está esperando. Filtrando DESPUÉS, el `DISTINCT ON` devuelve el
    `complete`, el WHERE externo lo descarta por estado, y el usuario
    desaparece del barrido: cero alertas, que es lo correcto.

    Verificación semántica (no cabe en un test unitario, queda anclada aquí):
    la query se ejecutó contra un Postgres real con las dos filas de este caso
    inyectadas vía CTE que sombrea `meal_plans` — 0 filas devueltas con el
    orden correcto, 1 fila (el `complete_partial` viejo) con el orden
    invertido."""
    sql = _capturar_sql_de_planes(monkeypatch)
    interior, exterior = _span_de_la_subconsulta_distinct_on(sql)

    # `complete_partial` solo aparece en la lista de estados activos: es el
    # marcador inequívoco de DÓNDE está aplicado el filtro.
    assert "'complete_partial'" not in interior, (
        "El filtro de generation_status está DENTRO del DISTINCT ON: un plan "
        f"superado ganaría cuando el vigente está `complete`.\n{interior}")
    assert "'complete_partial'" in exterior, exterior
    for estado in ("'generating'", "'generating_next'", "'partial'"):
        assert estado in exterior, (estado, exterior)


# ---------------------------------------------------------------------------
# Ronda 3 · FIX 1 (consecuencia) — alertas HUÉRFANAS.
#
# Con el acotamiento al plan vigente, un plan que YA tenía alerta abierta y
# luego deja de ser el vigente (el usuario regenera) SALE de la población
# barrida. Una resolución por-plan dentro del loop solo corre para lo que el
# barrido ve ⇒ su alerta quedaría abierta PARA SIEMPRE, y el modelo declarado
# en docs/system_alerts_resolution_table.md es Auto (implicit), que exige
# auto-resolución. Por eso la resolución es un único UPDATE de barrido al
# final, con lista de exclusión.
# ---------------------------------------------------------------------------


def test_cron_resuelve_alertas_huerfanas_de_planes_fuera_de_la_poblacion(monkeypatch):
    """Un solo plan overdue en la población ⇒ el barrido final resuelve TODA
    alerta `chunk_overdue` abierta salvo la suya. La alerta de un plan que ya
    no se barre (regeneración) cae ahí dentro: es lo único que puede cerrarla,
    porque ese plan_id ya no aparece en ninguna corrida."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    captured = []
    monkeypatch.setattr(ct, "execute_sql_write",
                        lambda sql, params=None, **k: captured.append({"sql": sql, "params": params}))

    def _fake_query(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [_cron_overdue_plan_row()]
        if "FROM plan_chunk_queue" in sql:
            return {"c": 0}
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query)
    ct._chunk_overdue_alert_job()

    sweeps = _sweeps(captured)
    assert len(sweeps) == 1, captured
    sql = " ".join(sweeps[0]["sql"].split())
    assert "resolved_at = NOW()" in sql, sql
    assert "resolved_at IS NULL" in sql, sql
    # La exclusión tiene que ser por LISTA de keys vivas; un UPDATE por
    # alert_key concreto no puede cerrar lo que ya no barre.
    assert "alert_key = ANY(" in sql, sql
    assert "NOT" in sql, sql
    assert sweeps[0]["params"][0] == [f"chunk_overdue:{_CRON_PLAN_ID}"], sweeps[0]


def test_cron_no_resuelve_nada_si_el_select_de_planes_falla(monkeypatch):
    """Si la consulta de planes revienta, el job sale SIN barrer. Resolver con
    la lista de exclusión vacía tras un SELECT fallido cerraría todas las
    alertas legítimas por un fallo transitorio de la DB.

    Nota de honestidad: este test PASA también contra el código pre-fix (donde
    no había barrido que pudiera dispararse de más). Su valor es como guard del
    barrido nuevo: mueve el `return` del except antes del sweep y se pone
    rojo — comprobado."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    captured = []
    monkeypatch.setattr(ct, "execute_sql_write",
                        lambda sql, params=None, **k: captured.append({"sql": sql, "params": params}))

    def _boom(sql, params=None, **kwargs):
        raise RuntimeError("boom: SELECT de meal_plans falló")

    monkeypatch.setattr(ct, "execute_sql_query", _boom)
    ct._chunk_overdue_alert_job()  # no debe lanzar

    assert captured == [], captured


def test_cron_documenta_el_acotamiento_con_los_numeros_medidos():
    """El docstring del job debe explicar POR QUÉ mira solo el plan vigente,
    con los números medidos. Sin eso, el próximo lector "arregla" el
    acotamiento creyendo que el cron mira demasiado poco."""
    import cron_tasks as ct
    doc = ct._chunk_overdue_alert_job.__doc__ or ""
    assert "vigente" in doc.lower(), doc
    assert "19" in doc and "23" in doc, doc
    assert "get_latest_meal_plan_with_id" in doc, doc


# ---------------------------------------------------------------------------
# Ronda 3 · FIX 2 — un chunk `stale` quedaba INVISIBLE.
#
# `/chunk-status` construía `upcoming_chunks` con `status IN
# ('pending','processing')`, pero el `in_flight_count` que alimenta el
# predicado cuenta `('pending','processing','stale')`. Y `stale` NO es
# terminal: `db_plans.py:1511` lo documenta ("el worker los re-pickea al
# refrescar pantry") y la consulta con la que el worker reclama trabajo lo
# incluye (`cron_tasks.py:8225`, `WHERE q.status IN ('pending','stale')`).
#
# Consecuencia: un chunk `stale` suprimía el aviso de ATRASADO (correcto, va a
# ejecutarse) pero NO pintaba pestaña fantasma ⇒ volvían a existir días
# encolados que el usuario no ve, que es literalmente el bug que esta feature
# existe para cerrar.
# ---------------------------------------------------------------------------


def _capturar_queries_de_chunk_status(monkeypatch, plan_data, upcoming_rows,
                                      counters=None, hoy=date(2026, 8, 10)):
    """Corre el endpoint y devuelve (queries emitidas, body)."""
    monkeypatch.setattr(chc, "rd_today", lambda: hoy)
    queries = []
    base = _fake_chunk_status_queries(plan_data, upcoming_rows, counters)

    def _fake(query, params=None, **kwargs):
        queries.append(query)
        return base(query, params, **kwargs)

    client = _build_test_client()
    from auth import get_verified_user_id
    client.app.dependency_overrides[get_verified_user_id] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/chunk-status")
    assert r.status_code == 200, r.text
    return queries, r.json()


_ESTADOS_RE = re.compile(r"status\s+IN\s*\(([^)]*)\)", re.IGNORECASE)


def _conjunto_de_estados(sql):
    m = _ESTADOS_RE.search(sql)
    assert m, sql
    return {s.strip().strip("'\"") for s in m.group(1).split(",")}


def test_chunk_status_upcoming_y_in_flight_usan_el_mismo_conjunto_de_estados(monkeypatch):
    """LA INVARIANTE. El conjunto de estados con el que se LISTAN los chunks
    futuros y el conjunto con el que se CUENTA el in-flight que suprime el
    aviso de atrasado tienen que ser el MISMO. Cualquier estado que esté solo
    en el contador es un día encolado que el usuario no ve y del que tampoco
    se le avisa — invisible por partida doble."""
    queries, _ = _capturar_queries_de_chunk_status(
        monkeypatch, _plan_3_dias_viejo(), _upcoming_rows_1_pending())

    upcoming_sql = [q for q in queries
                    if "id::text AS chunk_id" in q and "ORDER BY execute_after" in q]
    counters_sql = [q for q in queries if "AS in_flight_count" in q]
    assert len(upcoming_sql) == 1, queries
    assert len(counters_sql) == 1, queries

    estados_listados = _conjunto_de_estados(upcoming_sql[0])
    estados_contados = _conjunto_de_estados(counters_sql[0])
    assert estados_listados == estados_contados, (estados_listados, estados_contados)
    assert "stale" in estados_listados, estados_listados


def test_chunk_status_expone_un_chunk_stale(monkeypatch):
    """Un chunk `stale` (única fila de la cola) llega al payload. Es el caso
    real: el worker lo re-pickea al refrescar pantry, así que el día existe y
    hay que pintarlo.

    Nota de honestidad: bajo mocks la query no filtra de verdad, así que este
    test PASA también pre-fix. Los dientes del FIX 2 están en el test de
    arriba (los dos conjuntos de estados deben coincidir, y ese sí era rojo);
    este ancla que ningún post-filtro del endpoint descarte un `stale` camino
    del payload."""
    stale_rows = [{
        "chunk_id": _CHUNK_A,
        "week_number": 2,
        "days_offset": 3,
        "days_count": 4,
        "status": "stale",
        "execute_after": None,
    }]
    _, body = _capturar_queries_de_chunk_status(
        monkeypatch, _plan_3_dias_viejo(), stale_rows)
    assert body["upcoming_chunks"] == stale_rows


# ===========================================================================
# Ronda 4 · review de rama completa (B1-B5)
# ===========================================================================
#
# B1 y B3 se cierran JUNTOS sustituyendo el término "¿el plan aún debe días?"
# —que era un CONTEO de `_archived_days + days`— por una VENTANA DE FECHAS: el
# ciclo vigente pretende cubrir `total_days_requested` días desde su inicio, y
# solo hay atraso si HOY cae dentro de esa ventana.
#
# Por qué el conteo no podía sobrevivir: `_archived_days` NUNCA se vacía, ni
# siquiera cuando el plan RENUEVA (P0-1 RENEWAL, routers/plans.py) — mismo
# plan_id, `total_days_requested` intacto, `days=[]` y offsets desde 0. Dos
# ciclos comparten el mismo array de archivados, así que `len(archived)+len(days)`
# alcanza `total` y apaga el predicado PARA SIEMPRE.
#
# Y por qué la ventana necesita un ancla EXPLÍCITA (`_cycle_started_at`): medido
# contra los 24 planes de producción (scratchpad/forense_ciclo.py),
#   · `grocery_start_date == days[0].date` en 23/23 planes con dato ⇒ sigue a la
#     VENTANA ROLLING, no al ciclo: el shift lo reescribe a HOY en cada rotación.
#   · `cycle_start_date == created_at` en 19/23 (los otros 4 difieren 1 día por
#     TZ) ⇒ es el ancla INMUTABLE de creación y la renovación no la toca.
# Ninguno de los dos marca el inicio del ciclo VIGENTE. Por eso la renovación
# estampa el ancla, y los planes legacy degradan a la primera fecha entregada
# (que para un plan que nunca renovó ES el inicio del ciclo).
# ===========================================================================


def _d(iso):
    return {"date": iso, "meals": []}


def _plan_renovado(con_ancla=True):
    """Fixture EXACTA del revisor (scratchpad/probe_renewal.py), reachable por
    construcción: plan de 30 días creado el 2026-06-01 y consumido entero; el
    2026-07-01 `api_shift_plan` ve `days_since_creation >= total` ⇒ archiva los
    30 días, `days=[]`, `grocery_start_date=hoy` y encola el ciclo 2. El chunk 1
    genera 3 días (07-01..07-03) y ahí se queda: 27 días sin generar."""
    ciclo1 = [(date(2026, 6, 1) + timedelta(days=i)).isoformat() for i in range(30)]
    plan = {
        "total_days_requested": 30,
        "_archived_days": [_d(x) for x in ciclo1],
        "days": [_d(x) for x in ["2026-07-01", "2026-07-02", "2026-07-03"]],
    }
    if con_ancla:
        plan["_cycle_started_at"] = "2026-07-01"
    return plan


def test_b1_plan_renovado_declara_atrasado():
    """LA REGRESIÓN QUE B1 CIERRA. Con el conteo, este plan devolvía
    `(False, None)`: 30 archivados + 3 vivos >= 30 ⇒ "ya entregó todo", cuando
    en realidad debe 27 días y la cola está vacía. Las tres superficies quedaban
    MUDAS justo en el bug que la feature existe para cerrar."""
    plan = _plan_renovado()
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 7, 20)) == (True, "2026-07-04")


def test_b1_plan_renovado_el_indice_del_coach_tambien_lo_declara():
    """El coach llevaba su PROPIA copia del guard de conteo
    (`agent._build_pending_days_lines_block`), así que arreglar solo el
    predicado lo habría dejado mudo igual. Ambos leen ahora el mismo helper."""
    lines = chc.build_pending_plan_days_lines(_plan_renovado(), date(2026, 7, 20), 0)
    assert lines, "el índice del coach no declaró nada"
    assert any("ATRASADO" in l for l in lines), lines
    # Numeración por CICLO: el día que falta es el 4 del ciclo vigente, no el 34.
    assert "día 4" in lines[0], lines


def test_b1_la_renovacion_estampa_el_ancla_del_ciclo():
    """La rama P0-1 RENEWAL de `api_shift_plan` debe persistir `_cycle_started_at`
    junto al `grocery_start_date` que ya reescribe. Sin esa marca no existe forma
    de distinguir el ciclo 2 del ciclo 1: ambos comparten `_archived_days`,
    `total_days_requested` y una línea temporal de fechas CONTIGUA (la renovación
    empieza el día siguiente al último archivado), así que el ciclo no es
    recuperable de `plan_data` por fechas.
    tooltip-anchor: P2-CHUNK-OVERDUE-SIGNAL-CYCLE-ANCHOR"""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.find("[P0-1 RENEWAL] Plan semanal")
    assert i != -1, "no se encontró la rama P0-1 RENEWAL"
    bloque = src[max(0, i - 2500):i]
    assert "_cycle_started_at" in bloque, (
        "la renovación no estampa `_cycle_started_at`:\n" + bloque[-900:])


def test_b1_sin_ancla_un_plan_legacy_degrada_a_la_primera_fecha_entregada():
    """Planes anteriores al ancla (los 24 de producción hoy) usan la primera
    fecha entregada. Para un plan que NUNCA renovó eso ES el inicio del ciclo,
    así que el comportamiento no cambia: sigue detectando el atraso real."""
    plan = {
        "total_days_requested": 20,
        "_archived_days": [_d(x) for x in ["2026-07-24", "2026-07-25", "2026-07-26"]],
        "days": [_d(x) for x in ["2026-07-27", "2026-07-28"]],
    }
    # ciclo 07-24..08-12; hoy 07-30 está dentro y el último día vivo es 07-28.
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 7, 30)) == (True, "2026-07-29")


def test_b1_sin_ancla_un_plan_renovado_legacy_queda_mudo_a_sabiendas():
    """Honestidad: sin `_cycle_started_at` la renovación NO es recuperable, y el
    plan legacy renovado sigue mudo (el comportamiento de hoy, no una regresión).
    Se ancla para que nadie lo lea como que el fallback cubre la renovación."""
    assert chc.compute_chunk_overdue(
        _plan_renovado(con_ancla=False), 0, today=date(2026, 7, 20)) == (False, None)


# --- B3: caducidad de la ventana -------------------------------------------


def _plan_15d_abandonado():
    """Forma de `76a6836d` en producción (2026-08-04): plan de 15 días con
    2 archivados + 1 vivo, ciclo iniciado el 2026-08-02."""
    return {
        "total_days_requested": 15,
        "_archived_days": [_d("2026-08-02"), _d("2026-08-03")],
        "days": [_d("2026-08-04")],
    }


def test_b3_fuera_de_la_ventana_del_ciclo_no_hay_atraso():
    """B3 medido por el revisor: `76a6836d` devolvía `(True, …)` también a +30
    días, cuando ya no quedan días de plan y `api_shift_plan` no puede encolar
    nada — una alerta SIN camino de resolución, abierta para siempre. El ciclo
    (15 días desde el 2026-08-02) termina el 2026-08-17: pasado eso el plan
    terminó, no está atrasado."""
    plan = _plan_15d_abandonado()
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 9, 3)) == (False, None)
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 17)) == (False, None)


def test_b3_dentro_de_la_ventana_sigue_alertando():
    """La caducidad NO puede apagar la detección legítima: el mismo plan, dentro
    de su ventana, sigue declarando el atraso."""
    plan = _plan_15d_abandonado()
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 9)) == (True, "2026-08-05")
    assert chc.compute_chunk_overdue(plan, 0, today=date(2026, 8, 16)) == (True, "2026-08-05")


# --- B2: un plan pausado esperando al usuario NO está atrasado --------------
#
# Los chunks en `pending_user_action` no están en ('pending','processing','stale')
# ⇒ `in_flight_count = 0` ⇒ el predicado decía ATRASADO, y la pestaña prometía
# "el sistema lo reintenta solo la próxima vez que abras la app", que es FALSO:
# ese chunk espera una acción del USUARIO (consentimiento de nevera). Y el cron
# le mandaba una alerta al operador por algo que no es un fallo del sistema.
#
# El criterio del predicado es "¿hay algo que lo vaya a resolver?" — y la acción
# del usuario lo resuelve. Este conjunto queda DELIBERADAMENTE distinto del de
# `upcoming_chunks` (que NO incluye pending_user_action: esos días ya se pintan
# vía `paused_chunks`, y el frontend distingue 'pausado' de 'en proceso' con
# `puac > 0 && in_flight === 0`).


def test_b2_pausado_por_el_usuario_no_es_atrasado_en_chunk_status(monkeypatch):
    counters = {"in_flight_count": 0, "pending_user_action_count": 1,
                "failed_count": 0, "completed_count": 3}
    _, body = _capturar_queries_de_chunk_status(
        monkeypatch, _plan_3_dias_viejo(), [], counters=counters, hoy=date(2026, 8, 4))
    assert body["overdue"] is False, body
    assert body["overdue_since"] is None, body
    # Los contadores del payload NO cambian: el frontend los usa para distinguir
    # 'pausado' de 'en proceso'.
    assert body["in_flight_count"] == 0 and body["pending_user_action_count"] == 1


def test_b2_upcoming_chunks_NO_incluye_pending_user_action(monkeypatch):
    """Divergencia deliberada. El instinto de "alinear los dos filtros" es justo
    lo que hicimos en la ronda anterior por el motivo CONTRARIO; aquí el conjunto
    del predicado y el de la lista pintable dejan de coincidir a propósito."""
    queries, _ = _capturar_queries_de_chunk_status(
        monkeypatch, _plan_3_dias_viejo(), _upcoming_rows_1_pending())
    upcoming_sql = [q for q in queries
                    if "id::text AS chunk_id" in q and "ORDER BY execute_after" in q]
    assert len(upcoming_sql) == 1, queries
    assert "pending_user_action" not in upcoming_sql[0], upcoming_sql[0]


def test_b2_el_cron_cuenta_pending_user_action(monkeypatch):
    """El COUNT del cron incluye `pending_user_action` ⇒ el plan pausado no
    genera alerta de operador."""
    import cron_tasks as ct
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    captured = []
    monkeypatch.setattr(ct, "execute_sql_write",
                        lambda sql, params=None, **k: captured.append({"sql": sql, "params": params}))
    counts = []

    def _fake_query(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            return [_cron_overdue_plan_row()]
        if "FROM plan_chunk_queue" in sql:
            counts.append(sql)
            # 1 chunk pausado esperando al usuario, nada más en la cola.
            return {"c": 1 if "pending_user_action" in sql else 0}
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query)
    ct._chunk_overdue_alert_job()

    assert counts and "pending_user_action" in counts[0], counts
    assert _upserts(captured) == [], captured


def test_b2_el_coach_cuenta_pending_user_action(monkeypatch):
    import agent

    plan = _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15)
    vistas = []

    def _fake_execute(query, params=None, **kwargs):
        vistas.append(query)
        return {"c": 1 if "pending_user_action" in query else 0}

    monkeypatch.setattr("db.execute_sql_query", _fake_execute)
    out = agent._build_pending_days_lines_block(
        "user-1", plan, date(2026, 8, 10), plan_id="plan-abc-123")

    assert vistas and "pending_user_action" in vistas[0], vistas
    assert "ATRASADO" not in out, out
    assert "PENDIENTE" in out, out


# --- B4: el knob tiene que apagar las TRES superficies ----------------------


def test_b4_knob_off_apaga_el_cron_y_resuelve_lo_abierto(monkeypatch):
    """Con el knob apagado el cron no emite NADA. Y resuelve lo que quedó
    abierto: el rollback existe justamente para cortar una inundación de
    alertas (19 de 23 planes), así que dejarlas abiertas al apagar la señal
    frustraría el único uso del knob."""
    import cron_tasks as ct
    monkeypatch.setenv("MEALFIT_UPCOMING_DAYS_UI", "false")
    monkeypatch.setattr(chc, "rd_today", lambda: date(2026, 8, 10))

    captured = []
    monkeypatch.setattr(ct, "execute_sql_write",
                        lambda sql, params=None, **k: captured.append({"sql": sql, "params": params}))

    def _fake_query(sql, params=None, **kwargs):
        if "FROM meal_plans" in sql:
            raise AssertionError("con el knob OFF el cron no debería consultar planes")
        raise AssertionError(f"query inesperada: {sql}")

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query)
    ct._chunk_overdue_alert_job()

    assert _upserts(captured) == [], captured
    assert len(_sweeps(captured)) == 1, captured
    assert _sweeps(captured)[0]["params"][0] == [], captured


def test_b4_knob_off_apaga_el_indice_del_coach(monkeypatch):
    """Ojo con la forma de este test: `_build_pending_days_lines_block` fail-opena
    a "" ante CUALQUIER excepción, así que un fake que lanza haría pasar el test
    contra el código sin gatear (la excepción se traga y devuelve ""). Se cuentan
    las llamadas en vez de lanzar."""
    import agent
    monkeypatch.setenv("MEALFIT_UPCOMING_DAYS_UI", "false")

    llamadas = []

    def _spy(query, params=None, **kwargs):
        llamadas.append(query)
        return {"c": 0}

    monkeypatch.setattr("db.execute_sql_query", _spy)
    out = agent._build_pending_days_lines_block(
        "user-1", _plan(["2026-08-01", "2026-08-02", "2026-08-03"], 15),
        date(2026, 8, 10), plan_id="plan-abc-123")
    assert llamadas == [], llamadas
    assert out == "", out


# --- B5: orden determinista ------------------------------------------------


def test_b5_upcoming_chunks_tiene_desempate_determinista(monkeypatch):
    """`execute_after` es NULLABLE y ~12 paths la reescriben a `NOW()`: dos
    chunks empatados hacen que `upcoming[0]` (el que el frontend usa para
    `days_offset`/`days_count`/estado) salga al azar entre corridas."""
    queries, _ = _capturar_queries_de_chunk_status(
        monkeypatch, _plan_3_dias_viejo(), _upcoming_rows_1_pending())
    upcoming_sql = [q for q in queries
                    if "id::text AS chunk_id" in q and "ORDER BY execute_after" in q]
    assert len(upcoming_sql) == 1, queries
    orden = " ".join(upcoming_sql[0].split()).split("ORDER BY", 1)[1]
    assert "week_number" in orden, orden
    assert "days_offset" in orden, orden
