"""[P2-CHUNK-OVERDUE-SIGNAL] Predicado SSOT + payload de /chunk-status + cron + coach."""
from datetime import date
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


def _fake_chunk_status_queries(plan_data, upcoming_rows):
    """Dispatcher por-query: cada SELECT del endpoint tiene una firma
    textual distinta que permite enrutar sin tocar orden de invocación."""
    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans" in query:
            return {"user_id": _USER_A, "plan_data": plan_data}
        if "id::text AS chunk_id" in query and "status IN ('pending', 'processing')" in query:
            # [P2-CHUNK-OVERDUE-SIGNAL] query nueva de upcoming_chunks.
            return upcoming_rows
        if "SELECT execute_after FROM plan_chunk_queue" in query:
            return None  # next_chunk_eta (fetch_one), sin next chunk aquí
        if "status = 'pending_user_action'" in query:
            return []  # paused_rows
        if "COUNT(*) FILTER" in query:
            return {
                "in_flight_count": 0,
                "pending_user_action_count": 0,
                "failed_count": 0,
                "completed_count": 3,
            }
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


def test_cron_overdue_upsertea_y_autoresuelve(monkeypatch):
    """Corrida 1: 1 plan overdue (dates viejas, total 15, cola vacía) → upsert
    con alert_key='chunk_overdue:<id>'. Corrida 2 con el plan ya completo
    (total == len(days)) → UPDATE resolved_at (auto-resuelve, modelo Auto
    implicit: el job re-emite mientras la condición exista)."""
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

    assert len(captured) == 1, captured
    upsert = captured[0]
    assert "INSERT INTO system_alerts" in upsert["sql"]
    assert "ON CONFLICT" in upsert["sql"]
    assert upsert["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"

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

    assert len(captured) == 1, captured
    resolve = captured[0]
    assert "UPDATE system_alerts" in resolve["sql"]
    assert "resolved_at = NOW()" in resolve["sql"]
    assert resolve["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"


def test_cron_overdue_fail_open_por_plan(monkeypatch):
    """El primer plan lanza excepción en su COUNT in-flight; el segundo es
    overdue. El job NO debe abortar: el segundo plan recibe su alerta igual
    (fail-open POR PLAN, no fail-open global)."""
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

    assert len(captured) == 1, captured
    assert captured[0]["params"][0] == f"chunk_overdue:{_CRON_PLAN_ID}"


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
