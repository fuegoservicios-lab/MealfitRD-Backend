"""
[P1-2] Tests para `_nudge_chronic_zero_log_users` (push proactivo).

Antes el push solo se mandaba cuando el chunk N+1 ya estaba defiriéndose
(cron_tasks.py:6713-6739) — el usuario no se enteraba hasta que el plan ya
estaba en problemas. Ahora un cron cada N horas detecta usuarios con plan
activo y cero logs y les avisa antes de que el siguiente bloque tropiece.

Valida:
  A. Para cada candidato retornado por el SELECT se dispara push y se
     persiste `last_zero_log_nudge_at`.
  B. La query SELECT incluye los 3 filtros críticos: detection_days,
     cooldown, generation_status activo.
  C. Si el SELECT no retorna candidatos (todos en cooldown o todos con logs),
     no se dispara push.
  D. Si el push falla pero el SELECT retornó candidatos, el contador no se
     desborda y el error queda en logs (no propaga excepción).
"""
import sys
from unittest.mock import patch, MagicMock, call

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


def _candidate(user_id: str, last_nudge_at: str = ""):
    return {"user_id": user_id, "last_nudge_at": last_nudge_at}


def test_a_dispatches_push_and_persists_last_nudge_for_each_candidate():
    """[P1-2] Cada candidato recibe push + UPDATE de last_zero_log_nudge_at."""
    candidates = [
        _candidate("user-1"),
        _candidate("user-2"),
        _candidate("user-3"),
    ]

    write_calls = []
    def fake_write(sql, params=None, returning=False):
        write_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=candidates), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _nudge_chronic_zero_log_users
        sent = _nudge_chronic_zero_log_users()

    assert sent == 3, f"Debe enviar push a los 3 candidatos, got {sent}"
    assert mock_push.call_count == 3
    pushed_users = [c.kwargs["user_id"] for c in mock_push.call_args_list]
    assert sorted(pushed_users) == ["user-1", "user-2", "user-3"]

    # Cada user debe tener UN UPDATE de last_zero_log_nudge_at.
    nudge_updates = [
        c for c in write_calls
        if "UPDATE user_profiles" in c[0] and "last_zero_log_nudge_at" in c[0]
    ]
    assert len(nudge_updates) == 3, "Cada usuario debe persistir last_zero_log_nudge_at"
    pushed_in_updates = sorted(c[1][1] for c in nudge_updates)  # params=(now_iso, uid)
    assert pushed_in_updates == ["user-1", "user-2", "user-3"]


def test_b_select_query_filters_detection_days_cooldown_and_status():
    """[P1-2] Query SELECT debe filtrar detection_days, cooldown, y generation_status."""
    captured = {"sql": None, "params": None}

    def fake_query(sql, params=None, fetch_all=False, fetch_one=False):
        captured["sql"] = sql
        captured["params"] = params
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write"), \
         patch("cron_tasks._dispatch_push_notification"):
        from cron_tasks import _nudge_chronic_zero_log_users
        from constants import (
            CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS,
            CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS,
            CHUNK_ZERO_LOG_NUDGE_MAX_USERS,
        )
        _nudge_chronic_zero_log_users()

    sql = captured["sql"]
    # Filtro de detection days
    assert "consumed_at > NOW() - make_interval(days => %s)" in sql
    # Filtro de cooldown
    assert "make_interval(hours => %s)" in sql
    # Filtro de plan activo (no failed/expired)
    assert "generation_status" in sql
    assert "complete" in sql or "partial" in sql, "Debe filtrar status de plan activo"
    # Params en orden: detection_days, cooldown_hours, max_users
    assert captured["params"] == (
        CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS,
        CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS,
        CHUNK_ZERO_LOG_NUDGE_MAX_USERS,
    )


def test_c_no_candidates_no_push():
    """[P1-2] Sin candidatos (todos en cooldown o con logs) → 0 pushes, 0 writes."""
    write_calls = []
    with patch("cron_tasks.execute_sql_query", return_value=[]), \
         patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **k: write_calls.append(a)), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _nudge_chronic_zero_log_users
        sent = _nudge_chronic_zero_log_users()

    assert sent == 0
    assert mock_push.call_count == 0
    assert write_calls == []


def test_d_push_failure_for_one_user_does_not_block_others():
    """[P1-2] Si el push falla para un usuario, los demás no se ven afectados."""
    candidates = [_candidate("user-A"), _candidate("user-B"), _candidate("user-C")]

    push_calls = []
    def fake_push(**kwargs):
        push_calls.append(kwargs["user_id"])
        if kwargs["user_id"] == "user-B":
            raise RuntimeError("simulated webpush failure")

    write_calls = []
    def fake_write(sql, params=None, returning=False):
        write_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=candidates), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification", side_effect=fake_push):
        from cron_tasks import _nudge_chronic_zero_log_users
        sent = _nudge_chronic_zero_log_users()

    # Push se intentó para los 3
    assert push_calls == ["user-A", "user-B", "user-C"]
    # Solo 2 contaron como enviados (user-B falló)
    assert sent == 2
    # Solo A y C deben tener UPDATE de last_zero_log_nudge_at (B falló antes del UPDATE)
    nudge_updates = [
        c for c in write_calls
        if "UPDATE user_profiles" in c[0] and "last_zero_log_nudge_at" in c[0]
    ]
    updated_users = sorted(c[1][1] for c in nudge_updates)
    assert updated_users == ["user-A", "user-C"], \
        "user-B no debe tener UPDATE porque su push falló antes"


def test_e_persist_failure_does_not_double_count():
    """[P1-2] Si la persistencia de last_nudge falla, el push se cuenta como enviado
    pero se registra warning. Riesgo conocido: re-nudge dentro del cooldown si el
    UPDATE falla repetidamente — preferible al falso negativo de no contar el push."""
    candidates = [_candidate("user-X")]

    def fake_write(sql, params=None, returning=False):
        if "UPDATE user_profiles" in sql:
            raise RuntimeError("persist failure")
        return None

    with patch("cron_tasks.execute_sql_query", return_value=candidates), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification"):
        from cron_tasks import _nudge_chronic_zero_log_users
        sent = _nudge_chronic_zero_log_users()

    # Push se envió aunque la persistencia falló (decisión de diseño).
    assert sent == 1
