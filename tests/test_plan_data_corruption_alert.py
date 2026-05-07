"""
[P1-6] Tests del flujo de alerta cuando plan_data está corrupto.

Antes los helpers `_safe_lessons_list` / `_safe_lessons_dict` retornaban `[]` / `{}`
silenciosamente cuando el campo llegaba como tipo incorrecto (dict en vez de list,
str en vez de dict). El chunk N+1 corría sin lecciones del chunk N, perdiendo
contexto histórico, sin que el usuario o el sistema lo supieran.

Ahora una corrupción real (tipo incorrecto, NO None) dispara:
  1. INSERT en system_alerts con `alert_key=plan_data_corrupted:{plan_id}:{field}`
     y dedupe vía ON CONFLICT.
  2. Push notification al usuario (con cooldown 24h).

Valida:
  A. None y tipo correcto NO disparan alerta.
  B. Tipo incorrecto dispara INSERT + push.
  C. Si la alerta ya existe en cooldown 24h, NO se reenvía push (pero el INSERT sí
     actualiza triggered_at vía ON CONFLICT DO UPDATE).
  D. Sin plan_id, solo se loggea (no hay UNIQUE key para deduplicar).
  E. Si el INSERT falla, NO se envía push (no se quiere notificar sin trazabilidad).
"""
import sys
from unittest.mock import patch, MagicMock

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


def test_a_none_value_returns_empty_no_alert():
    """[P1-6] None es ausencia legítima (chunk nuevo) — NO disparar alerta."""
    write_calls = []
    with patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **k: write_calls.append(a)), \
         patch("cron_tasks.execute_sql_query", return_value=None), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _safe_lessons_list, _safe_lessons_dict
        assert _safe_lessons_list(None, plan_id="plan-x", user_id="user-x") == []
        assert _safe_lessons_dict(None, plan_id="plan-x", user_id="user-x") == {}

    assert write_calls == [], "None no debe persistir alerta"
    assert mock_push.call_count == 0


def test_b_correct_type_passes_through_no_alert():
    """[P1-6] Tipo correcto retorna el valor sin tocar system_alerts."""
    write_calls = []
    with patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **k: write_calls.append(a)), \
         patch("cron_tasks.execute_sql_query", return_value=None), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _safe_lessons_list, _safe_lessons_dict
        assert _safe_lessons_list([1, 2, 3], plan_id="plan-x", user_id="user-x") == [1, 2, 3]
        assert _safe_lessons_dict({"a": 1}, plan_id="plan-x", user_id="user-x") == {"a": 1}

    assert write_calls == []
    assert mock_push.call_count == 0


def test_c_wrong_type_inserts_alert_and_pushes():
    """[P1-6] Tipo incorrecto (corrupción real) dispara INSERT + push."""
    write_calls = []
    def fake_write(sql, params=None, returning=False):
        write_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=None), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _safe_lessons_list
        # str en vez de list → corrupción
        result = _safe_lessons_list("garbage", plan_id="plan-x", user_id="user-x")
        assert result == []

    insert_calls = [(s, p) for s, p in write_calls if "INSERT INTO system_alerts" in s]
    assert len(insert_calls) == 1, "Debe persistir alerta exactamente una vez"
    sql, params = insert_calls[0]
    assert "plan_data_corrupted" in sql or "plan_data_corrupted" in str(params)
    # alert_key debe contener plan_id + field_name
    assert "plan-x" in str(params[0]) or "plan-x" in str(params)

    assert mock_push.call_count == 1, "Debe enviarse push al usuario"
    push_kwargs = mock_push.call_args.kwargs
    assert push_kwargs["user_id"] == "user-x"
    assert "problema" in push_kwargs["body"].lower() or "problema" in push_kwargs["title"].lower()


def test_d_recent_alert_skips_push_but_still_inserts():
    """[P1-6] Si ya hubo alerta en cooldown 24h, INSERT actualiza vía ON CONFLICT pero
    NO se envía otro push (cooldown evita spam)."""
    write_calls = []
    def fake_write(sql, params=None, returning=False):
        write_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value={"triggered_at": "2026-05-01T20:00:00+00:00"}), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _safe_lessons_dict
        _safe_lessons_dict("string-cuando-debiera-ser-dict", plan_id="plan-y", user_id="user-y")

    # INSERT sí ocurre (refresca triggered_at via ON CONFLICT).
    insert_calls = [(s, p) for s, p in write_calls if "INSERT INTO system_alerts" in s]
    assert len(insert_calls) == 1, "INSERT debe ejecutarse para refrescar timestamp"
    # Pero push NO porque cooldown activo.
    assert mock_push.call_count == 0, "Push debe respetar cooldown 24h"


def test_e_no_plan_id_only_logs():
    """[P1-6] Sin plan_id, no podemos deduplicar — solo log, no INSERT, no push."""
    write_calls = []
    with patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **k: write_calls.append(a)), \
         patch("cron_tasks.execute_sql_query", return_value=None), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _safe_lessons_list
        _safe_lessons_list({"corrupted": True}, plan_id=None, user_id="user-x")

    assert write_calls == [], "Sin plan_id no debe haber INSERT"
    assert mock_push.call_count == 0, "Sin plan_id no debe haber push"


def test_f_insert_failure_blocks_push():
    """[P1-6] Si el INSERT en system_alerts falla, NO enviar push — preferimos
    notificación trazable a notificación huérfana."""
    def fake_write(sql, params=None, returning=False):
        if "INSERT INTO system_alerts" in sql:
            raise RuntimeError("simulated DB error")

    with patch("cron_tasks.execute_sql_query", return_value=None), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        from cron_tasks import _safe_lessons_list
        # No debe propagar excepción.
        result = _safe_lessons_list(42, plan_id="plan-z", user_id="user-z")

    assert result == []
    assert mock_push.call_count == 0, "Sin alerta persistida, no enviar push"
