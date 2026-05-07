"""
Tests P1-CHUNKS-1: dead-lettered chunks visibles al usuario incluso si push falla.

Bug original (audit P1-CHUNKS-1):
  Cuando un chunk supera `CHUNK_MAX_RECOVERY_ATTEMPTS`,
  `_escalate_unrecoverable_chunk` intenta `_dispatch_push_notification` para
  avisar al usuario. Si Firebase está saturado, el usuario removió permisos,
  o el OS bloquea la entrega, el push falla silenciosamente (daemon thread
  fire-and-forget) y solo se loguea WARNING. El chunk queda dead-lettered
  sin que el usuario lo sepa hasta que abra la app y note que el plan no
  avanza — días o semanas después.

Fix:
  1. `_user_action_required` en plan_data ahora trae payload enriquecido
     (title/body/cta/url/chunk_id/reason) que el frontend renderiza
     directamente. Antes solo guardaba {reason, week_number, requested_at}.
  2. `system_alerts` per-chunk con severity='critical' y dedupe por
     `alert_key=dead_lettered_chunk:{plan_id}:{week_number}`. Independiente
     del cron agregador `_alert_new_dead_lettered_chunks` (warning, hourly,
     min_count threshold) — aquí queremos un row inmediato por evento crítico.
  3. Endpoint `GET /api/plans/{plan_id}/chunk-status` expone
     `user_action_required` y `recovery_exhausted_chunks` para que el
     frontend muestre banner persistente con CTA "Regenerar plan" sin
     depender del push.
  4. Push se mueve DESPUÉS de la persistencia: un fallo del push no impide
     que el banner aparezca.
"""
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import cron_tasks


# ---------------------------------------------------------------------------
# 1. Persistencia de `_user_action_required` con payload enriquecido.
# ---------------------------------------------------------------------------
def _capture_sql_writes():
    """Helper: retorna (mock, lista_de_calls) para inspeccionar SQL escrito."""
    captured: list = []

    def _fake_write(query, params=None, *args, **kwargs):
        captured.append((query, params))

    return _fake_write, captured


def test_escalation_persiste_user_action_required_con_payload_completo():
    """[P1-CHUNKS-1] El UPDATE de meal_plans debe incluir title/body/cta/url
    + chunk_id en `_user_action_required`. Antes solo guardaba reason +
    week_number, dejando al frontend sin info para mostrar banner."""
    fake_write, captured = _capture_sql_writes()
    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification"):
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-1", user_id="user-1", plan_id="plan-1",
            week_number=2, recovery_attempts=2,
            escalation_reason="recovery_exhausted",
        )

    # Buscar el UPDATE que escribió `_user_action_required`.
    plan_updates = [
        (q, p) for (q, p) in captured
        if "UPDATE meal_plans" in q and "_user_action_required" in q
    ]
    assert len(plan_updates) == 1, (
        f"Esperaba 1 UPDATE de meal_plans con _user_action_required, "
        f"encontrados {len(plan_updates)}."
    )
    sql, params = plan_updates[0]
    # Verificar que el SQL incluye los campos enriquecidos:
    assert "'title'" in sql
    assert "'body'" in sql
    assert "'cta'" in sql
    assert "'url'" in sql
    assert "'chunk_id'" in sql
    assert "'reason_code'" in sql
    # Y que params tienen los valores de copy preformateado:
    # signature: (week, attempts, reason, reason, task, week, title, body, cta, url, plan_id)
    assert "task-1" in params  # chunk_id
    assert "plan-1" in params  # plan_id
    assert any("Tu plan necesita atención" in str(p) for p in params), (
        "title de recovery_exhausted debe estar en params"
    )
    assert any("Regenerar plan" in str(p) for p in params), (
        "cta debe ser 'Regenerar plan'"
    )


def test_escalation_copy_es_diferente_para_missing_anchor():
    """[P1-2] Cada escalation_reason tiene copy + url específicos."""
    fake_write, captured = _capture_sql_writes()
    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification"):
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-1", user_id="user-1", plan_id="plan-1",
            week_number=2, recovery_attempts=2,
            escalation_reason="unrecoverable_missing_anchor",
        )

    plan_updates = [
        (q, p) for (q, p) in captured if "_user_action_required" in q
    ]
    assert len(plan_updates) == 1
    _, params = plan_updates[0]
    assert any("regenerarse" in str(p).lower() for p in params)
    assert any("missing_anchor" in str(p) for p in params)


def test_escalation_copy_es_diferente_para_corrupted_date():
    fake_write, captured = _capture_sql_writes()
    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification"):
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-1", user_id="user-1", plan_id="plan-1",
            week_number=2, recovery_attempts=2,
            escalation_reason="unrecoverable_corrupted_date",
        )

    plan_updates = [
        (q, p) for (q, p) in captured if "_user_action_required" in q
    ]
    assert len(plan_updates) == 1
    _, params = plan_updates[0]
    assert any("corrupted_date" in str(p) for p in params)
    assert any("fecha de inicio" in str(p).lower() for p in params)


# ---------------------------------------------------------------------------
# 2. system_alerts per-chunk con severity='critical' y dedupe por alert_key.
# ---------------------------------------------------------------------------
def test_escalation_emite_system_alerts_critical():
    """[P1-CHUNKS-1] Visibilidad SRE: cada dead-letter genera un row con
    severity='critical' y alert_key dedupada por plan+week. Independiente
    del push — incluso si el push falla, SRE ve el evento."""
    fake_write, captured = _capture_sql_writes()
    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification"):
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-99", user_id="user-1", plan_id="plan-7",
            week_number=3, recovery_attempts=2,
            escalation_reason="recovery_exhausted",
        )

    alert_inserts = [
        (q, p) for (q, p) in captured
        if "INSERT INTO system_alerts" in q
    ]
    assert len(alert_inserts) == 1, (
        f"Esperaba 1 INSERT a system_alerts, encontrados {len(alert_inserts)}."
    )
    sql, params = alert_inserts[0]
    # severity='critical' hardcoded en el SQL:
    assert "'critical'" in sql
    assert "'dead_lettered_chunk'" in sql, "alert_type debe ser 'dead_lettered_chunk'"
    # alert_key con plan_id + week_number para dedupe:
    alert_key = params[0]
    assert alert_key == "dead_lettered_chunk:plan-7:3"
    # ON CONFLICT garantiza idempotencia (re-escalación NO duplica filas):
    assert "ON CONFLICT (alert_key) DO UPDATE" in sql


def test_escalation_alert_persiste_aunque_meal_plans_update_falle():
    """Defense-in-depth: si el UPDATE de meal_plans falla (lock contention,
    schema drift), system_alerts SIGUE persistiendo. SRE no pierde
    visibilidad por una falla parcial."""
    def _selective_fail(query, *args, **kwargs):
        if "UPDATE meal_plans" in query:
            raise RuntimeError("simulated lock contention")

    captured = []
    def _fake_write(query, params=None, *args, **kwargs):
        captured.append((query, params))
        if "UPDATE meal_plans" in query:
            raise RuntimeError("simulated lock contention")

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification"):
        # NO debe propagar la excepción.
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-1", user_id="user-1", plan_id="plan-1",
            week_number=2, recovery_attempts=2,
            escalation_reason="recovery_exhausted",
        )

    alert_inserts = [
        (q, p) for (q, p) in captured
        if "INSERT INTO system_alerts" in q
    ]
    assert len(alert_inserts) == 1, (
        "system_alerts debe persistirse incluso si UPDATE meal_plans falló."
    )


def test_escalation_continua_aunque_alert_falle():
    """Defense-in-depth inverso: si system_alerts falla (tabla missing,
    permisos), el push y los logs SIGUEN aplicándose. Cero acción del fix
    P1-CHUNKS-1 debe abortar el cleanup operacional."""
    def _selective_fail(query, *args, **kwargs):
        if "INSERT INTO system_alerts" in query:
            raise RuntimeError("permission denied")

    captured = []
    def _fake_write(query, params=None, *args, **kwargs):
        captured.append((query, params))
        if "INSERT INTO system_alerts" in query:
            raise RuntimeError("permission denied")

    push_mock = MagicMock()
    with patch.object(cron_tasks, "execute_sql_write", side_effect=_fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification", push_mock):
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-1", user_id="user-1", plan_id="plan-1",
            week_number=2, recovery_attempts=2,
            escalation_reason="recovery_exhausted",
        )

    # Push debe haberse intentado a pesar del fallo de system_alerts:
    push_mock.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Push fallido NO oculta la situación al usuario.
# ---------------------------------------------------------------------------
def test_user_action_required_persiste_aunque_push_falle():
    """[P1-CHUNKS-1] Caso crítico del audit: si el push falla
    (Firebase saturado, daemon thread muere silente, OS bloquea entrega),
    `_user_action_required` YA SE PERSISTIÓ antes del push, así que el
    frontend lo lee del polling de chunk-status y muestra el banner
    independientemente."""
    fake_write, captured = _capture_sql_writes()
    push_mock = MagicMock(side_effect=RuntimeError("Firebase down"))

    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification", push_mock):
        # NO debe propagar la excepción del push.
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-1", user_id="user-1", plan_id="plan-1",
            week_number=2, recovery_attempts=2,
            escalation_reason="recovery_exhausted",
        )

    push_mock.assert_called_once()  # Se intentó el push
    # Pero la persistencia ya había corrido — verifica que ambos UPDATEs
    # corrieron antes que el push:
    plan_updates = [(q, p) for (q, p) in captured if "_user_action_required" in q]
    alert_inserts = [(q, p) for (q, p) in captured if "INSERT INTO system_alerts" in q]
    assert len(plan_updates) == 1, "_user_action_required debe persistir antes del push"
    assert len(alert_inserts) == 1, "system_alerts debe insertarse antes del push"


def test_orden_persiste_antes_de_push_en_inspect_source():
    """Regression guard estructural: la implementación de
    `_escalate_unrecoverable_chunk` debe llamar al UPDATE de meal_plans y al
    INSERT de system_alerts ANTES del `_dispatch_push_notification`. Si
    alguien refactoriza y pone el push primero, este test reabre el alarm
    porque un fallo del push podría dejar el persist sin ejecutar."""
    import inspect
    src = inspect.getsource(cron_tasks._escalate_unrecoverable_chunk)
    plan_update_idx = src.find("_user_action_required")
    alert_idx = src.find("INSERT INTO system_alerts")
    push_idx = src.find("_dispatch_push_notification(")
    assert plan_update_idx != -1
    assert alert_idx != -1
    assert push_idx != -1
    assert plan_update_idx < push_idx, (
        "_user_action_required debe persistirse ANTES del push para que un "
        "push fallido no oculte la situación al usuario."
    )
    assert alert_idx < push_idx, (
        "system_alerts debe insertarse ANTES del push (mismo razón)."
    )


# ---------------------------------------------------------------------------
# 4. Sanity: la firma + reasons cubren los 3 escalation_reason canónicos.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reason", [
    "recovery_exhausted",
    "unrecoverable_missing_anchor",
    "unrecoverable_corrupted_date",
])
def test_escalation_funciona_para_los_3_reasons_canonicos(reason):
    """Sanity check: los 3 valores documentados en el docstring de
    `_escalate_unrecoverable_chunk` no levantan excepción y producen
    persist + alert + push intent. Si alguien añade un nuevo reason sin
    actualizar el if/elif del copy, este test sigue pasando con el default
    (recovery_exhausted-style copy) — eso es comportamiento aceptable, no
    un fallo."""
    fake_write, captured = _capture_sql_writes()
    push_mock = MagicMock()
    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification", push_mock):
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="t", user_id="u", plan_id="p",
            week_number=1, recovery_attempts=2, escalation_reason=reason,
        )
    plan_updates = [(q, p) for (q, p) in captured if "_user_action_required" in q]
    alert_inserts = [(q, p) for (q, p) in captured if "INSERT INTO system_alerts" in q]
    assert len(plan_updates) == 1
    assert len(alert_inserts) == 1
    push_mock.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Bug original reproducido: pre-fix push fallaba → usuario quedaba a oscuras.
# ---------------------------------------------------------------------------
def test_escenario_bug_original_push_falla_user_action_visible_para_polling():
    """
    PRE-FIX: el flag `_user_action_required` SÍ se persistía pero
    NO se exponía al frontend. Adicionalmente el copy era pobre
    (solo {reason, week_number, requested_at}) — el frontend tenía que
    construir el banner desde el reason crudo. Si el push fallaba, el usuario
    quedaba a ciegas.

    POST-FIX: payload enriquecido con title/body/cta/url permite al endpoint
    chunk-status retornar el dict tal cual; frontend renderiza banner sin
    duplicar mappings; system_alerts da visibilidad SRE; push fail no rompe
    nada.
    """
    fake_write, captured = _capture_sql_writes()
    push_mock = MagicMock(side_effect=RuntimeError("Firebase saturado"))

    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_write), \
         patch.object(cron_tasks, "_dispatch_push_notification", push_mock):
        cron_tasks._escalate_unrecoverable_chunk(
            task_id="task-x", user_id="user-y", plan_id="plan-z",
            week_number=5, recovery_attempts=2,
            escalation_reason="recovery_exhausted",
        )

    # 1. _user_action_required persistido con payload completo:
    plan_updates = [(q, p) for (q, p) in captured if "_user_action_required" in q]
    assert len(plan_updates) == 1
    _, params = plan_updates[0]
    # Hay title, body, cta, url, chunk_id en params:
    assert "task-x" in params
    assert "plan-z" in params
    assert any("Regenerar plan" in str(p) for p in params)
    # 2. system_alerts dedupable por alert_key:
    alert_inserts = [(q, p) for (q, p) in captured if "INSERT INTO system_alerts" in q]
    assert len(alert_inserts) == 1
    assert alert_inserts[0][1][0] == "dead_lettered_chunk:plan-z:5"
    # 3. Push se intentó pero falló — persistencia ya cubrió la visibilidad.
    push_mock.assert_called_once()
