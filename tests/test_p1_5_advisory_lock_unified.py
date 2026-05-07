"""[P1-5] Tests para `acquire_meal_plan_advisory_lock` unificado.

Antes había dos invocaciones de `pg_advisory_xact_lock` con hash functions
distintas (`hashtext` en routers/plans.py vs `hashtextextended` en
cron_tasks.py) y keys con formatos distintos. Aunque servían propósitos
distintos (catchup vs tz_resync), la divergencia hacía propenso a errores
añadir nuevos locks por meal_plan: cualquier nuevo call site debía adivinar
qué función usar para colisionar con los existentes.

Después: un único helper `acquire_meal_plan_advisory_lock(cursor, plan_id,
purpose)` en `db_plans.py` con:
  - Hash function fija: `hashtextextended(text, 0)`.
  - Espacio de keys: `meal_plan:<purpose>:<plan_id>`.
  - Set de purposes conocidos validado (warning en typo).

Ejecutar:
    cd backend && python -m pytest tests/test_p1_5_advisory_lock_unified.py -v
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ---------------------------------------------------------------------------
# 1. Helper emite SQL canónica con key namespaced
# ---------------------------------------------------------------------------
def test_emits_hashtextextended_sql_with_namespaced_key():
    from db_plans import acquire_meal_plan_advisory_lock

    cur = MagicMock()
    acquire_meal_plan_advisory_lock(cur, "plan-uuid-1", purpose="catchup")

    cur.execute.assert_called_once()
    sql, params = cur.execute.call_args.args
    assert "pg_advisory_xact_lock" in sql
    assert "hashtextextended" in sql
    assert "::text, 0" in sql, "seed=0 explícito para determinismo cross-deploy"
    assert params == ("meal_plan:catchup:plan-uuid-1",)


def test_default_purpose_is_general():
    from db_plans import acquire_meal_plan_advisory_lock

    cur = MagicMock()
    acquire_meal_plan_advisory_lock(cur, "plan-X")

    sql, params = cur.execute.call_args.args
    assert params == ("meal_plan:general:plan-X",)


def test_accepts_uuid_object_as_plan_id():
    """meal_plan_id puede llegar como UUID o str; el helper lo coerce a text."""
    import uuid
    from db_plans import acquire_meal_plan_advisory_lock

    cur = MagicMock()
    plan_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    acquire_meal_plan_advisory_lock(cur, plan_uuid, purpose="tz_resync")

    sql, params = cur.execute.call_args.args
    assert params == (f"meal_plan:tz_resync:{plan_uuid}",)


# ---------------------------------------------------------------------------
# 2. Determinismo: misma key produce misma llamada
# ---------------------------------------------------------------------------
def test_same_inputs_produce_identical_sql_call():
    """Dos invocaciones con mismos args deben emitir SQL idéntico (lock colisiona)."""
    from db_plans import acquire_meal_plan_advisory_lock

    cur1, cur2 = MagicMock(), MagicMock()
    acquire_meal_plan_advisory_lock(cur1, "plan-7", purpose="catchup")
    acquire_meal_plan_advisory_lock(cur2, "plan-7", purpose="catchup")

    assert cur1.execute.call_args.args == cur2.execute.call_args.args


def test_different_purposes_produce_different_keys():
    """Distintos purposes sobre el mismo plan_id NO deben colisionar."""
    from db_plans import acquire_meal_plan_advisory_lock

    cur1, cur2 = MagicMock(), MagicMock()
    acquire_meal_plan_advisory_lock(cur1, "plan-7", purpose="catchup")
    acquire_meal_plan_advisory_lock(cur2, "plan-7", purpose="tz_resync")

    params1 = cur1.execute.call_args.args[1]
    params2 = cur2.execute.call_args.args[1]
    assert params1 != params2


def test_different_plan_ids_produce_different_keys():
    from db_plans import acquire_meal_plan_advisory_lock

    cur1, cur2 = MagicMock(), MagicMock()
    acquire_meal_plan_advisory_lock(cur1, "plan-A", purpose="catchup")
    acquire_meal_plan_advisory_lock(cur2, "plan-B", purpose="catchup")

    assert cur1.execute.call_args.args[1] != cur2.execute.call_args.args[1]


# ---------------------------------------------------------------------------
# 3. Validación de purpose desconocido (warning, no excepción)
# ---------------------------------------------------------------------------
def test_unknown_purpose_emits_warning_but_proceeds(caplog):
    import logging
    from db_plans import acquire_meal_plan_advisory_lock

    cur = MagicMock()
    with caplog.at_level(logging.WARNING, logger="db_plans"):
        acquire_meal_plan_advisory_lock(cur, "plan-Z", purpose="typo_porpose")

    # SQL se ejecuta igual (no bloqueante).
    cur.execute.assert_called_once()
    # Pero hay warning estructurado.
    matching = [r for r in caplog.records if "[P1-5]" in r.message and "typo_porpose" in r.message]
    assert matching, "Warning esperado para purpose desconocido"


def test_known_purposes_do_not_emit_warning(caplog):
    import logging
    from db_plans import acquire_meal_plan_advisory_lock, _MEAL_PLAN_LOCK_PURPOSES

    with caplog.at_level(logging.WARNING, logger="db_plans"):
        for purpose in _MEAL_PLAN_LOCK_PURPOSES:
            cur = MagicMock()
            acquire_meal_plan_advisory_lock(cur, "plan-K", purpose=purpose)

    typo_warnings = [r for r in caplog.records if "P1-5" in r.message and "purpose desconocido" in r.message]
    assert not typo_warnings


# ---------------------------------------------------------------------------
# 4. Migración: routers/plans.py y cron_tasks.py usan el helper
# ---------------------------------------------------------------------------
def test_routers_plans_uses_helper():
    """Verifica que routers/plans.py importa y usa el helper P1-5 en lugar de
    invocar `pg_advisory_xact_lock(hashtext(...))` directamente."""
    plans_router_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "routers", "plans.py",
    )
    with open(plans_router_path, encoding="utf-8") as f:
        content = f.read()
    assert "acquire_meal_plan_advisory_lock" in content
    assert "purpose=\"catchup\"" in content or "purpose='catchup'" in content
    # No debe quedar la invocación directa con hashtext.
    assert "hashtext(%s)::bigint" not in content, (
        "routers/plans.py todavía invoca hashtext directamente; debe usar el helper P1-5."
    )


def test_cron_tasks_uses_helper_for_tz_resync():
    cron_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(cron_path, encoding="utf-8") as f:
        content = f.read()
    # El helper se importa al menos una vez.
    assert "acquire_meal_plan_advisory_lock" in content
    # purpose='tz_resync' presente.
    assert "purpose=\"tz_resync\"" in content or "purpose='tz_resync'" in content


# ---------------------------------------------------------------------------
# 5. Espacio de purposes documenta los namespaces conocidos
# ---------------------------------------------------------------------------
def test_known_purposes_set_contains_production_namespaces():
    from db_plans import _MEAL_PLAN_LOCK_PURPOSES

    assert "catchup" in _MEAL_PLAN_LOCK_PURPOSES, "purpose 'catchup' usado en routers/plans.py"
    assert "tz_resync" in _MEAL_PLAN_LOCK_PURPOSES, "purpose 'tz_resync' usado en cron_tasks.py"
    assert "general" in _MEAL_PLAN_LOCK_PURPOSES, "purpose 'general' default"
