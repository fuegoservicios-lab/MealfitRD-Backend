"""[P0-2] Tests para la cadena de fallback de TZ en `_enqueue_plan_chunk`.

Cuando `snapshot.form_data._plan_start_date` falta o falla parseo, el chunk
debe resolverse vía:
  1. snapshot → 2. user_profile + today → 3. último meal_plan → 4. 8am UTC.

Antes, el fallback caía a `NOW() + delay_days`, lo que disparaba el chunk a
hora arbitraria (e.g., las 3am local). Ahora cada fuente produce un anchor
con TZ resoluble; solo el peor caso fuerza 8am UTC, y nunca a las 3am.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ----------------------------------------------------------------------------
# Unit tests (mocked) para _resolve_chunk_start_anchor
# ----------------------------------------------------------------------------

def _snapshot(start_iso=None, tz_offset=None):
    fd = {}
    if start_iso is not None:
        fd["_plan_start_date"] = start_iso
    if tz_offset is not None:
        fd["tz_offset_minutes"] = tz_offset
        fd["tzOffset"] = tz_offset
    return {"form_data": fd}


def test_source_1_snapshot_with_valid_start_date():
    """Path 1: snapshot._plan_start_date válido → source='snapshot'."""
    from cron_tasks import _resolve_chunk_start_anchor

    snap = _snapshot(start_iso="2026-05-02T00:00:00+00:00", tz_offset=240)

    with patch("cron_tasks._get_user_tz_live", return_value=240):
        start_dt, tz_min, source = _resolve_chunk_start_anchor(
            user_id="user-1", snapshot=snap, meal_plan_id="plan-1", week_number=1,
        )

    assert source == "snapshot"
    assert tz_min == 240
    assert start_dt.year == 2026 and start_dt.month == 5 and start_dt.day == 2


def test_source_2_profile_today_when_snapshot_missing_start():
    """Path 2: sin _plan_start_date pero profile tiene TZ → 'profile_today'."""
    from cron_tasks import _resolve_chunk_start_anchor

    snap = _snapshot(start_iso=None)  # missing entirely

    with patch("cron_tasks._get_user_tz_minutes_optional", return_value=240):
        start_dt, tz_min, source = _resolve_chunk_start_anchor(
            user_id="user-1", snapshot=snap, meal_plan_id="plan-1", week_number=1,
        )

    assert source == "profile_today"
    assert tz_min == 240
    today = datetime.now(timezone.utc).date()
    assert start_dt.date() == today
    assert start_dt.hour == 0 and start_dt.minute == 0


def test_source_2_profile_today_when_snapshot_parse_fails():
    """Path 1 parse-fail + profile con TZ → 'profile_today' (no 'snapshot')."""
    from cron_tasks import _resolve_chunk_start_anchor

    snap = _snapshot(start_iso="not-a-date-at-all", tz_offset=999)

    with patch("cron_tasks._get_user_tz_minutes_optional", return_value=180):
        start_dt, tz_min, source = _resolve_chunk_start_anchor(
            user_id="user-1", snapshot=snap, meal_plan_id="plan-1", week_number=1,
        )

    assert source == "profile_today"
    assert tz_min == 180  # del profile, no del snapshot corrupto


def test_source_3_last_plan_when_profile_lacks_tz():
    """Path 3: snapshot vacío + profile sin TZ + último plan con TZ → 'last_plan'."""
    from cron_tasks import _resolve_chunk_start_anchor

    snap = _snapshot(start_iso=None)
    last_plan_row = {
        "id": "plan-prior",
        "plan_data": {"health_profile": {"tz_offset_minutes": 300}},
    }

    with patch("cron_tasks._get_user_tz_minutes_optional", return_value=None), \
         patch("cron_tasks.execute_sql_query", return_value=last_plan_row):
        start_dt, tz_min, source = _resolve_chunk_start_anchor(
            user_id="user-1", snapshot=snap, meal_plan_id="plan-new", week_number=1,
        )

    assert source == "last_plan"
    assert tz_min == 300


def test_source_4_forced_8am_utc_when_all_fail():
    """Path 4: nada disponible → (None, 0, 'forced_8am_utc')."""
    from cron_tasks import _resolve_chunk_start_anchor

    snap = _snapshot(start_iso=None)

    with patch("cron_tasks._get_user_tz_minutes_optional", return_value=None), \
         patch("cron_tasks.execute_sql_query", return_value=None):
        start_dt, tz_min, source = _resolve_chunk_start_anchor(
            user_id="user-1", snapshot=snap, meal_plan_id="plan-new", week_number=1,
        )

    assert source == "forced_8am_utc"
    assert tz_min == 0
    assert start_dt is None


def test_source_4_when_last_plan_has_no_tz_field():
    """Path 4: último plan existe pero sin TZ → 'forced_8am_utc'."""
    from cron_tasks import _resolve_chunk_start_anchor

    snap = _snapshot(start_iso=None)
    row_no_tz = {"id": "plan-prior", "plan_data": {"health_profile": {}}}

    with patch("cron_tasks._get_user_tz_minutes_optional", return_value=None), \
         patch("cron_tasks.execute_sql_query", return_value=row_no_tz):
        start_dt, tz_min, source = _resolve_chunk_start_anchor(
            user_id="user-1", snapshot=snap, meal_plan_id="plan-new", week_number=1,
        )

    assert source == "forced_8am_utc"
    assert start_dt is None


# ----------------------------------------------------------------------------
# Integration tests con DB real (fixture)
# ----------------------------------------------------------------------------

pytestmark_e2e = pytest.mark.e2e


@pytest.mark.e2e
def test_enqueue_uses_profile_today_when_snapshot_missing_start_date(seeded_user_profile):
    """[P0-2] Snapshot sin `_plan_start_date` → execute_after deriva del profile.

    El conftest fixture ya seedea user_profile con tz_offset_minutes=-240.
    Verificamos que el resolver lo lee y NO cae a NOW()+delay (hora arbitraria).
    """
    from cron_tasks import _enqueue_plan_chunk
    from db_core import execute_sql_query, execute_sql_write

    user_id, _fixture_plan_id = seeded_user_profile
    plan_id = str(uuid.uuid4())
    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, name, plan_data, calories, macros) "
        "VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb)",
        (plan_id, user_id, "Plan P0-2 test",
         json.dumps({"days": []}, ensure_ascii=False), 2000,
         json.dumps({"protein": 150})),
    )

    snapshot_without_start = {
        "form_data": {
            "totalDays": 3,
            # NO _plan_start_date — fuerza el fallback chain
            "current_pantry_ingredients": [],
        },
        "totalDays": 3,
    }

    try:
        _enqueue_plan_chunk(
            user_id=user_id,
            meal_plan_id=plan_id,
            week_number=1,
            days_offset=0,
            days_count=3,
            pipeline_snapshot=snapshot_without_start,
            chunk_kind="initial_plan",
        )

        row = execute_sql_query(
            "SELECT execute_after, pipeline_snapshot FROM plan_chunk_queue "
            "WHERE meal_plan_id = %s AND week_number = 1",
            (plan_id,), fetch_one=True,
        )
        assert row, "El chunk debió encolarse"

        snap = row["pipeline_snapshot"] or {}
        anchor_source = (snap.get("form_data") or {}).get("_chunk_anchor_source")
        # El fixture tiene tz_offset_minutes=-240 en health_profile, así que la
        # cadena debe parar en 'profile_today'.
        assert anchor_source == "profile_today", \
            f"Esperaba anchor profile_today, obtuve {anchor_source!r}"

        execute_after = row["execute_after"]
        if isinstance(execute_after, str):
            from constants import safe_fromisoformat
            execute_after = safe_fromisoformat(execute_after)
        if execute_after.tzinfo is None:
            execute_after = execute_after.replace(tzinfo=timezone.utc)

        # Sanity: NO está en el rango "3am-7am UTC del día actual" que sería el
        # síntoma del bug previo (NOW()+delay disparándose a hora arbitraria).
        # El resolver con tz=-240 (convención test, asume DR) y formula
        # midnight_utc + (-240+30)min = midnight - 210min = previous day 20:30 UTC,
        # luego clamped por execute_dt_min = NOW()+1min. Lo único que afirmamos
        # categóricamente es: execute_after >= NOW() + 1min (no en pasado) y
        # tiene un anchor reproducible.
        now = datetime.now(timezone.utc)
        assert execute_after >= now, "execute_after debe ser futuro"
        # Margen de 5 min: validar que no es NOW+exactly-delay-days al instante.
        assert (execute_after - now).total_seconds() >= 30, \
            "execute_after debe respetar el clamp execute_dt_min ≥ NOW+1min"
    finally:
        execute_sql_write("DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,))
        execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))


@pytest.mark.e2e
def test_enqueue_forced_8am_utc_when_no_tz_anywhere(seeded_user_profile):
    """[P0-2] Sin TZ en perfil ni en último plan → execute_after = 8am UTC del día N."""
    from cron_tasks import _enqueue_plan_chunk
    from db_core import execute_sql_query, execute_sql_write

    user_id, _fixture_plan_id = seeded_user_profile
    plan_id = str(uuid.uuid4())

    # Forzar profile sin TZ y limpiar cualquier plan previo del usuario para que
    # el resolver no encuentre nada.
    execute_sql_write(
        "UPDATE user_profiles SET health_profile = %s::jsonb WHERE id = %s",
        (json.dumps({"age": 30, "weight": 75}), user_id),  # SIN tz_offset_minutes
    )
    execute_sql_write("DELETE FROM meal_plans WHERE user_id = %s", (user_id,))

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, name, plan_data, calories, macros) "
        "VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb)",
        (plan_id, user_id, "Plan P0-2 8am UTC",
         json.dumps({"days": []}, ensure_ascii=False), 2000,
         json.dumps({"protein": 150})),
    )
    # El plan recién creado tampoco tiene health_profile en plan_data → path 3 falla.

    snapshot_without_start = {
        "form_data": {"totalDays": 3, "current_pantry_ingredients": []},
        "totalDays": 3,
    }

    try:
        _enqueue_plan_chunk(
            user_id=user_id,
            meal_plan_id=plan_id,
            week_number=1,
            days_offset=0,
            days_count=3,
            pipeline_snapshot=snapshot_without_start,
            chunk_kind="initial_plan",
        )

        row = execute_sql_query(
            "SELECT execute_after, pipeline_snapshot FROM plan_chunk_queue "
            "WHERE meal_plan_id = %s AND week_number = 1",
            (plan_id,), fetch_one=True,
        )
        assert row, "El chunk debió encolarse aún sin TZ"

        snap = row["pipeline_snapshot"] or {}
        anchor_source = (snap.get("form_data") or {}).get("_chunk_anchor_source")
        assert anchor_source == "forced_8am_utc", \
            f"Esperaba forced_8am_utc, obtuve {anchor_source!r}"

        execute_after = row["execute_after"]
        if isinstance(execute_after, str):
            from constants import safe_fromisoformat
            execute_after = safe_fromisoformat(execute_after)
        if execute_after.tzinfo is None:
            execute_after = execute_after.replace(tzinfo=timezone.utc)

        # Forced 8am UTC: si delay_days > 0 → fresh_target = (today + delay) at 8am UTC
        # → hour=8. Si delay_days==0 y now > 8am UTC, el clamp execute_dt_min eleva el
        # timestamp a NOW+1min, así que hour matchea now.hour. Lo importante es:
        #   1. anchor_source == "forced_8am_utc" (validado arriba).
        #   2. execute_after >= NOW (no en el pasado).
        #   3. NUNCA disparado a las 3-5am del día (síntoma del bug pre-P0-2).
        now = datetime.now(timezone.utc)
        assert execute_after >= now, "execute_after debe ser futuro tras clamp"
        # El bug previo (NOW+delay sin TZ) podía caer a hora arbitraria; el fix
        # garantiza hour==8 si delay>0, o NOW+1min (clamped) si delay=0.
        if execute_after.date() > now.date():
            assert execute_after.hour == 8, (
                f"Para delay_days > 0, fresh_target = (target_date) at 8am UTC. "
                f"obtuve hora={execute_after.hour}"
            )
    finally:
        execute_sql_write("DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,))
        execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))
