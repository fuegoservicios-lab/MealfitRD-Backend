"""[P2-1] Tests para `_record_tz_fallback` con dedupe per (plan_id, reason).

Antes cada chunk que caía al fallback TZ emitía una línea WARNING; un plan
de 30 días con `_plan_start_date` corrupto generaba ~8 líneas idénticas
modulo `week_number`.

Después: dedupe por `(plan_id, reason)` con TTL 1h.
  - Primera ocurrencia → WARNING accionable.
  - Repeticiones dentro del TTL → solo se acumulan en el ring buffer.
  - Tras expirar el TTL, vuelve a emitir WARNING (señal de problema persistente).
  - Ring buffer expuesto vía /api/system/tz-fallback-health.

Ejecutar:
    cd backend && python -m pytest tests/test_p2_1_tz_fallback_aggregated_logging.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _reset_tz_fallback_buffers():
    """Limpia los buffers entre tests para aislamiento."""
    from cron_tasks import _TZ_FALLBACK_EVENTS, _TZ_FALLBACK_DEDUPE_KEYS
    _TZ_FALLBACK_EVENTS.clear()
    _TZ_FALLBACK_DEDUPE_KEYS.clear()
    yield
    _TZ_FALLBACK_EVENTS.clear()
    _TZ_FALLBACK_DEDUPE_KEYS.clear()


# ---------------------------------------------------------------------------
# 1. Primera ocurrencia emite WARNING; el buffer se llena
# ---------------------------------------------------------------------------
def test_first_occurrence_emits_warning_and_records_event(caplog):
    import logging
    from cron_tasks import _record_tz_fallback, _TZ_FALLBACK_EVENTS

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        emitted = _record_tz_fallback(
            user_id="user-1",
            meal_plan_id="plan-A",
            week_number=2,
            reason="parse_plan_start_date_failed",
            detail="err=ValueError:bad iso",
        )

    assert emitted is True
    assert len(_TZ_FALLBACK_EVENTS) == 1
    matching = [r for r in caplog.records if "[P0-4/TZ-FALLBACK]" in r.message]
    assert len(matching) == 1, "Primera ocurrencia debe emitir exactamente 1 WARNING"
    assert "plan-A" in matching[0].message
    assert "parse_plan_start_date_failed" in matching[0].message


# ---------------------------------------------------------------------------
# 2. Repeticiones (mismo plan, mismo reason) NO emiten WARNING extra
# ---------------------------------------------------------------------------
def test_repeated_same_plan_same_reason_dedupes_warning(caplog):
    import logging
    from cron_tasks import _record_tz_fallback, _TZ_FALLBACK_EVENTS

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        # Plan 30d → 8 chunks caen al fallback. Solo el primero debe emitir WARNING.
        emissions = []
        for week in range(2, 10):
            emissions.append(
                _record_tz_fallback(
                    user_id="user-1",
                    meal_plan_id="plan-A",
                    week_number=week,
                    reason="missing_plan_start_date_in_snapshot",
                )
            )

    assert emissions[0] is True, "Primer chunk emite WARNING"
    assert all(e is False for e in emissions[1:]), "Resto NO emite WARNING"

    warnings = [r for r in caplog.records if "[P0-4/TZ-FALLBACK]" in r.message]
    assert len(warnings) == 1, f"Esperaba 1 WARNING, obtuvo {len(warnings)}"

    # Pero el ring buffer SÍ acumula los 8 eventos.
    assert len(_TZ_FALLBACK_EVENTS) == 8


# ---------------------------------------------------------------------------
# 3. Distintos planes con mismo reason emiten cada uno su WARNING
# ---------------------------------------------------------------------------
def test_different_plans_same_reason_each_emits_warning(caplog):
    import logging
    from cron_tasks import _record_tz_fallback

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        e1 = _record_tz_fallback("u1", "plan-A", 2, "missing_plan_start_date_in_snapshot")
        e2 = _record_tz_fallback("u2", "plan-B", 2, "missing_plan_start_date_in_snapshot")
        e3 = _record_tz_fallback("u3", "plan-C", 2, "missing_plan_start_date_in_snapshot")

    assert all([e1, e2, e3])
    warnings = [r for r in caplog.records if "[P0-4/TZ-FALLBACK]" in r.message]
    assert len(warnings) == 3


# ---------------------------------------------------------------------------
# 4. Mismo plan con distintos reasons emiten cada uno su WARNING
# ---------------------------------------------------------------------------
def test_same_plan_different_reasons_each_emits_warning(caplog):
    import logging
    from cron_tasks import _record_tz_fallback

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        e1 = _record_tz_fallback("u1", "plan-A", 2, "missing_plan_start_date_in_snapshot")
        e2 = _record_tz_fallback("u1", "plan-A", 3, "parse_plan_start_date_failed")

    assert e1 is True and e2 is True
    warnings = [r for r in caplog.records if "[P0-4/TZ-FALLBACK]" in r.message]
    assert len(warnings) == 2


# ---------------------------------------------------------------------------
# 5. TTL expirado: vuelve a emitir WARNING (problema persistente)
# ---------------------------------------------------------------------------
def test_warning_re_emitted_after_dedupe_ttl_expires(caplog):
    import logging
    import time as _t
    from cron_tasks import (
        _record_tz_fallback,
        _TZ_FALLBACK_DEDUPE_KEYS,
        _TZ_FALLBACK_DEDUPE_TTL_SECONDS,
    )

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        first = _record_tz_fallback("u1", "plan-A", 2, "missing_plan_start_date_in_snapshot")

    assert first is True

    # Forzar "expiración" del TTL adelantando el timestamp guardado.
    expired_ts = _t.time() - _TZ_FALLBACK_DEDUPE_TTL_SECONDS - 60
    _TZ_FALLBACK_DEDUPE_KEYS[("plan-A", "missing_plan_start_date_in_snapshot")] = expired_ts

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        second = _record_tz_fallback("u1", "plan-A", 5, "missing_plan_start_date_in_snapshot")

    assert second is True, "Tras TTL expirado, debe re-emitir WARNING"


# ---------------------------------------------------------------------------
# 6. Ring buffer poda eventos viejos
# ---------------------------------------------------------------------------
def test_ring_buffer_prunes_events_older_than_window():
    import time as _t
    from cron_tasks import (
        _record_tz_fallback,
        _TZ_FALLBACK_EVENTS,
        _TZ_FALLBACK_WINDOW_SECONDS,
    )

    # Inyectar evento "antiguo" (25h atrás).
    old_ts = _t.time() - _TZ_FALLBACK_WINDOW_SECONDS - 3600
    _TZ_FALLBACK_EVENTS.append((old_ts, "old-user", "old-plan", 1, "missing_plan_start_date_in_snapshot"))

    # Añadir evento nuevo → poda automática del antiguo.
    _record_tz_fallback("new-user", "new-plan", 2, "missing_plan_start_date_in_snapshot")

    plan_ids = [e[2] for e in _TZ_FALLBACK_EVENTS]
    assert "old-plan" not in plan_ids
    assert "new-plan" in plan_ids


# ---------------------------------------------------------------------------
# 7. Ring buffer cap previene crecimiento ilimitado
# ---------------------------------------------------------------------------
def test_ring_buffer_caps_at_max_records():
    import time as _t
    from cron_tasks import (
        _record_tz_fallback,
        _TZ_FALLBACK_EVENTS,
        _TZ_FALLBACK_MAX_RECORDS,
    )

    now = _t.time()
    # Llenar al cap.
    for i in range(_TZ_FALLBACK_MAX_RECORDS):
        _TZ_FALLBACK_EVENTS.append((now - 0.001, f"u-{i}", f"plan-{i}", 1, "x"))

    pre = len(_TZ_FALLBACK_EVENTS)
    assert pre == _TZ_FALLBACK_MAX_RECORDS

    # Añadir uno más → poda al cap.
    _record_tz_fallback("u-overflow", "plan-overflow", 1, "missing_plan_start_date_in_snapshot")

    assert len(_TZ_FALLBACK_EVENTS) <= _TZ_FALLBACK_MAX_RECORDS


# ---------------------------------------------------------------------------
# 8. Endpoint /api/system/tz-fallback-health agrega correctamente
# ---------------------------------------------------------------------------
def test_health_endpoint_aggregates_by_reason_and_top_plans():
    from cron_tasks import _record_tz_fallback
    from routers.system import get_tz_fallback_health

    # Plan A: 5 hits del mismo reason.
    for week in range(2, 7):
        _record_tz_fallback("u1", "plan-A", week, "missing_plan_start_date_in_snapshot")
    # Plan B: 3 hits, otro reason.
    for week in range(2, 5):
        _record_tz_fallback("u2", "plan-B", week, "parse_plan_start_date_failed")
    # Plan C: 1 hit.
    _record_tz_fallback("u3", "plan-C", 2, "missing_plan_start_date_in_snapshot")

    res = get_tz_fallback_health()

    assert res["success"] is True
    assert res["total_24h"] == 9
    assert res["by_reason"]["missing_plan_start_date_in_snapshot"] == 6
    assert res["by_reason"]["parse_plan_start_date_failed"] == 3
    assert res["unique_plans_24h"] == 3
    assert res["unique_users_24h"] == 3
    # plan-A tiene más hits, debe estar primero en top_plans.
    assert res["top_plans_24h"][0]["plan_id"] == "plan-A"
    assert res["top_plans_24h"][0]["count"] == 5


def test_health_endpoint_returns_zeros_when_buffer_empty():
    from routers.system import get_tz_fallback_health

    res = get_tz_fallback_health()

    assert res["success"] is True
    assert res["total_24h"] == 0
    assert res["by_reason"] == {}
    assert res["top_plans_24h"] == []


# ---------------------------------------------------------------------------
# 9. WARNING incluye el detail string cuando se proporciona
# ---------------------------------------------------------------------------
def test_detail_appended_to_warning_message(caplog):
    import logging
    from cron_tasks import _record_tz_fallback

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        _record_tz_fallback(
            "u1", "plan-D", 2,
            reason="parse_plan_start_date_failed",
            detail="err=ValueError:invalid_iso_format",
        )

    matching = [r for r in caplog.records if "[P0-4/TZ-FALLBACK]" in r.message]
    assert matching
    assert "err=ValueError:invalid_iso_format" in matching[0].message
