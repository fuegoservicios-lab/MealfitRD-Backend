"""[P2-2] Tests para `CHUNK_LEARNING_MODE='safety_margin'`.

El modo existía en producción pero no estaba documentado ni testeado
explícitamente. Este test fija el contrato:

  - "strict" (default): chunk N+1 se programa para iniciar JUSTO al terminar N
    (delay = days_offset, modulo CHUNK_PROACTIVE_MARGIN_DAYS y retry boost).
  - "safety_margin": chunk N+1 se adelanta `ceil(days_count/2)` días. Para
    planes ≥15d, el chunk final recibe adelanto adicional de 3 días.

Cualquier cambio futuro al cálculo debe romper estos tests deliberadamente.

Ejecutar:
    cd backend && python -m pytest tests/test_p2_2_chunk_learning_mode_safety_margin.py -v
"""
import math
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ---------------------------------------------------------------------------
# 1. Modo "strict": delay = days_offset (chunk programado para su día)
# ---------------------------------------------------------------------------
@patch("cron_tasks.CHUNK_LEARNING_MODE", "strict")
def test_strict_mode_delay_equals_days_offset_for_initial_plan_no_margin():
    """Sin proactive margin y chunk_kind=initial_plan, strict NO adelanta."""
    from cron_tasks import _compute_chunk_delay_days

    with patch("cron_tasks.CHUNK_PROACTIVE_MARGIN_DAYS", 0):
        delay, mode, offset, count = _compute_chunk_delay_days(
            days_offset=3, days_count=3, week_number=2,
            pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
        )

    assert mode == "strict"
    assert delay == 3, "strict + margen 0 → chunk se ejecuta al iniciar su ventana"


@patch("cron_tasks.CHUNK_LEARNING_MODE", "strict")
def test_strict_mode_applies_proactive_margin_to_initial_plan_chunks():
    """Con CHUNK_PROACTIVE_MARGIN_DAYS=1, chunks initial_plan se adelantan 1 día."""
    from cron_tasks import _compute_chunk_delay_days

    with patch("cron_tasks.CHUNK_PROACTIVE_MARGIN_DAYS", 1):
        delay, _, _, _ = _compute_chunk_delay_days(
            days_offset=3, days_count=3, week_number=2,
            pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
        )

    assert delay == 2, "strict con margen 1 → chunk adelantado 1 día"


@patch("cron_tasks.CHUNK_LEARNING_MODE", "strict")
def test_strict_mode_failed_retry_forces_minimum_one_day_advance():
    """`for_failed_retry=True` garantiza al menos 1 día de adelanto en strict."""
    from cron_tasks import _compute_chunk_delay_days

    with patch("cron_tasks.CHUNK_PROACTIVE_MARGIN_DAYS", 0):
        delay, _, _, _ = _compute_chunk_delay_days(
            days_offset=5, days_count=3, week_number=3,
            pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
            for_failed_retry=True,
        )

    assert delay == 4, "Retry de chunk failed: delay = days_offset - 1"


@patch("cron_tasks.CHUNK_LEARNING_MODE", "strict")
def test_strict_mode_first_chunk_offset_zero_has_zero_delay():
    """Chunk inicial (offset 0) corre inmediatamente en strict."""
    from cron_tasks import _compute_chunk_delay_days

    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=0, days_count=3, week_number=1,
        pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
    )

    assert delay == 0


# ---------------------------------------------------------------------------
# 2. Modo "safety_margin": delay = days_offset - ceil(days_count/2)
# ---------------------------------------------------------------------------
@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_safety_margin_advances_chunk_by_half_of_days_count():
    """Chunk de 3 días con offset=3 → delay=2 (1 día de adelanto, ceil(3/2)=2)."""
    from cron_tasks import _compute_chunk_delay_days

    delay, mode, _, _ = _compute_chunk_delay_days(
        days_offset=3, days_count=3, week_number=2,
        pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
    )

    assert mode == "safety_margin"
    assert delay == 3 - math.ceil(3 / 2)
    assert delay == 1, "Chunk 3d con offset 3 se adelanta a delay=1 en safety_margin"


@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_safety_margin_advances_4day_chunk_by_two_days():
    """Chunk de 4 días con offset=4 → delay=2 (2 días de adelanto)."""
    from cron_tasks import _compute_chunk_delay_days

    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=4, days_count=4, week_number=2,
        pipeline_snapshot={"totalDays": 8}, chunk_kind="initial_plan",
    )

    assert delay == 4 - math.ceil(4 / 2)
    assert delay == 2


@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_safety_margin_offset_zero_has_zero_delay():
    """max(0, ...) garantiza que offset=0 no produce delay negativo."""
    from cron_tasks import _compute_chunk_delay_days

    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=0, days_count=3, week_number=1,
        pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
    )

    assert delay == 0


# ---------------------------------------------------------------------------
# 3. Modo "safety_margin" + plan ≥15d: heurística "GAP B" para chunk final
# ---------------------------------------------------------------------------
@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_safety_margin_plan_15d_final_chunk_advanced_by_three_days():
    """Plan 15d, chunk final → delay = days_offset - 3 (heurística GAP B)."""
    from cron_tasks import _compute_chunk_delay_days
    from constants import PLAN_CHUNK_SIZE

    total_days = 15
    total_weeks = math.ceil(total_days / PLAN_CHUNK_SIZE)
    final_week = total_weeks  # último chunk
    days_offset = (final_week - 1) * PLAN_CHUNK_SIZE  # offset del último chunk

    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=days_offset, days_count=PLAN_CHUNK_SIZE, week_number=final_week,
        pipeline_snapshot={"totalDays": total_days}, chunk_kind="initial_plan",
    )

    assert delay == max(0, days_offset - 3), \
        f"Chunk final de plan 15d debe usar heurística -3; obtuvo delay={delay}"


@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_safety_margin_plan_30d_final_chunk_advanced_by_three_days():
    """Mismo comportamiento para planes más largos."""
    from cron_tasks import _compute_chunk_delay_days
    from constants import PLAN_CHUNK_SIZE

    total_days = 30
    total_weeks = math.ceil(total_days / PLAN_CHUNK_SIZE)
    final_week = total_weeks
    days_offset = (final_week - 1) * PLAN_CHUNK_SIZE

    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=days_offset, days_count=PLAN_CHUNK_SIZE, week_number=final_week,
        pipeline_snapshot={"totalDays": total_days}, chunk_kind="initial_plan",
    )

    assert delay == max(0, days_offset - 3)


@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_safety_margin_plan_under_15d_does_not_apply_final_chunk_heuristic():
    """Plan 7d (< 15) NO aplica el adelanto extra de 3 días."""
    from cron_tasks import _compute_chunk_delay_days

    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=3, days_count=4, week_number=2,
        pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
    )

    # Aplica solo la fórmula básica de safety_margin: max(0, 3 - ceil(4/2)) = max(0, 1) = 1
    assert delay == 1


@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_safety_margin_plan_15d_non_final_chunk_uses_basic_formula():
    """Chunk intermedio (no el final) de plan 15d usa la fórmula básica."""
    from cron_tasks import _compute_chunk_delay_days

    # Plan 15d → 5 chunks (week 1..5). week 2 NO es final (final >= 5-1 = 4 → weeks 4 y 5).
    # Probamos week 2 (claramente intermedio).
    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=3, days_count=3, week_number=2,
        pipeline_snapshot={"totalDays": 15}, chunk_kind="initial_plan",
    )

    # Fórmula básica: max(0, 3 - ceil(3/2)) = max(0, 1) = 1.
    assert delay == 1


# ---------------------------------------------------------------------------
# 4. Clamp y degradación defensiva
# ---------------------------------------------------------------------------
@patch("cron_tasks.CHUNK_LEARNING_MODE", "strict")
def test_delay_clamped_to_180_days_max():
    """`min(delay_days, 180)` previene overflow en planes hipotéticamente largos."""
    from cron_tasks import _compute_chunk_delay_days

    delay, _, _, _ = _compute_chunk_delay_days(
        days_offset=500, days_count=3, week_number=200,
        pipeline_snapshot={"totalDays": 500}, chunk_kind="initial_plan",
    )

    assert delay == 180


@patch("cron_tasks.CHUNK_LEARNING_MODE", "definitely_not_a_valid_mode")
def test_unknown_mode_falls_through_to_safety_margin_branch():
    """Cualquier valor no-'strict' cae al else (la rama safety_margin). El validador
    de constants.py normaliza al boot; este test cubre el camino interno por si
    alguien patchea el módulo en runtime con un valor ad-hoc."""
    from cron_tasks import _compute_chunk_delay_days

    delay, mode, _, _ = _compute_chunk_delay_days(
        days_offset=3, days_count=3, week_number=2,
        pipeline_snapshot={"totalDays": 7}, chunk_kind="initial_plan",
    )

    assert mode == "definitely_not_a_valid_mode"
    # Aplica fórmula de safety_margin: 3 - ceil(3/2) = 1.
    assert delay == 1


# ---------------------------------------------------------------------------
# 5. Determinismo: mismas inputs → mismo output
# ---------------------------------------------------------------------------
@patch("cron_tasks.CHUNK_LEARNING_MODE", "safety_margin")
def test_same_inputs_produce_same_output():
    from cron_tasks import _compute_chunk_delay_days

    args = dict(
        days_offset=6, days_count=4, week_number=3,
        pipeline_snapshot={"totalDays": 15}, chunk_kind="initial_plan",
    )
    a = _compute_chunk_delay_days(**args)
    b = _compute_chunk_delay_days(**args)

    assert a == b


# ---------------------------------------------------------------------------
# 6. Constante validada al boot: solo strict/safety_margin
# ---------------------------------------------------------------------------
def test_constants_module_normalizes_invalid_env_to_strict():
    """Verificación del contrato de constants.py: cualquier env var no-válida
    queda como 'strict' tras el boot."""
    import importlib

    with patch.dict(os.environ, {"CHUNK_LEARNING_MODE": "totally_bogus"}, clear=False):
        import constants as _c
        importlib.reload(_c)
        assert _c.CHUNK_LEARNING_MODE == "strict"

    # Re-cargo con un valor válido para no contaminar otros tests.
    with patch.dict(os.environ, {"CHUNK_LEARNING_MODE": "strict"}, clear=False):
        import constants as _c
        importlib.reload(_c)
