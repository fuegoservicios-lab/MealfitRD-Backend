"""[P1-COH-BENIGN-SKIP · 2026-05-25] Regression test parser-based.

Ancla el filtro `_BENIGN_SKIP_REASONS` en `_shopping_coherence_alert_job`
para que `skip_reason='below_min_plans'` NO se cuente como failure del
clasificador `_is_failure` (alimenta la alert
`shopping_coherence_alert_job_failures_burst`).

Pre-fix: cualquier `skip_reason != None` se contaba como fallo → alert
ruidosa cuando el cron skippeaba por baja actividad legítima (menos planes
nuevos en 24h que el umbral `min_plans`). En prod llegó a `count=14` antes
del fix.

Las 4 skip_reason restantes (`db_core_import_failed`,
`db_not_initialized`, `fetch_plans_failed`,
`guard_persist_import_failed`) SÍ son fallos reales y se siguen contando.

Detalle: `project_p1_coh_benign_skip_2026_05_25.md`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


def test_benign_skip_frozenset_present(cron_source: str) -> None:
    """`_BENIGN_SKIP_REASONS` debe declarar al menos `below_min_plans`."""
    assert "_BENIGN_SKIP_REASONS" in cron_source, (
        "Falta el frozenset `_BENIGN_SKIP_REASONS` del clasificador P1-COH-BENIGN-SKIP. "
        "Renombrarlo o eliminarlo rompe la defensa contra alert espuria."
    )
    pattern = re.compile(
        r"_BENIGN_SKIP_REASONS\s*=\s*frozenset\(\s*\{[^\}]*\"below_min_plans\"[^\}]*\}\s*\)"
    )
    assert pattern.search(cron_source), (
        "El frozenset `_BENIGN_SKIP_REASONS` debe contener `below_min_plans` literal. "
        "Si se cambia a un knob env-driven, actualizar este test."
    )


def test_is_failure_excludes_benign_skip(cron_source: str) -> None:
    """`_is_failure` DEBE consultar `_BENIGN_SKIP_REASONS` antes de marcar
    `_tick_skip_reason is not None` como fallo."""
    # Match: skip_reason is not None AND skip_reason not in _BENIGN_SKIP_REASONS
    pattern = re.compile(
        r"_tick_skip_reason\s+is\s+not\s+None"
        r"\s*\n?\s*and\s+_tick_skip_reason\s+not\s+in\s+_BENIGN_SKIP_REASONS"
    )
    assert pattern.search(cron_source), (
        "El clasificador `_is_failure` perdió el guard `_tick_skip_reason not in "
        "_BENIGN_SKIP_REASONS`. Revertir el cambio rompería P1-COH-BENIGN-SKIP "
        "(alert espuria volvería)."
    )


def test_marker_anchor_present(cron_source: str) -> None:
    """El comment-marker `[P1-COH-BENIGN-SKIP · 2026-05-25]` debe estar
    presente para que un grep encuentre el origen del filtro."""
    assert "[P1-COH-BENIGN-SKIP · 2026-05-25]" in cron_source, (
        "Tooltip-anchor `[P1-COH-BENIGN-SKIP · 2026-05-25]` desapareció de "
        "cron_tasks.py. Si renombras el marker, actualizar este test."
    )


def test_other_failure_reasons_still_count(cron_source: str) -> None:
    """Las 4 skip_reasons que SÍ son fallos siguen presentes en el código —
    cierra el riesgo de que alguien ampliara `_BENIGN_SKIP_REASONS` a todas
    (vaciando la alert)."""
    real_failures = (
        "db_core_import_failed",
        "db_not_initialized",
        "fetch_plans_failed",
        "guard_persist_import_failed",
    )
    for reason in real_failures:
        assert reason in cron_source, (
            f"skip_reason `{reason}` desapareció de cron_tasks.py. "
            "Si era intencional, actualizar este test + memoria."
        )
    # Y que esos 4 NO estén en _BENIGN_SKIP_REASONS
    for reason in real_failures:
        in_benign = re.search(
            r"_BENIGN_SKIP_REASONS\s*=\s*frozenset\([^)]*"
            + re.escape(reason),
            cron_source,
        )
        assert not in_benign, (
            f"skip_reason `{reason}` fue añadido a `_BENIGN_SKIP_REASONS` — "
            "esto silenciaría un fallo real del cron. Revisar."
        )
