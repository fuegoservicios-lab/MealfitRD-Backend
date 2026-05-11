"""[P2-NEW-7 · 2026-05-11] Misfire grace amplio para crones agregadores
delay-tolerant (P3-B coherence metrics, sweep alerts, GC orphans).

Bug original (audit 2026-05-11):
    Audit DB 2026-05-11 detectó que crones agregadores tipo
    `aggregate_coherence_block_history_metrics` ocasionalmente
    aparecían en `system_alerts.scheduler_missed_*` aunque su
    output es delay-tolerant: un agregador horario que corre 5min
    tarde sigue produciendo telemetría válida.

    El default global `MEALFIT_SCHEDULER_MISFIRE_GRACE_S` se calibró
    para chunks que requieren ejecución prompt (cooking timers).
    Aplicar el mismo grace a agregadores genera false-MISSED bajo
    carga del pool.

Fix:
    `_aggregator_misfire_grace_s()` knob helper + `misfire_grace_time`
    explícito aplicado a 3 crones delay-tolerant:
      - aggregate_coherence_block_history_metrics
      - resolve_stale_scheduler_alerts
      - gc_orphan_chunk_telemetry
    (gc_orphan_conversation_summaries P2-NEW-6 también lo usa).

Estrategia del test (parser-based):
    1. Helper `_aggregator_misfire_grace_s` definido.
    2. Knob `MEALFIT_AGGREGATOR_MISFIRE_GRACE_S` referenciado.
    3. Los 3 callsites mencionados usan `misfire_grace_time=...()`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_FP = _REPO_ROOT / "backend" / "cron_tasks.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _CRON_FP.read_text(encoding="utf-8")


def test_aggregator_misfire_grace_helper_defined(src: str):
    """`_aggregator_misfire_grace_s` debe ser una función top-level."""
    assert re.search(
        r"^def\s+_aggregator_misfire_grace_s\s*\(",
        src,
        re.MULTILINE,
    ), (
        "P2-NEW-7 regresión: helper `_aggregator_misfire_grace_s` no "
        "existe. Sin él, los callsites pierden el override del grace."
    )


def test_knob_referenced(src: str):
    """Knob `MEALFIT_AGGREGATOR_MISFIRE_GRACE_S` debe leerse via `_env_int`."""
    assert re.search(
        r"_env_int\(\s*[\"\']MEALFIT_AGGREGATOR_MISFIRE_GRACE_S[\"\']",
        src,
    ), (
        "P2-NEW-7 regresión: knob `MEALFIT_AGGREGATOR_MISFIRE_GRACE_S` "
        "no se lee. Sin knob, no se puede ajustar el grace sin redeploy."
    )


@pytest.mark.parametrize("job_id", [
    "aggregate_coherence_block_history_metrics",
    "resolve_stale_scheduler_alerts",
    "gc_orphan_chunk_telemetry",
])
def test_aggregator_callsite_uses_misfire_grace(src: str, job_id: str):
    """Cada cron agregador delay-tolerant debe registrarse con
    `misfire_grace_time=_aggregator_misfire_grace_s()`."""
    # Localizar el bloque `_add_job_jittered(... id=<job_id> ...)`.
    pattern = re.compile(
        rf"_add_job_jittered\([^)]*id\s*=\s*[\"']{re.escape(job_id)}[\"'][^)]*\)",
        re.DOTALL,
    )
    m = pattern.search(src)
    assert m, (
        f"P2-NEW-7 regresión: job `{job_id}` no se encuentra en "
        "register_plan_chunk_scheduler. ¿Fue renombrado?"
    )
    block = m.group(0)
    assert "misfire_grace_time" in block, (
        f"P2-NEW-7 regresión: cron `{job_id}` ya no especifica "
        "`misfire_grace_time`. Volverá a generar false-MISSED bajo carga."
    )
    assert "_aggregator_misfire_grace_s" in block, (
        f"P2-NEW-7 regresión: cron `{job_id}` usa misfire_grace pero "
        "NO via el helper SSOT `_aggregator_misfire_grace_s()`. "
        "Hardcodear el valor rompe la configurabilidad via knob."
    )
