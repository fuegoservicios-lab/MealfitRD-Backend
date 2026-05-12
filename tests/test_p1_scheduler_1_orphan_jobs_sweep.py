"""[P1-SCHEDULER-1 · 2026-05-12] Sexto sweep en
`_resolve_stale_scheduler_alerts`: orphan job_id detection.

Bug observado (audit 2026-05-11):
    Los sweeps existentes (TTL standard, TTL one-off UUID, cascade
    parent-stabilization, cascade hard-cap, missed-hard-cap) cierran
    alerts por edad o por estado del parent. NINGUNO cubre el caso
    "job_id que ya NO existe en el scheduler vivo" (renombrado entre
    deploys, removido de `register_plan_chunk_scheduler`, one-off no-UUID
    completado). El listener EVENT_JOB_EXECUTED (P1-NEW-2) nunca dispara
    para esos jobs → la alert sigue abierta 12-24h hasta el TTL → alert
    fatigue.

Fix:
    Sexto sweep snapshot `scheduler.get_jobs()` y resuelve alerts cuyos
    job_id NO están en el set vivo. Cierra inmediatamente (no TTL-bound).
    Kill switch via `MEALFIT_SCHEDULER_ORPHAN_SWEEP_ENABLED` (default True).

Lo que este test enforza:
    A) Anchor `P1-SCHEDULER-1` presente en `cron_tasks.py`.
    B) Función `_resolve_stale_scheduler_alerts` contiene una invocación
       a `scheduler.get_jobs()` (vía lazy import de `app`).
    C) Knob `MEALFIT_SCHEDULER_ORPHAN_SWEEP_ENABLED` aparece como gate.
    D) El tick observable `_scheduler_alerts_sweep_tick` incluye
       `swept_orphan_jobs` en metadata.
    E) La key del prefijo (`scheduler_missed_` y `scheduler_error_`) se
       strip-ea para parsear el job_id ANTES del SELECT contra
       active_job_ids.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _isolate_sweep(src: str) -> str:
    """Aísla `_resolve_stale_scheduler_alerts` hasta la siguiente `def `."""
    m = re.search(
        r"def\s+_resolve_stale_scheduler_alerts\b(.*?)(?=^def\s+\w)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None, "Función _resolve_stale_scheduler_alerts no encontrada."
    return m.group(1)


def test_a_anchor_present(cron_src: str):
    sweep = _isolate_sweep(cron_src)
    assert "P1-SCHEDULER-1" in sweep, (
        "P1-SCHEDULER-1: anchor desapareció del sweep orphan. Restaurar "
        "el bloque comentario que lo declara."
    )


def test_b_get_jobs_invoked(cron_src: str):
    sweep = _isolate_sweep(cron_src)
    assert "scheduler.get_jobs(" in sweep or ".get_jobs()" in sweep, (
        "P1-SCHEDULER-1: el sweep no invoca `scheduler.get_jobs()`. "
        "Sin ese snapshot no hay forma de detectar orphan job_ids."
    )


def test_c_kill_switch_knob_present(cron_src: str):
    sweep = _isolate_sweep(cron_src)
    assert "MEALFIT_SCHEDULER_ORPHAN_SWEEP_ENABLED" in sweep, (
        "P1-SCHEDULER-1: knob de kill-switch ausente. Restaurar para "
        "poder desactivar el sweep sin redeploy si un refactor del "
        "scheduler rompe la fiabilidad de get_jobs()."
    )


def test_d_tick_metadata_includes_orphan_count(cron_src: str):
    sweep = _isolate_sweep(cron_src)
    assert "swept_orphan_jobs" in sweep, (
        "P1-SCHEDULER-1: el tick observable `_scheduler_alerts_sweep_tick` "
        "no incluye `swept_orphan_jobs` en metadata. Imposible correlacionar "
        "post-mortem el efecto del orphan sweep."
    )


def test_e_alert_key_prefixes_stripped(cron_src: str):
    """El sweep parsea el job_id del alert_key strip-eando el prefix."""
    sweep = _isolate_sweep(cron_src)
    assert "scheduler_missed_" in sweep and "scheduler_error_" in sweep, (
        "P1-SCHEDULER-1: el sweep no menciona los prefijos canónicos "
        "`scheduler_missed_` / `scheduler_error_` — sin esos no puede "
        "parsear job_id desde alert_key."
    )


def test_f_excludes_cascade_parent(cron_src: str):
    """`scheduler_cascade_missed` NO debe entrar al orphan sweep — es
    singleton, no per-job; su cierre lo maneja el cascade sweep #4 (hard
    cap) o #3 (stabilization)."""
    sweep = _isolate_sweep(cron_src)
    # Buscar el bloque del orphan sweep específicamente.
    orphan_block = re.search(
        r"P1-SCHEDULER-1.*?(?=\[P2-B-OBS|\Z)", sweep, re.DOTALL
    )
    assert orphan_block is not None, "Bloque orphan sweep no aislable."
    block = orphan_block.group(0)
    # Debe contener una exclusión explícita.
    assert (
        "scheduler_cascade_missed" in block
        and ("alert_key <> 'scheduler_cascade_missed'" in block
             or "!= 'scheduler_cascade_missed'" in block)
    ), (
        "P1-SCHEDULER-1: el orphan sweep debe excluir `scheduler_cascade_missed`. "
        "Sino podría resolver el parent prematuramente."
    )
