"""[P2-NEW-16 · 2026-05-11] GC sweep para `plan_quality_emit_lock:*`.

Bug original (re-audit 2026-05-11):
    P3-NEW-9 escribe `app_kv_store.plan_quality_emit_lock:<user_id>:<plan_id>`
    en cada emit de `plan_quality_degraded`. La fila se reusa por
    (user × plan) — bounded — pero planes eliminados (P0-HIST-1 restore)
    dejan keys huérfanas. Sin techo natural a largo plazo.

Fix:
    Nueva función `_sweep_stale_emit_locks_kv` registrada como cron
    cada `MEALFIT_PLAN_QUALITY_EMIT_LOCK_SWEEP_INTERVAL_MIN` (default 360min).
    DELETE rows con `key LIKE 'plan_quality_emit_lock:%' AND updated_at <
    NOW() - INTERVAL N hours` (default 24h, clamp [1, 168]).

Estrategia del test (parser-based):
    1. Función `_sweep_stale_emit_locks_kv` existe.
    2. Lee TTL via `_env_int` con default 24h y clamp [1, 168].
    3. DELETE filtra por `key LIKE 'plan_quality_emit_lock:%'` AND
       `updated_at < NOW() - make_interval(hours => %s)`.
    4. Tick observable `_sweep_stale_emit_locks_kv_tick`.
    5. Best-effort try/except.
    6. Cron registrado con id `sweep_stale_emit_locks_kv`.
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


def test_function_defined(src: str):
    assert "def _sweep_stale_emit_locks_kv(" in src, (
        "P2-NEW-16 regresión: `_sweep_stale_emit_locks_kv` ya no está "
        "definido. Sin él, KV crece sin techo."
    )


def test_ttl_env_with_clamp(src: str):
    func_start = src.find("def _sweep_stale_emit_locks_kv(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    assert re.search(
        r'_env_int\(\s*[\'"]MEALFIT_PLAN_QUALITY_EMIT_LOCK_TTL_HOURS[\'"]\s*,\s*24\s*\)',
        body,
    ), (
        "P2-NEW-16 regresión: knob TTL default ya no es 24h. Cambiar "
        "requiere actualizar este test."
    )
    # Hard clamp [1, 168].
    assert re.search(r"max\(1,\s*min\(.*?,\s*168\)\)", body), (
        "P2-NEW-16 regresión: el TTL ya no tiene clamp [1, 168]. Sin "
        "clamp, un knob mal configurado podría borrar todas las locks "
        "frescas (ttl=0) o nunca (ttl=999999)."
    )


def test_delete_filters_correctly(src: str):
    func_start = src.find("def _sweep_stale_emit_locks_kv(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    assert "DELETE FROM app_kv_store" in body, (
        "P2-NEW-16 regresión: el sweep ya no hace DELETE. Si se cambió "
        "a UPDATE (reset), las keys persisten — bloat continúa."
    )
    assert "key LIKE 'plan_quality_emit_lock:%%'" in body, (
        "P2-NEW-16 regresión: el filtro de keys ya no es "
        "`plan_quality_emit_lock:%%`. Sin él, el DELETE podría borrar "
        "otras keys (CB rows, etc.)."
    )
    assert "make_interval(hours => %s)" in body, (
        "P2-NEW-16 regresión: el filtro de edad ya no usa "
        "`make_interval(hours => %s)`."
    )


def test_tick_observable(src: str):
    func_start = src.find("def _sweep_stale_emit_locks_kv(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    assert "_sweep_stale_emit_locks_kv_tick" in body, (
        "P2-NEW-16 regresión: el tick observable ya no se emite. Sin él, "
        "no podemos distinguir `cron vivo + 0 deletes` de `cron caído`."
    )


def test_best_effort_try_except(src: str):
    func_start = src.find("def _sweep_stale_emit_locks_kv(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    assert "try:" in body and "except Exception" in body, (
        "P2-NEW-16 regresión: el sweep ya no está envuelto en "
        "try/except. Sin él, un fallo del DELETE crashea el cron "
        "tick entero."
    )


def test_cron_registered(src: str):
    pattern = re.compile(
        r'if\s+not\s+scheduler\.get_job\(\s*[\'"]sweep_stale_emit_locks_kv[\'"]\s*\)',
    )
    assert pattern.search(src), (
        "P2-NEW-16 regresión: el cron `sweep_stale_emit_locks_kv` ya no "
        "se registra en `register_plan_chunk_scheduler`. Sin esto, "
        "el sweep nunca se ejecuta."
    )
    # Y el knob de intervalo debe estar.
    assert "MEALFIT_PLAN_QUALITY_EMIT_LOCK_SWEEP_INTERVAL_MIN" in src, (
        "P2-NEW-16 regresión: el knob de intervalo no aparece en el "
        "registro del cron."
    )
