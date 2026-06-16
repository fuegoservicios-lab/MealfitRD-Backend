"""[P2-NEW-10 · 2026-05-11] Auto-resolve para `plan_quality_degraded:*`
y `post_swap_critical_divergence_*`.

Bug original (audit 2026-05-11):
    Ambos alert_keys eran modelo `Auto (implicit) + Manual cleanup`.
    Un usuario con 3 regen degradadas + 1 exitosa dejaba 3 alerts
    huérfanas hasta cleanup manual SRE → alert fatigue creciente.

Fix:
    Nueva función `_resolve_stale_plan_quality_alerts` registrada como
    cron (interval default 60min) hace 2 sweeps:
      1. `plan_quality_degraded:*`: cierra si existe meal_plan posterior
         del MISMO user_id con generation_status='complete' AND sin
         `_is_fallback` AND sin `_review_failed_but_delivered`.
      2. `post_swap_critical_divergence_*`: cierra por edad >=
         MEALFIT_POST_SWAP_DIVERGENCE_AUTO_RESOLVE_HOURS (default 24h,
         hard-floor 6h alineado con cooldown del productor).

    Kill switch `MEALFIT_PLAN_QUALITY_AUTO_RESOLVE_ENABLED=True` default.
    Tick observable `_plan_quality_alerts_sweep_tick`.

Estrategia del test (parser-based sobre cron_tasks.py):
    1. La función `_resolve_stale_plan_quality_alerts` existe.
    2. Lee el knob via `_env_bool` con default True.
    3. Lee el TTL post_swap via `_env_int` con default 24 y hard-floor 6.
    4. Hace UPDATE plan_quality_degraded con la condición correcta.
    5. Hace UPDATE post_swap por edad absoluta.
    6. Emite tick `_plan_quality_alerts_sweep_tick`.
    7. Registrada en `register_plan_chunk_scheduler` con id estable.
    8. CLAUDE.md fila `plan_quality_degraded` menciona P2-NEW-10.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_FP = _REPO_ROOT / "backend" / "cron_tasks.py"
_CLAUDE_FP = _REPO_ROOT / "CLAUDE.md"
# [P3-CLAUDEMD-CAP refactor] La tabla canónica de ~32 `alert_key` (con las
# filas `plan_quality_degraded` y `post_swap_critical_divergence_*` y sus
# markers P-fix) se movió de CLAUDE.md a este doc dedicado para respetar el
# cap de tamaño de CLAUDE.md. La doc de la sección "Política de system_alerts
# resolution" en CLAUDE.md apunta explícitamente a este archivo.
_ALERT_TABLE_FP = _REPO_ROOT / "backend" / "docs" / "system_alerts_resolution_table.md"


@pytest.fixture(scope="module")
def src() -> str:
    return _CRON_FP.read_text(encoding="utf-8")


def test_function_exists(src: str):
    assert "def _resolve_stale_plan_quality_alerts(" in src, (
        "P2-NEW-10 regresión: `_resolve_stale_plan_quality_alerts` ya no "
        "existe. Sin ella, los alerts plan_quality_degraded:* quedan "
        "abiertos indefinidamente."
    )


def test_kill_switch_default_true(src: str):
    """Knob `MEALFIT_PLAN_QUALITY_AUTO_RESOLVE_ENABLED` con default True."""
    func_start = src.find("def _resolve_stale_plan_quality_alerts(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    pattern = re.compile(
        r'_env_bool\(\s*[\'"]MEALFIT_PLAN_QUALITY_AUTO_RESOLVE_ENABLED[\'"]\s*,\s*True\s*\)',
    )
    assert pattern.search(body), (
        "P2-NEW-10 regresión: el knob "
        "MEALFIT_PLAN_QUALITY_AUTO_RESOLVE_ENABLED ya no se lee con "
        "default True. Si se removió, perdemos el kill switch."
    )


def test_post_swap_ttl_with_hard_floor(src: str):
    """TTL post_swap default 24h, hard-floor 6h (cooldown del productor)."""
    func_start = src.find("def _resolve_stale_plan_quality_alerts(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    assert re.search(
        r'_env_int\(\s*[\'"]MEALFIT_POST_SWAP_DIVERGENCE_AUTO_RESOLVE_HOURS[\'"]\s*,\s*24\s*\)',
        body,
    ), (
        "P2-NEW-10 regresión: TTL post_swap default ya no es 24h."
    )
    # Hard floor 6h (cooldown del productor).
    assert re.search(r"post_swap_h\s*<\s*6", body), (
        "P2-NEW-10 regresión: el hard-floor de 6h sobre post_swap_h ya "
        "no existe. Sin él, un knob mal configurado a <6h causaría "
        "re-emisión inmediata por el productor (cooldown=6h)."
    )


def test_plan_quality_sweep_semantic_condition(src: str):
    """El sweep #1 cierra si existe meal_plan posterior limpio."""
    func_start = src.find("def _resolve_stale_plan_quality_alerts(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    # Tokens críticos del UPDATE.
    for tok in (
        "alert_key LIKE 'plan_quality_degraded:%%'",
        "split_part(a.alert_key, ':', 2)",
        "generation_status",
        "'complete'",
        "_is_fallback",
        "_review_failed_but_delivered",
    ):
        assert tok in body, (
            f"P2-NEW-10 regresión: el sweep #1 ya no contiene el token "
            f"`{tok}`. Sin él, la semántica `plan posterior limpio` se "
            f"rompe y el sweep podría cerrar alerts incorrectamente."
        )


def test_post_swap_sweep_by_age(src: str):
    """El sweep #2 cierra post_swap_critical_divergence_* por edad."""
    func_start = src.find("def _resolve_stale_plan_quality_alerts(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    assert "alert_key LIKE 'post_swap_critical_divergence_%%'" in body, (
        "P2-NEW-10 regresión: el sweep #2 ya no filtra por "
        "`post_swap_critical_divergence_*`."
    )
    assert "make_interval(hours => %s)" in body, (
        "P2-NEW-10 regresión: el sweep #2 ya no usa `make_interval(hours...)` "
        "para el filtro por edad."
    )


def test_tick_observable_emitted(src: str):
    """El tick `_plan_quality_alerts_sweep_tick` se emite SIEMPRE."""
    func_start = src.find("def _resolve_stale_plan_quality_alerts(")
    func_end = src.find("\ndef ", func_start + 1)
    body = src[func_start:func_end]
    assert "_plan_quality_alerts_sweep_tick" in body, (
        "P2-NEW-10 regresión: el tick observable "
        "`_plan_quality_alerts_sweep_tick` ya no se emite. Sin él, no "
        "podemos distinguir `cron vivo + 0 hits` de `cron caído`."
    )
    # Tick emitido tanto en el kill-switch path como en el sweep normal.
    tick_count = body.count("_plan_quality_alerts_sweep_tick")
    assert tick_count >= 2, (
        f"P2-NEW-10 regresión: solo {tick_count} menciones del tick "
        "(esperado ≥2 — uno en kill-switch, uno en normal path)."
    )


def test_registered_in_scheduler(src: str):
    """El cron debe estar registrado en `register_plan_chunk_scheduler`."""
    pattern = re.compile(
        r'if\s+not\s+scheduler\.get_job\(\s*[\'"]resolve_stale_plan_quality_alerts[\'"]\s*\)',
    )
    assert pattern.search(src), (
        "P2-NEW-10 regresión: el cron `resolve_stale_plan_quality_alerts` "
        "ya no se registra en `register_plan_chunk_scheduler`. Sin esto, "
        "el sweep nunca se ejecuta."
    )
    # También debe usar _add_job_jittered con la función y un knob de intervalo.
    sched_idx = src.find('"resolve_stale_plan_quality_alerts"')
    assert sched_idx > 0
    window = src[max(0, sched_idx - 600):sched_idx + 400]
    assert "_resolve_stale_plan_quality_alerts" in window, (
        "P2-NEW-10 regresión: el bloque del scheduler ya no referencia la "
        "función `_resolve_stale_plan_quality_alerts`."
    )
    assert "MEALFIT_PLAN_QUALITY_AUTO_RESOLVE_INTERVAL_MIN" in window, (
        "P2-NEW-10 regresión: el intervalo del cron ya no está controlado "
        "por el knob `MEALFIT_PLAN_QUALITY_AUTO_RESOLVE_INTERVAL_MIN`."
    )


def test_claude_md_documents_p2_new_10():
    """La tabla canónica de alerts debe mencionar P2-NEW-10 en las 2 filas.

    [P3-CLAUDEMD-CAP refactor] La tabla de ~32 `alert_key` se movió de
    CLAUDE.md a `backend/docs/system_alerts_resolution_table.md`. Las filas
    `plan_quality_degraded` y `post_swap_critical_divergence_*` (con el marker
    P2-NEW-10 que las conecta al cron auto-resolve) viven ahora ahí. Leemos
    ese doc canónico, no CLAUDE.md (que solo retiene la invariante I5)."""
    alert_table = _ALERT_TABLE_FP.read_text(encoding="utf-8")
    # La fila plan_quality_degraded debe mencionar el marker.
    pq_idx = alert_table.find("plan_quality_degraded")
    assert pq_idx > 0, (
        "P2-NEW-10 regresión: la fila `plan_quality_degraded` desapareció de "
        "la tabla canónica de alerts."
    )
    row = alert_table[pq_idx:pq_idx + 2500]
    assert "P2-NEW-10" in row, (
        "P2-NEW-10 regresión: la fila `plan_quality_degraded` en la tabla "
        "canónica de alerts ya no menciona el marker. Sin él, el modelo de "
        "resolution documentado (Auto explicit) no se conecta al cron que lo "
        "implementa."
    )
    # post_swap row debe estar adyacente y también mencionar P2-NEW-10.
    post_idx = alert_table.find("post_swap_critical_divergence")
    assert post_idx > 0, (
        "P2-NEW-10 regresión: la fila `post_swap_critical_divergence_*` "
        "desapareció de la tabla canónica de alerts."
    )
    post_row = alert_table[post_idx:post_idx + 1500]
    assert "P2-NEW-10" in post_row, (
        "P2-NEW-10 regresión: la fila `post_swap_critical_divergence_*` "
        "no menciona el marker. Sin él, alguien podría asumir que la "
        "alerta sigue manual y duplicar lógica."
    )
