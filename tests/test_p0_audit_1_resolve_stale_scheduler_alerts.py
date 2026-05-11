"""[P0-AUDIT-1 · 2026-05-10] Defense-in-depth sweep para `scheduler_missed_*`
y `scheduler_error_*` alerts huérfanas.

Bug original (audit 2026-05-10):
    Producción acumulaba 26 alerts `scheduler_missed_*` unresolved (5+
    horas) pese a P1-NEW-2 (listener auto-resolve via `EVENT_JOB_EXECUTED`).
    El listener funciona para jobs recurrentes que vuelven a ejecutar,
    pero NO cubre:
      - Jobs one-off con `job_id` UUID (no recurren — ej. observado:
        `scheduler_missed_f71bdd2b0a5a408b9aac5b2b30df3989`).
      - Jobs renombrados o eliminados entre deploys (alert_key viejo
        no matchea job_id nuevo).
      - Race entre MISSED y EXECUTED (worker reinició entre eventos).
      - Deploy lag: binario aún sin P1-NEW-2.

Fix:
    1. Nueva función `_resolve_stale_scheduler_alerts` en cron_tasks.py
       que UPDATEea filas con `resolved_at IS NULL` cuando
       `triggered_at < NOW() - MEALFIT_SCHEDULER_ALERT_TTL_H` y
       `alert_key LIKE 'scheduler_missed_%' OR 'scheduler_error_%'`.
    2. Registrada en `register_plan_chunk_scheduler` cada
       `MEALFIT_SCHEDULER_ALERT_SWEEP_INTERVAL_MIN` (default 60 min).
    3. Knobs auto-registrados via `_env_int`.

Estrategia del test (parser estático sobre cron_tasks.py):
    1. Función existe con la signatura esperada.
    2. UPDATE filtra por namespace y TTL via `make_interval`.
    3. UPDATE filtra por `resolved_at IS NULL` (no pisa cerradas).
    4. NO toca `scheduler_cascade_missed` (que es critical, manual).
    5. Registrada en `register_plan_chunk_scheduler` con id estable.
    6. Knobs `MEALFIT_SCHEDULER_ALERT_TTL_H` y
       `MEALFIT_SCHEDULER_ALERT_SWEEP_INTERVAL_MIN` referenciados.
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


def _extract_function_body(src: str, name: str) -> str:
    """Extrae el cuerpo de una función top-level hasta el siguiente
    `def ` top-level. Parser estático que evita imports."""
    m = re.search(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"No se encontró función top-level `{name}` en cron_tasks.py."
    return m.group(1)


def test_resolve_stale_scheduler_alerts_function_exists(cron_src: str):
    """`_resolve_stale_scheduler_alerts` debe existir como función
    top-level — sin ella el sweep no corre y las alerts huérfanas
    siguen acumulando."""
    pattern = re.compile(
        r"^def\s+_resolve_stale_scheduler_alerts\s*\(\s*\)",
        re.MULTILINE,
    )
    assert pattern.search(cron_src), (
        "P0-AUDIT-1 regresión: `_resolve_stale_scheduler_alerts` "
        "ya no existe en cron_tasks.py. Sin ella, las alerts huérfanas "
        "del scheduler (P1-NEW-2 long-tail) se acumulan indefinidamente."
    )


def test_sweep_filters_only_scheduler_namespace(cron_src: str):
    """El UPDATE debe filtrar por `alert_key LIKE 'scheduler_missed_%'`
    Y/O `'scheduler_error_%'`. Tocar otros namespaces (ej.
    `inventory_rpc_fallback`, `corrupted_plan_data`) sería peligroso."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert "scheduler_missed_" in body, (
        "P0-AUDIT-1 regresión: sweep no cubre `scheduler_missed_*`."
    )
    assert "scheduler_error_" in body, (
        "P0-AUDIT-1 regresión: sweep no cubre `scheduler_error_*`. "
        "Dejar errores stuck deja triage cost creciente."
    )


def test_sweep_does_not_touch_cascade_or_other_alerts(cron_src: str):
    """El sweep NO debe tocar `scheduler_cascade_missed` (critical,
    requiere revisión manual) ni otros alert_keys (chunk_*,
    inventory_*, etc.). Defensa contra `LIKE 'scheduler_%'` over-broad."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # El sweep no debe matchear el wildcard genérico 'scheduler_%'.
    assert "'scheduler_%%'" not in body and '"scheduler_%%"' not in body, (
        "P0-AUDIT-1 regresión: sweep usa wildcard genérico "
        "`scheduler_%` que mataría `scheduler_cascade_missed` "
        "(critical, manual). Restringir a `scheduler_missed_%` y "
        "`scheduler_error_%` explícitos."
    )


def test_sweep_filters_unresolved_only(cron_src: str):
    """UPDATE debe incluir `resolved_at IS NULL` — sin esto pisamos
    timestamps de alerts cerradas manualmente."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    pattern = re.compile(r"resolved_at\s+IS\s+NULL", re.IGNORECASE)
    assert pattern.search(body), (
        "P0-AUDIT-1 regresión: sweep no filtra `resolved_at IS NULL`. "
        "Sin esto, cada run pisa timestamps de alerts ya cerradas "
        "manualmente — perdemos historial real de resolución."
    )


def test_sweep_uses_ttl_with_make_interval(cron_src: str):
    """El TTL debe aplicarse via `make_interval(hours => %s)` con el
    parámetro del knob. Hardcodear el TTL (ej. INTERVAL '24 hours')
    rompe la configurabilidad via env var."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    pattern = re.compile(r"make_interval\(\s*hours\s*=>", re.IGNORECASE)
    assert pattern.search(body), (
        "P0-AUDIT-1 regresión: sweep no usa `make_interval(hours => ...)`. "
        "Sin parametrizar el TTL, el knob "
        "`MEALFIT_SCHEDULER_ALERT_TTL_H` no tiene efecto."
    )


def test_sweep_reads_ttl_knob(cron_src: str):
    """El knob `MEALFIT_SCHEDULER_ALERT_TTL_H` debe leerse en
    `_resolve_stale_scheduler_alerts` via `_env_int` (auto-registro
    en `_KNOBS_REGISTRY`)."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    pattern = re.compile(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_ALERT_TTL_H["\']\s*,'
    )
    assert pattern.search(body), (
        "P0-AUDIT-1 regresión: `MEALFIT_SCHEDULER_ALERT_TTL_H` ya no "
        "se lee via `_env_int`. Sin esto, el knob no aparece en "
        "`/admin/knobs` y rollback en caliente no funciona."
    )


def test_sweep_registered_in_scheduler(cron_src: str):
    """El cron debe estar registrado en `register_plan_chunk_scheduler`
    con `id='resolve_stale_scheduler_alerts'` (id estable para
    APScheduler matching)."""
    # Localizar la función `register_plan_chunk_scheduler`.
    m = re.search(
        r"^def\s+register_plan_chunk_scheduler\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        cron_src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "register_plan_chunk_scheduler no encontrada."
    body = m.group(1)
    pattern = re.compile(
        r"id\s*=\s*[\"']resolve_stale_scheduler_alerts[\"']",
    )
    assert pattern.search(body), (
        "P0-AUDIT-1 regresión: cron `resolve_stale_scheduler_alerts` "
        "ya no se registra en `register_plan_chunk_scheduler`. "
        "Sin registro, APScheduler no lo dispara — los sweeps "
        "no corren y las alerts huérfanas siguen acumulando."
    )


def test_sweep_uses_jittered_helper(cron_src: str):
    """Registro debe usar `_add_job_jittered` (SSOT P0-NEW-2) para no
    re-introducir bursts del scheduler — el sweep mismo no debe
    contribuir a la cascada que arregla."""
    m = re.search(
        r"^def\s+register_plan_chunk_scheduler\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        cron_src,
        re.DOTALL | re.MULTILINE,
    )
    assert m
    body = m.group(1)
    # Buscar el bloque que registra `resolve_stale_scheduler_alerts` y
    # confirmar que usa _add_job_jittered antes.
    sweep_block = re.search(
        r"_add_job_jittered\([^)]*_resolve_stale_scheduler_alerts",
        body,
        re.DOTALL,
    )
    assert sweep_block, (
        "P0-AUDIT-1 regresión: el registro del sweep no pasa por "
        "`_add_job_jittered`. Sin jitter, el cron compite con los "
        "otros crons del minuto y puede contribuir a `scheduler_missed_*`."
    )


def test_sweep_interval_knob_referenced(cron_src: str):
    """El interval debe ser configurable via
    `MEALFIT_SCHEDULER_ALERT_SWEEP_INTERVAL_MIN`."""
    pattern = re.compile(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_ALERT_SWEEP_INTERVAL_MIN["\']\s*,'
    )
    assert pattern.search(cron_src), (
        "P0-AUDIT-1 regresión: `MEALFIT_SCHEDULER_ALERT_SWEEP_INTERVAL_MIN` "
        "no se lee. Sin knob para el interval, ajustar la frecuencia "
        "del sweep en producción requiere redeploy."
    )


def test_sweep_handles_exceptions_gracefully(cron_src: str):
    """Sweep debe tener try/except — un fallo del sweep NO puede
    pausar el resto de los crons."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert "except" in body, (
        "P0-AUDIT-1 regresión: sweep sin manejo de excepciones. Un "
        "fallo del UPDATE (DB blip, lock contention) propaga y "
        "APScheduler reportará el cron como ERROR — ironía: el "
        "sweep que arregla scheduler_error_* genera uno nuevo."
    )
