"""[P1-AUDIT-1 · 2026-05-12] Cron `drain_pending_facts_queue` reemplaza el
trigger DB con URL+secret hardcoded.

Pre-fix, la función SECURITY DEFINER `public.trigger_process_pending_facts_webhook`
tenía:

    webhook_url    := 'https://mealfit-rd.vercel.app/api/webhooks/process-pending-facts';
    webhook_secret := 'mealfit_secure_webhook_secret_2026';

ambos en cleartext dentro de pg_proc — visibles a cualquier rol con
`SELECT` sobre `pg_catalog.pg_proc`. La URL apuntaba a Vercel; el backend
ya está en Easypanel/Nixpacks. Cero rotabilidad sin DDL.

Fix:
  - Migración `p1_audit_1_drop_dead_webhook_trigger.sql` dropea
    trigger + función (idempotente).
  - Cron `drain_pending_facts_queue` en `cron_tasks.py` polea la tabla
    cada N min (knob `MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN`, clamp
    [1, 30]) y delega a `process_pending_queue_sync(user_id)` — mismo
    procesador que el webhook usaba.

Lo que este test enforza:
  A) `cron_tasks.py` define la función `_drain_pending_facts_queue`.
  B) `cron_tasks.py::register_plan_chunk_scheduler` registra el job
     `drain_pending_facts_queue` con `_add_job_jittered`.
  C) El cron lee el knob `MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN`
     (defensa contra "fix" que hardcodee el intervalo).
  D) La migración SSOT existe y contiene los DROP statements.
  E) El cuerpo del cron invoca `process_pending_queue_sync` (mismo
     procesador del webhook).
  F) El anchor `P1-AUDIT-1` aparece en el cuerpo del cron — sin él, un
     refactor cosmético pierde el contexto.

Tooltip-anchor sync: `P1-AUDIT-1-DRAIN` en `cron_tasks.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"
_MIGRATIONS_DIR = _BACKEND_ROOT.parent / "supabase" / "migrations"
_MIGRATION_FILE = _MIGRATIONS_DIR / "p1_audit_1_drop_dead_webhook_trigger.sql"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A) La función cron existe.
# ---------------------------------------------------------------------------

def test_a_drain_function_defined(cron_src: str):
    assert re.search(r"^def _drain_pending_facts_queue\b", cron_src, re.MULTILINE), (
        "P1-AUDIT-1 regresión: cron_tasks.py no define "
        "`_drain_pending_facts_queue`. Esta función reemplaza el trigger DB "
        "+ webhook que tenían URL y secret hardcoded en pg_proc."
    )


# ---------------------------------------------------------------------------
# B) El cron está registrado en register_plan_chunk_scheduler.
# ---------------------------------------------------------------------------

def test_b_drain_registered_in_scheduler(cron_src: str):
    # Extraer el cuerpo completo de register_plan_chunk_scheduler.
    m = re.search(
        r"def register_plan_chunk_scheduler\b.*?(?=^def\s)",
        cron_src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, (
        "P1-AUDIT-1: no pude localizar `register_plan_chunk_scheduler` en "
        "cron_tasks.py — estructura del archivo cambió, actualizar este test."
    )
    body = m.group(0)
    assert "_drain_pending_facts_queue" in body, (
        "P1-AUDIT-1 regresión: register_plan_chunk_scheduler no referencia "
        "`_drain_pending_facts_queue`. Sin el registro en el scheduler, el "
        "cron no corre y `pending_facts_queue` acumula entries sin drenar "
        "tras eliminar el trigger DB."
    )
    assert "drain_pending_facts_queue" in body, (
        "P1-AUDIT-1 regresión: el job id `drain_pending_facts_queue` no "
        "aparece en register_plan_chunk_scheduler — el `id` del scheduler "
        "es crítico para idempotencia (`scheduler.get_job(...)`) y para "
        "que `_scheduler_alert_listener` correlacione missed events."
    )
    assert "_add_job_jittered(" in body, (
        "P1-AUDIT-1: register_plan_chunk_scheduler debe usar "
        "`_add_job_jittered` (SSOT de jitter). Búsqueda débil — el block "
        "ya lo usa para los otros crones."
    )


# ---------------------------------------------------------------------------
# C) Knob respetado, no hardcoded.
# ---------------------------------------------------------------------------

def test_c_knob_used_for_interval(cron_src: str):
    """El intervalo debe venir de `MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN`
    (registrado en `_KNOBS_REGISTRY` via `_env_int`), NO hardcoded.
    """
    assert "MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN" in cron_src, (
        "P1-AUDIT-1 regresión: cron_tasks.py no referencia el knob "
        "`MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN`. El intervalo debe ser "
        "operacional (ajustable sin redeploy) y registrado en "
        "`_KNOBS_REGISTRY` — el patrón ya está documentado en CLAUDE.md "
        "sección 'Convenciones del repo'."
    )


def test_c2_knob_clamped(cron_src: str):
    """El knob debe estar clamped a un rango razonable (defensa contra
    operador que setea 0 o 99999)."""
    # Búsqueda flexible: `max(1, min(_facts_drain_interval, 30))` o variante.
    pattern = re.compile(
        r"max\(\s*\d+\s*,\s*min\([^,]*_facts_drain[^,]*,\s*\d+\s*\)\s*\)",
        re.IGNORECASE,
    )
    assert pattern.search(cron_src), (
        "P1-AUDIT-1 regresión: el knob "
        "`MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN` no tiene clamp visible. "
        "Patrón esperado: `max(1, min(_facts_drain_interval, 30))`. Sin "
        "clamp, un operador puede setear 0 → APScheduler rechaza el "
        "interval. O 99999 → cola nunca se drena."
    )


# ---------------------------------------------------------------------------
# D) Migración SSOT existe + drops esperados.
# ---------------------------------------------------------------------------

def test_d_migration_file_exists():
    assert _MIGRATION_FILE.exists(), (
        f"P1-AUDIT-1 regresión: falta la migración SSOT "
        f"`{_MIGRATION_FILE.relative_to(_BACKEND_ROOT.parent)}`. Sin ella, "
        f"un deploy fresh recrearía el trigger con el secret hardcoded "
        f"si alguien re-ejecuta el bootstrap SQL antiguo."
    )


def test_d2_migration_drops_trigger_and_function():
    src = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"DROP\s+TRIGGER\s+IF\s+EXISTS\s+process_facts_on_insert",
        src,
        re.IGNORECASE,
    ), (
        "P1-AUDIT-1: la migración debe `DROP TRIGGER IF EXISTS "
        "process_facts_on_insert ON public.pending_facts_queue` "
        "(idempotente)."
    )
    assert re.search(
        r"DROP\s+FUNCTION\s+IF\s+EXISTS\s+public\.trigger_process_pending_facts_webhook",
        src,
        re.IGNORECASE,
    ), (
        "P1-AUDIT-1: la migración debe `DROP FUNCTION IF EXISTS "
        "public.trigger_process_pending_facts_webhook()` (idempotente)."
    )


def test_d3_migration_preserves_table():
    """La tabla `pending_facts_queue` NO debe ser droppeada — el backend
    todavía INSERTea via `enqueue_pending_fact()` y el cron la drena."""
    src = _MIGRATION_FILE.read_text(encoding="utf-8")
    # Permitido: COMMENT ON TABLE, DROP TRIGGER/FUNCTION. Prohibido: DROP TABLE.
    prohibited = re.compile(
        r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:public\.)?pending_facts_queue",
        re.IGNORECASE,
    )
    assert not prohibited.search(src), (
        "P1-AUDIT-1: la migración NO debe droppear `pending_facts_queue` — "
        "el backend la usa via `db_facts.py::enqueue_pending_fact` (overflow "
        "path fact_extractor:439) y el cron `_drain_pending_facts_queue` "
        "la lee. Solo se eliminan trigger + función."
    )


# ---------------------------------------------------------------------------
# E) El cron delega al procesador correcto.
# ---------------------------------------------------------------------------

def test_e_drain_uses_process_pending_queue_sync(cron_src: str):
    """El cron debe importar y llamar `process_pending_queue_sync` (mismo
    procesador del webhook). Si alguien lo reescribe inline, el lock
    `acquire_fact_lock(user_id)` se pierde y dos crones pueden procesar
    el mismo user concurrentemente."""
    # Extraer la función _drain_pending_facts_queue.
    m = re.search(
        r"def _drain_pending_facts_queue\b.*?(?=^def\s)",
        cron_src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "P1-AUDIT-1: no pude localizar el cuerpo de _drain_pending_facts_queue."
    body = m.group(0)
    assert "process_pending_queue_sync" in body, (
        "P1-AUDIT-1 regresión: `_drain_pending_facts_queue` no delega a "
        "`fact_extractor.process_pending_queue_sync`. Reescribirlo inline "
        "pierde el `acquire_fact_lock(user_id)` que serializa por usuario; "
        "dos crones podrían procesar el mismo user concurrentemente y "
        "duplicar facts."
    )


def test_f_anchor_p1_audit_1_present(cron_src: str):
    """Anchor textual: `P1-AUDIT-1` debe permanecer en el código para que un
    refactor cosmético no borre el contexto del fix."""
    m = re.search(
        r"def _drain_pending_facts_queue\b.*?(?=^def\s)",
        cron_src,
        re.DOTALL | re.MULTILINE,
    )
    body = m.group(0) if m else ""
    assert "P1-AUDIT-1" in body, (
        "P1-AUDIT-1 regresión: cuerpo de `_drain_pending_facts_queue` perdió "
        "la referencia textual `P1-AUDIT-1`. Sin el anchor, futuro "
        "mantenedor no entiende por qué este cron existe y podría "
        "eliminarlo creyendo que es duplicación."
    )
