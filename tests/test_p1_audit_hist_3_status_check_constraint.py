"""[P1-AUDIT-HIST-3 · 2026-05-09] Tests static analysis sobre la
migración SSOT que añade CHECK constraint a `plan_chunk_queue.status`.

Bug original (audit Historial 2026-05-09):
    `plan_chunk_queue.status` (varchar) NO tenía CHECK constraint. La
    DB de producción MealFitRD tenía un row con `status='complete'`
    (sin `d`) coexistiendo con el canónico `'completed'`. Todas las
    queries del código filtran por `status='completed'` (con `d`):
      - routers/plans.py:3349 (tier_breakdown del Dashboard)
      - routers/plans.py:4249 (history-status-summary, P0-AUDIT-HIST-2)
      - routers/plans.py:5262 (regen-degraded chunks)
      - cron_tasks.py:3560 / 4599 (synthesis source filter)
    Resultado: el row con `'complete'` quedaba SILENCIOSAMENTE excluido
    de TODAS esas queries → undercounts en analytics, agregadores del
    Historial subreportando, etc.

Fix:
    Migración SSOT
    `migrations/p1_audit_hist_3_plan_chunk_queue_status_check.sql`:
      1. Normaliza `'complete'` → `'completed'` (drift conocido).
      2. Cualquier otro valor inválido → `'cancelled'` (terminal seguro)
         con `dead_letter_reason` preservando el valor original.
      3. ADD CONSTRAINT con whitelist de los 7 estados canónicos.

    Aplicada al remoto (project mpoodlmnzaeuuazsazbj) y validada vía
    MCP Supabase: la constraint existe + UPDATE con valor inválido es
    rechazado por `check_violation`.

Cobertura (static analysis del SQL — coherente con el patrón de
P1-HIST-AUDIT-6 / P2-HIST-AUDIT-3):
    1. Migración existe en migrations/.
    2. Header documenta marker P1-AUDIT-HIST-3.
    3. SQL contiene normalización de 'complete' → 'completed'.
    4. SQL contiene defensiva para otros valores no-canónicos.
    5. SQL define CHECK con los 7 estados canónicos exactos.
    6. SQL es idempotente (DROP CONSTRAINT IF EXISTS antes del ADD).
    7. CHECK whitelist matchea SSOT del código (drift detection).
    8. COMMENT ON CONSTRAINT documental presente.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_MIGRATION_PATH = (
    _BACKEND_ROOT.parent
    / "migrations"
    / "p1_audit_hist_3_plan_chunk_queue_status_check.sql"
)

# SSOT canónico de estados — cualquier divergencia entre este set
# y el CHECK constraint del SQL fallará el test (drift detection).
_CANONICAL_STATES = frozenset({
    "pending",
    "processing",
    "stale",
    "failed",
    "pending_user_action",
    "completed",
    "cancelled",
})


def _migration_text() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Migración existe y marker presente
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_PATH.exists(), (
        f"No se encontró la migración P1-AUDIT-HIST-3 en "
        f"{_MIGRATION_PATH}. Path esperado por convención del repo "
        f"(SSOT migrations en migrations/)."
    )


def test_marker_present_in_header():
    text = _migration_text()
    assert "P1-AUDIT-HIST-3" in text, (
        "La migración debe contener el marker `P1-AUDIT-HIST-3` para "
        "trazabilidad cross-archivo (memoria, app.py:_LAST_KNOWN_PFIX)."
    )


# ---------------------------------------------------------------------------
# 2. Normalización: 'complete' → 'completed'
# ---------------------------------------------------------------------------
def test_migration_normalizes_complete_to_completed():
    """El UPDATE de drift conocido debe estar presente. Sin esto, la
    creación del CHECK fallaría (constraint violation en filas
    existentes) y la migración no se aplicaría."""
    text = _migration_text()
    # Buscar el UPDATE específico — flexible en whitespace.
    assert re.search(
        r"UPDATE\s+plan_chunk_queue\s+SET\s+status\s*=\s*'completed'"
        r"[^;]*WHERE\s+status\s*=\s*'complete'",
        text,
        re.IGNORECASE | re.DOTALL,
    ), (
        "Migración debe normalizar `status='complete'` (sin d, drift) "
        "a `'completed'` ANTES del CHECK. Sin esto, la creación de la "
        "constraint falla con check_violation en la DB de producción."
    )


def test_migration_defensively_normalizes_unknown_values():
    """Cualquier otro valor inválido (no en la whitelist) debe ir a
    `'cancelled'` con `dead_letter_reason` preservando el valor
    original — defense-in-depth para data drift desconocida."""
    text = _migration_text()
    # El segundo UPDATE debe filtrar por NOT IN (canónicos) y setear
    # status='cancelled'.
    assert re.search(
        r"UPDATE\s+plan_chunk_queue\s+SET\s+status\s*=\s*'cancelled'"
        r"[\s\S]*?WHERE\s+status\s+NOT\s+IN",
        text,
        re.IGNORECASE,
    ), (
        "Migración debe tener UPDATE defensivo para valores no-canónicos "
        "que NO sean `'complete'` (e.g. typos futuros del código): "
        "mapearlos a `'cancelled'` (terminal seguro) preservando el "
        "valor original en `dead_letter_reason`."
    )

    # Verificar que el valor original se preserva en dead_letter_reason.
    assert "dead_letter_reason" in text and "p1_audit_hist_3_drift_normalize" in text, (
        "El UPDATE defensivo debe preservar el valor original en "
        "`dead_letter_reason` con prefijo `p1_audit_hist_3_drift_normalize_from_` "
        "para post-mortem."
    )


# ---------------------------------------------------------------------------
# 3. CHECK constraint con whitelist canónica
# ---------------------------------------------------------------------------
def test_check_constraint_includes_all_canonical_states():
    """El CHECK debe incluir EXACTAMENTE los 7 estados canónicos —
    ni uno más (e.g. 'complete' permitido por error sería re-introducir
    el drift), ni uno menos (e.g. olvidar 'pending_user_action' rompe
    los crons de pausa pantry).
    """
    text = _migration_text()

    # Buscar el bloque CHECK (status IN (...)).
    m = re.search(
        r"CHECK\s*\(\s*status\s+IN\s*\(([^)]+)\)\s*\)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    assert m is not None, (
        "Migración debe contener `CHECK (status IN (...))` con la "
        "whitelist canónica."
    )

    found_states = set(re.findall(r"'([^']+)'", m.group(1)))
    assert found_states == _CANONICAL_STATES, (
        f"DRIFT entre la migración y el SSOT canónico.\n"
        f"  Migración: {sorted(found_states)}\n"
        f"  SSOT:      {sorted(_CANONICAL_STATES)}\n"
        f"Si añadiste o quitaste un estado, sincroniza ambos sets "
        f"en el mismo commit (este test + el código que setea/lee "
        f"esos valores)."
    )


def test_check_constraint_named_canonically():
    """El nombre `plan_chunk_queue_status_check` permite que un
    operador lo identifique vía `pg_get_constraintdef` por nombre.
    Sin nombre canónico, Postgres genera uno aleatorio que dificulta
    drop/replace en futuras migraciones."""
    text = _migration_text()
    assert re.search(
        r"ADD\s+CONSTRAINT\s+plan_chunk_queue_status_check",
        text,
        re.IGNORECASE,
    ), (
        "La constraint debe nombrarse `plan_chunk_queue_status_check` "
        "para idempotencia (DROP IF EXISTS lo encuentra) y "
        "diagnóstico operacional."
    )


# ---------------------------------------------------------------------------
# 4. Idempotencia
# ---------------------------------------------------------------------------
def test_migration_is_idempotent_drop_before_add():
    """`DROP CONSTRAINT IF EXISTS` antes del `ADD CONSTRAINT` permite
    re-aplicar la migración sin romper. Sin esto, una segunda
    ejecución falla con "constraint already exists"."""
    text = _migration_text()
    drop_idx = text.find("DROP CONSTRAINT IF EXISTS plan_chunk_queue_status_check")
    add_idx = text.find("ADD CONSTRAINT plan_chunk_queue_status_check")
    assert drop_idx > -1, (
        "Migración debe contener `DROP CONSTRAINT IF EXISTS "
        "plan_chunk_queue_status_check` para idempotencia."
    )
    assert add_idx > -1, "Migración debe contener `ADD CONSTRAINT ...`."
    assert drop_idx < add_idx, (
        "El DROP debe preceder al ADD. Si están al revés, la primera "
        "ejecución falla con duplicate constraint y la segunda lo "
        "elimina + re-crea — comportamiento inverso al esperado."
    )


def test_migration_normalize_uses_coalesce_for_dead_letter_reason():
    """`COALESCE(dead_letter_reason, ...)` preserva razones previas si
    un cron ya marcó la fila — evita pisar evidencia útil."""
    text = _migration_text()
    assert re.search(
        r"dead_letter_reason\s*=\s*COALESCE\s*\(\s*dead_letter_reason\s*,",
        text,
    ), (
        "El UPDATE defensivo debe usar `COALESCE(dead_letter_reason, "
        "'p1_audit_hist_3_...')` para preservar razones previas."
    )


# ---------------------------------------------------------------------------
# 5. Documentación operacional
# ---------------------------------------------------------------------------
def test_comment_on_constraint_present():
    """`COMMENT ON CONSTRAINT` documenta el WHY para que un operador
    que inspecciona via `\\d plan_chunk_queue` o `pg_get_constraintdef`
    encuentre la motivación sin leer git log."""
    text = _migration_text()
    assert "COMMENT ON CONSTRAINT plan_chunk_queue_status_check" in text, (
        "La constraint debe tener COMMENT documental con el marker "
        "P1-AUDIT-HIST-3 y la explicación del drift que cierra."
    )
    # El comentario debe citar el marker.
    comment_idx = text.find("COMMENT ON CONSTRAINT")
    comment_block = text[comment_idx:comment_idx + 1500]
    assert "P1-AUDIT-HIST-3" in comment_block


# ---------------------------------------------------------------------------
# 6. Drift detection vs el código
# ---------------------------------------------------------------------------
def test_canonical_states_match_code_set_clauses():
    """El set canónico de estados de la migración debe corresponder
    con todos los `SET status = '...'` del código de producción.

    Esto detecta cuando el código añade un estado (e.g.
    `preempted_quota`) sin actualizar la constraint — ese cambio
    fallaría en deploy con check_violation.
    """
    backend = _BACKEND_ROOT
    code_files = [
        backend / "routers" / "plans.py",
        backend / "cron_tasks.py",
        backend / "db_plans.py",
        backend / "services.py",
    ]
    found = set()
    for f in code_files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        # Captura `SET status = 'X'` (no `status = 'X'` en WHERE,
        # que tiene patrones distintos en lectura).
        for m in re.finditer(r"SET\s+status\s*=\s*'([a-z_]+)'", text):
            found.add(m.group(1))

    # Los `SET` deben ser un SUBSET de los canónicos. Si el código
    # mete un valor fuera del set, este test detecta el drift.
    extras = found - _CANONICAL_STATES
    assert not extras, (
        f"El código backend setea `status` a valores que NO están en "
        f"la whitelist canónica de la migración: {sorted(extras)}. "
        f"Si esos valores son legítimos, añádelos a la migración. "
        f"Si son typos, corrige el código."
    )
