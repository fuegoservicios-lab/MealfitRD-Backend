"""[P1-NEXT-1 · 2026-05-11] Lock-the-contract: el helper
`update_meal_plan_data` (db_plans.py) DEBE adquirir
`acquire_meal_plan_advisory_lock(cursor, plan_id, purpose='general')`
ANTES del `UPDATE meal_plans SET plan_data = %s WHERE id = %s` en su
cuerpo, dentro de la misma transacción (`conn.transaction()`).

Cierra el gap detectado en el audit 2026-05-11:
    P1-NEW-C (test que vigila routers/plans.py) scaneaba el patrón
    literal `UPDATE meal_plans SET plan_data = %s::jsonb` y dejaba
    pasar las llamadas del helper `update_meal_plan_data` porque el
    helper usa el adapter `Jsonb(...)` (sin `::jsonb` literal). Los
    4 callsites prod del helper (/recipe/expand, /recalculate-
    shopping-list, proactive_agent JIT week-2, tools.modify_single_meal)
    estaban entonces fuera del scope del test → corrían el UPDATE
    full-overwrite SIN advisory lock → race lost-update silente
    contra `_chunk_worker` T2 que también full-overwrite plan_data
    tras cada chunk con purpose='general'.

Fix P1-NEXT-1: el lock vive ADENTRO del helper (no en cada caller)
para que TODOS los callsites lo hereden automáticamente, presente
y futuros.

Drift detection:
    - Si el helper pierde el `acquire_meal_plan_advisory_lock` en una
      refactorización → falla.
    - Si el UPDATE migra fuera del `with conn.transaction()` → falla.
    - Si el `purpose` cambia (e.g. `'recalc'` en lugar de `'general'`),
      dos writers en buckets distintos no se serializan → falla.

Whitelist:
    No prevista — `update_meal_plan_data` SIEMPRE debe lockear. Si
    el helper desaparece (todos los callsites migran a `jsonb_set`),
    el test queda como skip natural y se puede eliminar manualmente.

Tooltip-anchor: P1-NEXT-1-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_DB_PLANS = Path(__file__).resolve().parent.parent / "db_plans.py"


def _read_db_plans_source() -> str:
    if not _DB_PLANS.exists():
        raise FileNotFoundError(f"{_DB_PLANS} no encontrado.")
    return _DB_PLANS.read_text(encoding="utf-8")


def _extract_function_body(source: str, fn_name: str) -> str:
    """Aísla el cuerpo de `def fn_name(...)` hasta el siguiente top-level
    `def ` o `class `. Defensivo contra refactor que añada decoradores."""
    pattern = re.compile(
        rf"^def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    assert m, (
        f"No se encontró `def {fn_name}(...)` en db_plans.py. "
        "El test P1-NEXT-1 perdió su anchor — verifica que el helper "
        "no fue renombrado o movido. Si fue borrado tras migrar todos "
        "los callsites a jsonb_set, este test queda como skip natural."
    )
    body_start = m.start()
    next_def_pattern = re.compile(r"^(def |class )", re.MULTILINE)
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[body_start:next_def.start()]
    return source[body_start:]


# ---------------------------------------------------------------------------
# 1. acquire_meal_plan_advisory_lock con purpose='general' está presente
# ---------------------------------------------------------------------------
def test_helper_acquires_general_purpose_lock():
    """`update_meal_plan_data` DEBE invocar
    `acquire_meal_plan_advisory_lock(..., purpose='general')` en su cuerpo.
    """
    source = _read_db_plans_source()
    body = _extract_function_body(source, "update_meal_plan_data")

    lock_re = re.compile(
        r"acquire_meal_plan_advisory_lock\s*\([^)]*purpose\s*=\s*['\"]general['\"]",
        re.DOTALL,
    )
    assert lock_re.search(body), (
        "P1-NEXT-1 violation: `update_meal_plan_data` no contiene "
        "`acquire_meal_plan_advisory_lock(...purpose='general'...)` en "
        "su cuerpo. Sin ese lock, los 4 callsites prod (/recipe/expand, "
        "/recalculate-shopping-list, proactive_agent JIT, "
        "tools.modify_single_meal) hacen full-overwrite de plan_data en "
        "race contra `_chunk_worker` T1/T2 que también full-overwrite con "
        "purpose='general' → lost-update silente. Fix: dentro del bloque "
        "`with connection_pool.connection() as conn: with conn.transaction(): "
        "with conn.cursor() as cursor:`, llamar "
        "`acquire_meal_plan_advisory_lock(cursor, plan_id, purpose=\"general\")` "
        "ANTES del UPDATE."
    )


# ---------------------------------------------------------------------------
# 2. El lock está ANTES del UPDATE en orden textual (dentro del cuerpo)
# ---------------------------------------------------------------------------
def test_lock_is_before_update_in_body():
    """Defensa contra refactor que invierta el orden — el lock pierde
    su valor si se adquiere DESPUÉS del UPDATE.
    """
    source = _read_db_plans_source()
    body = _extract_function_body(source, "update_meal_plan_data")

    lock_re = re.compile(
        r"acquire_meal_plan_advisory_lock\s*\([^)]*purpose\s*=\s*['\"]general['\"]",
        re.DOTALL,
    )
    update_re = re.compile(
        r"UPDATE\s+meal_plans\s+SET\s+plan_data\s*=\s*%s\s+WHERE",
        re.IGNORECASE,
    )

    lock_match = lock_re.search(body)
    update_match = update_re.search(body)
    assert lock_match, "Lock call missing (cubre test #1)"
    assert update_match, (
        "No se encontró `UPDATE meal_plans SET plan_data = %s WHERE` en "
        "el cuerpo del helper. ¿El UPDATE fue migrado a jsonb_set "
        "quirúrgico o a otro patrón? Si es así, este test ya no aplica "
        "— borrarlo o restringirlo al nuevo patrón."
    )
    assert lock_match.start() < update_match.start(), (
        "P1-NEXT-1 violation: en `update_meal_plan_data`, el "
        "`acquire_meal_plan_advisory_lock(...purpose='general'...)` "
        "aparece DESPUÉS del UPDATE. Un lock posterior al UPDATE es "
        "no-op para serialización: dos writers concurrentes ya pisaron "
        "plan_data antes de que cualquiera tome el lock. Mover el lock "
        "ANTES del UPDATE."
    )


# ---------------------------------------------------------------------------
# 3. El lock + UPDATE viven dentro del mismo `with conn.transaction()`
# ---------------------------------------------------------------------------
def test_lock_and_update_share_transaction_block():
    """`pg_advisory_xact_lock` se libera al COMMIT/ROLLBACK. Si lock y
    UPDATE están en transacciones distintas, el lock se libera antes
    del UPDATE y la serialización se rompe.
    """
    source = _read_db_plans_source()
    body = _extract_function_body(source, "update_meal_plan_data")

    # Buscar el patrón `with conn.transaction():` y verificar que
    # tanto lock como UPDATE estén dentro del mismo bloque indentado.
    tx_match = re.search(r"with\s+conn\.transaction\s*\(\s*\)", body)
    assert tx_match, (
        "P1-NEXT-1 violation: `update_meal_plan_data` no contiene "
        "`with conn.transaction():`. El lock advisory + UPDATE deben "
        "compartir transacción explícita para que el lock no se libere "
        "prematuramente. Fix: envolver el bloque del cursor en "
        "`with conn.transaction():` (mismo patrón que update_plan_data_atomic)."
    )

    lock_re = re.compile(
        r"acquire_meal_plan_advisory_lock\s*\([^)]*purpose\s*=\s*['\"]general['\"]",
        re.DOTALL,
    )
    update_re = re.compile(
        r"UPDATE\s+meal_plans\s+SET\s+plan_data\s*=\s*%s\s+WHERE",
        re.IGNORECASE,
    )

    # Ambos DEBEN aparecer después del `with conn.transaction()`.
    tx_end = tx_match.end()
    lock_m = lock_re.search(body, pos=tx_end)
    update_m = update_re.search(body, pos=tx_end)

    assert lock_m and update_m, (
        "P1-NEXT-1 violation: el `acquire_meal_plan_advisory_lock` o el "
        "`UPDATE meal_plans` no aparecen DESPUÉS del "
        "`with conn.transaction():`. Confirmar que ambos viven dentro "
        "del mismo bloque transaccional explícito."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link slug del marker (alineado a P1-NEW-C / P2-HIST-AUDIT-14)
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p1_next_1"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p1_next_1`) para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee cuando "
        "el marker se bumpee a `P1-NEXT-1 · 2026-05-11`."
    )
