"""[P2-DOC-1 · 2026-05-12] Helper SSOT `record_meal_plan_audit_backup` en
`backend/db_meal_plans_audit.py`.

Vector observado (audit 2026-05-11):
    Tabla `meal_plans_audit` con `n_live_tup=1` (single row legacy)
    pese a estar referenciada por SOP P3-AUDIT-6. El SOP exigía SQL
    inline pre-mutación pero los SRE evitaban hacerlo bajo presión.

Fix:
    Helper Python `record_meal_plan_audit_backup(meal_plan_id, action,
    actor, note)` que:
      - valida UUID + action contra el enum del CHECK constraint;
      - snapshot automático del `plan_data` actual;
      - INSERT a `meal_plans_audit` con RETURNING id;
      - retorna BIGINT o None si falló (caller decide si abortar).

    NO se invoca automáticamente desde `update_meal_plan_data` — la
    convención sigue siendo "operacional manual": SRE elige cuándo
    backup-ear.

Lo que este test enforza:
    A) Módulo `db_meal_plans_audit.py` existe.
    B) Función `record_meal_plan_audit_backup` declarada con la firma
       documentada `(meal_plan_id, action, actor, note=None)`.
    C) Validación de UUID via `uuid.UUID(meal_plan_id)`.
    D) Enum cerrado `_VALID_ACTIONS` declarado y contiene los 4 valores
       del CHECK constraint en DB.
    E) La función `list_recent_audit_backups` existe (utilidad SRE).
    F) Anchor `P2-DOC-1` en el módulo.
    G) Runbook `runbook_system_alerts_sops_2026_05_11.md` referencia el
       helper Python en paso 3 del SOP P3-AUDIT-6.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _BACKEND_ROOT / "db_meal_plans_audit.py"
_MIGRATION = (
    _BACKEND_ROOT.parent / "supabase" / "migrations"
    / "p2_new_5_meal_plans_audit_table.sql"
)
_RUNBOOK = (
    Path.home()
    / ".claude" / "projects"
    / "c--Users-angel-OneDrive-Escritorio-MealfitRD-IA"
    / "memory" / "runbook_system_alerts_sops_2026_05_11.md"
)

_EXPECTED_ACTIONS = (
    "corruption_repair",
    "manual_rollback",
    "pre_delete_backup",
    "schema_migration",
)


@pytest.fixture(scope="module")
def module_src() -> str:
    assert _MODULE_PATH.exists(), (
        "P2-DOC-1: módulo `backend/db_meal_plans_audit.py` no encontrado."
    )
    return _MODULE_PATH.read_text(encoding="utf-8")


def test_a_function_defined_with_documented_signature(module_src: str):
    """`record_meal_plan_audit_backup(meal_plan_id, action, actor, note=None)`."""
    # La firma debe matchear documented contract.
    pattern = re.compile(
        r"def\s+record_meal_plan_audit_backup\s*\(\s*"
        r"meal_plan_id\s*:\s*str\s*,\s*"
        r"action\s*:\s*str\s*,\s*"
        r"actor\s*:\s*str\s*,\s*"
        r"note\s*:\s*Optional\[str\]\s*=\s*None\s*,?\s*\)",
        re.MULTILINE,
    )
    assert pattern.search(module_src), (
        "P2-DOC-1: firma de `record_meal_plan_audit_backup` no matchea "
        "el contrato documentado `(meal_plan_id, action, actor, note=None)`."
    )


def test_b_uuid_validation(module_src: str):
    """El helper debe validar UUID via `uuid.UUID(meal_plan_id)` ANTES
    de hacer el SELECT."""
    assert "_uuid.UUID(meal_plan_id)" in module_src or "uuid.UUID(meal_plan_id)" in module_src, (
        "P2-DOC-1: validación de UUID ausente. Sin ella, un meal_plan_id "
        "garbage llega al cast SQL y produce error 500 con detalle del schema."
    )


def test_c_valid_actions_enum_matches_check_constraint(module_src: str):
    """`_VALID_ACTIONS` declarado y contiene los 4 valores del CHECK."""
    pattern = re.compile(r"_VALID_ACTIONS\s*=\s*\(([\s\S]*?)\)")
    m = pattern.search(module_src)
    assert m, "P2-DOC-1: `_VALID_ACTIONS` no declarado."
    block = m.group(1)
    for action in _EXPECTED_ACTIONS:
        assert f'"{action}"' in block, (
            f"P2-DOC-1: action {action!r} ausente en `_VALID_ACTIONS`. "
            f"El CHECK constraint en DB rechazaría inserciones con valores "
            f"distintos a los 4 listados."
        )


def test_d_check_constraint_matches_migration(module_src: str):
    """Cross-check: los 4 valores del enum local DEBEN coincidir con
    el CHECK constraint de la migración SSOT. Si la migración añade
    un valor nuevo sin extender este enum, los SRE no podrán usarlo
    via el helper."""
    assert _MIGRATION.exists(), "Migración SSOT meal_plans_audit no encontrada."
    mig_src = _MIGRATION.read_text(encoding="utf-8")
    check_match = re.search(
        r"action\s+TEXT\s+NOT\s+NULL\s+CHECK\s*\(\s*action\s+IN\s*\(([\s\S]*?)\)\s*\)",
        mig_src,
        re.IGNORECASE,
    )
    assert check_match, "P2-DOC-1: CHECK constraint en migración no aislable."
    constraint_block = check_match.group(1)
    for action in _EXPECTED_ACTIONS:
        assert f"'{action}'" in constraint_block, (
            f"P2-DOC-1: action {action!r} ausente del CHECK constraint en "
            f"`p2_new_5_meal_plans_audit_table.sql`. Drift entre helper "
            f"Python y schema DB."
        )


def test_e_list_helper_exists(module_src: str):
    """`list_recent_audit_backups(meal_plan_id, user_id, limit)` para
    forensics post-incidente."""
    assert "def list_recent_audit_backups(" in module_src, (
        "P2-DOC-1: `list_recent_audit_backups` ausente. Sin él, el SRE "
        "debe hacer SQL manual contra la tabla audit para ver historial."
    )


def test_f_anchor_present(module_src: str):
    assert "P2-DOC-1" in module_src, (
        "P2-DOC-1: anchor desapareció del módulo."
    )


def test_g_runbook_references_helper():
    """El runbook P3-AUDIT-6 (paso 3 "Backup defensivo") debe mencionar
    el helper Python como opción preferida."""
    if not _RUNBOOK.exists():
        pytest.skip(f"Runbook no encontrado en {_RUNBOOK}; skip (no en CI).")
    runbook_src = _RUNBOOK.read_text(encoding="utf-8")
    assert "record_meal_plan_audit_backup" in runbook_src, (
        "P2-DOC-1: runbook P3-AUDIT-6 no referencia el helper Python. "
        "Sin esa referencia, los SRE seguirán haciendo SQL manual."
    )
    assert "db_meal_plans_audit" in runbook_src, (
        "P2-DOC-1: runbook no menciona el módulo `db_meal_plans_audit`."
    )
