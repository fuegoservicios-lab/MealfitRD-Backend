"""[P0-PROD-AUDIT-1 · 2026-05-23] Guard que el runbook de rollback de
migraciones existe + cubre los 3 escenarios canónicos.

Gap original (audit 2026-05-23, B-P0-4):
    La política de migraciones es "forward-only" (documentada en CLAUDE.md
    "DDL en runtime"), pero NO existía SOP escrito para reaccionar a una
    migración rota en producción. Cuando un incidente ocurre, el SRE
    improvisa — y la improvisación cuesta MTTR alto.

Fix:
    [`docs/runbooks/migration_rollback.md`] documenta el SOP para los 3
    escenarios canónicos:
      1. Migración falla durante apply (transacción rolled back, sin daño).
      2. Migración aplica con éxito pero introduce regresión.
      3. Migración aplica con éxito pero corrupta datos (worst case).

    Este test ancla la existencia + cobertura mínima del runbook.

Por qué un test del doc (no solo el doc en sí):
    Sin enforcement, runbooks se desactualizan: alguien borra la sección
    "Escenario 3" en un refactor cosmético y nadie lo nota hasta que el
    incidente real ocurre. Este test parsea el doc y falla loud si las
    secciones canónicas desaparecen.

Tooltip-anchor: P0-PROD-AUDIT-1-MIGRATION-RUNBOOK | audit 2026-05-23.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_RUNBOOK_PATH = _BACKEND_ROOT / "docs" / "runbooks" / "migration_rollback.md"


def test_runbook_exists() -> None:
    """El runbook debe existir en `docs/runbooks/migration_rollback.md`.

    Si moviste el doc, actualizar este test Y el cross-link en
    CLAUDE.md / `docs/runbooks/README.md`.
    """
    assert _RUNBOOK_PATH.exists(), (
        f"Runbook ausente en {_RUNBOOK_PATH}. Cierre del gap B-P0-4 perdido. "
        f"Restaurar via git revert del commit que lo borró."
    )


def test_runbook_covers_three_canonical_scenarios() -> None:
    """El runbook debe cubrir los 3 escenarios canónicos. Si alguien
    consolida/refactoriza y elimina uno (e.g. "scenario 3 se solapaba con
    P3-AUDIT-6"), este test falla loud.

    Los 3 escenarios son ortogonales — borrar uno deja un blind spot.
    """
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    required_scenarios = [
        ("Escenario 1: migración falla durante apply",
         "Cubre el caso happy: tx rollback automático, sin daño. SOP: investigar SQLSTATE + fix forward."),
        ("Escenario 2: migración aplica con éxito pero introduce regresión",
         "Cubre el caso de schema OK pero código no lo maneja. SOP: feature flag / knob / migración compensatoria."),
        ("Escenario 3: migración corruptó datos (worst case)",
         "Cubre el caso peor: restore selectivo. Cross-link al SOP P3-AUDIT-6 para meal_plans."),
    ]
    missing = []
    for header, description in required_scenarios:
        if header not in text:
            missing.append(f"  - `{header}` — {description}")
    if missing:
        pytest.fail(
            f"\n[P0-PROD-AUDIT-1] Runbook migration_rollback.md NO cubre los 3 "
            f"escenarios canónicos. Secciones ausentes:\n\n"
            + "\n".join(missing)
            + "\n\nRestaurar las secciones. Si decidiste consolidar, documentar "
            "la decisión en el mismo PR + actualizar este test."
        )


def test_runbook_covers_preflight_checklist() -> None:
    """El runbook debe tener un checklist de pre-flight ANTES de aplicar la
    migración productiva. Es la defensa más barata: la mayoría de incidents
    se evitan en este paso (validar idempotencia, SSOT cross-repo, test
    parser-based existe, sanity check).

    Si alguien borra el checklist, los incidents recurrirán por causas que
    el checklist habría capturado.
    """
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "pre-flight checklist" in text.lower() or "preflight checklist" in text.lower(), (
        "Runbook migration_rollback.md no incluye el SOP de pre-flight "
        "checklist. Esto es la defensa más barata contra incidents. "
        "Restaurar la sección con los items canónicos (IF NOT EXISTS, "
        "SSOT cross-repo, test parser-based, sanity check post-apply)."
    )
    # Sanity de items específicos del checklist (los más críticos).
    required_items = [
        "IF NOT EXISTS",
        "SSOT",
        "Idempotente",
        "branch",  # SOP de testing en branch de Supabase
    ]
    missing_items = [item for item in required_items if item not in text]
    assert not missing_items, (
        f"Pre-flight checklist incompleto — items ausentes: {missing_items}. "
        f"Estos son canónicos del repo (ver CLAUDE.md "
        f"'DDL en runtime' + 'SSOT de migrations')."
    )


def test_runbook_links_to_p3_audit_6_for_data_corruption() -> None:
    """El escenario 3 (data corruption) debe cross-linkar al SOP P3-AUDIT-6
    para el caso específico de `meal_plans` (que tiene tabla de audit
    `meal_plans_audit` con snapshots restorables).

    Sin el cross-link, alguien podría improvisar restore de meal_plans
    sin usar el audit existente — pérdida de datos vs restore selectivo.
    """
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "P3-AUDIT-6" in text or "meal_plans_audit" in text, (
        "Runbook NO menciona P3-AUDIT-6 ni meal_plans_audit en el "
        "escenario de data corruption. Cross-link perdido — restore "
        "selectivo de meal_plans requiere conocer este SOP existente."
    )


def test_runbook_forbids_manual_drop_anti_pattern() -> None:
    """El runbook debe documentar EXPLÍCITAMENTE que ejecutar
    `DROP COLUMN`/`DROP TABLE`/`DROP CONSTRAINT` desde el dashboard SQL
    editor para "deshacer" una migración es anti-patrón.

    Este es el error más común de SREs no familiarizados con la política
    forward-only. Sin la advertencia explícita en el runbook, alguien
    lo intentará bajo presión de incidente.
    """
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "Anti-patrón" in text and "DROP" in text, (
        "Runbook NO documenta el anti-patrón de DROP manual para 'rollback'. "
        "Restaurar la sección — es la advertencia más crítica del SOP "
        "(error humano más común bajo presión de incidente)."
    )


def test_runbook_is_referenced_from_claude_md_or_docs_index() -> None:
    """El runbook debe ser descubrible: o desde CLAUDE.md o desde el index
    de runbooks (`docs/runbooks/README.md`). Si solo vive como archivo
    suelto, un SRE bajo presión NO lo encontrará.
    """
    runbook_name = "migration_rollback.md"

    claude_md = _BACKEND_ROOT / "CLAUDE.md"
    runbooks_index = _BACKEND_ROOT / "docs" / "runbooks" / "README.md"

    found_in_claude = claude_md.exists() and runbook_name in claude_md.read_text(encoding="utf-8")
    found_in_index = runbooks_index.exists() and runbook_name in runbooks_index.read_text(encoding="utf-8")

    assert found_in_claude or found_in_index, (
        f"Runbook `{runbook_name}` no referenciado desde CLAUDE.md ni desde "
        f"docs/runbooks/README.md. Añadir cross-link en al menos uno — sin "
        f"descubribilidad, el runbook es inútil bajo presión de incidente."
    )
