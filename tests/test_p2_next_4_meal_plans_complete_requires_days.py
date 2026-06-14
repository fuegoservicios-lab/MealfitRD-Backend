"""[P2-NEXT-4 · 2026-05-11] Migración SSOT que enforza CHECK constraint
`meal_plans_complete_requires_days`: si `generation_status='complete'`
entonces `jsonb_array_length(plan_data->'days') > 0`.

Cierra el modo de corrupción detectado en el audit 2026-05-11:
    Plan `005c5a99-3bd2-41fb-a311-bf025c930b3d` tenía
    `generation_status='complete'` con `jsonb_array_length(days)=0`
    durante ~14h en prod. Root cause: chunk worker T1 marcó status
    como complete pero el merge `plan_data.days = merged_days` se
    perdió (race / rollback parcial intermedio entre estado del chunk
    y plan_data).

Fix data + DDL:
    - Plan 005c5a99 backupado en `meal_plans_audit` (action=
      corruption_repair) y migrado a `generation_status='abandoned'`.
    - Migración `p2_next_4_meal_plans_complete_requires_days.sql`
      añade el CHECK constraint con sanity check pre-deploy + VALIDATE.

Drift detection:
    - Migración borrada/renombrada → falla.
    - CHECK constraint pierde el guard `jsonb_array_length(...) > 0` →
      falla.
    - Sanity check pre-deploy removido → falla (sin él, una re-aplicación
      después de un nuevo bug del chunk worker fallaría con un error
      críptico de constraint en lugar del mensaje accionable).

Tooltip-anchor: P2-NEXT-4-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATIONS_DIR = _REPO_ROOT / "migrations"
_MIGRATION_FILE = _MIGRATIONS_DIR / "p2_next_4_meal_plans_complete_requires_days.sql"


# ---------------------------------------------------------------------------
# 1. La migración existe
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_FILE.exists(), (
        f"P2-NEXT-4 violation: migración SSOT no encontrada en {_MIGRATION_FILE}. "
        "El CHECK constraint vive en runtime de prod pero sin SSOT en /migrations, "
        "un cluster nuevo (staging/local) lo perdería. Crear la migración y aplicarla."
    )


# ---------------------------------------------------------------------------
# 2. Migración define el CHECK constraint con la regla correcta
# ---------------------------------------------------------------------------
def test_constraint_rule_matches_invariant():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")

    # ADD CONSTRAINT con el nombre estable
    assert re.search(
        r"ADD\s+CONSTRAINT\s+meal_plans_complete_requires_days",
        source,
        re.IGNORECASE,
    ), (
        "P2-NEXT-4 violation: migración no añade el constraint con el "
        "nombre canónico `meal_plans_complete_requires_days`. Nombre "
        "estable es load-bearing para `pg_get_constraintdef` queries y "
        "para rollback (DROP CONSTRAINT IF EXISTS)."
    )

    # Regla: status='complete' → jsonb_array_length(days) > 0
    assert re.search(r"generation_status['\"]?\s*[)]?\s*!=\s*['\"]complete['\"]", source, re.IGNORECASE), (
        "P2-NEXT-4 violation: el CHECK no menciona la condición "
        "`generation_status != 'complete'`. Sin ella el constraint "
        "permitiría status arbitrarios con days=0 (no es la invariante "
        "deseada)."
    )
    assert re.search(
        r"jsonb_array_length\s*\(\s*COALESCE\s*\(\s*plan_data\s*->\s*['\"]days['\"]",
        source,
        re.IGNORECASE,
    ), (
        "P2-NEXT-4 violation: el CHECK no usa "
        "`jsonb_array_length(COALESCE(plan_data->'days', '[]'::jsonb)) > 0`. "
        "Sin COALESCE, planes con `days=null` (caso patológico distinto de "
        "`days=[]`) pasarían el check silenciosamente."
    )


# ---------------------------------------------------------------------------
# 3. Sanity check pre-deploy bloquea aplicar si hay violators
# ---------------------------------------------------------------------------
def test_pre_deploy_sanity_check_present():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(r"DO\s*\$\$", source), (
        "P2-NEXT-4 violation: migración no tiene bloque DO $$ con "
        "sanity check de violators pre-deploy. Sin él, una re-aplicación "
        "después de un nuevo bug del chunk worker fallaría con un error "
        "críptico de PostgreSQL en lugar del mensaje explicativo."
    )
    assert re.search(r"RAISE\s+EXCEPTION", source, re.IGNORECASE), (
        "P2-NEXT-4 violation: el bloque DO $$ no `RAISE EXCEPTION` "
        "cuando detecta violators. Sin esto, una aplicación con plans "
        "violatorios procede silently → constraint NOT VALID en una "
        "tabla con violadores es estado inestable."
    )


# ---------------------------------------------------------------------------
# 4. Idempotencia: DROP CONSTRAINT IF EXISTS antes del ADD
# ---------------------------------------------------------------------------
def test_migration_is_idempotent():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"DROP\s+CONSTRAINT\s+IF\s+EXISTS\s+meal_plans_complete_requires_days",
        source,
        re.IGNORECASE,
    ), (
        "P2-NEXT-4 violation: migración no es idempotente — falta "
        "`DROP CONSTRAINT IF EXISTS meal_plans_complete_requires_days` "
        "antes del ADD. Sin esto, re-aplicar la migración (recovery, "
        "staging refresh) falla con 'constraint already exists'."
    )


# ---------------------------------------------------------------------------
# 5. VALIDATE separado del ADD (permite NOT VALID + cleanup gradual)
# ---------------------------------------------------------------------------
def test_validate_constraint_separate_statement():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"VALIDATE\s+CONSTRAINT\s+meal_plans_complete_requires_days",
        source,
        re.IGNORECASE,
    ), (
        "P2-NEXT-4 violation: migración no incluye `VALIDATE CONSTRAINT` "
        "explícito tras el `NOT VALID`. Mejor práctica: ADD NOT VALID "
        "(barato, no scan) → VALIDATE (separado, scan completo). "
        "Permite el sanity check entre ambos pasos."
    )


# ---------------------------------------------------------------------------
# 6. COMMENT presente para forensics post-incident
# ---------------------------------------------------------------------------
def test_constraint_has_comment_with_pfix_marker():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"COMMENT\s+ON\s+CONSTRAINT\s+meal_plans_complete_requires_days",
        source,
        re.IGNORECASE,
    ), (
        "P2-NEXT-4 violation: no hay `COMMENT ON CONSTRAINT` que "
        "documente el origen. Un operador SRE que vea el constraint "
        "fallar en runtime debe poder hacer `\\d+ meal_plans` y "
        "encontrar el P-fix de origen en una línea."
    )
    assert "P2-NEXT-4" in source, (
        "P2-NEXT-4 violation: COMMENT debe mencionar el marker P2-NEXT-4."
    )


# ---------------------------------------------------------------------------
# 7. Cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p2_next_4"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "Filename debe contener slug `p2_next_4` para cross-link."
    )
