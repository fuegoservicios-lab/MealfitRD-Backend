"""[P3-MULTIPLIER-DB-CHECK · 2026-05-14] Migración SSOT que enforza
CHECK constraint `meal_plans_calc_household_multiplier_range`: si
`plan_data->>'calc_household_multiplier'` está presente, debe ser
numérico en `[1, 100]`.

Motivación (audit 2026-05-14):
    El clamp `[1, MEALFIT_MAX_HOUSEHOLD_SIZE=20]` vive en
    `compute_household_multiplier` ([constants.py:2136](backend/constants.py))
    y se aplica también inline en `/recalculate-shopping-list`
    (P3-PDF-POLISH-4-B-RECALC). Pero el enforcement es PURAMENTE en
    código — una regresión en uno de los callsites (alguien añadiendo
    un 4to write path sin pasar por el helper SSOT, o un knob mal
    calibrado que sube el cap a 999) persistiría valores absurdos en
    `plan_data.calc_household_multiplier` sin que la DB lo bloquee.

Fix:
    Migración SSOT en
    `migrations/p3_multiplier_db_check_2026_05_14.sql` con
    el mismo patrón que P2-NEXT-4 (`meal_plans_complete_requires_days`):
      - Sanity check pre-deploy (RAISE EXCEPTION si hay violators).
      - DROP CONSTRAINT IF EXISTS + ADD NOT VALID (idempotente, no scan).
      - VALIDATE CONSTRAINT separado.
      - COMMENT ON CONSTRAINT con marker P3-MULTIPLIER-DB-CHECK.

    Rango [1, 100]: lower 1 (planner requiere >=1 persona); upper 100
    (5× cap default knob, permite que MEALFIT_MAX_HOUSEHOLD_SIZE se
    ajuste hasta 100 sin BD bloquear).

    NULL-friendly: planes legacy pre-P1-3 (sin la key) pasan el check
    porque `plan_data->>'calc_household_multiplier'` es NULL para
    JSONB sin la key.

Drift detection (parser-based):
    1. La migración existe en `migrations/`.
    2. Constraint name `meal_plans_calc_household_multiplier_range`.
    3. Rango [1, 100] explícito en el CHECK.
    4. Sanity check DO $$ con RAISE EXCEPTION para violators.
    5. Idempotencia (DROP IF EXISTS + ADD).
    6. VALIDATE separado.
    7. COMMENT con marker P3-MULTIPLIER-DB-CHECK.
    8. Cross-link slug.

Whitelist:
    No prevista. Si el rango sube (p.ej. el knob `MEALFIT_MAX_HOUSEHOLD_SIZE`
    pasa de 100 a 200), actualizar la migración (DROP + ADD nuevo) +
    este test + COMMENT.

Tooltip-anchor: P3-MULTIPLIER-DB-CHECK-START | gap audit 2026-05-14
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATIONS_DIR = _REPO_ROOT / "migrations"
_MIGRATION_FILE = _MIGRATIONS_DIR / "p3_multiplier_db_check_2026_05_14.sql"


# ---------------------------------------------------------------------------
# 1. Migración existe
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_FILE.exists(), (
        f"P3-MULTIPLIER-DB-CHECK violation: migración SSOT no encontrada "
        f"en {_MIGRATION_FILE}. El CHECK constraint vive en runtime de "
        f"prod (aplicado vía Supabase MCP el 2026-05-14) pero sin SSOT "
        f"en /migrations, un cluster nuevo (staging/local) lo "
        f"perdería. Crear la migración con el contenido canónico."
    )


# ---------------------------------------------------------------------------
# 2. Constraint name canónico
# ---------------------------------------------------------------------------
def test_constraint_name_is_canonical():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"ADD\s+CONSTRAINT\s+meal_plans_calc_household_multiplier_range",
        source,
        re.IGNORECASE,
    ), (
        "P3-MULTIPLIER-DB-CHECK violation: migración no añade el constraint "
        "con el nombre canónico `meal_plans_calc_household_multiplier_range`. "
        "Nombre estable es load-bearing para `pg_get_constraintdef` queries "
        "y para rollback (DROP CONSTRAINT IF EXISTS)."
    )


# ---------------------------------------------------------------------------
# 3. Rango [1, 100] explícito en el CHECK
# ---------------------------------------------------------------------------
def test_check_range_is_1_to_100():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    # Tolerar variantes de espacio. La regla NUMERIC BETWEEN 1 AND 100.
    assert re.search(
        r"::numeric\s+BETWEEN\s+1\s+AND\s+100",
        source,
        re.IGNORECASE,
    ), (
        "P3-MULTIPLIER-DB-CHECK violation: el CHECK no enforza "
        "`::numeric BETWEEN 1 AND 100`. Rango fuera del contrato:\n"
        "  - Lower 1: planner requiere >=1 persona.\n"
        "  - Upper 100: 5× cap default del knob MEALFIT_MAX_HOUSEHOLD_SIZE.\n"
        "Si cambiaste el rango por una razón legítima (escalado de "
        "producto), actualizar también el COMMENT y este test."
    )


# ---------------------------------------------------------------------------
# 4. Guard regex para no-numéricos (defense-in-depth contra strings)
# ---------------------------------------------------------------------------
def test_check_guards_against_non_numeric():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    # El CHECK debe usar `~ '^-?\d+(\.\d+)?$'` antes del cast a numeric.
    # Sin él, un cast directo lanzaría exception en runtime con error
    # críptico cuando alguien persistiera un string non-numeric.
    assert re.search(
        r"~\s*['\"]?\^-\?",
        source,
    ), (
        "P3-MULTIPLIER-DB-CHECK violation: el CHECK no tiene el guard "
        "regex `~ '^-?\\d+(\\.\\d+)?$'` antes del cast a `::numeric`. "
        "Sin él, persistir `calc_household_multiplier=\"abc\"` levantaría "
        "exception PostgreSQL en runtime con mensaje críptico. El regex "
        "deja que el constraint rechace el valor con mensaje accionable "
        "(constraint violation, no cast error)."
    )


# ---------------------------------------------------------------------------
# 5. NULL-friendly: planes legacy sin la key pasan el check
# ---------------------------------------------------------------------------
def test_check_is_null_friendly():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    # El CHECK debe tener `NOT (plan_data ? 'calc_household_multiplier')`
    # como cláusula de bypass para planes legacy.
    assert re.search(
        r"NOT\s*\(\s*plan_data\s*\?\s*['\"]calc_household_multiplier['\"]",
        source,
        re.IGNORECASE,
    ), (
        "P3-MULTIPLIER-DB-CHECK violation: el CHECK no tiene la cláusula "
        "`NOT (plan_data ? 'calc_household_multiplier')` que permite a "
        "planes legacy pre-P1-3 (sin la key) pasar el check. Sin esta "
        "rama, aplicar el constraint a una BD con históricos los "
        "invalidaría retroactivamente."
    )


# ---------------------------------------------------------------------------
# 6. Sanity check pre-deploy con RAISE EXCEPTION
# ---------------------------------------------------------------------------
def test_pre_deploy_sanity_check_present():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(r"DO\s*\$\$", source), (
        "P3-MULTIPLIER-DB-CHECK violation: migración no tiene bloque "
        "DO $$ con sanity check de violators pre-deploy. Sin él, una "
        "re-aplicación después de un bug del helper SSOT fallaría con "
        "error críptico de PostgreSQL en lugar del mensaje explicativo."
    )
    assert re.search(r"RAISE\s+EXCEPTION", source, re.IGNORECASE), (
        "P3-MULTIPLIER-DB-CHECK violation: el bloque DO $$ no "
        "`RAISE EXCEPTION` cuando detecta violators. Sin esto, una "
        "aplicación con plans violatorios procede silently → constraint "
        "NOT VALID en una tabla con violadores es estado inestable."
    )


# ---------------------------------------------------------------------------
# 7. Idempotencia: DROP CONSTRAINT IF EXISTS antes del ADD
# ---------------------------------------------------------------------------
def test_migration_is_idempotent():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"DROP\s+CONSTRAINT\s+IF\s+EXISTS\s+meal_plans_calc_household_multiplier_range",
        source,
        re.IGNORECASE,
    ), (
        "P3-MULTIPLIER-DB-CHECK violation: migración no es idempotente — "
        "falta `DROP CONSTRAINT IF EXISTS meal_plans_calc_household_multiplier_range` "
        "antes del ADD. Sin esto, re-aplicar la migración (recovery, "
        "staging refresh) falla con 'constraint already exists'."
    )


# ---------------------------------------------------------------------------
# 8. VALIDATE separado del ADD (NOT VALID → VALIDATE staged)
# ---------------------------------------------------------------------------
def test_validate_constraint_separate_statement():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"VALIDATE\s+CONSTRAINT\s+meal_plans_calc_household_multiplier_range",
        source,
        re.IGNORECASE,
    ), (
        "P3-MULTIPLIER-DB-CHECK violation: migración no incluye "
        "`VALIDATE CONSTRAINT` explícito tras el `NOT VALID`. Mejor "
        "práctica (espejo de P2-NEXT-4): ADD NOT VALID (barato, no scan) "
        "→ VALIDATE (separado, scan completo). Permite el sanity check "
        "entre ambos pasos sin abortar la migración en caso de re-aplicación."
    )


# ---------------------------------------------------------------------------
# 9. COMMENT con marker P3-MULTIPLIER-DB-CHECK
# ---------------------------------------------------------------------------
def test_constraint_has_comment_with_pfix_marker():
    source = _MIGRATION_FILE.read_text(encoding="utf-8")
    assert re.search(
        r"COMMENT\s+ON\s+CONSTRAINT\s+meal_plans_calc_household_multiplier_range",
        source,
        re.IGNORECASE,
    ), (
        "P3-MULTIPLIER-DB-CHECK violation: no hay `COMMENT ON CONSTRAINT` "
        "que documente el origen. Un operador SRE que vea el constraint "
        "fallar en runtime debe poder hacer `\\d+ meal_plans` y "
        "encontrar el P-fix de origen en una línea."
    )
    assert "P3-MULTIPLIER-DB-CHECK" in source, (
        "P3-MULTIPLIER-DB-CHECK violation: COMMENT debe mencionar el "
        "marker `P3-MULTIPLIER-DB-CHECK` para forensics post-incident."
    )


# ---------------------------------------------------------------------------
# 10. Cross-link slug (P2-HIST-AUDIT-14): este ES el marker del bundle
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p3_multiplier_db_check"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        f"Filename DEBE contener `{expected_slug}` para que el cross-link "
        f"`test_p2_hist_audit_14_marker_test_link` lo matchee con el "
        f"marker activo `P3-MULTIPLIER-DB-CHECK · YYYY-MM-DD`."
    )
