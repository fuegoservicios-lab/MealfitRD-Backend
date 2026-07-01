"""[P1-EXTENDED-MICROS-GUARD · 2026-07-01] (audit objetivo v2 · P1-2 · completitud micros extendidos)

Gap que cierra: las 8 columnas EXTENDIDAS del panel (zinc/folato/vitA/vitC/vitE/vitK/selenio/omega3)
se poblaron solo `WHERE fdc_id IS NOT NULL`, SIN el DO-block cero-NULL que protege a las 10 BASE
(p1_micronutrient_catalog_backfill_2026_06_24.sql). Una celda NULL descarta el aporte del ingrediente
→ cobertura por-nutriente <0.6 → `estimado_bajo` → el micro-closer SALTA ese micro. Verificación en
vivo del audit: quedaba exactamente 1 fila ('Gandules', vitE/vitK — staple del moro DD), oculta porque
el script de cobertura redondeaba 201/202 a "100%".

Fix (migración p1_extended_micros_zero_null_guard_2026_07_01.sql, aplicada a Neon el 2026-07-01):
  - Backfill Gandules vitE=0.2 / vitK=5 (proxy leguminosa conservador-bajo, COALESCE idempotente).
  - DO-block cero-NULL espejo del base: regresión futura (alimento nuevo sin panel extendido) revienta
    la migración — completar backfill, NO relajar el guard.
  - scripts/check_extended_micro_coverage.py: stdout utf-8 (crasheaba en cp1252) + columna `nulls` con
    conteo exacto (el `:.0%` ocultaba filas sueltas) + exit 2 si CUALQUIER fila NULL (contrato cero-NULL).

Tests parser-based (sin DB): la migración existe en AMBOS dirs SSOT, idéntica, cubre las 8 columnas
con RAISE EXCEPTION; el script ops tiene los 3 fixes.
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_WORKSPACE = _BACKEND.parent
_MIG_NAME = "p1_extended_micros_zero_null_guard_2026_07_01.sql"
_MIG_BACKEND = _BACKEND / "migrations" / _MIG_NAME
_MIG_ROOT = _WORKSPACE / "migrations" / _MIG_NAME
_CHECK_SCRIPT = (_BACKEND / "scripts" / "check_extended_micro_coverage.py").read_text(encoding="utf-8")

_EXTENDED_COLS = (
    "zinc_mg_per_100g", "folate_mcg_dfe_per_100g", "vitamin_a_mcg_rae_per_100g", "vitamin_c_mg_per_100g",
    "vitamin_e_mg_per_100g", "vitamin_k_mcg_per_100g", "selenium_mcg_per_100g", "omega3_ala_g_per_100g",
)


def test_migration_exists_in_backend_dir():
    assert _MIG_BACKEND.exists(), f"falta {_MIG_NAME} en backend/migrations/"


def test_migration_ssot_synced_both_dirs():
    """[P3-MIGRATIONS-SSOT] toda migración vive IDÉNTICA en migrations/ y backend/migrations/."""
    if not _MIG_ROOT.exists():
        import pytest
        pytest.skip("workspace-root migrations/ no disponible en este checkout (repo backend standalone)")
    assert _MIG_BACKEND.read_text(encoding="utf-8") == _MIG_ROOT.read_text(encoding="utf-8"), \
        "drift entre migrations/ y backend/migrations/ — sincronizar ambos dirs"


def test_do_block_covers_all_8_extended_columns():
    sql = _MIG_BACKEND.read_text(encoding="utf-8")
    assert "DO $$" in sql and "RAISE EXCEPTION" in sql, "falta el DO-block cero-NULL"
    for col in _EXTENDED_COLS:
        assert f"{col} IS NULL" in sql, f"el DO-block no cubre {col}"


def test_migration_backfills_gandules_idempotent():
    sql = _MIG_BACKEND.read_text(encoding="utf-8")
    assert "Gandules" in sql
    assert "COALESCE(vitamin_e_mg_per_100g" in sql and "COALESCE(vitamin_k_mcg_per_100g" in sql, \
        "el backfill debe ser COALESCE (idempotente, no pisa valores existentes)"


def test_check_script_utf8_and_exact_null_counts():
    assert 'sys.stdout.reconfigure(encoding="utf-8"' in _CHECK_SCRIPT, \
        "el script ops crasheaba en consola Windows cp1252 (falta reconfigure)"
    assert "n_null = total - g" in _CHECK_SCRIPT, \
        "la tabla debe mostrar el conteo EXACTO de NULLs (el % redondeado ocultó Gandules)"
    assert "n_any_null" in _CHECK_SCRIPT and "sys.exit(2)" in _CHECK_SCRIPT, \
        "cualquier fila NULL debe salir con exit 2 (contrato cero-NULL)"


def test_marker_anchor_present():
    assert "P1-EXTENDED-MICROS-GUARD" in _MIG_BACKEND.read_text(encoding="utf-8")
    assert "P1-EXTENDED-MICROS-GUARD" in _CHECK_SCRIPT
