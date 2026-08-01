"""[P1-CULINARY-CONTRACT · 2026-07-31] Capa determinista de coherencia culinaria
(F1 del spec): migración de metadata + scan V1/V2/V3 + 3 superficies en warn."""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_MIG = _BACKEND / "migrations" / "p1_culinary_metadata_master_ingredients_2026_07_31.sql"
_MIG_ROOT = _BACKEND.parent / "migrations" / _MIG.name


def test_migracion_existe_en_ambos_dirs_identica():
    assert _MIG.exists(), "falta la migración en backend/migrations/"
    assert _MIG_ROOT.exists(), "falta la copia en migrations/ (P3-MIGRATIONS-SSOT)"
    assert _MIG.read_bytes() == _MIG_ROOT.read_bytes(), "las dos copias divergen"


def test_migracion_idempotente_y_con_sanity():
    sql = _MIG.read_text(encoding="utf-8")
    assert sql.count("ADD COLUMN IF NOT EXISTS") >= 2
    assert "prep_methods text[]" in sql and "ready_to_eat boolean" in sql
    assert "DO $$" in sql, "falta el bloque sanity DO $$"
    # Regex (no substring literal): las filas de Backfill 1 combinan la categoría con
    # el filtro IS NULL en la misma cláusula WHERE ("WHERE category = 'X' AND
    # prep_methods IS NULL"), así que la subcadena "WHERE prep_methods IS NULL" sola
    # nunca aparece verbatim aunque el filtro SÍ esté aplicado.
    assert re.search(r"WHERE\b.*prep_methods IS NULL", sql), (
        "el backfill debe filtrar IS NULL para ser re-ejecutable sin pisar "
        "overrides manuales posteriores")
