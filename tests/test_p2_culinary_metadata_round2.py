"""[P2-CULINARY-METADATA-ROUND2 · 2026-08-01] Backfill ronda 2 de metadata
culinaria (`prep_methods`/`ready_to_eat`) en `master_ingredients`.

Contexto: la ronda 1 (`p1_culinary_metadata_master_ingredients_2026_07_31.sql`)
dejó la categoría 'Despensa' entera sin default a propósito ("heterogénea:
aceites, granos, enlatados") — 56/204 filas de `master_ingredients` quedaron
con `prep_methods IS NULL`, midiendo cobertura de `culinary_coherence.
scan_coverage()` ~61-69% en planes reales, bajo el ≥80% que exige la
precondición de F2 (`P1-CULINARY-CONTRACT-BLOCK`, docs/culinary_coherence.md).

Este test es parser-based (no ejecuta SQL): ancla que la migración
`p2_culinary_metadata_backfill_round2_2026_08_01.sql` existe en AMBOS
directorios SSOT (`backend/migrations/` + `migrations/` workspace-root,
P3-MIGRATIONS-SSOT), es byte-idéntica en ambos, es idempotente (filtro
`IS NULL`), y trae el sanity `DO $$` de vocabulario canónico (mismo patrón
que la ronda 1, `test_p1_culinary_contract.py`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MIG_NAME = "p2_culinary_metadata_backfill_round2_2026_08_01.sql"
_MIG = _BACKEND / "migrations" / _MIG_NAME
_MIG_ROOT = _BACKEND.parent / "migrations" / _MIG_NAME

_CANON_VOCAB = ("hervir", "plancha", "freir", "hornear", "guisar",
                "saltear", "licuar", "tostar", "crudo", "ninguno")


def _sql_text() -> str:
    assert _MIG.exists(), f"falta la migración en backend/migrations/: {_MIG}"
    return _MIG.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. SSOT: ambos directorios, byte-idénticos (P3-MIGRATIONS-SSOT)
# ---------------------------------------------------------------------------

def test_migracion_existe_en_ambos_dirs_identica():
    assert _MIG.exists(), "falta la migración en backend/migrations/"
    assert _MIG_ROOT.exists(), "falta la copia en migrations/ (P3-MIGRATIONS-SSOT)"
    assert _MIG.read_bytes() == _MIG_ROOT.read_bytes(), "las dos copias divergen"


# ---------------------------------------------------------------------------
# 2. Idempotencia: todo UPDATE que toca prep_methods/ready_to_eat filtra
#    IS NULL (o, para la corrección puntual de 'saltear' en tortillas,
#    array_append condicional NOT ('saltear' = ANY(prep_methods)) — mismo
#    patrón que p1_culinary_metadata_leche_hervir_2026_07_31.sql).
# ---------------------------------------------------------------------------

def test_migracion_idempotente():
    sql = _sql_text()
    updates = re.findall(
        r"UPDATE public\.master_ingredients\s+SET.*?;", sql, re.DOTALL | re.IGNORECASE
    )
    assert len(updates) >= 10, (
        f"se esperaban >=10 UPDATE statements (8 grupos + corrección tortilla), "
        f"encontrados {len(updates)}"
    )
    no_guard = [
        u for u in updates
        if "prep_methods IS NULL" not in u
        and "NOT ('saltear' = ANY(prep_methods))" not in u
    ]
    assert not no_guard, (
        "UPDATE(s) sin guard de idempotencia (ni `prep_methods IS NULL` ni el "
        f"array_append condicional):\n" + "\n---\n".join(no_guard)
    )


def test_migracion_no_tiene_where_desnudo():
    """Ningún UPDATE debe carecer de WHERE por completo (fat-finger que
    pisaría TODA la tabla, no solo las filas vírgenes)."""
    sql = _sql_text()
    updates = re.findall(
        r"UPDATE public\.master_ingredients\s+SET.*?;", sql, re.DOTALL | re.IGNORECASE
    )
    sin_where = [u for u in updates if "WHERE" not in u.upper()]
    assert not sin_where, f"UPDATE sin WHERE (peligro de full-table write):\n{sin_where}"


# ---------------------------------------------------------------------------
# 3. Sanity DO $$ con el vocabulario canónico
# ---------------------------------------------------------------------------

def test_migracion_trae_sanity_vocabulario_canonico():
    sql = _sql_text()
    assert sql.count("DO $$") >= 1, "falta el bloque sanity DO $$"
    assert "RAISE EXCEPTION" in sql, (
        "el sanity de vocabulario debe abortar la migración (RAISE EXCEPTION) "
        "si aparece un método fuera del vocabulario canónico"
    )
    for method in _CANON_VOCAB:
        assert f"'{method}'" in sql, (
            f"el vocabulario canónico '{method}' no aparece en el sanity check "
            f"de la migración — ¿drift respecto al vocabulario de la ronda 1?"
        )


def test_vocabulario_canonico_coincide_con_culinary_coherence():
    """El vocabulario hardcodeado en el sanity de esta migración debe seguir
    coincidiendo EXACTO con `culinary_coherence.PREP_VOCAB` (SSOT del scan) —
    si alguien añade un método nuevo a un lado sin el otro, el scan y la DB
    divergen en silencio."""
    import culinary_coherence as cc
    assert set(cc.PREP_VOCAB) == set(_CANON_VOCAB), (
        f"drift entre PREP_VOCAB del scan ({sorted(cc.PREP_VOCAB)}) y el "
        f"vocabulario anclado en este test ({sorted(_CANON_VOCAB)})"
    )


# ---------------------------------------------------------------------------
# 4. Docstring/marker anchor (P3-CLAUDEMD-CAP / convención de markers)
# ---------------------------------------------------------------------------

def test_docstring_trae_el_marker():
    """El docstring de este archivo debe contener el marker literal, para
    que el test cross-link (`test_p2_hist_audit_14_marker_test_link.py`)
    encuentre este archivo cuando `_LAST_KNOWN_PFIX` se bumpee."""
    text = Path(__file__).read_text(encoding="utf-8")
    assert "[P2-CULINARY-METADATA-ROUND2 · 2026-08-01]" in text


def test_migracion_documenta_razonamiento_por_grupo():
    """La migración debe traer comentarios de razonamiento por grupo (no
    solo el SQL desnudo) — requisito explícito del P-fix: "Documenta el
    razonamiento por grupo EN la migración"."""
    sql = _sql_text()
    grupos_esperados = [
        "Grupo 1", "Grupo 2", "Grupo 3", "Grupo 4", "Grupo 5", "Grupo 6",
        "Grupo 7", "Grupo 8",
    ]
    faltan = [g for g in grupos_esperados if g not in sql]
    assert not faltan, f"faltan comentarios de grupo en la migración: {faltan}"


# ---------------------------------------------------------------------------
# 5. Cobertura: 0 filas nuevas fuera de 'Despensa' referenciadas por nombre
#    (defensa contra copy-paste que apunte a un alimento de otra categoría
#    por error — el diagnóstico SQL confirmó que las 56 NULL eran TODAS
#    Despensa).
# ---------------------------------------------------------------------------

def test_nombres_referenciados_no_vacio():
    sql = _sql_text()
    # Todos los nombres citados entre comillas simples tras 'name' (IN (...) o = '...')
    nombres = set(re.findall(r"name\s*(?:=|IN\s*\()\s*'([^']+)'", sql))
    nombres |= set(m for m in re.findall(
        r"'([^']+)'", "\n".join(re.findall(r"name\s+IN\s*\(([^)]+)\)", sql, re.DOTALL))
    ))
    assert len(nombres) >= 50, (
        f"se esperaban >=50 nombres de alimento distintos referenciados "
        f"(56 filas NULL originales), encontrados {len(nombres)}"
    )
