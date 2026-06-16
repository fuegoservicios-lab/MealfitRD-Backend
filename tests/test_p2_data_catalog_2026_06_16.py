"""[P2-DATA-CATALOG · 2026-06-16] Lote de datos del audit fresco (catálogo master_ingredients).

- P2-1: "Yema de huevo" separada de "Huevo" (entero) — la yema real es ~3x grasa/colesterol.
- P2-2: "Leche descremada" separada de "Leche" (entera) + reactiva el swap de dislipidemia.
- P2-3: "Yogur griego entero" separado de "Yogurt griego sin azúcar" (nonfat).
- P2-4: ningún alimento alto en grasa (fats>5) queda sin satfat (NULL) → falso 'ok' a dislipidémicos.

Las aserciones de DATOS necesitan Neon (skip sin NEON_DATABASE_URL → se validan en VPS/CI con DB).
Las de archivos/migración son deterministas.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent          # workspace root
_BE = Path(__file__).resolve().parent.parent                   # backend/

try:
    import psycopg
    import dotenv
    dotenv.load_dotenv(_BE / ".env")
    _NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
except Exception:
    _NEON = None

_skip = pytest.mark.skipif(not _NEON, reason="sin NEON_DATABASE_URL → datos del catálogo se validan en VPS/CI")


def _q(sql, params=None):
    with psycopg.connect(_NEON) as conn:
        return conn.execute(sql, params or ()).fetchall()


# ───────────────────────── migraciones en AMBOS dirs (SSOT) ─────────────────────────
@pytest.mark.parametrize("fname", [
    "p2_1_yema_de_huevo_ingredient_2026_06_16.sql",
    "p2_2_leche_descremada_ingredient_2026_06_16.sql",
    "p2_3_yogur_griego_entero_ingredient_2026_06_16.sql",
])
def test_migration_in_both_dirs(fname):
    a = _BE / "migrations" / fname
    b = _ROOT / "migrations" / fname
    assert a.exists() and b.exists(), f"{fname} debe existir en backend/migrations Y migrations/ (SSOT)"
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8"), f"{fname} difiere entre dirs"


def test_populate_has_usda_queries():
    src = (_BE / "scripts" / "populate_nutrition_db.py").read_text(encoding="utf-8")
    assert '"Yema de huevo": "egg yolk' in src
    assert '"Leche descremada": "milk nonfat' in src
    assert '"Yogur griego entero": "yogurt greek plain whole' in src


# ───────────────────────── P2-1/2/3: filas nuevas separadas (datos) ─────────────────────────
@_skip
def test_yema_separated_from_whole_egg():
    rows = _q("SELECT name, fats_g_per_100g, cholesterol_mg_per_100g FROM master_ingredients WHERE name='Yema de huevo'")
    assert rows, "falta la fila Yema de huevo"
    _, fats, chol = rows[0]
    assert float(fats) > 15, f"la yema debe tener grasa alta (real ~26g), no la del entero (~9.5): {fats}"
    assert float(chol) > 600, f"la yema debe tener colesterol alto (~1085mg), no el del entero (372): {chol}"
    # 'yema de huevo' ya NO debe ser alias del entero
    huevo = _q("SELECT aliases FROM master_ingredients WHERE name='Huevo'")[0][0]
    assert "yema de huevo" not in huevo


@_skip
def test_leche_descremada_separated():
    rows = _q("SELECT fats_g_per_100g FROM master_ingredients WHERE name='Leche descremada'")
    assert rows and float(rows[0][0]) < 1.0, "leche descremada debe tener grasa ~0, no la de la entera (3.25)"
    leche = _q("SELECT aliases FROM master_ingredients WHERE name='Leche'")[0][0]
    assert "leche descremada" not in leche


@_skip
def test_yogur_entero_separated():
    rows = _q("SELECT fats_g_per_100g FROM master_ingredients WHERE name='Yogur griego entero'")
    assert rows and float(rows[0][0]) > 3, "yogur griego entero debe tener grasa ~4, no la del nonfat (0.37)"
    nonfat = _q("SELECT aliases FROM master_ingredients WHERE name='Yogurt griego sin azúcar'")[0][0]
    assert "yogur griego" not in nonfat, "el bare 'yogur griego' debe quedar en el entero, no en el nonfat"


# ───────────────────────── P2-2: swap de dislipidemia reactivado ─────────────────────────
def test_dyslipidemia_leche_swap_reactivated():
    import condition_rules as cr
    subs = cr.collect_substitutions({"medicalConditions": ["Colesterol alto"]})
    repls = {s["replacement"] for s in subs}
    assert "Leche descremada" in repls, "el swap dislipidemia leche entera→descremada debe estar reactivado"


# ───────────────────────── P2-4: guard de cobertura satfat ─────────────────────────
# Whitelist documentada: alimentos donde USDA genuinamente NO reporta satfat desglosado. HOY vacía —
# 'Almendras fileteadas' (Foundation fdc 2261420 sin satfat) se completó con el valor SR de almendras
# (3.802 g/100g, colesterol 0). Si una fila futura genuinamente carece de satfat en USDA, añadirla aquí
# con su razón en vez de inventar un valor.
_SATFAT_NULL_WHITELIST: set = set()


@_skip
def test_no_high_fat_row_missing_satfat():
    rows = _q("SELECT name FROM master_ingredients WHERE fats_g_per_100g > 5 AND saturated_fat_g_per_100g IS NULL")
    offenders = [r[0] for r in rows if r[0] not in _SATFAT_NULL_WHITELIST]
    assert not offenders, (
        f"alimentos altos en grasa SIN satfat (dato faltante → falso 'ok' a dislipidémicos): {offenders}. "
        f"Correr backfill_p4_dash_cols.py o añadir a la whitelist documentada si USDA no lo reporta.")
