"""[P1-UNKNOWN-CATALOG-FILTER · 2026-06-15] El log `unknown_ingredients` debe señalar GAPS REALES del
catálogo de macros, no falsos-positivos del mapa de variedad.

Contexto del bug (forense en prod 2026-06-14): las 11 entradas de `unknown_ingredients` eran TODAS
falsos-positivos — ingredientes que SÍ resuelven al catálogo de macros ("leche descremada"→Leche,
"semillas de chía"→Semillas de chía, "queso"/"lonjas de queso"→Queso blanco) o condimentos de cero
macros ("edulcorante al gusto"). El log se alimentaba del canonicalizador de VARIEDAD
(`canonical_bases = GLOBAL_REVERSE_MAP.values()`), que es MÁS ESTRECHO que el catálogo de macros →
ruido en la señal de "qué ingrediente agregar".

Fix (3 partes):
  1. `services._save_plan_and_track_background` enruta los no-canónicos por el resolver del catálogo
     (`IngredientNutritionDB.lookup`) ANTES de loguearlos; solo los que TAMPOCO resuelven son gaps reales.
     Knob de rollback: MEALFIT_UNKNOWN_LOG_USE_CATALOG.
  2. Alias genérico "queso" → "Queso blanco" (migración p1_queso_generic_alias_2026_06_15.sql) para que
     el genérico resuelva (seguro: el orden longest-first + match exacto protege los quesos específicos).
  3. "edulcorante"/"estevia"/etc en IGNORED_TRACKING_TERMS (condimentos de cero macros).

Además persiste `resolution_coverage` en plan_data al guardar (punto único confiable; antes 0/16 planes
en prod lo tenían). Marker [P1-PERSIST-RESOLUTION-COVERAGE].
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SERVICES = _REPO_ROOT / "backend" / "services.py"
_CONSTANTS = _REPO_ROOT / "backend" / "constants.py"
_MIG_ROOT = _REPO_ROOT / "migrations" / "p1_queso_generic_alias_2026_06_15.sql"
_MIG_BACKEND = _REPO_ROOT / "backend" / "migrations" / "p1_queso_generic_alias_2026_06_15.sql"


# ───────────────────────── parser-based: el fix está en el código ─────────────────────────

def test_services_routes_noncanonical_through_catalog_before_logging():
    """El filtro del catálogo (`_catalog_db.lookup`) debe aparecer ANTES del
    `log_unknown_ingredients` — si no, se loguean falsos-positivos."""
    src = _SERVICES.read_text(encoding="utf-8")
    assert "P1-UNKNOWN-CATALOG-FILTER" in src
    idx_filter = src.find("_genuine = [n for n in non_canonical if _catalog_db.lookup(n) is None]")
    idx_log = src.find("log_unknown_ingredients(user_id, non_canonical")
    assert idx_filter != -1, "filtro del catálogo ausente en services.py"
    assert idx_log != -1, "callsite de log_unknown_ingredients ausente"
    assert idx_filter < idx_log, "el filtro del catálogo debe ejecutarse ANTES de loguear unknowns"


def test_services_filter_is_knob_gated():
    src = _SERVICES.read_text(encoding="utf-8")
    assert "MEALFIT_UNKNOWN_LOG_USE_CATALOG" in src, "knob de rollback del filtro ausente"


def test_services_persists_resolution_coverage_at_save():
    """[P1-PERSIST-RESOLUTION-COVERAGE] coverage persistido en plan_data al guardar, knob-gated,
    sin recomputar si upstream ya lo puso."""
    src = _SERVICES.read_text(encoding="utf-8")
    assert "P1-PERSIST-RESOLUTION-COVERAGE" in src
    assert 'plan_data["resolution_coverage"]' in src
    assert "MEALFIT_PERSIST_RESOLUTION_COVERAGE" in src
    assert '"resolution_coverage" not in plan_data' in src, (
        "debe respetar el coverage que upstream (clinical layer) ya haya puesto — no recomputar")


def test_edulcorante_ignored_as_condiment():
    src = _CONSTANTS.read_text(encoding="utf-8")
    # En el set IGNORED_TRACKING_TERMS.
    block = re.search(r"IGNORED_TRACKING_TERMS\s*=\s*\{(.*?)\}", src, re.DOTALL)
    assert block, "IGNORED_TRACKING_TERMS no encontrado"
    assert '"edulcorante"' in block.group(1), "edulcorante debe estar en IGNORED_TRACKING_TERMS"


def test_queso_alias_migration_exists_idempotent_and_synced():
    for p in (_MIG_ROOT, _MIG_BACKEND):
        assert p.exists(), f"migración ausente: {p}"
    a = _MIG_ROOT.read_text(encoding="utf-8")
    b = _MIG_BACKEND.read_text(encoding="utf-8")
    assert a == b, "migración no sincronizada entre migrations/ y backend/migrations/ (SSOT)"
    assert "array_append(aliases, 'queso')" in a
    assert "NOT ('queso' = ANY(" in a, "la migración debe ser idempotente (guard NOT ... ANY)"
    assert "RAISE EXCEPTION" in a, "la migración debe tener sanity check"


# ───────────────────────── funcional: el resolver con el alias es correcto y seguro ─────────────────────────

def _injected_queso_db():
    from nutrition_db import IngredientNutritionDB
    rows = [
        {"name": "Queso blanco", "aliases": ["queso", "queso blanco fresco", "queso de freir"],
         "kcal_per_100g": 298.5, "protein_g_per_100g": 18.1, "carbs_g_per_100g": 2.98, "fats_g_per_100g": 23.8},
        {"name": "Queso crema", "aliases": ["cream cheese", "queso crema philadelphia"],
         "kcal_per_100g": 356.3, "protein_g_per_100g": 6.15, "carbs_g_per_100g": 5.52, "fats_g_per_100g": 34.4},
        {"name": "Queso mozzarella", "aliases": ["mozzarella", "queso mozarela"],
         "kcal_per_100g": 297.3, "protein_g_per_100g": 22.2, "carbs_g_per_100g": 2.4, "fats_g_per_100g": 22.1},
        {"name": "Leche", "aliases": ["leche descremada", "leche entera", "leche liquida"],
         "kcal_per_100g": 61, "protein_g_per_100g": 3.15, "carbs_g_per_100g": 4.8, "fats_g_per_100g": 3.25},
        {"name": "Semillas de chia", "aliases": ["chia", "semilla de chia"],
         "kcal_per_100g": 510.7, "protein_g_per_100g": 16.5, "carbs_g_per_100g": 42.1, "fats_g_per_100g": 30.7},
    ]
    return IngredientNutritionDB(rows=rows)


def test_generic_queso_resolves_to_queso_blanco():
    db = _injected_queso_db()
    for s in ("queso", "lonjas/pedazos de queso", "lonja/pedazo de queso", "2 lonjas de queso"):
        r = db.lookup(s)
        assert r is not None and r.name == "Queso blanco", f"{s!r} debió resolver a Queso blanco, dio {r}"


def test_specific_cheeses_do_not_collapse_to_blanco():
    """SEGURIDAD del alias genérico: los quesos específicos NO deben colapsar a Queso blanco."""
    db = _injected_queso_db()
    for s, expected in (("queso crema", "Queso crema"),
                        ("queso crema philadelphia", "Queso crema"),
                        ("queso mozzarella", "Queso mozzarella"),
                        ("mozzarella", "Queso mozzarella")):
        r = db.lookup(s)
        assert r is not None and r.name == expected, f"{s!r} debió resolver a {expected}, dio {r}"


def test_leche_descremada_and_chia_resolve_via_alias():
    db = _injected_queso_db()
    assert (db.lookup("leche descremada") or None) and db.lookup("leche descremada").name == "Leche"
    assert db.lookup("leche descremada (61ml)").name == "Leche"
    assert db.lookup("semillas de chia (36g)").name == "Semillas de chia"
