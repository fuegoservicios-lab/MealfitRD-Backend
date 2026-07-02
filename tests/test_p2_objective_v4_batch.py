"""[P2-OBJECTIVE-V4-BATCH · 2026-07-02] Test ancla del batch de 13 P2 del audit objetivo v4.

Ejes: macros (#5 carb-floor updates, #6 kcal backstop, #7 doc-drift, #9 resolución de platos
compuestos), creatividad/slots (#11 raw-staple gate, #12 telemetría de evasiones, #13 dish-quality
pressure en swap), presupuesto/lista (#1 parity de pisos, #2 size_grams, #3 pkg-sync report,
#4 brands apply-immediate), ops (#8 nightly loud-skip) y datos (#10 cerrado por verificación —
0 NULLs en las 18 columnas de micros del catálogo).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"


def _read(rel: str, base: Path = _BACKEND) -> str:
    return (base / rel).read_text(encoding="utf-8")


_GO = _read("graph_orchestrator.py")
_PL = _read("routers/plans.py")
_TO = _read("tools.py")
_AG = _read("agent.py")
_SC = _read("shopping_calculator.py")
_NDB = _read("nutrition_db.py")
_SUP = _read("routers/supermarket.py")


# ════════════════════════════════════════════════════════════════════════════
# Marker + presencia de los 13 sub-markers
# ════════════════════════════════════════════════════════════════════════════
def test_marker_bumped():
    assert "P2-OBJECTIVE-V4-BATCH · 2026-07-02" in _read("app.py")


def test_all_submarkers_present():
    hay = _GO + _PL + _TO + _AG + _SC + _NDB + _SUP
    hay += _read(".github/workflows/macro-benchmark-nightly.yml")
    hay += _read("scripts/sync_supermarket_prices_report_2026_07_02.py")
    for marker in (
        "P2-CARB-FLOOR-UPDATES", "P2-KCAL-GATE-BACKSTOP", "P2-KNOB-DOC-DRIFT",
        "P2-COMPOUND-DISH-RESOLUTION", "P2-RAW-STAPLE-PRESSURE", "P2-SLOT-EVASION-TELEMETRY",
        "P2-UPDATE-DISHQUALITY-PRESSURE", "P2-BRANDPREF-SIZE-COLUMN",
        "P2-SUPERMARKET-PKG-SYNC", "P2-NIGHTLY-BENCH-ACTIVATE",
    ):
        assert marker in hay, f"falta el sub-marker {marker}"


# ════════════════════════════════════════════════════════════════════════════
# #1 P2-BUDGET-FLOOR-SSOT — paridad de pisos frontend ↔ backend
# ════════════════════════════════════════════════════════════════════════════
def test_budget_floor_parity_frontend_backend():
    """Los pisos DOP por ciclo viven duplicados (formValidation.js ↔ nutrition_calculator)
    sincronizados A MANO — este test es el candado del drift (audit v4 presupuesto gap #8)."""
    import nutrition_calculator as nc
    backend_floors = {int(k): float(v) for k, v in nc._BUDGET_CYCLE_FLOOR_DEFAULTS_DOP.items()}
    js = _read("src/config/formValidation.js", base=_FRONTEND)
    m = re.search(r"DOP\s*:\s*\{([^}]*)\}", js, re.DOTALL)
    assert m, "no se encontró el bloque DOP de BUDGET_MIN_TOTAL en formValidation.js"
    js_vals = {k: float(v) for k, v in re.findall(r"(weekly|biweekly|monthly)\s*:\s*(\d+)", m.group(1))}
    mapping = {"weekly": 7, "biweekly": 15, "monthly": 30}
    for js_key, days in mapping.items():
        assert js_vals.get(js_key) == backend_floors.get(days), (
            f"drift de piso {js_key}: frontend={js_vals.get(js_key)} vs "
            f"backend[{days}d]={backend_floors.get(days)} — sincronizar AMBOS lados"
        )


# ════════════════════════════════════════════════════════════════════════════
# #5 carb-floor en updates · #6 kcal backstop
# ════════════════════════════════════════════════════════════════════════════
def test_carb_floor_updates_wired():
    for src, label, occurrences in ((_PL, "plans.py", 1), (_TO, "tools.py", 2)):
        assert src.count('MEALFIT_CARB_FLOOR_UPDATES", "true"') >= occurrences or \
            src.count("MEALFIT_CARB_FLOOR_UPDATES") >= occurrences, \
            f"{label}: falta el knob del carb-floor de updates"
        assert "_close_carb_gap_for_day" in src, f"{label}: no invoca el closer aditivo"
    # regen-day lo OMITE por diseño (pantry-strict) — documentado junto al micro-closer.
    assert "mismo diseño para el floor ADITIVO de carbos" in _PL
    # tools corre el floor en las DOS pasadas del patrón closer-order.
    assert "_floor_pre" in _TO and "_floor_cm" in _TO
    assert "_adj_pre or _floor_pre" in _TO and "_adj_cm or _floor_cm" in _TO
    assert "_adj_swap or _floor_swap" in _PL


def test_kcal_backstop_knobs_and_wiring(go):
    assert go.BAND_GATE_KCAL_BACKSTOP is True
    assert go.BAND_GATE_KCAL_THRESHOLD == pytest.approx(0.5)
    i = _GO.index("_kcal_trigger = bool(BAND_GATE_KCAL_BACKSTOP")
    window = _GO[i:i + 1200]
    assert "_agg_trigger or _per_macro_trigger or _kcal_trigger" in window


# ════════════════════════════════════════════════════════════════════════════
# #7 doc-drift — los literals stale NO deben reaparecer
# ════════════════════════════════════════════════════════════════════════════
def test_stale_default_off_comments_gone():
    for stale in (
        "Default OFF (CARB_FLOOR_ENABLED",
        "default OFF — opt-in tras medir la distribución",
        "Default OFF (opt-in). Anchor: P2-DISH-QUALITY-GATE",
        "(cada sub-check tras su knob, default OFF)",
        "gated OFF hasta confirmar el A/B",
        "knob, default OFF): condición declarada",
    ):
        assert stale not in _GO, f"comment stale reapareció en graph_orchestrator: {stale!r}"
    assert "influye en `_get_fast_filtered_catalogs` (catálogo de" not in _PL, \
        "el comment falso de budget→catálogo reapareció en plans.py"


# ════════════════════════════════════════════════════════════════════════════
# #9 resolución de platos compuestos (biblioteca G17 revivida como RESOLUCIÓN)
# ════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def test_compound_dish_lookup_resolves_moro(monkeypatch):
    import nutrition_db as ndb
    monkeypatch.setattr(ndb, "_UNIFIED_RESOLVER_ENABLED", False)  # determinista/offline
    db = ndb.IngredientNutritionDB(rows=[])  # catálogo vacío → fuerza el tier compuesto
    info = db.lookup("moro")
    assert info is not None, "el tier compuesto debe resolver 'moro'"
    assert info.source == "dominican_dish_decomp"
    assert info.kcal > 100 and info.carbs > 10
    macros = db.macros_from_ingredient_string("300g de moro")
    assert macros and macros["grams"] == 300
    assert macros["kcal"] == pytest.approx(info.kcal * 3, rel=0.01)


def test_compound_dish_knob_off(monkeypatch):
    import nutrition_db as ndb
    monkeypatch.setattr(ndb, "_UNIFIED_RESOLVER_ENABLED", False)
    monkeypatch.setattr(ndb, "_COMPOUND_DISH_ENABLED", False)
    db = ndb.IngredientNutritionDB(rows=[])
    assert db.lookup("moro") is None


def test_compound_dish_only_when_catalog_misses(monkeypatch):
    """Si el catálogo SÍ resuelve, el tier compuesto no interviene (fallback FINAL)."""
    import nutrition_db as ndb
    monkeypatch.setattr(ndb, "_UNIFIED_RESOLVER_ENABLED", False)
    row = {"name": "Moro casero", "aliases": ["moro"], "kcal_per_100g": 999.0,
           "protein_g_per_100g": 1.0, "carbs_g_per_100g": 1.0, "fats_g_per_100g": 1.0}
    db = ndb.IngredientNutritionDB(rows=[row])
    info = db.lookup("moro")
    assert info is not None and info.kcal == pytest.approx(999.0), \
        "el row del catálogo debe ganar sobre la biblioteca compuesta"


# ════════════════════════════════════════════════════════════════════════════
# #11 raw-staple gate · #12 telemetría de evasiones · #13 dish pressure en swap
# ════════════════════════════════════════════════════════════════════════════
def test_raw_staple_gate_defaults_and_wiring(go):
    assert go.RAW_STAPLE_SOFT_GATE_ENABLED is True
    assert go.RAW_STAPLE_REJECT_RATIO == pytest.approx(0.5)
    i = _GO.index("if RAW_STAPLE_SOFT_GATE_ENABLED:")
    window = _GO[i:i + 3000]
    assert "_raw_staple_advisory_final" in window, "falta el degrade a advisory en intento final"
    assert "raw_staple_ratio" in window


def test_slot_evasion_telemetry_wired():
    assert "P2-SLOT-EVASION-TELEMETRY" in _GO
    assert '"name_evaded": not _name_flagged' in _GO


def test_swap_dish_quality_pressure_wired():
    i = _AG.index("P2-UPDATE-DISHQUALITY-PRESSURE")
    window = _AG[i:i + 2600]
    assert '"MEALFIT_SWAP_DISH_QUALITY_PRESSURE", "true"' in window, "knob default ON"
    assert "strict_pantry and not clean_ingredients" in window, "debe respetar el guard pantry-strict"
    assert 'raise ValueError("DISH_QUALITY: "' in window, \
        "la presión usa ValueError (exento del CB por P2-CB-GUARDRAIL-NOT-FAILURE)"


# ════════════════════════════════════════════════════════════════════════════
# #2 size_grams · #3 pkg-sync report · #4 apply-immediate · #8 nightly
# ════════════════════════════════════════════════════════════════════════════
def test_size_grams_column_wired():
    assert "size_grams" in _SUP and '"size_grams"' in _SUP, "router: falta size_grams en whitelist"
    assert "size_grams::float8" in _SUP, "router: falta en _SELECT_COLS"
    # Overlay de costeo: size_grams explícito primero, parser como fallback.
    i = _SC.index("def fetch_brand_pref_packages(")
    body = _SC[i:_SC.index("\ndef ", i + 10)]
    assert "size_grams" in body, "el overlay debe preferir size_grams explícito"
    # Migración SSOT en AMBOS dirs (P3-MIGRATIONS-SSOT).
    for base in (_BACKEND / "migrations", _BACKEND.parent / "migrations"):
        p = base / "p2_supermarket_size_grams_2026_07_02.sql"
        assert p.exists(), f"migración ausente en {base}"
        assert "ADD COLUMN IF NOT EXISTS size_grams" in p.read_text(encoding="utf-8")
    # Admin UI del /supermercado con el campo.
    sup_page = _read("src/pages/SupermarketPage.jsx", base=_FRONTEND)
    assert "size_grams" in sup_page, "SupermarketPage: falta el campo de tamaño de envase"


def test_pkg_sync_report_script_is_report_first():
    src = _read("scripts/sync_supermarket_prices_report_2026_07_02.py")
    assert "DRY-RUN" in src and "--commit" in src
    assert "PRICE_GUARD = 2.5" in src, "guard bidireccional 2.5× del commit"
    assert "brand IS NULL" in src, "solo genéricos del súper"


def test_brands_apply_immediate_wired():
    brands = _read("src/components/dashboard/SupermarketBrands.jsx", base=_FRONTEND)
    assert "onPrefApplied" in brands and "applyTimerRef" in brands, "falta el debounce del re-costeo"
    assert "P2-BRANDS-APPLY-IMMEDIATE" in brands
    dash = _read("src/pages/Dashboard.jsx", base=_FRONTEND)
    assert "onPrefApplied" in dash and "P2-BRANDS-APPLY-IMMEDIATE" in dash
    assert "al instante" in brands, "la nota de UI debe reflejar el apply inmediato"


def test_nightly_loud_skip_and_doc():
    yml = _read(".github/workflows/macro-benchmark-nightly.yml")
    assert "GITHUB_STEP_SUMMARY" in yml, "el skip debe escribir al step summary (visible)"
    assert "docs/nightly_benchmark_activation.md" in yml
    assert "if: failure()" in yml, "el fallo debe dejar resumen de REGRESIÓN"
    doc = _read("docs/nightly_benchmark_activation.md")
    for secret in ("DEEPSEEK_API_KEY", "NEON_DATABASE_URL_POOLED", "NEON_DATABASE_URL"):
        assert secret in doc


# ════════════════════════════════════════════════════════════════════════════
# #10 cerrado por VERIFICACIÓN — 0 NULLs en las 18 columnas de micros (Neon)
# ════════════════════════════════════════════════════════════════════════════
_MICRO_COLS = (
    "fiber_g_per_100g", "sodium_mg_per_100g", "vitamin_d_mcg_per_100g", "calcium_mg_per_100g",
    "iron_mg_per_100g", "vitamin_b12_mcg_per_100g", "potassium_mg_per_100g", "magnesium_mg_per_100g",
    "saturated_fat_g_per_100g", "zinc_mg_per_100g", "folate_mcg_dfe_per_100g",
    "vitamin_a_mcg_rae_per_100g", "vitamin_c_mg_per_100g", "vitamin_e_mg_per_100g",
    "vitamin_k_mcg_per_100g", "selenium_mcg_per_100g", "omega3_ala_g_per_100g", "sugars_g_per_100g",
)


def test_catalog_micros_fully_populated():
    """El gap #10 del plan v4 ('~14 sin fdc_id salen estimado') quedó CERRADO por datos
    (verificado 2026-07-02: 202/202 filas completas). Este candado detecta regresión: una fila
    nueva/UPDATE con NULL en cualquiera de las 18 columnas reabre el `estimado_bajo` que el
    micro-closer salta a propósito."""
    try:
        from db import execute_sql_query
        checks = ", ".join(f"COUNT(*) FILTER (WHERE {c} IS NULL) AS n{i}"
                           for i, c in enumerate(_MICRO_COLS))
        row = execute_sql_query(f"SELECT COUNT(*) AS total, {checks} FROM master_ingredients",
                                fetch_one=True)
    except Exception as exc:  # pragma: no cover - entorno sin Neon
        pytest.skip(f"Neon no disponible en este entorno: {exc}")
    assert row and row.get("total", 0) > 0
    nulls = {c: row.get(f"n{i}") for i, c in enumerate(_MICRO_COLS) if row.get(f"n{i}")}
    assert not nulls, f"columnas de micros con NULLs (reabre estimado_bajo): {nulls}"
