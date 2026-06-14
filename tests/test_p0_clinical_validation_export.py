"""[P0-CLINICAL-VALIDATION · 2026-06-14] Anchors del export de validación clínica (gap P0 del audit:
"precisión autoafirmada, sin validación externa/humana"). El script muestrea planes reales y compara
target vs entregado(claim del LLM) vs recomputado desde ingredientes (catálogo = ground-truth). Aquí
cubrimos la lógica pura + anchors parser-based (un renombre rompe el test antes que producción)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_SCRIPT = _BACKEND / "scripts" / "clinical_validation_export.py"


def _load():
    spec = importlib.util.spec_from_file_location("clinical_validation_export", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists_and_imports():
    assert _SCRIPT.exists(), "scripts/clinical_validation_export.py debe existir"
    mod = _load()
    for fn in ("_num", "_delivered_day_claimed", "_recompute_day_from_ingredients", "_open_pools", "main"):
        assert hasattr(mod, fn), f"falta {fn}"


def test_num_parses_macro_strings():
    mod = _load()
    assert mod._num("154g") == 154.0
    assert mod._num("2050 kcal") == 2050.0
    assert mod._num(120) == 120.0
    assert mod._num(None) == 0.0
    assert mod._num("P:30g") == 30.0


def test_delivered_day_claimed_sums_meal_headers():
    mod = _load()
    day = {"meals": [
        {"protein": 30, "carbs": 40, "fats": 10, "cals": 350},
        {"protein": "25g", "carbs": "50g", "fats": "12g", "cals": "420"},
    ]}
    agg = mod._delivered_day_claimed(day)
    assert agg["protein"] == 55.0
    assert agg["carbs"] == 90.0
    assert agg["fats"] == 22.0
    assert agg["kcal"] == 770.0


def test_recompute_uses_catalog_db_stub():
    """El recompute suma macros_from_ingredient_string del catálogo (ground-truth determinista)."""
    mod = _load()

    class _StubDB:
        def macros_from_ingredient_string(self, s):
            s = s.lower()
            if "pollo" in s:
                return {"protein": 30.0, "carbs": 0.0, "fats": 3.0, "kcal": 150.0}
            if "arroz" in s:
                return {"protein": 4.0, "carbs": 45.0, "fats": 0.5, "kcal": 200.0}
            return {}  # no resuelve

    day = {"meals": [{"ingredients": ["150g de pollo", "1 taza de arroz", "vegetales al gusto"]}]}
    agg, resolved, total = mod._recompute_day_from_ingredients(day, _StubDB())
    assert total == 3 and resolved == 2  # 'vegetales' no resuelve
    assert agg["protein"] == 34.0
    assert agg["carbs"] == 45.0


def test_anchor_marker_and_query_filter_present():
    """El script usa el filtro correcto (status chunked, no 'complete') + el marker."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "P0-CLINICAL-VALIDATION" in src
    assert "<> 'failed'" in src  # excluye solo failed (no exige 'complete', que no existe en chunked)
    assert "macros_from_ingredient_string" in src
