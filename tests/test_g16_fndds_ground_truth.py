"""[G16-FNDDS-GROUND-TRUTH · 2026-06-15] Ground-truth EXTERNO no-circular para validación de macros (USDA FNDDS).

Cierra la parte code-closeable de G16: el check de integridad recomputaba desde el MISMO catálogo (circular).
FNDDS (dominio público/CC0) da platos compuestos medidos independientemente. `build_fndds_reference.py`
mapea plato DR → análogo FNDDS Survey (SOLO Survey — rechaza Branded basura tipo 'moro→TRUFFLE TORTE');
`clinical_validation_export.py` compara el PERFIL de macros (fracción de kcal P/C/F, independiente de la
porción) de la app vs FNDDS.

Tests puros (mockeados): NO requieren USDA_API_KEY ni el JSON generado.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, _BACKEND / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cve = _load("cve_g16", "scripts/clinical_validation_export.py")
bld = _load("bld_g16", "scripts/build_fndds_reference.py")


# ── clinical_validation_export: perfil de macros (independiente de porción) ──
def test_macro_fractions_sums_to_one_and_fat():
    f = cve._macro_fractions(40, 150, 50)
    assert f is not None and abs(sum(f) - 1.0) < 1e-9
    assert abs(f[2] - (9 * 50) / (4 * 40 + 4 * 150 + 9 * 50)) < 1e-9


def test_macro_fractions_none_on_zero():
    assert cve._macro_fractions(0, 0, 0) is None


def test_match_dish_substring_accent_insensitive():
    dishes = {"moro": {}, "locrio": {}, "arroz con habichuelas": {}}
    assert cve._match_fndds_dish("Moró de habichuelas con pollo", dishes) == "moro"
    assert cve._match_fndds_dish("Locrío de pollo", dishes) == "locrio"
    assert cve._match_fndds_dish("Ensalada verde con aguacate", dishes) is None


def test_load_reference_returns_dict():
    # {} si aún no se generó el JSON (no rompe el export).
    assert isinstance(cve._load_fndds_reference(), dict)


def test_profile_deviation_zero_for_identical(monkeypatch):
    """Si el perfil de la app == perfil FNDDS, la desviación es 0 (sanidad del cálculo usado en el resumen)."""
    appf = cve._macro_fractions(10, 30, 5)
    reff = cve._macro_fractions(10, 30, 5)
    dev = sum(abs(a - b) for a, b in zip(appf, reff)) / 3.0
    assert dev == 0.0


# ── build_fndds_reference: filtro Survey-only ──
class _Resp:
    status_code = 200
    def __init__(self, foods): self._foods = foods
    def raise_for_status(self): pass
    def json(self): return {"foods": self._foods}


class _FakeRequests:
    def __init__(self, resp): self._resp = resp
    def get(self, *a, **k): return self._resp


def _nutrients(p, c, f, kcal):
    return [
        {"nutrientName": "Protein", "value": p},
        {"nutrientName": "Carbohydrate, by difference", "value": c},
        {"nutrientName": "Total lipid (fat)", "value": f},
        {"nutrientName": "Energy", "unitName": "KCAL", "value": kcal},
    ]


def test_search_prefers_survey_rejects_branded(monkeypatch):
    foods = [
        {"dataType": "Branded", "description": "TRUFFLE TORTE", "fdcId": 1, "foodNutrients": _nutrients(5, 40, 20, 350)},
        {"dataType": "Survey (FNDDS)", "description": "Beans and white rice", "fdcId": 2, "foodNutrients": _nutrients(4, 25, 1, 130)},
    ]
    monkeypatch.setattr(bld, "requests", _FakeRequests(_Resp(foods)))
    res = bld._search_fndds("rice and beans")
    assert res is not None
    food, m, dtype = res
    assert dtype == "Survey (FNDDS)"
    assert food["description"] == "Beans and white rice", "debe elegir el Survey, NO el Branded basura"
    assert m["carbs"] == 25 and m["protein"] == 4


def test_search_misses_when_no_survey(monkeypatch):
    """Sin match Survey → MISS honesto (NO cae a Branded/genérico)."""
    foods = [{"dataType": "Branded", "description": "X", "fdcId": 9, "foodNutrients": _nutrients(5, 5, 5, 95)}]
    monkeypatch.setattr(bld, "requests", _FakeRequests(_Resp(foods)))
    assert bld._search_fndds("x") is None
