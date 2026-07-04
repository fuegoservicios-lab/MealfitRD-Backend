"""[P1-BUDGET-DRIVER-AWARE · 2026-07-04] Test ancla del pase de abaratamiento
driver-aware (audit v7 · P1-2).

Gap que cierra: la convergencia de presupuesto (P1-BUDGET-CONVERGENCE) corría
CIEGA al driver de costo — barría días en orden contra la tabla estática de 8
pares (`_BUDGET_CHEAP_EQUIVALENTS`, cap 3 subs) sin mirar qué ítems encarecen
realmente la lista costeada. Un plan `excedido` cuyos ítems caros no estaban
entre los 8 pares (quesos madurados, cortes de res premium, berries) recibía
0 sustituciones y permanecía excedido tras la "convergencia".

Ahora `_apply_budget_driver_aware_pass`:
  1. rankea los ítems de la lista SEMANAL costeada por `estimated_cost_rd`,
  2. resuelve la familia culinaria de cada driver (`_BUDGET_DRIVER_FAMILIES`,
     misma-categoría SIEMPRE),
  3. valida el candidato contra el precio VIVO del catálogo master (≥30% más
     barato, sin alérgenos, sin dislikes) y sustituye en el plan con la misma
     maquinaria honesta del pase estático (nota, nombre, pasos de receta).

El pase estático queda como backstop en la convergencia (corre después).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as _f:
    _GO_SRC = _f.read()


# ---------------------------------------------------------------------------
# 1. Anclas de source: knobs + wiring
# ---------------------------------------------------------------------------

def test_knob_defaults():
    assert re.search(
        r'BUDGET_DRIVER_AWARE_ENABLED\s*=\s*_env_bool\("MEALFIT_BUDGET_DRIVER_AWARE",\s*True\)',
        _GO_SRC,
    ), "knob maestro debe existir con default ON (rollback sin redeploy)"
    assert '_env_int("MEALFIT_BUDGET_DRIVER_AWARE_MAX_SUBS", 5' in _GO_SRC
    assert '_env_int("MEALFIT_BUDGET_DRIVER_AWARE_TOP_ITEMS", 8' in _GO_SRC
    assert '_env_float(\n    "MEALFIT_BUDGET_DRIVER_AWARE_MIN_SAVING_PCT", 0.30' in _GO_SRC or \
        '_env_float("MEALFIT_BUDGET_DRIVER_AWARE_MIN_SAVING_PCT", 0.30' in _GO_SRC


def test_families_table_exists():
    assert "_BUDGET_DRIVER_FAMILIES = (" in _GO_SRC
    # Familias nuevas del audit v7 (los modos de fallo citados: quesos
    # madurados, res premium, berries) deben estar cubiertas.
    fam_start = _GO_SRC.index("_BUDGET_DRIVER_FAMILIES = (")
    fam_block = _GO_SRC[fam_start:_GO_SRC.index("def _apply_budget_driver_aware_pass")]
    for token in ("cheddar", "churrasco", "fresas?", "salm[oó]n"):
        assert token in fam_block, f"familia esperada ausente: {token}"


def test_wired_in_convergence_before_static_backstop():
    """En el bloque de convergencia el pase driver-aware corre ANTES del
    backstop estático force=True."""
    conv = _GO_SRC.index('str(_bc_rec0.get("status") or "") == "excedido"')
    window = _GO_SRC[conv:conv + 2500]
    da = window.find("_apply_budget_driver_aware_pass(")
    static = window.find("_apply_budget_cheapen_pass(_bc_days, form_data, force=True)")
    assert da != -1, "driver-aware no cableado en la convergencia"
    assert static != -1, "backstop estático eliminado de la convergencia"
    assert da < static, "driver-aware debe correr ANTES del backstop estático"


# ---------------------------------------------------------------------------
# 2. Funcional (sin DB: precio del catálogo mockeado)
# ---------------------------------------------------------------------------

_PRICES = {
    # keys accent-stripped lowercase (contrato de _budget_build_master_price_map)
    "queso cheddar": 400.0,
    "queso blanco": 90.0,
    "salmon": 500.0,
    "filete de pescado blanco": 150.0,
    "fresas": 300.0,
    "lechosa": 40.0,
    "almendras": 450.0,
    "mani": 80.0,
}


def _mk_days():
    return [
        {
            "day": 1,
            "meals": [
                {
                    "name": "Salmón a la plancha con queso cheddar",
                    "ingredients": [
                        "120 g de salmón",
                        "40 g de queso cheddar",
                        "100 g de arroz blanco",
                    ],
                    "ingredients_raw": [
                        "120 g de salmón",
                        "40 g de queso cheddar",
                        "100 g de arroz blanco",
                    ],
                    "recipe": [
                        "El toque de fuego: sella el salmón 4 min por lado.",
                        "Sirve con el queso cheddar rallado por encima.",
                    ],
                }
            ],
        }
    ]


_WEEKLY = [
    {"name": "Salmón", "estimated_cost_rd": 850.0},
    {"name": "Queso cheddar", "estimated_cost_rd": 420.0},
    {"name": "Arroz blanco", "estimated_cost_rd": 30.0},
]


@pytest.fixture()
def _go(monkeypatch):
    import graph_orchestrator as go

    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: dict(_PRICES))
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MAX_SUBS", 5)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_TOP_ITEMS", 8)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MIN_SAVING_PCT", 0.30)
    return go


def test_substitutes_ranked_drivers(_go):
    days = _mk_days()
    subs = _go._apply_budget_driver_aware_pass(days, {}, _WEEKLY)
    assert subs == 2
    meal = days[0]["meals"][0]
    joined = " | ".join(meal["ingredients"])
    assert "Filete de pescado blanco" in joined
    assert "Queso blanco" in joined
    assert "salmón" not in joined.lower()
    assert "cheddar" not in joined.lower()
    # Marcado honesto + nombre + raw + pasos reescritos.
    assert any("→ Filete de pescado blanco" in n for n in meal["_budget_substitutions"])
    assert "cheddar" not in meal["name"].lower()
    assert "cheddar" not in " ".join(meal["ingredients_raw"]).lower()
    steps = " ".join(meal["recipe"]).lower()
    assert "salmón" not in steps and "salmon" not in steps


def test_min_saving_threshold_skips(_go, monkeypatch):
    """Candidato que no ahorra ≥30% → skip (mismo umbral del pase estático)."""
    prices = dict(_PRICES)
    prices["queso blanco"] = 380.0  # 380 >= 400*0.7 → no ahorra
    monkeypatch.setattr(_go, "_budget_build_master_price_map", lambda: prices)
    days = _mk_days()
    subs = _go._apply_budget_driver_aware_pass(
        days, {}, [{"name": "Queso cheddar", "estimated_cost_rd": 420.0}]
    )
    assert subs == 0
    assert "cheddar" in " ".join(days[0]["meals"][0]["ingredients"]).lower()


def test_dislike_blocks_candidate(_go):
    days = _mk_days()
    subs = _go._apply_budget_driver_aware_pass(
        days,
        {"dislikes": ["queso blanco", "filete de pescado blanco"]},
        _WEEKLY,
    )
    assert subs == 0


def test_allergy_scan_blocks_candidate(_go, monkeypatch):
    monkeypatch.setattr(
        _go, "_scan_allergen_violations", lambda plan, allergies: ["violación"]
    )
    days = _mk_days()
    subs = _go._apply_budget_driver_aware_pass(
        days, {"allergies": ["pescado"]}, _WEEKLY
    )
    assert subs == 0


def test_cap_respected(_go, monkeypatch):
    monkeypatch.setattr(_go, "BUDGET_DRIVER_AWARE_MAX_SUBS", 1)
    days = _mk_days()
    subs = _go._apply_budget_driver_aware_pass(days, {}, _WEEKLY)
    assert subs == 1


def test_knob_off_is_noop(_go, monkeypatch):
    monkeypatch.setattr(_go, "BUDGET_DRIVER_AWARE_ENABLED", False)
    days = _mk_days()
    assert _go._apply_budget_driver_aware_pass(days, {}, _WEEKLY) == 0


def test_unpriced_or_empty_list_is_noop(_go):
    days = _mk_days()
    assert _go._apply_budget_driver_aware_pass(days, {}, []) == 0
    assert _go._apply_budget_driver_aware_pass(days, {}, None) == 0
    assert (
        _go._apply_budget_driver_aware_pass(
            days, {}, [{"name": "Salmón", "estimated_cost_rd": 0}]
        )
        == 0
    )


def test_driver_without_family_is_noop(_go):
    """Un driver caro sin familia curada (ya-económico: pollo) no se toca."""
    days = _mk_days()
    subs = _go._apply_budget_driver_aware_pass(
        days, {}, [{"name": "Pechuga de pollo", "estimated_cost_rd": 900.0}]
    )
    assert subs == 0


def test_fail_open_on_price_map_error(_go, monkeypatch):
    def _boom():
        raise RuntimeError("catálogo caído")

    monkeypatch.setattr(_go, "_budget_build_master_price_map", _boom)
    days = _mk_days()
    assert _go._apply_budget_driver_aware_pass(days, {}, _WEEKLY) == 0
    assert "salmón" in " ".join(days[0]["meals"][0]["ingredients"]).lower()
