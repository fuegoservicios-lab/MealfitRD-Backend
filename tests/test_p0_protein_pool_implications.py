"""[P0-PROTEIN-POOL-IMPLICATIONS · 2026-05-16] Fix del matcher de
`_apply_protein_pool_scrub` que eliminaba ingredientes legítimos cuando
el LLM expandía el canonical del pool con un descriptor familiar.

Bug original (plan aeb25e1c, día 2):
  Pool = ['Chuleta', 'Claras de Huevo', 'Queso Blanco Fresco']
  LLM ingrediente: "300 g de chuleta de cerdo (lomo, sin grasa visible)"
  Scrub removía por "cerdo" → receta sin proteína → lista de compras sin
  chuleta → usuario percibe app rota.

Causa raíz: el matcher trataba 'cerdo' y 'chuleta' como independientes,
ignorando la relación familiar (chuleta es chuleta de cerdo en RD).

Fix: `_POOL_IMPLICATIONS` map en `prompts/day_generator.py`. Cuando una
key del pool aparece (e.g., 'chuleta'), las restricted keys del MISMO
grupo proteico ('cerdo') quedan auto-autorizadas. Aplicado en
`_apply_protein_pool_scrub` via union con `implied_authorized`.

Test funcional crítico: los 3 casos representativos:
  1. **Bug original**: pool=[Chuleta], ing="chuleta de cerdo" → preservado.
  2. **Caso paralelo**: pool=[Pechuga de Pollo], ing="pechuga de pollo" → preservado.
  3. **NO regresión**: pool=[Lentejas], ing="cerdo en cubos" → eliminado (cerdo sigue
     siendo restricted si no hay implicación en el pool).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DAY_GEN_PY = _BACKEND_ROOT / "prompts" / "day_generator.py"
_GO_PY = _BACKEND_ROOT / "graph_orchestrator.py"


# ---------------------------------------------------------------------------
# 1. Parser: existencia + estructura del mapping
# ---------------------------------------------------------------------------
def test_pool_implications_map_defined():
    """`_POOL_IMPLICATIONS` debe estar definido en day_generator.py como
    dict de pool_key → set de restricted_keys auto-autorizadas."""
    src = _DAY_GEN_PY.read_text(encoding="utf-8")
    assert "_POOL_IMPLICATIONS = {" in src, (
        "Mapping `_POOL_IMPLICATIONS` no encontrado en prompts/day_generator.py"
    )


def test_chuleta_implies_cerdo():
    """Caso del bug original: pool 'chuleta' debe auto-autorizar 'cerdo'.
    Sin esto, el scrub elimina "chuleta de cerdo" del LLM."""
    src = _DAY_GEN_PY.read_text(encoding="utf-8")
    # El mapping debe tener 'chuleta' → {'cerdo'} (mínimo).
    chuleta_block = re.search(
        r"'chuleta':\s*\{[^}]*'cerdo'[^}]*\}",
        src,
    )
    assert chuleta_block, (
        "`_POOL_IMPLICATIONS['chuleta']` no incluye 'cerdo'. Este es el "
        "case canónico del bug aeb25e1c — sin esta implicación, pool "
        "Chuleta + ingrediente 'chuleta de cerdo' = scrub remueve."
    )


def test_implications_cover_key_protein_families():
    """Las familias proteicas comunes en RD deben tener implicaciones:
    cerdo (chuleta/lomo/tocineta), pollo (pechuga/muslo), res (bistec/lomo),
    pescado (tilapia/mero/salmón)."""
    src = _DAY_GEN_PY.read_text(encoding="utf-8")
    expected_pool_keys = [
        "chuleta",       # → cerdo
        "pechuga de pollo",  # → pollo
        "muslo de pollo",    # → pollo
        "bistec",        # → res
        "tilapia",       # → pescado
        "mero",          # → pescado
    ]
    for pool_key in expected_pool_keys:
        assert f"'{pool_key}':" in src, (
            f"Pool key {pool_key!r} no listado en _POOL_IMPLICATIONS — "
            f"esto deja casos análogos al bug original sin cubrir."
        )


def test_scrub_uses_implications():
    """`_apply_protein_pool_scrub` debe importar + usar `_POOL_IMPLICATIONS`
    para computar `implied_authorized` antes de calcular `unauthorized_keys`."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Import del mapping
    assert "from prompts.day_generator import _RESTRICTED_PROTEIN_KEYS, _POOL_IMPLICATIONS" in src, (
        "Scrub no importa _POOL_IMPLICATIONS — fix no aplicado."
    )
    # Computa implied_authorized
    assert "implied_authorized" in src, (
        "Variable `implied_authorized` ausente en el scrub — el mapping no "
        "se está aplicando."
    )
    # `unauthorized_keys` excluye los implied
    assert "and k not in implied_authorized" in src, (
        "El cálculo de `unauthorized_keys` no resta los implied — fix "
        "incompleto."
    )


# ---------------------------------------------------------------------------
# 2. Funcional: el bug ya no se reproduce + no hay regresión
# ---------------------------------------------------------------------------
def _load_scrub():
    """Lazy import del scrub para no disparar inits costosos."""
    os.environ.setdefault("GEMINI_API_KEY", "dummy")
    os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "dummy")
    os.environ.setdefault("CRON_SECRET", "dummy")
    sys.path.insert(0, str(_BACKEND_ROOT))
    from graph_orchestrator import _apply_protein_pool_scrub
    return _apply_protein_pool_scrub


def test_bug_original_chuleta_preserved():
    """Caso del bug aeb25e1c: pool=[Chuleta], ingrediente con descriptor
    "chuleta de cerdo (lomo, sin grasa visible)" — DEBE preservarse."""
    scrub = _load_scrub()
    day_result = {
        "meals": [{
            "name": "Ropa Vieja de Chuleta con Víveres Verdes",
            "ingredients": [
                "300 g de chuleta de cerdo (lomo, sin grasa visible)",
                "200 g de yuca",
                "1 cda de aceite de oliva",
            ],
            "recipe": ["Sazonar la chuleta", "Cocinar a fuego medio"],
        }]
    }
    skeleton = {"protein_pool": ["Chuleta", "Claras de Huevo", "Queso Blanco Fresco"]}
    scrub(day_result, skeleton, 2, "TEST-P0-FIX")
    remaining = day_result["meals"][0]["ingredients"]
    assert any("chuleta" in ing.lower() for ing in remaining), (
        f"REGRESIÓN: chuleta fue eliminada del ingrediente. "
        f"Pool=[Chuleta] debe auto-autorizar 'cerdo' via _POOL_IMPLICATIONS. "
        f"Ingredientes restantes: {remaining}"
    )


def test_pechuga_de_pollo_preserved_when_pool_has_pechuga():
    """Caso paralelo: pool=[Pechuga de Pollo], ingrediente "pechuga de pollo"
    — DEBE preservarse. Sin implication, 'pollo' restricted lo eliminaría."""
    scrub = _load_scrub()
    day_result = {
        "meals": [{
            "name": "Pechuga a la Plancha",
            "ingredients": ["200 g de pechuga de pollo", "1 taza de arroz"],
            "recipe": ["Cocinar la pechuga de pollo"],
        }]
    }
    skeleton = {"protein_pool": ["Pechuga de Pollo", "Yogurt Griego"]}
    scrub(day_result, skeleton, 1, "TEST-P0-FIX")
    remaining = day_result["meals"][0]["ingredients"]
    assert any("pechuga" in ing.lower() for ing in remaining), (
        f"REGRESIÓN: pechuga fue eliminada. Pool=[Pechuga de Pollo] debe "
        f"auto-autorizar 'pollo'. Ingredientes restantes: {remaining}"
    )


def test_no_regression_cerdo_blocked_when_pool_excludes_cerdo_family():
    """NO regresión: si pool NO tiene chuleta/lomo/tocineta, 'cerdo' sigue
    siendo restricted. El LLM intentando colar cerdo debe ser bloqueado."""
    scrub = _load_scrub()
    day_result = {
        "meals": [{
            "name": "Lentejas Estofadas (LLM coló cerdo)",
            "ingredients": [
                "200 g de lentejas",
                "150 g de cerdo en cubos",  # LLM violó el pool
            ],
            "recipe": ["Cocinar las lentejas con el cerdo"],
        }]
    }
    skeleton = {"protein_pool": ["Lentejas", "Yogurt Griego"]}
    scrub(day_result, skeleton, 3, "TEST-P0-FIX")
    remaining = day_result["meals"][0]["ingredients"]
    full_text = " ".join(remaining).lower()
    assert "cerdo" not in full_text, (
        f"REGRESIÓN PELIGROSA: cerdo sobrevivió al scrub aunque pool=[Lentejas] "
        f"NO tiene implicación. El fix está sobre-autorizando. "
        f"Ingredientes restantes: {remaining}"
    )
    # Lentejas SÍ debe seguir
    assert any("lentejas" in ing.lower() for ing in remaining), (
        f"Las lentejas (proteína legítima del pool) fueron eliminadas — bug nuevo."
    )


def test_bistec_implies_res():
    """Pool=[Bistec], ingrediente "bistec de res" — preservar."""
    scrub = _load_scrub()
    day_result = {
        "meals": [{
            "name": "Bistec Encebollado",
            "ingredients": ["250 g de bistec de res", "1 cebolla"],
            "recipe": ["Sellar el bistec"],
        }]
    }
    skeleton = {"protein_pool": ["Bistec", "Queso Blanco"]}
    scrub(day_result, skeleton, 1, "TEST-P0-FIX")
    remaining = day_result["meals"][0]["ingredients"]
    assert any("bistec" in ing.lower() for ing in remaining), (
        f"Bistec eliminado. Pool=[Bistec] debe auto-autorizar 'res'. "
        f"Ingredientes restantes: {remaining}"
    )


def test_tilapia_implies_pescado():
    """Pool=[Tilapia], ingrediente "filete de tilapia (pescado fresco)"
    — preservar. La implicación tilapia → pescado evita que 'pescado' como
    descriptor regional restrinja."""
    scrub = _load_scrub()
    day_result = {
        "meals": [{
            "name": "Tilapia al Horno",
            "ingredients": ["200 g de filete de tilapia (pescado fresco)", "1 limón"],
            "recipe": ["Marinar la tilapia"],
        }]
    }
    skeleton = {"protein_pool": ["Tilapia", "Lentejas"]}
    scrub(day_result, skeleton, 1, "TEST-P0-FIX")
    remaining = day_result["meals"][0]["ingredients"]
    assert any("tilapia" in ing.lower() for ing in remaining), (
        f"Tilapia eliminada. Pool=[Tilapia] debe auto-autorizar 'pescado'. "
        f"Ingredientes restantes: {remaining}"
    )
