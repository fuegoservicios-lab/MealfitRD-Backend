"""[FINAL-REVIEW-P2 · 2026-08-03] `_budget_food_matches_key_set` documentaba 2
diferencias con `_resolve_brand_pref` (shopping_calculator:3406) y omitía la 3ª: no
aplicaba `_brand_prep_distinct_conflict` ("mantequilla de maní" ≠ "maní"). En el path
de Nevera (`owned_keys`, Task 15 de esta rama — código NUEVO, sin comportamiento
previo que preservar) eso hacía que con 'maní' ya comprado la convergencia de
presupuesto se SALTARA la sustitución de 'Mantequilla de maní' tratándola como "ya
tengo esto" — perdiendo ahorro real. Mismo vector medido en
P1-BRAND-PREF-PREP-DISTINCT (RD$110; también trigo/Harina de trigo, coco/Leche de
coco).

Fix en dos partes:
  (a) `_budget_food_is_owned` (nuevo wrapper, path de Nevera) aplica el guard —
      'Mantequilla de maní' YA NO se trata como poseída solo porque 'maní' está en
      la Nevera, así que la sustitución económica SÍ corre. 'Maní' a secas sigue
      saltándose (es literalmente lo que ya tiene).
  (b) `_budget_food_is_brand_pinned` (path de marcas fijadas, preexistente) sigue
      SIN el guard — byte-idéntico al pre-fix; alinearlo es cambio a medir aparte.

Ver informe de la ola: `.superpowers/sdd/2026-08-02-solver-seeder-v7-gaps/
final-review-p2-fix-report.md`.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_marker_and_helper_anchored():
    assert "_brand_prep_distinct_conflict" in _GO_SRC
    assert "def _budget_food_is_owned(" in _GO_SRC
    assert "skip_prep_conflict" in _GO_SRC


def test_owned_keys_does_not_skip_prep_distinct_substitution():
    """'maní' en la Nevera NO debe saltar la sustitución económica de 'Mantequilla
    de maní' — son productos distintos (P1-BRAND-PREF-PREP-DISTINCT)."""
    owned = {"mani"}
    assert go._budget_food_is_owned("Mantequilla de maní", owned) is False


def test_owned_keys_still_skips_the_base_food_itself():
    """'Maní' a secas SÍ debe saltarse — es literalmente lo que ya está en la Nevera."""
    owned = {"mani"}
    assert go._budget_food_is_owned("Maní", owned) is True


def test_owned_keys_still_skips_unrelated_containment():
    """El guard nuevo NO debe romper contención legítima ('coco' owned salta
    'Aceite de coco' si NO es un caso base↔preparación reconocido... aquí probamos
    el caso positivo simple de contención sin prefijo de preparación."""
    owned = {"pollo"}
    assert go._budget_food_is_owned("Pechuga de pollo", owned) is True


def test_pinned_path_untouched_byte_identical():
    """El path de marcas fijadas (pre-existente, P1-BUDGET-RESPECT-BRAND-PIN) sigue
    SIN el guard de preparación distinta — comportamiento previo intacto: un pin de
    'maní' sigue colapsando contra 'Mantequilla de maní' exactamente como antes de
    esta ola (caso existente de test_p1_budget_respect_brand_pin.py re-corrido aquí
    para dejar la paridad explícita)."""
    pinned = {"mani"}
    assert go._budget_food_is_brand_pinned("Mantequilla de maní", pinned) is True


def test_matches_key_set_default_kwarg_preserves_prior_behavior():
    """Sin pasar `skip_prep_conflict`, `_budget_food_matches_key_set` se comporta
    EXACTAMENTE como antes del fix (default False) — solo el path que pasa
    `skip_prep_conflict=True` (Nevera) cambia de comportamiento."""
    keys = {"mani"}
    assert go._budget_food_matches_key_set("Mantequilla de maní", keys) is True
    assert go._budget_food_matches_key_set(
        "Mantequilla de maní", keys, skip_prep_conflict=True
    ) is False
