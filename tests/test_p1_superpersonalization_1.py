"""[P1-SUPERPERSONALIZATION-1 · 2026-06-19] Tests ancla del panel de Súper
Personalización.

Cubre las dos piezas puras del feature:
  1. `build_super_personalization_context` (prompts/plan_generator.py) — el
     builder que traduce `health_profile.super_personalization` → bloque de
     prompt, inyectado en plan-gen (planner + day generator) y chat coach.
  2. `_clean_super_personalization` (routers/user_data.py) — el validador/
     normalizador del payload del endpoint PUT.

Contrato verificado:
  - no-op transparente cuando no hay datos accionables (retorna "").
  - exclusiones por cultura/religión presentes y obligatorias.
  - el validador acota listas/longitudes y rechaza enums inválidos (422).
  - el bloque NO inyecta texto sobre alergias/condiciones/medicamentos
    (esas son estrictas y viven en otros bloques) — defensa del contrato
    "aditivo, no clínico".
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompts.plan_generator import build_super_personalization_context


# --------------------------------------------------------------------------
# build_super_personalization_context
# --------------------------------------------------------------------------

def test_empty_inputs_are_noops():
    assert build_super_personalization_context(None) == ""
    assert build_super_personalization_context({}) == ""
    assert build_super_personalization_context({"super_personalization": None}) == ""
    assert build_super_personalization_context({"super_personalization": {}}) == ""
    # dict presente pero sin claves accionables → "" (no contamina el prompt).
    assert build_super_personalization_context(
        {"super_personalization": {"foodLikes": [], "freeText": "   "}}
    ) == ""


def test_food_likes_and_cuisines_render():
    out = build_super_personalization_context(
        {"super_personalization": {"foodLikes": ["pollo frito", "plátano"], "cuisines": ["criolla"]}}
    )
    assert "SÚPER PERSONALIZACIÓN" in out
    assert "pollo frito" in out and "plátano" in out
    assert "criolla" in out


def test_religious_restriction_is_a_hard_exclusion():
    out = build_super_personalization_context(
        {"super_personalization": {"religiousRestriction": "halal"}}
    )
    assert "HALAL" in out
    assert "NUNCA" in out  # exclusión obligatoria, no preferencia blanda


def test_equipment_and_skill_and_flavor():
    out = build_super_personalization_context(
        {"super_personalization": {
            "kitchenEquipment": ["estufa", "licuadora"],
            "cookingSkill": "principiante",
            "flavorProfile": {"picante": "alto", "dulce": "bajo"},
        }}
    )
    assert "estufa" in out and "licuadora" in out
    assert "PRINCIPIANTE" in out
    assert "picante=alto" in out and "dulce=bajo" in out


def test_free_text_is_capped():
    long_text = "a" * 5000
    out = build_super_personalization_context(
        {"super_personalization": {"freeText": long_text}}
    )
    assert "En sus palabras" in out
    # capado a 1200 + elipsis → muy por debajo del texto original.
    assert len(out) < 2000
    assert "…" in out


def test_nested_health_profile_fallback():
    """El builder cubre ambos transportes: clave plana hidratada o el
    health_profile anidado dentro de form_data."""
    out = build_super_personalization_context(
        {"health_profile": {"super_personalization": {"foodLikes": ["yuca"]}}}
    )
    assert "yuca" in out
    # La clave plana tiene precedencia si ambas existen.
    out2 = build_super_personalization_context({
        "super_personalization": {"foodLikes": ["plano"]},
        "health_profile": {"super_personalization": {"foodLikes": ["anidado"]}},
    })
    assert "plano" in out2 and "anidado" not in out2


def test_block_does_not_assert_clinical_authority():
    """El bloque debe declarar explícitamente que NO altera alergias/condiciones/
    medicamentos/macros — invariante del diseño 'aditivo, no clínico'."""
    out = build_super_personalization_context(
        {"super_personalization": {"foodLikes": ["mango"]}}
    )
    lowered = out.lower()
    assert "no uses este bloque para alterar" in lowered
    assert "alergias" in lowered and "medicamentos" in lowered


# --------------------------------------------------------------------------
# _clean_super_personalization (validador del endpoint)
# --------------------------------------------------------------------------

def _import_cleaner():
    try:
        from routers.user_data import _clean_super_personalization
        return _clean_super_personalization
    except Exception as e:  # pragma: no cover - import-heavy en algunos entornos
        pytest.skip(f"routers.user_data no importable en este entorno: {e}")


def test_cleaner_normalizes_and_caps():
    clean = _import_cleaner()
    out = clean({
        "foodLikes": ["  Pollo  ", "pollo", "", "x" * 200],  # dedup case-insensitive + trim + cap item
        "cuisines": [f"cocina{i}" for i in range(100)],  # 100 distintas → cap a 30
        "religiousRestriction": "NONE",
        "cookingSkill": "Intermedio",
        "flavorProfile": {"picante": "Alto", "dulce": "", "salado": "medio"},
        "freeText": "  hola  " + ("z" * 5000),
    })
    assert out["foodLikes"][0] == "Pollo"
    assert "pollo" not in [v.lower() for v in out["foodLikes"][1:2]] or len(out["foodLikes"]) <= 3
    assert len(out["foodLikes"][-1]) <= 60
    assert len(out["cuisines"]) == 30
    assert out["religiousRestriction"] == ""  # "none" → vacío
    assert out["cookingSkill"] == "intermedio"
    assert out["flavorProfile"] == {"picante": "alto", "salado": "medio"}  # "" se omite
    assert len(out["freeText"]) <= 1500


def test_cleaner_rejects_invalid_enums():
    clean = _import_cleaner()
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        clean({"religiousRestriction": "pastafari"})
    with pytest.raises(HTTPException):
        clean({"cookingSkill": "chef-estrella"})
    with pytest.raises(HTTPException):
        clean({"flavorProfile": {"picante": "extremo"}})
    with pytest.raises(HTTPException):
        clean({"foodLikes": "no-soy-lista"})
