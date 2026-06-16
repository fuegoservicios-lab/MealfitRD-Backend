"""[P1-DM2-GLYCEMIC-ONLY · 2026-06-15] (gap-audit P1-1) El soft-reject DM2 solo degrada glucémicos.

Antes el soft-reject DM2 degradaba CUALQUIER `critical` del revisor a 'high' para diabéticos (salvo los
flags deterministas schema/alérgeno/renal/dieta) → un critical de seguridad NO-glucémico phraseado por el
LLM se entregaba con banner en vez de caer al fallback. `_critical_is_purely_glycemic(issues)` exige
evidencia glucémica positiva Y ausencia de otra preocupación de seguridad antes de degradar.

Validación determinista del helper puro + parser-anchor sobre el source.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_GO_PATH = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# ---------------------------------------------------------------------------
# _critical_is_purely_glycemic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("issues", [
    ["El plan tiene demasiada azúcar añadida para un diabético"],
    ["Contiene miel y plátano maduro (carga glucémica alta)"],
    ["Uso de jugo de naranja y pan blanco eleva el índice glucémico"],
    ["Exceso de carbohidratos simples"],
])
def test_purely_glycemic_returns_true(go, issues):
    assert go._critical_is_purely_glycemic(issues) is True


@pytest.mark.parametrize("issues", [
    ["El plan no respeta la dieta vegana declarada"],          # dieta
    ["Contiene un alérgeno declarado: maní"],                   # alérgeno
    ["Huevo crudo licuado en el batido (riesgo de salmonella)"],  # food-safety
    ["Déficit calórico en embarazo"],                           # embarazo
    ["Contiene gluten y el usuario es celíaco"],                # celíaca
])
def test_non_glycemic_safety_returns_false(go, issues):
    assert go._critical_is_purely_glycemic(issues) is False


def test_mixed_glycemic_plus_safety_returns_false(go):
    # señal glucémica PERO también un blocker de seguridad → no degradar
    assert go._critical_is_purely_glycemic(
        ["Mucha azúcar añadida", "además contiene camarón, un alérgeno declarado"]
    ) is False


def test_no_glycemic_signal_returns_false(go):
    # sin evidencia glucémica positiva → conservador, no degradar
    assert go._critical_is_purely_glycemic(["El plan está mal estructurado"]) is False
    assert go._critical_is_purely_glycemic([]) is False


# ---------------------------------------------------------------------------
# Post-review: precisión de marcadores
# ---------------------------------------------------------------------------
def test_raw_food_word_does_not_block_glycemic(go):
    # 'crudo'/'cruda' anclado → ya NO bloquea criticals glucémicos legítimos (era falso fallback)
    assert go._critical_is_purely_glycemic(["Usa miel cruda, mucha carga glucémica"]) is True
    assert go._critical_is_purely_glycemic(["vegetales crudos con mucha azúcar añadida"]) is True


def test_real_raw_food_safety_still_blocks(go):
    assert go._critical_is_purely_glycemic(["Huevo crudo en el batido (riesgo de salmonella)"]) is False
    assert go._critical_is_purely_glycemic(["Ceviche de pescado sin cocer"]) is False


def test_drug_interaction_blocks_downgrade(go):
    assert go._critical_is_purely_glycemic(
        ["Mucha azúcar, pero también posible interacción con la metformina"]
    ) is False


def test_hta_dyslipidemia_still_degrades_per_g9(go):
    # G9 (CLAUDE.md): HTA/dislipidemia son advisory (NO fallback) → el degrade glucémico SÍ aplica
    assert go._critical_is_purely_glycemic(
        ["Demasiada azúcar y también exceso de sodio para la hipertensión"]
    ) is True
    assert go._critical_is_purely_glycemic(
        ["Carga glucémica alta y mucha grasa saturada/colesterol"]
    ) is True


# ---------------------------------------------------------------------------
# Knob + parser-anchor
# ---------------------------------------------------------------------------
def test_knob_default_on_and_registered(go):
    assert go.DM2_DOWNGRADE_GLYCEMIC_ONLY is True
    names = [k[0] for k in go._SAFETY_CRITICAL_KNOBS]
    assert "MEALFIT_DM2_DOWNGRADE_GLYCEMIC_ONLY" in names


def test_soft_reject_gated_by_glycemic_only():
    src = _GO_PATH.read_text(encoding="utf-8")
    assert "_critical_is_purely_glycemic(issues)" in src, \
        "la condición del soft-reject DM2 debe gatear por _critical_is_purely_glycemic"
    assert "DM2_DOWNGRADE_GLYCEMIC_ONLY" in src
