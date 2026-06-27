"""[P1-DM2-GLYCEMIC-GUARD · 2026-06-27] El revisor médico rechazaba planes DM2 por carga glucémica
(toronja↔antidiabéticos = CYP3A4/hipoglucemia; refinados de alto IG). Guard DETERMINISTA calorie-neutral
vía el motor de sustitución de condición (aplica en S1 Guard 3 + en las superficies de UPDATE):
  - toronja/pomelo → fruta baja en IG (evita la interacción farmacológica; decisión del owner: SIEMPRE en DM2).
  - arroz blanco / pan blanco / pan rallado / tortilla de trigo / harina refinada → su versión INTEGRAL.
+ directiva al prompt (porción de almidón ≤150 g/comida + prohibir toronja).

El cap DURO de porción de víveres (batata >150g) queda como follow-up (necesita compensación calórica para
no chocar con el band gate de macros). Este guard cubre los críticos calorie-neutral (toronja + refinados).

NB: la resolubilidad de los reemplazos al catálogo la cubre test_p2_subs_resolubility_contract (necesita DB).
Verificado en la DB viva: Fresa/Arroz integral/Pan integral/Avena resuelven.
"""
from __future__ import annotations

from pathlib import Path

import condition_rules as cr


def _dm2_rule():
    return next(r for r in cr.CONDITION_RULES if r.id == "dm2")


def test_dm2_rule_includes_glycemic_subs():
    dm2 = _dm2_rule()
    flat = " ".join(str(s) for s in dm2.substitutions).lower()
    # toronja → fruta segura
    assert "toronja" in flat and "pomelo" in flat and "fresa" in flat
    # refinados → integral
    assert "arroz blanco" in flat and "arroz integral" in flat
    assert "pan blanco" in flat and "pan integral" in flat
    assert "tortilla de trigo" in flat
    # las subs de azúcar previas SIGUEN presentes (no se perdieron al combinar)
    assert "azucar" in flat or "miel" in flat


def test_dm2_prompt_has_portion_and_grapefruit_directive():
    dm2 = _dm2_rule()
    pb = dm2.prompt_block.lower()
    assert "150" in pb                       # cap de porción de almidón
    assert "toronja" in pb and "pomelo" in pb
    assert "indice glucemico" in pb or "índice glucémico" in pb or "ig" in pb


def test_glycemic_subs_preserve_quantity():
    """Los swaps de refinado→integral y toronja→fruta son preserve_qty=True (calorie-neutral): el 4º campo
    de cada tupla es True (mismo gramaje, distinto alimento)."""
    for tokens, repl, label, preserve in cr._DM2_GLYCEMIC_SUBS:
        assert preserve is True, (repl, "debe preservar cantidad (calorie-neutral)")


def test_marker_anchor_present():
    src = (Path(cr.__file__).resolve()).read_text(encoding="utf-8")
    assert "P1-DM2-GLYCEMIC-GUARD" in src
