"""[P1-COHERENCE-SEVERE-DIRECTIONAL · 2026-07-10] La severidad de coherencia debe ser DIRECCIONAL:
solo la SUB-oferta (la lista tiene MENOS de lo que las recetas piden → el usuario no puede cocinar)
puede ser egregia o material; la SOBRE-oferta es redondeo de paquete (compras 1 lb de res aunque la
receta use 120g → ratio 3.8x) y va a telemetría, no a reject.

Evidencia viva (corr=2ac209c7, 2026-07-10 00:16-00:26, post count-material): 3 intentos quemados y
entrega degradada con banner porque Res/Cerdo/Queso fresco aparecían con ratios act/exp en bandas
'2-4' y '>=4' — TODOS sobre-oferta de empaque. delta_pct = abs(act-exp)/exp trataba "compraste de
más" igual que "te falta comida". La única sub-oferta real era marginal (banda 0.5-0.9).

Con el fix la corrida habría entregado el intento 1 directo. Rollback:
MEALFIT_REVIEW_COHERENCE_SEVERE_UNDERSUPPLY_ONLY=false → comportamiento abs() previo exacto.

tooltip-anchor: P1-COHERENCE-SEVERE-DIRECTIONAL
"""
from pathlib import Path

import graph_orchestrator as go

_GO_SRC = Path(go.__file__).read_text(encoding="utf-8")


def _over(ratio):
    # lista tiene ratio× lo que piden las recetas (sobre-oferta)
    return {"food": "Res", "magnitude": True, "expected_qty": 100.0,
            "actual_qty": 100.0 * ratio, "delta_pct": abs(ratio - 1.0)}


def _under(missing_frac):
    # a la lista le FALTA missing_frac de lo que piden las recetas (sub-oferta)
    return {"food": "Pollo", "magnitude": True, "expected_qty": 100.0,
            "actual_qty": 100.0 * (1.0 - missing_frac), "delta_pct": missing_frac}


def test_oversupply_is_never_material_for_count():
    # el caso vivo: 2 entradas >=4x (sobre-oferta empaque) + 1 sub-oferta marginal
    block = [_over(4.2), _over(5.0), _under(0.15)]
    assert go._coherence_material_divergence_count(block, 0.25) == 0


def test_undersupply_material_counts():
    block = [_under(0.30), _over(4.0)]
    assert go._coherence_material_divergence_count(block, 0.25) == 1


def test_egregious_rule_is_directional():
    # helper puro para la regla egregia: solo sub-oferta >= severe_delta
    assert go._coherence_divergence_is_egregious(_under(0.60), 0.50) is True
    assert go._coherence_divergence_is_egregious(_under(0.30), 0.50) is False
    assert go._coherence_divergence_is_egregious(_over(4.0), 0.50) is False   # sobre-oferta JAMÁS egregia
    assert go._coherence_divergence_is_egregious(_over(10.0), 0.50) is False


def test_presence_missing_still_always_material():
    # alimento AUSENTE de la lista (presence, sin magnitude flag) sigue siendo material siempre
    block = [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier",
              "expected_qty": 400.0, "actual_qty": 0.0}]
    assert go._coherence_material_divergence_count(block, 0.25) == 1


def test_directional_knob_off_restores_abs_behavior(monkeypatch):
    monkeypatch.setattr(go, "COHERENCE_SEVERE_UNDERSUPPLY_ONLY", False)
    # con el knob OFF, la sobre-oferta vuelve a contar (paridad abs() previa)
    block = [_over(4.0), _over(5.0)]
    assert go._coherence_material_divergence_count(block, 0.25) == 2
    assert go._coherence_divergence_is_egregious(_over(4.0), 0.50) is True


def test_wired_into_review_severe_rule():
    assert "MEALFIT_REVIEW_COHERENCE_SEVERE_UNDERSUPPLY_ONLY" in _GO_SRC
    i = _GO_SRC.index("MEALFIT_REVIEW_COHERENCE_SEVERE_MIN_COUNT")
    window = _GO_SRC[i:i + 3000]
    assert "_coherence_divergence_is_egregious" in window, (
        "la regla egregia del severe-only debe usar el predicado direccional"
    )
