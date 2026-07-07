"""[P1-COHERENCE-SEVERE-NO-NOISE · 2026-07-07] El bloqueo severo del T2 escalaba
por SOBRE-oferta de envase.

Logs en vivo (plan 72c8b965 semana 2):
  🛒 [COH-GUARD/pkg-noise] Filtradas 63 de 130. Restantes accionables: 67
  🛒 [COH-GUARD/warn] 68 divergencias (presence=1, magnitude=67).
     Hipótesis: {'unknown': 67, 'pantry_overdeduct': 1}
  [P2-COHERENCE-1] T2 block_severe_only escaló warn→block → Shopping list falló 3 veces

Los 67 `unknown` son sobre-oferta de envase de STAPLES (arroz/lechuga/ajo/calamar
comprados por paquete) que `pkg-noise` no filtra (lo limitamos a condimentos para no
tocar proteínas). `_has_severe_divergence` los marcaba severos vía |delta|>0.50 SIN
importar la hipótesis → block T2 en falso → 3 retries + re-encolado (quema DeepSeek).

El docstring de `_has_severe_divergence` YA declaraba `unknown`/`pantry_overdeduct`
NO-severas; el fix alinea el código: el check de magnitud las excluye. Solo
cap_swallowed (falta real) + magnitudes severas de tipos accionables (yield_uncovered)
escalan. La sobre-oferta de envase nunca hace el plan incocinable.
tooltip-anchor: P1-COHERENCE-SEVERE-NO-NOISE
"""
from __future__ import annotations

from pathlib import Path

import pytest

import shopping_calculator as sc

_SRC = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")


def test_marker_present():
    assert "P1-COHERENCE-SEVERE-NO-NOISE" in _SRC


def test_unknown_oversupply_not_severe():
    """`unknown` de magnitud con |delta|>0.50 (sobre-oferta de envase) NO es severo."""
    divs = [{"food": "Arroz", "hypothesis": "unknown", "magnitude": True,
             "expected_qty": 500, "actual_qty": 2000, "delta_pct": 3.0}]
    assert sc._has_severe_divergence(divs) is False


def test_pantry_overdeduct_not_severe():
    """`pantry_overdeduct` (artefacto conocido del aggregator) NO fuerza retry T2."""
    divs = [{"food": "Habichuelas", "hypothesis": "pantry_overdeduct", "magnitude": True,
             "expected_qty": 900, "actual_qty": 200, "delta_pct": -0.78}]
    assert sc._has_severe_divergence(divs) is False


def test_prod_case_67_unknown_not_severe():
    """El caso exacto de prod: 67 unknown (sobre-oferta) + 1 pantry_overdeduct → NO severo."""
    divs = [{"food": f"Staple{i}", "hypothesis": "unknown", "magnitude": True,
             "expected_qty": 100, "actual_qty": 900, "delta_pct": 8.0} for i in range(67)]
    divs.append({"food": "X", "hypothesis": "pantry_overdeduct", "magnitude": True,
                 "expected_qty": 900, "actual_qty": 200, "delta_pct": -0.78})
    assert sc._has_severe_divergence(divs) is False, (
        "67 divergencias de sobre-oferta de envase NO deben bloquear el chunk T2"
    )


# --- Regresión: lo que SÍ sigue siendo severo ---
def test_cap_swallowed_still_severe():
    """Falta real (receta pide pollo, lista lo omite) SÍ escala."""
    divs = [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False}]
    assert sc._has_severe_divergence(divs) is True


def test_yield_uncovered_high_delta_still_severe():
    """yield_uncovered con |delta|>0.50 sigue siendo severo (no se toca)."""
    divs = [{"food": "Pollo", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.75}]
    assert sc._has_severe_divergence(divs) is True


def test_real_undersupply_via_cap_swallowed_severe():
    """Una FALTA real mezclada entre sobre-oferta SÍ escala (no se pierde señal)."""
    divs = [{"food": f"S{i}", "hypothesis": "unknown", "magnitude": True,
             "expected_qty": 100, "actual_qty": 900, "delta_pct": 8.0} for i in range(30)]
    divs.append({"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False})
    assert sc._has_severe_divergence(divs) is True
