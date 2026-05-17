"""[P2-STAPLE-DIVERSITY · 2026-05-16] Regression guard: el PLANNER_SYSTEM_PROMPT
DEBE incluir regla de diversidad cross-day de staples (yogurt griego, avena,
claras, etc.) para evitar repetición que el reviewer médico rechaza.

Bug observado en plan_id=`fbd014b2-594d-4ad9-aa08-db7bf027a099` (2026-05-16 02:09:41):
  - Self-critique detectó: "Staples repetidos detectados: avena en 2 días,
    claras de huevo en 3 días, yogurt griego en 2 días"
  - Reviewer médico rechazó plan con "frecuencia repetitiva en múltiples comidas".

Root cause: el PLANNER_SYSTEM_PROMPT tenía regla de DIVERSIDAD para desayunos
(Categoría A-E) pero NO para staples cross-day. El planner asignaba pools que
incluían yogurt/avena/claras en los 3 días → day_generator usaba lo asignado
→ staples repetidos.

Fix: nueva sección `DIVERSIDAD OBLIGATORIA DE STAPLES CROSS-DAY` con cap
máximo 2 días por staple + alternativas válidas.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANNER_PATH = _BACKEND_ROOT / "prompts" / "planner.py"


def _read_planner() -> str:
    return _PLANNER_PATH.read_text(encoding="utf-8")


def test_staple_diversity_marker_present():
    """El prompt debe incluir marker `P2-STAPLE-DIVERSITY` para trazabilidad."""
    text = _read_planner()
    assert "P2-STAPLE-DIVERSITY" in text, (
        "Falta marker `P2-STAPLE-DIVERSITY` en planner.py. "
        "Convención del repo: cada regla nueva del prompt debe tener marker."
    )


def test_section_header_present():
    """El prompt debe tener la sección `DIVERSIDAD OBLIGATORIA DE STAPLES CROSS-DAY`."""
    text = _read_planner()
    assert "DIVERSIDAD OBLIGATORIA DE STAPLES" in text, (
        "Falta sección `DIVERSIDAD OBLIGATORIA DE STAPLES CROSS-DAY` en "
        "PLANNER_SYSTEM_PROMPT. Sin esta sección el planner sigue asignando "
        "los mismos staples a los 3 días."
    )


@pytest.mark.parametrize("staple", [
    "yogurt griego",
    "avena",
    "claras de huevo",
    "casabe",
    "queso",
])
def test_staple_listed(staple: str):
    """Cada staple crítico debe estar listado explícitamente. Una regla "
    "genérica ('no repitas staples') no es suficiente — el LLM necesita la "
    "lista exacta."""
    text = _read_planner().lower()
    assert staple.lower() in text, (
        f"Staple `{staple}` no listado explícitamente. Sin la lista, el "
        f"planner racionaliza excepciones."
    )


def test_max_2_dias_constraint():
    """La regla debe usar `MÁXIMO 2 días` como cap explícito (no "
    "preferiblemente, evita, etc.)."""
    text = _read_planner()
    # Buscar la sección.
    idx = text.find("DIVERSIDAD OBLIGATORIA DE STAPLES")
    assert idx > 0, "Sección no encontrada para localizar el bloque."
    block = text[idx : idx + 2000]
    # Debe aparecer "MÁXIMO 2 días" al menos 3 veces (yogurt, avena, claras como mínimo).
    matches = block.count("MÁXIMO 2 días")
    assert matches >= 3, (
        f"Solo {matches} ocurrencias de 'MÁXIMO 2 días' en la sección. "
        f"La regla debe ser explícita para cada staple, no genérica."
    )


def test_alternativas_validas_provided():
    """El prompt debe ofrecer alternativas válidas para que el LLM sepa qué "
    "USAR cuando el cap se aplique."""
    text = _read_planner().lower()
    # Buscar al menos 2 ejemplos de alternativas en el bloque.
    idx = text.find("diversidad obligatoria de staples")
    assert idx > 0
    block = text[idx : idx + 2500]
    alternatives_found = sum(
        1 for alt in ("batido proteico", "casabe", "fruta", "sándwich")
        if alt in block
    )
    assert alternatives_found >= 2, (
        f"Solo {alternatives_found} alternativas encontradas en la regla. "
        f"Sin alternativas concretas, el LLM puede aplicar la regla "
        f"defaulteando al primer staple no-listed (drift)."
    )
