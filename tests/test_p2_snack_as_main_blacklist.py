"""[P2-SNACK-AS-MAIN-BLACKLIST · 2026-05-16] Regression guard: el prompt del
day_generator DEBE prohibir explícitamente que ingredientes-snack (galletas de
soda, plátano chips, palitos de pan, etc.) sean el componente principal por
peso de un desayuno/almuerzo/cena.

Bug observado en plan_id=`fbd014b2-594d-4ad9-aa08-db7bf027a099` (2026-05-16 02:15:38):
  - Cena Día 3 generada con "Galletas de soda 105g" como ingrediente
    predominante por peso.
  - Revisor médico rechazó: "El plan presenta una calidad nutricional
    cuestionable en la cena del Día 3, basándose excesivamente en galletas de
    soda (105g) como componente principal."

Root cause: el prompt del day_generator mencionaba galletas de soda solo en
la regla #14 (cap de sodio por DÍA, no por meal). El LLM cumplía técnicamente
(1 alimento salty) pero el meal quedaba nutricionalmente pobre.

Fix: regla nueva `e) INGREDIENTES-SNACK PROHIBIDOS COMO COMPONENTE PRINCIPAL`
en el bloque de slots del prompt, con alternativas válidas (casabe, pan
integral, tostones caseros).
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _BACKEND_ROOT / "prompts" / "day_generator.py"


def _read_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def test_snack_blacklist_marker_present():
    """El prompt debe incluir el marker `P2-SNACK-AS-MAIN-BLACKLIST` para que
    sea trazable. Sin el marker, alguien puede borrarlo en un refactor
    cosmético sin saber su origen."""
    text = _read_prompt()
    assert "P2-SNACK-AS-MAIN-BLACKLIST" in text, (
        "Falta marker `P2-SNACK-AS-MAIN-BLACKLIST` en day_generator.py. "
        "Convención del repo: cada regla del prompt debe tener marker para "
        "trazabilidad."
    )


@pytest.mark.parametrize("forbidden_snack", [
    "galletas de soda",
    "plátano chips",
    "palitos de pan",
])
def test_forbidden_snack_listed(forbidden_snack: str):
    """Cada ingrediente-snack crítico debe estar listado explícitamente como
    prohibido en el prompt. Si solo se menciona "snacks" genéricamente, el LLM
    puede racionalizar excepciones."""
    text = _read_prompt().lower()
    assert forbidden_snack.lower() in text, (
        f"`{forbidden_snack}` no aparece explícitamente como prohibido en el "
        f"prompt. El LLM necesita la lista exacta para constraint efectivo."
    )


@pytest.mark.parametrize("valid_alternative", [
    "casabe",
    "pan integral",
    "tostones caseros",
])
def test_valid_alternatives_provided(valid_alternative: str):
    """El prompt debe ofrecer alternativas válidas (casabe, pan integral,
    tostones caseros) para que el LLM sepa qué USAR en lugar de los snacks
    prohibidos. Una blacklist sin sustitutos genera regen loops."""
    text = _read_prompt().lower()
    assert valid_alternative.lower() in text, (
        f"`{valid_alternative}` no aparece como alternativa válida en el "
        f"prompt. La blacklist necesita sustitutos explícitos."
    )


def test_blacklist_constraint_strict_wording():
    """La regla debe usar lenguaje imperativo (`NUNCA`, `PROHIBIDO`) y no
    sugerir ('evita', 'preferiblemente'). LLMs ignoran constraints débiles
    bajo presión de coherencia narrativa."""
    text = _read_prompt()
    # Buscar el bloque de la regla.
    idx = text.find("P2-SNACK-AS-MAIN-BLACKLIST")
    assert idx > 0, "Marker no encontrado para localizar el bloque."
    # Ventana ~800 chars desde el marker hacia adelante.
    block = text[idx : idx + 800]
    assert ("NUNCA" in block or "PROHIBIDO" in block.upper()), (
        "El bloque de blacklist NO usa lenguaje imperativo "
        "(NUNCA/PROHIBIDO). Los LLMs ignoran 'evita'/'preferiblemente' bajo "
        "presión narrativa — la regla debe ser absoluta."
    )
