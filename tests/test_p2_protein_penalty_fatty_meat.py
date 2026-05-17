"""[P2-PROTEIN-PENALTY-FATTY-MEAT · 2026-05-16] Regression guard: cuando el
goal del user es `gain_muscle`, las proteínas de carne fresca GRASA (chuleta,
costilla, panceta, lechón, pernil) DEBEN ser penalizadas (×0.3) en la
selección del anti-mode-collapse.

Bug observado en plan_id=`fbd014b2-594d-4ad9-aa08-db7bf027a099` (2026-05-16 02:08:52):
  - Planner eligió pool `['Lentejas', 'Chuleta', 'Pavo']` para goal=gain_muscle.
  - Día 2 generó "Chuletas Krunsh y Tostones de Airfryer".
  - PROTEIN-RECIPE-VIOLATION strippeó chuleta del ingredients.
  - Receta quedó como "Chuleta al Airfryer..." SIN chuleta.
  - Revisor médico rechazó: "deficiencia nutricional severa en el almuerzo del
    Día 2, donde solo se incluyen tostones sin una fuente de proteína adecuada".

Root cause: la lista `_PROCESSED_MEAT_KEYWORDS` solo cubría embutidos
(salami, longaniza, jamón, etc.) pero NO carne fresca grasa (chuleta de cerdo,
costilla). Para `gain_muscle` el ratio proteína/grasa es subóptimo:
  - Chuleta de cerdo: ~250kcal/100g, 20g grasa.
  - Pechuga de pollo: 165kcal/100g, 3.6g grasa.

Fix: nueva categoría `_FATTY_FRESH_MEAT_KEYWORDS` con penalty ×0.3 que SOLO
aplica a `gain_muscle`. Menos agresivo que el ×0.1 de embutidos porque son
carnes frescas con valor nutricional legítimo en perfiles 'balanced'/cultural.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AI_HELPERS_PATH = _BACKEND_ROOT / "ai_helpers.py"


def _read_ai_helpers() -> str:
    return _AI_HELPERS_PATH.read_text(encoding="utf-8")


def test_fatty_fresh_meat_keywords_tuple_exists():
    """`_FATTY_FRESH_MEAT_KEYWORDS` debe existir como tuple con al menos
    `chuleta` y `costilla` (las 2 más críticas observadas en el bug)."""
    text = _read_ai_helpers()
    m = re.search(
        r"_FATTY_FRESH_MEAT_KEYWORDS\s*=\s*\(([^)]+)\)",
        text,
        re.DOTALL,
    )
    assert m, (
        "Falta tuple `_FATTY_FRESH_MEAT_KEYWORDS` en ai_helpers.py. "
        "P2-PROTEIN-PENALTY-FATTY-MEAT requiere esta lista."
    )
    body = m.group(1).lower()
    for kw in ("chuleta", "costilla", "panceta"):
        assert f'"{kw}"' in body or f"'{kw}'" in body, (
            f"`{kw}` falta en `_FATTY_FRESH_MEAT_KEYWORDS`. Sin él, el "
            f"planner puede elegirlo en pool gain_muscle."
        )


def test_goals_penalize_fatty_fresh_includes_gain_muscle():
    """El set `_GOALS_PENALIZE_FATTY_FRESH` debe incluir `gain_muscle`."""
    text = _read_ai_helpers()
    m = re.search(
        r"_GOALS_PENALIZE_FATTY_FRESH\s*=\s*\{([^}]+)\}",
        text,
    )
    assert m, "Falta set `_GOALS_PENALIZE_FATTY_FRESH`."
    body = m.group(1)
    assert '"gain_muscle"' in body or "'gain_muscle'" in body, (
        "`gain_muscle` debe estar en `_GOALS_PENALIZE_FATTY_FRESH`. "
        "Sin él, el penalty no se aplica."
    )


def test_penalty_multiplier_03():
    """El multiplier para fatty fresh meat debe ser ×0.3 (menos agresivo que
    ×0.1 de embutidos procesados — cerdo magro es legítimo en otros goals)."""
    text = _read_ai_helpers()
    # Buscar el bloque del penalty fatty.
    m = re.search(
        r"_GOALS_PENALIZE_FATTY_FRESH.*?protein_weights\[i\]\s*\*=\s*([\d.]+)",
        text,
        re.DOTALL,
    )
    assert m, "No encontré el bloque de penalty para fatty fresh meat."
    mult = float(m.group(1))
    assert 0.2 <= mult <= 0.5, (
        f"Multiplier {mult} fuera del rango razonable [0.2, 0.5]. "
        f"×0.1 es demasiado agresivo (carne fresca no es embutido). "
        f"×0.5 es demasiado suave (Chuleta sigue siendo elegida). "
        f"Default sugerido: 0.3."
    )


def test_existing_processed_meat_penalty_preserved():
    """Defensiva: el penalty original `_PROCESSED_MEAT_KEYWORDS` (embutidos)
    NO debe haberse roto al añadir el nuevo penalty fatty."""
    text = _read_ai_helpers()
    assert "_PROCESSED_MEAT_KEYWORDS" in text, (
        "Penalty original de embutidos eliminado por error."
    )
    assert "_GOALS_PENALIZE_PROCESSED" in text, (
        "Set `_GOALS_PENALIZE_PROCESSED` eliminado por error."
    )
    # Verificar que salami/longaniza/etc siguen en la lista.
    m = re.search(
        r"_PROCESSED_MEAT_KEYWORDS\s*=\s*\(([^)]+)\)",
        text,
        re.DOTALL,
    )
    assert m, "Tuple `_PROCESSED_MEAT_KEYWORDS` desapareció."
    for kw in ("salami", "chorizo", "tocineta"):
        assert kw in m.group(1).lower(), (
            f"`{kw}` removido de `_PROCESSED_MEAT_KEYWORDS`. "
            f"Regresión: estos embutidos deben seguir penalizados ×0.1."
        )
