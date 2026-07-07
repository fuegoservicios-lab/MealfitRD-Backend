"""[P1-BLENDER-STEP-COHERENCE · 2026-07-06] Pasos de receta robóticos en
preparaciones LICUADAS (batido/jugo/smoothie).

Detectado en la review visual del "Batido verde" del plan de 30 días:
  Paso 2: "Incorpora queso a la preparación y mézclalo antes de servir."
  Paso 3: "Incorpora también agua fría al plato antes de servir, integrándolo
           de forma coherente."
Ambos leen robóticos y contradicen el Montaje (que ya licúa todo). Dos causas:

 A. Falso positivo de `agua fría` en el reverse-coherence guard
    (`_ensure_ingredients_used_in_recipe`): "1 taza de agua fría" → el único
    token ≥4 chars es "fria" (temperatura), que no aparece literal en los
    pasos (el Montaje dice "el agua") → paso espurio. Fix: agua/hielo se
    saltan (nunca son un ingrediente "olvidado").

 B. El protein-closer (`_closer_protein_step_text`) usaba el wording de plato
    ("Incorpora ... mézclalo") en un licuado. Fix: rama `blended` →
    "Agrega X a la licuadora y licúa hasta integrar".

En un licuado, un ingrediente ausente va A LA LICUADORA, no "al plato".
tooltip-anchor: P1-BLENDER-STEP-COHERENCE
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser-based: contrato del wording
# ---------------------------------------------------------------------------
def test_marker_present():
    assert "P1-BLENDER-STEP-COHERENCE" in _SRC


def test_water_ice_skip_in_reverse_coherence():
    """El reverse-coherence guard DEBE saltar agua/hielo antes de flaggearlos
    como ingredientes no usados."""
    assert 'bare_low in ("agua", "hielo")' in _SRC and 'bare_low.startswith("agua ")' in _SRC


def test_closer_text_has_blended_param():
    """`_closer_protein_step_text` DEBE aceptar `blended` y ramificar a licuadora."""
    assert "def _closer_protein_step_text(nm: str, no_cook: bool, blended: bool = False)" in _SRC
    assert "a la licuadora y licúa hasta integrar" in _SRC


# ---------------------------------------------------------------------------
# 2. Funcional: protein-closer text (pura)
# ---------------------------------------------------------------------------
def test_closer_blended_wording():
    txt = g._closer_protein_step_text("queso blanco", no_cook=True, blended=True)
    assert txt == "Agrega queso blanco a la licuadora y licúa hasta integrar.", txt


def test_closer_non_blended_backcompat():
    """Sin blended (default), el wording de lácteo blando no cambia."""
    txt = g._closer_protein_step_text("queso blanco", no_cook=True, blended=False)
    assert txt == "Incorpora queso blanco a la preparación y mézclalo antes de servir.", txt


def test_closer_cook_wording_backcompat():
    """Proteína cocinable sin blended → sigue el wording de cocción."""
    txt = g._closer_protein_step_text("pechuga de pollo", no_cook=False, blended=False)
    assert "Cocina pechuga de pollo" in txt, txt


# ---------------------------------------------------------------------------
# 3. Funcional: reverse-coherence guard en batidos
# ---------------------------------------------------------------------------
def test_agua_fria_no_genera_paso_espurio():
    """Un batido con 'agua fría' cuyo Montaje ya licúa el agua NO debe recibir
    el paso robótico 'Incorpora también agua fría al plato...'."""
    meal = {
        "name": "Batido verde",
        "ingredients": ["2½ naranjas", "1 taza de agua fría", "Hielo al gusto"],
        "recipe": [
            "MISE EN PLACE: Pela la naranja.",
            "MONTAJE: Coloca en la licuadora la naranja, el agua y el hielo. Licúa hasta homogéneo.",
        ],
    }
    added = g._ensure_ingredients_used_in_recipe(meal)
    assert added == 0, f"agua/hielo no deben generar pasos (added={added})"
    assert not any("agua fría al plato" in s for s in meal["recipe"]), meal["recipe"]
    assert not any("agua fria al plato" in s.lower() for s in meal["recipe"]), meal["recipe"]


def test_ingrediente_ausente_en_batido_usa_wording_licuadora():
    """Si un ingrediente REAL falta en un licuado, el paso añadido usa el
    wording de licuadora, no 'al plato'."""
    meal = {
        "name": "Batido verde",
        "ingredients": ["2½ puñado de espinacas frescas", "1 taza de agua fría"],
        "recipe": ["MONTAJE: Coloca en la licuadora el agua. Licúa hasta homogéneo."],
    }
    added = g._ensure_ingredients_used_in_recipe(meal)
    assert added == 1, f"espinaca ausente debe añadir 1 paso (added={added})"
    joined = " ".join(meal["recipe"])
    assert "a la licuadora antes de licuar" in joined, meal["recipe"]
    assert "al plato antes de servir" not in joined, meal["recipe"]
