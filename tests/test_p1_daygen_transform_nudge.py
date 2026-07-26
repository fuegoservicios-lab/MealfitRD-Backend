"""[P1-DAYGEN-TRANSFORM-NUDGE · 2026-07-09] Nudge §19 en el system prompt del day-gen para exigir ≥1
preparación transformada real y NO caer en "puros staples servidos".

Forense plan f19d55a6 (gain_muscle, gen en vivo): intento #1 rechazado HIGH por el gate TRANSFORM_SOFT_GATE
(transform_meals=0 < MEALFIT_TRANSFORM_GATE_MIN_COUNT=1) → "El plan no incluye NINGUNA preparación
transformada" → retry COMPLETO pagado. El prompt pedía creatividad pero no exigía un MÍNIMO de
transformadas; el LLM emitía proteína-plancha + carbo-hervido + veg-suelto. El nudge (estático a
import-time, prompt-cache-safe, espejo del §17/§18) enseña la regla de conteo que cierra el gate desde el
intento 1.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "prompts", "day_generator.py"), encoding="utf-8") as f:
    _DG = f.read()


def test_marker_present():
    assert "P1-DAYGEN-TRANSFORM-NUDGE" in _DG


def test_section_19_content_in_source():
    assert "PREPARACIONES TRANSFORMADAS" in _DG
    # menciona ejemplos concretos de preparación transformada dominicana
    assert "locrio" in _DG.lower()
    assert "staple" in _DG.lower()


def test_prompt_constant_contains_nudge():
    """El nudge debe estar EN el system prompt construido (append estático a import-time)."""
    import prompts.day_generator as dg
    assert "PREPARACIONES TRANSFORMADAS" in dg.DAY_GENERATOR_SYSTEM_PROMPT
    # regla de conteo: al menos 1 transformada por día
    assert "AL MENOS una preparación transformada" in dg.DAY_GENERATOR_SYSTEM_PROMPT


def test_nudge_preserves_macros_contract():
    """El nudge aclara que transformar es la TÉCNICA, no cambia los macros (evita que el LLM re-dimensione)."""
    import prompts.day_generator as dg
    assert "NO cambia los macros" in dg.DAY_GENERATOR_SYSTEM_PROMPT
