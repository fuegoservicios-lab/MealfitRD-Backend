"""[P1-FRUIT-SAVORY-BURN-FIX · 2026-07-29] El corrector de pareo fruta-dulce+salado debe tener la
ÚLTIMA palabra, no la primera.

`_fruit_savory_autofix` corría SOLO pre-motor. Los pases tardíos meten proteína en el plato y le
reescriben el nombre (`_repair_protein_floor_post_caps`, el protein-closer, el namefix de proteína
fantasma), así que el pareo puede NACER después de que el corrector ya pasó — y entonces nadie lo
corrige y el reviewer lo convierte en rechazo HIGH.

Medido en vivo (corr=5cbced82, 2026-07-29): el intento #1 llegó al reviewer con banda **1.00**
(12/12 celdas, 18:48:51), presupuesto dentro y micros al 79%, y se descartó ENTERO por
'PAREO CHOCANTE FRUTA+SALADO' → retry completo 18:49:08 → 18:52:27 (~3.4 min, planificador +
day-gen + assemble). El corrector determinista resuelve ese plato en milisegundos y con coste LLM
cero.

Es la MISMA clase (y la misma cura) que `P1-SAMEDAY-BURN-FIX · 2026-07-11`, que ya re-corre su
autofix tras el chain por haber quemado 2 intentos con banda 1.00 en corr=57a373e0. Este archivo
ancla el contrato de ORDEN, que es lo único que hace útil a un corrector idempotente.
"""
from __future__ import annotations

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knob_exists_and_defaults_on():
    import graph_orchestrator as go
    assert 'FRUIT_SAVORY_LATE_REFIX_ENABLED = _env_bool("MEALFIT_FRUIT_SAVORY_LATE_REFIX", True)' in _GO
    assert go.FRUIT_SAVORY_LATE_REFIX_ENABLED is True


def _line_of(needle: str) -> int:
    """Nº de línea (1-indexed) de una aparición que DEBE ser única.

    Dos anclas de este mismo test fallaron por no serlo: `[AGENTE REVISOR MÉDICO] Verificando plan`
    aparece también en un comentario muy anterior, y el import de la chain de calidad existe en dos
    callsites. Con `str.index` el orden salía invertido aunque el código estuviera bien — un rojo
    que miente. El ancla se elige por UNICIDAD, no por conveniencia, y aquí se enforza."""
    lines = _GO.split("\n")
    hits = [i + 1 for i, ln in enumerate(lines) if needle in ln]
    assert len(hits) == 1, (
        f"ancla NO única ({len(hits)} apariciones en L{hits[:5]}): {needle!r}. "
        f"Elige un literal que solo exista en el callsite que quieres anclar.")
    return hits[0]


def test_refix_runs_after_the_quality_chain_and_before_the_reviewer():
    """El ORDEN es todo el fix: correr el corrector otra vez ANTES del chain no arregla nada,
    porque el pareo todavía no ha nacido."""
    l_pre = _line_of("_fs_fixed = _fruit_savory_autofix(days, form_data)")
    l_chain = _line_of('await _adb(_apqfc, result, surface="assemble-tail")')
    l_refix = _line_of("if FRUIT_SAVORY_LATE_REFIX_ENABLED:")
    l_review = _line_of('logger.info(f"🩺 [AGENTE REVISOR MÉDICO] Verificando plan')
    assert l_pre < l_chain < l_refix < l_review, (
        f"orden roto: pre-motor L{l_pre} · chain L{l_chain} · refix L{l_refix} · reviewer L{l_review}")


def test_refix_mirrors_its_sibling_block():
    """Convive con P1-SAMEDAY-BURN-FIX en el mismo punto y con la misma forma: si alguien mueve uno,
    que el otro quede visible al lado (no en un rincón distinto del archivo)."""
    i_sameday = _GO.index("[P1-SAMEDAY-BURN-FIX] re-autofix post-chain no-op")
    i_fruit = _GO.index("[P1-FRUIT-SAVORY-BURN-FIX] re-autofix post-chain no-op")
    assert 0 < (i_fruit - i_sameday) < 2500, "los dos re-fixes post-chain viven juntos"


def test_refix_is_failsafe():
    """Un corrector de conveniencia JAMÁS puede tumbar la entrega: su excepción se loguea y sigue."""
    seg = _GO[_GO.index("if FRUIT_SAVORY_LATE_REFIX_ENABLED:"):]
    seg = seg[:1400]
    assert "except Exception as _fs_le:" in seg
    assert "logger.warning" in seg, "fail-open pero NUNCA mudo"


def test_autofix_is_idempotent_enough_to_rerun():
    """Re-correrlo sobre un plan ya limpio no debe cambiar nada (es lo que permite llamarlo 2×)."""
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Pollo Guisado con Arroz",
         "ingredients": ["150 g de pollo", "100 g de arroz blanco"], "recipe": ["Guisa el pollo."]}]}]
    import copy
    before = copy.deepcopy(days)
    n1 = go._fruit_savory_autofix(days, {})
    n2 = go._fruit_savory_autofix(days, {})
    assert n1 == 0 and n2 == 0, "plato sin fruta dulce: nada que corregir"
    assert days == before, "y el plan queda intacto"
