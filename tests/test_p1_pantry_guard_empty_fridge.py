"""[P1-PANTRY-GUARD-EMPTY-FRIDGE · 2026-07-11] Nevera consultada y VACÍA desactiva la
validación estricta de despensa (y la rotación implícita).

Caso vivo (renovación del owner, 15:23-15:33): Nevera real = 0 filas, el cliente envió
`current_pantry_ingredients: []` explícito, y aun así el guard estricto validó contra
una lista fantasma (fallback legacy `current_shopping_list`) → "Ingredientes
COMPLETAMENTE INEXISTENTES: [todo el plan]" → 2 intentos quemados → entrega degradada
con banda 1.00. El mismo fallback marcaba la generación como "ROTACIÓN estricta".

Contrato:
1. `current_pantry_ingredients` PRESENTE y vacío = "nevera consultada y vacía" (señal
   moderna de verdad) → el fallback legacy NO reactiva la validación (`has_pantry` cae
   a False) ni la rama de rotación estricta. Flags explícitos (_is_rotation_reroll /
   _strict_pantry_required) siguen mandando.
2. Key AUSENTE → comportamiento legacy intacto (clientes viejos con solo
   current_shopping_list conservan su semántica).
3. Instrumentación: al validar, el log nombra fuente + tamaño + muestra +
   update_reason (la lista fantasma del caso vivo era irrastreable).

tooltip-anchor: P1-PANTRY-GUARD-EMPTY-FRIDGE
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_empty_fridge_disables_legacy_fallback_in_review():
    m = re.search(
        r'_fridge_checked_empty = \("current_pantry_ingredients" in form_data\s*'
        r'and not form_data\.get\("current_pantry_ingredients"\)\)',
        _GO_SRC,
    )
    assert m, "señal 'nevera consultada y vacía' desapareció del review-gate"
    _i = _GO_SRC.find("_fridge_checked_empty")
    assert "has_pantry = False" in _GO_SRC[_i:_i + 2000], (
        "con nevera vacía explícita, el fallback current_shopping_list no debe "
        "reactivar la validación estricta (caso vivo: TODO inexistente → degradado)"
    )


def test_explicit_flags_still_win():
    i = _GO_SRC.find("[P1-PANTRY-GUARD-EMPTY-FRIDGE] Nevera consultada y vacía")
    window = _GO_SRC[max(0, i - 800):i]
    assert "not (is_rotation or is_strict_required)" in window, (
        "_is_rotation_reroll/_strict_pantry_required explícitos siguen forzando "
        "validación aunque la nevera esté vacía (contratos de swap/rotación)"
    )


def test_rotation_prompt_respects_empty_fridge():
    m = re.search(
        r'_rot_fridge_empty = \("current_pantry_ingredients" in actual_form_data\s*'
        r'and not actual_form_data\.get\("current_pantry_ingredients"\)\)',
        _GO_SRC,
    )
    assert m, "la rama de rotación (prompt estricto) debe respetar la nevera vacía"
    assert "is_rotation = (not _rot_fridge_empty) and bool(" in _GO_SRC


def test_validation_logs_source():
    assert "fuente={_pg_src}" in _GO_SRC and "update_reason={form_data.get('update_reason')!r}" in _GO_SRC, (
        "la validación debe nombrar su fuente — la lista fantasma del caso vivo "
        "fue irrastreable sin esto"
    )


def test_marker_anchored_in_source():
    assert _GO_SRC.count("P1-PANTRY-GUARD-EMPTY-FRIDGE") >= 3
