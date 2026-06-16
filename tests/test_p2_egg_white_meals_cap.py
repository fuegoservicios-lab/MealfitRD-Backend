"""[P2-EGG-WHITE-MEALS-CAP · 2026-05-16] Regression guard: además del cap por
meal (6 claras) y por día (12 claras), añadir cap de FRECUENCIA — máximo 2
meals/día con claras como proteína base.

Bug observado en plan_id=`fbd014b2-594d-4ad9-aa08-db7bf027a099` (2026-05-16 02:15:38):
  - Reviewer médico rechazó: "La frecuencia de consumo de claras de huevo es
    excesivamente alta en múltiples comidas, lo que requiere una advertencia
    sobre la necesidad de cocción completa para evitar la deficiencia de biotina."
  - El plan tenía claras en 3-4 meals/día (consolidation logueó "4.5 claras
    de huevo" como base en múltiples meals).
  - Caps existentes (PER_MEAL=6, PER_DAY=12) NO limitaban frecuencia: la suma
    diaria estaba dentro de 12 (e.g., 3 meals × 4 = 12) pero la frecuencia era
    excesiva.

Fix: nuevo cap `MEALFIT_MAX_MEALS_WITH_EGG_WHITES` (default 2). Si el día
tiene >2 meals con claras, los meals en exceso ven sus claras recortadas a 1
simbólica (no se elimina el ingrediente para no romper la receta).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_graph() -> str:
    return _GRAPH_PATH.read_text(encoding="utf-8")


def test_knob_defined():
    """`MAX_MEALS_WITH_EGG_WHITES` debe estar definido como knob via `_env_int`."""
    text = _read_graph()
    m = re.search(
        r'MAX_MEALS_WITH_EGG_WHITES\s*=\s*_env_int\(\s*["\']MEALFIT_MAX_MEALS_WITH_EGG_WHITES["\']\s*,\s*(\d+)',
        text,
    )
    assert m, (
        "Falta knob `MAX_MEALS_WITH_EGG_WHITES = _env_int(\"MEALFIT_MAX_MEALS_WITH_EGG_WHITES\", ...)`. "
        "P2-EGG-WHITE-MEALS-CAP requiere este cap para evitar reviewer rechazos."
    )
    default = int(m.group(1))
    assert 1 <= default <= 3, (
        f"Default {default} fuera del rango razonable [1, 3]. "
        f"1 es muy restrictivo (incluso desayuno regular bloquea); "
        f"3+ no cierra el bug observado."
    )


def test_cap_applied_in_egg_white_block():
    """El cap por frecuencia debe aplicarse DESPUÉS de los otros 2 caps "
    "(PER_MEAL y PER_DAY) — orden: primero recortar cantidades, luego "
    "recortar frecuencia."""
    text = _read_graph()
    # El cap nuevo debe aparecer después de "MAX_EGG_WHITES_PER_DAY" pero
    # antes del log final. El knob PER_DAY se declara como asignación
    # `MAX_EGG_WHITES_PER_DAY  = _env_int("MEALFIT_MAX_EGG_WHITES_PER_DAY",  12)`.
    idx_max_day = text.find('MAX_EGG_WHITES_PER_DAY  = _env_int')
    idx_max_meals = text.find("MAX_MEALS_WITH_EGG_WHITES")
    assert idx_max_day > 0 and idx_max_meals > 0, "Knobs no encontrados."
    assert idx_max_meals > idx_max_day, (
        "Knob MAX_MEALS_WITH_EGG_WHITES debe estar declarado DESPUÉS de "
        "MAX_EGG_WHITES_PER_DAY (los 3 forman familia EGG-WHITE-CAP)."
    )


def test_loop_for_meals_with_eggw_count():
    """El bloque debe iterar sobre `result.days[*].meals` contando los que "
    "tienen claras, NO solo aplicar al primero."""
    text = _read_graph()
    # Buscar el bloque de IMPLEMENTACIÓN del cap nuevo. El marker
    # `[P2-EGG-WHITE-MEALS-CAP` aparece dos veces: en la declaración del knob
    # (comentario, ~línea 509) y en el pass de implementación (~línea 11188,
    # anotado "Tercer pass"). El loop vive en el segundo, así que anclamos a
    # ese marker concreto.
    idx = text.find("[P2-EGG-WHITE-MEALS-CAP · 2026-05-16] Tercer pass")
    assert idx > 0, (
        "Bloque de implementación `[P2-EGG-WHITE-MEALS-CAP ... Tercer pass` "
        "no encontrado en cuerpo."
    )
    # Ventana suficiente para abarcar el for-loop.
    block = text[idx : idx + 2000]
    assert "_meals_with_eggw_list" in block, (
        "El bloque no construye lista de meals con claras (`_meals_with_eggw_list`). "
        "Sin esto, no puede contar la frecuencia."
    )
    assert "len(_meals_with_eggw_list)" in block, (
        "El bloque no verifica `len(_meals_with_eggw_list) > MAX_MEALS_WITH_EGG_WHITES`. "
        "Sin esa check, el cap no se aplica."
    )


def test_recorta_a_1_simbolica_no_elimina():
    """El cap recorta a 1 clara simbólica en lugar de eliminar el ingrediente. "
    "Eliminar rompería la receta (ej. \"Revoltillo de Claras\" sin claras)."""
    text = _read_graph()
    # Anclar al bloque de implementación (segundo marker, "Tercer pass"), no
    # al comentario del knob.
    idx = text.find("[P2-EGG-WHITE-MEALS-CAP · 2026-05-16] Tercer pass")
    block = text[idx : idx + 2000]
    # Debe asignar `f"1 {_rest}"` (1 clara simbólica) en lugar de empty string
    assert re.search(r'"1 \{_rest\}"', block) or "f\"1 {_rest}\"" in block, (
        "El cap por frecuencia NO recorta a `1 {_rest}` (clara simbólica). "
        "Eliminar el ingrediente entero rompe la receta downstream."
    )


def test_log_consolidado_3_caps():
    """El log final debe consolidar los 3 caps (PER_MEAL, PER_DAY, MEALS_COUNT) "
    "en una sola línea con info útil para SRE."""
    text = _read_graph()
    # Buscar el log [EGG-WHITE-CAP] post-fix. El log es un f-string multilínea
    # que contiene comillas internas, así que usamos DOTALL (`.` matchea
    # newlines) en vez de `[^"]*` (que se cortaba en la primera comilla).
    log_match = re.search(
        r'\[EGG-WHITE-CAP\].*?MAX_MEALS_WITH_EGG_WHITES',
        text,
        re.DOTALL,
    )
    assert log_match, (
        "El log [EGG-WHITE-CAP] no menciona `MAX_MEALS_WITH_EGG_WHITES`. "
        "Post-fix el log debe reportar los 3 caps aplicados."
    )
