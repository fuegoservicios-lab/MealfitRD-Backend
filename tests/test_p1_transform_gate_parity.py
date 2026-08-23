"""[P1-TRANSFORM-GATE-PARITY · 2026-07-29] El gate de "preparación transformada" rechazaba planes
pidiendo algo que su propio detector no sabía contar.

Tres listas del MISMO concepto, distintas entre sí:
  · prompt §19 (prompts/day_generator.py): «guisos, locrios (almuerzo), panqueques/arepitas,
    bollitos/buñuelos, revoltillos, tortitas/croquetas, mangú, ensaladas COMPUESTAS»
  · mensaje de rechazo (graph_orchestrator): «panqueques de avena, arepitas, bollitos de yuca,
    revoltillo, guiso, locrio de almuerzo»
  · detector `_TRANSFORM_NAME_TOKENS`: SIN guiso, guisado, locrio, revoltillo ni tortilla.

Medido en vivo (corr=5cbced82, 2026-07-29): un plan con 'Revoltillo…', 'Bulgur GUISADO Estilo
Pilaf', 'Nabo GUISADO con Huevo Pochado' y 'TORTILLA de Vegetales al Estilo Criollo' puntuó
`transform_meals=0` y fue rechazado con severidad HIGH pidiéndole exactamente eso. **Un rechazo que
el LLM no puede obedecer**: puede escribir el guiso que el mensaje le pide y volver a puntuar 0. Y
como el gate es HIGH, cuesta un retry completo (~3.4 min + 2 llamadas LLM).

El match pasa a FRONTERA DE PALABRA a la vez, porque añadir "guis" por substring habría marcado como
transformado cualquier plato con GUISAntes de guarnición (plato frío) — se reusa el `\\bguis(?!ant)`
que ya vivía en `_meal_is_hot_cooked`.
"""
from __future__ import annotations

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "prompts", "day_generator.py"), encoding="utf-8") as f:
    _DG = f.read()


def _tf(name: str) -> bool:
    import graph_orchestrator as go
    from constants import strip_accents as sa
    return go._name_is_transformed(sa(name.lower()))


@pytest.mark.parametrize("name", [
    # los platos REALES de la corrida rechazada
    "Revoltillo de Huevo con Vegetales",
    "Bulgur Guisado Estilo Pilaf con Puerro",
    "Nabo Guisado con Huevo Pochado y Kale Salteado",
    "Tortilla de Vegetales al Estilo Criollo",
    # el resto de familias que el prompt promete
    "Locrio de Pollo",
    "Moro de Guandules",
    "Buñuelos de Yuca",
    "Asopao de Camarones",
    "Panqueques de Avena",
    "Arepitas de Yuca",
])
def test_promised_families_now_count(name):
    assert _tf(name), f"el prompt y el mensaje de rechazo prometen que {name!r} cuenta"


@pytest.mark.parametrize("name", [
    "Pollo a la Plancha con Arroz Blanco",
    "Pechuga al Horno con Ensalada Verde",
    "Filete de Pescado al Vapor con Batata",
    "Yogurt Griego con Fresas",
])
def test_plain_staples_still_do_not_count(name):
    """El gate existe para rechazar el plan de puros staples servidos: no puede volverse trivial."""
    assert not _tf(name), f"{name!r} es un staple servido, no una preparación transformada"


def test_guisantes_does_not_count_as_guiso():
    """`guis` ⊂ GUISAntes: 'Batata con Guisantes' es un plato FRÍO. Sin el `(?!ant)`, añadir el
    token 'guiso' habría marcado como transformado cualquier plato con guisantes de guarnición —
    la enésima mordida de la clase 'res'⊂'fresco'."""
    assert not _tf("Batata con Guisantes")
    assert not _tf("Ensalada de Guisantes y Zanahoria")
    assert _tf("Pollo Guisado con Guisantes"), "pero un guiso REAL sí, aunque lleve guisantes"


def test_detector_covers_what_the_rejection_message_promises():
    """Paridad bidireccional: cada familia nombrada en el mensaje de rechazo debe ser detectable.
    Si alguien añade un ejemplo al mensaje sin añadirlo al detector, vuelve el rechazo imposible."""
    # [reapuntado 2026-08-23, P1-COUNTRY-*] Los ejemplos del mensaje se componen por
    # país unas líneas ANTES de la frase (rama `transform_minimum` de
    # `_review_country_feedback`): la ventana arranca en la rama, no en la frase,
    # o los ejemplos quedan fuera del recorte. Las familias ancladas son las DO.
    i = _GO.index('if kind == "transform_minimum"')
    msg = _GO[i:_GO.index('if kind == ', i + 10)]
    assert "El plan no incluye NINGUNA preparación transformada" in msg
    for familia, ejemplo in (("panqueque", "Panqueques de Avena"), ("arepita", "Arepitas de Yuca"),
                             ("bollito", "Bollitos de Yuca"), ("revoltillo", "Revoltillo de Huevo"),
                             ("guiso", "Pollo Guisado"), ("locrio", "Locrio de Pollo")):
        assert familia in msg, f"el mensaje ya no nombra {familia!r} — actualiza este test"
        assert _tf(ejemplo), f"el mensaje promete {familia!r} y el detector no lo cuenta"


def test_detector_covers_what_the_prompt_promises():
    """Misma paridad contra el prompt §19, que es lo que el LLM lee al generar."""
    assert "PREPARACIONES TRANSFORMADAS" in _DG
    i = _DG.index("PREPARACIONES TRANSFORMADAS")
    blk = _DG[i:i + 900]
    for familia, ejemplo in (("guiso", "Pollo Guisado"), ("locrio", "Locrio de Pollo"),
                             ("revoltillo", "Revoltillo de Huevo"), ("mangú", "Mangú con Cebolla"),
                             ("croqueta", "Croquetas de Pollo al Horno")):
        assert familia in blk, f"el prompt ya no nombra {familia!r} — actualiza este test"
        assert _tf(ejemplo), f"el prompt promete {familia!r} y el detector no lo cuenta"


def test_word_boundary_knob_and_rollback():
    import graph_orchestrator as go
    assert 'TRANSFORM_GATE_WORD_BOUNDARY = _env_bool("MEALFIT_TRANSFORM_GATE_WORD_BOUNDARY", True)' in _GO
    assert go.TRANSFORM_GATE_WORD_BOUNDARY is True


def test_counter_uses_the_ssot_helper():
    """El contador del report debe pasar por `_name_is_transformed`, no re-implementar el match."""
    seg = _GO[_GO.index("def compute_dish_quality_report"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "_name_is_transformed(_nm_tf)" in seg
    assert "any(t in _nm_tf for t in _TRANSFORM_NAME_TOKENS)" not in seg


def test_reconcile_efficacy_probe_measures_after_the_pass():
    """[P2-RECONCILE-EFFICACY-PROBE] `_trace_misalign` solo corría ANTES del reconciliador, así que
    nadie sabía cuántas divergencias quedaban abiertas tras su paso — el dato que hace falta para
    decidir si el pase mide en la etapa equivocada. Medir antes de mover."""
    i_pre = _GO.index('_trace_misalign(result.get("days"), "pre_reconcile")')
    i_pass = _GO.index("_reconcile_display_raw_lines(result.get(\"days\") or [])")
    i_post = _GO.index('_trace_misalign(result.get("days"), "post_reconcile")')
    assert i_pre < i_pass < i_post, "la sonda va DESPUÉS del pase que debe evaluar"
    assert 'RECONCILE_EFFICACY_PROBE = _env_bool("MEALFIT_RECONCILE_EFFICACY_PROBE", True)' in _GO
