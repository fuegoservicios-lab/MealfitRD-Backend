# -*- coding: utf-8 -*-
"""[P1-SLOT-FLOOR-UNCLOSED + P1-PLANNER-RETRY-REASON · 2026-09-05] Dos silencios medidos en el plan vivo
18326457, ambos del mismo tipo: el código sabía que algo había ido mal y no lo escribía.

  · El piso de proteína por franja solo registra sus ÉXITOS. La merienda del día 3 quedó en 13 g sobre un
    reparto de 20 (135 g/día × 15 %), por debajo del piso del 70 %, y la pasada no dejó ni una línea: en el
    journal, «no había nada que cerrar» y «no pude cerrarlo» se ven exactamente igual.
  · El planificador reintentaba diciendo «Reintento #N...» y nada más. Sus tres fallos seguidos abren el
    disyuntor de `glm-5.3-flash` (32 veces el 2026-09-05) y mandan el esqueleto al modelo de respaldo, sin que
    quede registrado si fue un timeout, un JSON roto o un 400 del proveedor.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_el_piso_avisa_cuando_no_puede_cerrar():
    i = _SRC.index("def _repair_light_slot_protein")
    cuerpo = _SRC[i:i + 6000]
    assert "P1-SLOT-FLOOR-UNCLOSED" in cuerpo, "la rama de fallo tiene su marcador"
    j = cuerpo.index("P1-SLOT-FLOOR-UNCLOSED")
    # La ventana es de 1.800 y no de 900 a propósito: el comentario que explica la rama ocupa ~800
    # caracteres y dejaba el `logger.warning` FUERA del trozo que este test mira — el mismo modo de fallo
    # que ya nos comió una tarde. Se ancla el aviso, no la longitud de su comentario.
    aviso = cuerpo[j:j + 1800]
    for dato in ("_slot_target", "candidatos=", "kcal ", "dieta="):
        assert dato in aviso, f"el aviso debe llevar {dato} para distinguir la causa"
    assert "logger.warning" in aviso, "es warning, no info: un plato corto no es una curiosidad"


def test_la_rama_de_fallo_cuelga_del_mismo_if_que_la_de_exito():
    """Si el `else` se separa del `if _g > 0`, el aviso dejaría de correr y nadie lo notaría."""
    i = _SRC.index("def _repair_light_slot_protein")
    cuerpo = _SRC[i:i + 6000]
    a = cuerpo.index("if _g > 0:")
    b = cuerpo.index("else:", a)
    assert b - a < 700, "el else sigue pegado al if del cierre"
    assert cuerpo[a:b].count("logger.info") == 1, "la rama de éxito conserva su único log"


def test_el_piso_sigue_siendo_fail_safe():
    """Un aviso nuevo no puede convertir la pasada en una fuente de excepciones."""
    assert go._repair_light_slot_protein(None, None, None) == 0
    assert go._repair_light_slot_protein([{"meals": [{}]}], {}, {"mainGoal": "gain_muscle"}) == 0


def test_el_reintento_del_planificador_dice_el_motivo():
    i = _SRC.index("P1-PLANNER-RETRY-REASON")
    bloque = _SRC[i:i + 1200]
    assert "retry_state.outcome.exception()" in bloque, "el motivo sale de la excepción real"
    assert "__name__" in bloque, "el tipo de la excepción distingue timeout de JSON roto"
    assert "retry_state.outcome.failed" in bloque, "sin resultado fallido se cae al mensaje de siempre"


def test_el_lambda_no_revienta_sin_outcome():
    """tenacity puede llamar a before_sleep con un outcome vacío; el aviso no puede tumbar el planificador."""
    class _RS:
        attempt_number = 2
        outcome = None
    i = _SRC.index("before_sleep=lambda retry_state:", _SRC.index("P1-PLANNER-RETRY-REASON"))
    fin = _SRC.index("\n    )", i)
    fuente = _SRC[i:fin].split("before_sleep=", 1)[1].strip().rstrip(",")
    fn = eval(fuente, {"logger": go.logger})  # noqa: S307 - fuente propia del repo
    fn(_RS())  # no debe lanzar
