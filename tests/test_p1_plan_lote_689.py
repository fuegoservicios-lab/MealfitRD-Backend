"""[P1-PLAN-LOTE-689 · 2026-09-29] En modo voz, la cantidad que varía mucho SÍ se pregunta.

El 686 le dijo al coach que no preguntara cantidades («usa la porción típica») porque el dueño pidió anotar rápido.
Su ejemplo siguiente pedía lo contrario: «si le digo que me comí 2 huevos con pan integral, que me diga "¿cuántas
lonjas?", ya que le fue ambiguo». La regla: la cantidad dicha u obvia no se pregunta; la de algo que varía mucho de
una persona a otra (pan, arroz, tortillas, casabe, jugo) se pregunta UNA vez antes de registrar.
"""
from __future__ import annotations

from prompts.chat_agent import CHAT_VOICE_MODE_RECORDATORIO, _CHAT_CALL_MODE_RULES


def test_la_regla_pregunta_solo_la_cantidad_que_varia_mucho():
    assert "La CANTIDAD: si la dijo o es obvia" in _CHAT_CALL_MODE_RULES
    assert "pan, arroz, tortillas, casabe, jugo" in _CHAT_CALL_MODE_RULES
    assert "¿Cuántas lonjas de pan?" in _CHAT_CALL_MODE_RULES
    assert "pregunta UNA vez antes de registrar" in _CHAT_CALL_MODE_RULES
    assert "la cantidad que no dijo NO se pregunta" not in _CHAT_CALL_MODE_RULES, "la regla del 686 queda sustituida"


def test_registrar_de_inmediato_es_cuando_ya_sabe_que_y_cuanto():
    assert "ya sabes qué y cuánto, llama a la herramienta de registro DE INMEDIATO" in _CHAT_CALL_MODE_RULES


def test_el_recordatorio_final_dice_lo_mismo():
    assert "falta la cantidad de algo que varía mucho" in CHAT_VOICE_MODE_RECORDATORIO
    assert "registra con su respuesta" in CHAT_VOICE_MODE_RECORDATORIO
    assert len(CHAT_VOICE_MODE_RECORDATORIO) < 700
