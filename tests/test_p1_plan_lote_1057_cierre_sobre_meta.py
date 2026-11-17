# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1057 · 2026-11-17] Cierre del coach cuando el día pasa de la meta.

Captura del dueño (16-nov): tras la cena el día quedó 1 399 kcal sobre la meta y el coach solo dio la cifra («superaste
tu meta… por ~1399 kcal»). La regla D (CIERRE) gana una frase: con el día >10 % por encima de su meta de kcal, UNA idea
concreta y amable para la próxima comida o para mañana, sin regañar ni culpa y nunca saltarse comidas para compensar.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FRASE = "Si lo anotado deja el día >10 % por encima de su meta de kcal"


def test_la_regla_esta_en_los_prompts_del_chat():
    from prompts import chat_agent as ca

    for nombre in ("CHAT_STREAM_INLINE_PROMPT",):
        texto = getattr(ca, nombre)
        assert FRASE in texto, nombre
        assert "sin regañar ni culpa" in texto
        assert "nunca saltarse comidas para compensar" in texto


def test_la_regla_sobrevive_al_manual_compacto():
    from prompts.chat_agent import CHAT_STREAM_INLINE_PROMPT, build_tools_instructions_stream_estaticas
    from prompts.coach_lean import adelgazar

    lean = adelgazar(CHAT_STREAM_INLINE_PROMPT + "\n" + build_tools_instructions_stream_estaticas(), frozenset())
    assert FRASE in lean


def test_el_modo_voz_sigue_sin_consejos_que_no_pidio():
    # V8 manda sobre los topes y el formato de arriba: en voz no se comenta si se pasó (salvo que lo pregunte).
    from prompts import chat_agent as ca

    voz = [v for k, v in vars(ca).items() if isinstance(v, str) and "MODO VOZ (MANDA SOBRE" in v]
    assert voz and all("SIN CONSEJOS QUE NO PIDIÓ" in v for v in voz)
