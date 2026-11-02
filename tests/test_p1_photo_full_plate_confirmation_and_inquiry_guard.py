# -*- coding: utf-8 -*-
"""[BB-016 · 2026-11-02] Pruebas de regresión para:
1. Confirmación visible con alimentos principales del plato completo (no solo el chip/ingrediente contestado).
2. Detección de estado pasivo y preguntas informativas para evitar falsos positivos de [P1-DIARY-CLAIM-VERIFY].
"""
from __future__ import annotations

import pytest
from diario_afirmacion import (
    participio_describe_estado,
    mensaje_sin_nada_que_registrar,
)
from agent import _diary_claim_sentences, _reply_claims_diary_write


def test_participio_describe_estado_handles_referencias_pasivas():
    # Referencias relativas o de estado previo: NO afirman un nuevo registro
    assert participio_describe_estado("Sí: por lo que quedó ")
    assert participio_describe_estado("Sí: lo blanco que mencionaste quedó ")
    assert participio_describe_estado("Sí: lo blanco que mencionaste quedó como ")
    assert participio_describe_estado("según lo que está ")
    assert participio_describe_estado("como ya quedó ")
    assert participio_describe_estado("en lo que figura ")
    assert participio_describe_estado("con el desayuno ")
    assert participio_describe_estado("ya tienes tu desayuno ")

    # Afirmaciones activas de registro en este turno: SÍ deben verificarse (retornar False aquí)
    assert not participio_describe_estado("Anoté tu desayuno: ")
    assert not participio_describe_estado("Tu cena quedó ")
    assert not participio_describe_estado("Quedó como tu cena de hoy: ")


def test_mensaje_sin_nada_que_registrar_identifica_consultas_puras():
    # Preguntas informativas donde el usuario indaga sobre ingredientes, visión o diario existente
    assert mensaje_sin_nada_que_registrar("Sabes lo que es lo blanco que está en el pan")
    assert mensaje_sin_nada_que_registrar("¿Qué es lo blanco del pan?")
    assert mensaje_sin_nada_que_registrar("¿Qué pusiste en mi desayuno?")
    assert mensaje_sin_nada_que_registrar("¿Sabes qué ingredientes tiene?")
    assert mensaje_sin_nada_que_registrar("¿Puedo comer esto?")
    assert mensaje_sin_nada_que_registrar("¿Puedo comer pizza hoy?")

    # Mensajes con verbos de consumo o solicitudes de registro: NO son consultas vacías
    assert not mensaje_sin_nada_que_registrar("Desayuné 2 huevos con pan")
    assert not mensaje_sin_nada_que_registrar("Me comí un bowl de pasta")
    assert not mensaje_sin_nada_que_registrar("Registra un filete de carne")
    assert not mensaje_sin_nada_que_registrar("Anota mi cena")
    assert not mensaje_sin_nada_que_registrar("¿Me anotas la cena?")


def test_diary_claim_sentences_does_not_flag_passive_cottage_cheese_answer():
    resp_1 = "Sí: por lo que quedó anotado, lo blanco en el pan es **queso cottage**."
    assert not _reply_claims_diary_write(resp_1)
    assert _diary_claim_sentences(resp_1) == []

    resp_2 = "Sí: lo blanco que mencionaste quedó anotado como **queso cottage** en tu desayuno."
    assert not _reply_claims_diary_write(resp_2)
    assert _diary_claim_sentences(resp_2) == []

    # Pero una afirmación genuina sin tool sí se sigue detectando
    resp_real_claim = "Anoté tu desayuno: 2 panes con queso y 4 huevos, aprox. 500 kcal."
    assert _reply_claims_diary_write(resp_real_claim)
    assert len(_diary_claim_sentences(resp_real_claim)) > 0


def test_rules_mandate_full_plate_component_confirmation():
    from prompts.chat_agent import _CHAT_VOICE_RULES, _PLATO_INSTRUCCION
    from respuestas_de_la_foto import instruccion

    assert "menciona brevemente los componentes principales del plato completo" in _CHAT_VOICE_RULES
    assert "NUNCA nombres únicamente el chip o ingrediente que el usuario contestó" in _CHAT_VOICE_RULES
    assert "componentes principales del plato completo" in _PLATO_INSTRUCCION

    dummy_vision = {"respuestas": "Mantequilla de maní", "items": [{"kind": "plato", "description": "pan y huevos"}]}
    instr = instruccion(dummy_vision)
    assert "menciona los componentes principales del plato completo" in instr
    assert "NUNCA nombres únicamente el chip o ingrediente que el usuario contestó" in instr
