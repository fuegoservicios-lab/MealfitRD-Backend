# -*- coding: utf-8 -*-
"""
Tests unitarios para la suite de correcciones y mejoras del Agente IA v2:
- Títulos dinámicos y extracción limpia de platos desde adjuntos
- Puntuación en _significant_food_tokens y marcos descriptivos de plato
- Clasificador de sentimiento (extra_body y fast-path de chips de fotos)
- Preservación de [SYSTEM_TITLE] en podado de mensajes y memoria
- Sincronización proactiva desde el diario (mark_nudge_logged_from_diary)
- Endpoint GET /api/diary/chat-photos
"""
import pytest
from unittest.mock import MagicMock, patch
import inspect

from constants import (
    _significant_food_tokens,
    meal_name_backed_by_description,
    derive_meal_name_from_description,
)
from agent import _trim_dangling_title_words, _rag_es_casual
from sentiment_classifier import classify_sentiment
import routers.diary as diary_router


def test_significant_food_tokens_strips_punctuation():
    desc = "arroz blanco, trozos de cerdo asado con cuerito crujiente y yuca hervida."
    tokens = _significant_food_tokens(desc)
    assert "blanco" in tokens
    assert "cerdo" in tokens
    assert "asado" in tokens
    assert "yuca" in tokens
    # Ningún token debe terminar con coma o punto
    for t in tokens:
        assert not t.endswith(",")
        assert not t.endswith(".")


def test_meal_name_backed_by_description_with_commas():
    meal_name = "Arroz blanco con cerdo asado y yuca"
    desc = "Una porción abundante de arroz blanco, trozos de cerdo asado con cuerito crocante y yuca hervida."
    # Antes del fix fallaba porque 'blanco,' no matcheaba 'blanco'
    assert meal_name_backed_by_description(meal_name, desc) is True


def test_derive_meal_name_from_description_strips_frame():
    desc = "Una porción abundante de arroz blanco con cerdo asado y yuca hervida. Estimación: 840 kcal"
    derived = derive_meal_name_from_description(desc, max_words=8)
    # Debe quitar "Una porción abundante de" y recortar antes de la estimación
    assert not derived.lower().startswith("una porción")
    assert not derived.lower().startswith("plato")
    assert "arroz blanco" in derived.lower()

    nested_desc = "Plato servido con una porción abundante de arroz blanco, trozos de cerdo asado con cuerito crocante."
    derived_nested = derive_meal_name_from_description(nested_desc, max_words=8)
    assert not derived_nested.lower().startswith("plato")
    assert not derived_nested.lower().startswith("una porción")
    assert derived_nested.lower().startswith("arroz blanco")


def test_trim_dangling_title_words():
    assert _trim_dangling_title_words("1.5 tazas · Con cuerito y") == "1.5 tazas · Con cuerito"
    assert _trim_dangling_title_words("Consulta sobre el") == "Consulta"
    assert _trim_dangling_title_words("Almuerzo con arroz de") == "Almuerzo con arroz"
    assert _trim_dangling_title_words("Almuerzo saludable") == "Almuerzo saludable"
    assert _trim_dangling_title_words("") == ""


def test_sentiment_fast_path_for_photo_chips():
    chip_msg = "1.5 tazas · Con cuerito y grasa"
    result = classify_sentiment(chip_msg)
    assert result["sentiment"] == "neutral"


def test_rag_casual_fast_path_for_photo_chips():
    chip_msg = "1.5 tazas · Con cuerito y grasa"
    # Respuestas de opciones de chips de fotos no deben disparar RAG innecesario
    assert _rag_es_casual(chip_msg) is True


def test_chat_photos_endpoint_registered():
    routes = [route.path for route in diary_router.router.routes]
    assert "/api/diary/chat-photos" in routes


def test_system_title_preserved_in_db_chat_delete_old_messages():
    import db_chat
    src = inspect.getsource(db_chat.delete_old_messages)
    assert "NOT LIKE '[SYSTEM_TITLE]" in src or "NOT LIKE '[SYSTEM_TITLE]%%'" in src


def test_log_consumed_meal_syncs_proactive_nudge():
    import db_facts
    src = inspect.getsource(db_facts.log_consumed_meal)
    assert "mark_nudge_logged_from_diary" in src
