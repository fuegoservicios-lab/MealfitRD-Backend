# -*- coding: utf-8 -*-
"""[P1-ADMIN-CHAT-FILTER-SYSTEM-TITLE · 2026-10-20]
Pruebas para:
1. Exclusión de mensajes [SYSTEM_TITLE] en la vista de detalle de conversaciones del admin.
2. Blindaje de idioma español en avisos proactivos cuando locale es es-DO.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock
import pytest

for mod in ("fastapi", "fastapi.responses"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import admin_prueba_detalle as apd


def test_conversacion_excluye_system_title(monkeypatch):
    """El tag interno [SYSTEM_TITLE] NUNCA debe aparecer como mensaje del coach en la vista de admin."""
    uid = "11111111-1111-4111-8111-111111111111"
    sid = "22222222-2222-4222-8222-222222222222"

    filas_fake = [
        {
            "id": "m1", "role": "user", "content": "Hola coach",
            "created_at": "2026-10-19T09:00:00+00:00", "feedback": None, "attachments": []
        },
        {
            "id": "m2", "role": "model", "content": "[SYSTEM_TITLE] Desayuno saludable",
            "created_at": "2026-10-19T09:00:01+00:00", "feedback": None, "attachments": []
        },
        {
            "id": "m3", "role": "model", "content": "¡Hola! ¿En qué puedo ayudarte hoy?",
            "created_at": "2026-10-19T09:00:05+00:00", "feedback": None, "attachments": []
        },
    ]

    monkeypatch.setattr(apd, "execute_sql_query", lambda q, p, **k: filas_fake)

    res = apd.conversacion(uid, sid)
    assert res is not None
    mensajes = res["mensajes"]

    # Debe contener m1 y m3, pero m2 ([SYSTEM_TITLE]) DEBE haber sido filtrado
    textos = [m["texto"] for m in mensajes]
    assert "Hola coach" in textos
    assert "¡Hola! ¿En qué puedo ayudarte hoy?" in textos
    assert not any("[SYSTEM_TITLE]" in t for t in textos)
    assert len(mensajes) == 2


def test_proactive_prompt_contiene_regla_espanol():
    """El prompt base proactivo prohíbe explícitamente responder en inglés cuando el usuario está en español."""
    from prompts.proactive import PROACTIVE_PROMPT
    assert "PROHIBIDO escribir en inglés" in PROACTIVE_PROMPT
    assert "español" in PROACTIVE_PROMPT.lower()
