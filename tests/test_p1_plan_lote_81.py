# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-81 · 2026-09-17] El idioma del MENSAJE manda, con una directiva determinista por turno: en la batería final
el dueño (app en es-DO) escribió en inglés y recibió español (1 de 4 corridas). La directiva nativa del idioma detectado se
añade al prompt del turno; el español y lo irreconocible no añaden nada."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("texto,loc", [
    ("hi! what's for dinner today?", "en-US"),
    ("can you tell me how much protein I have left today?", "en-US"),
    ("oi! comi dois ovos com pão no café da manhã", "pt-BR"),
    ("salut, qu'est-ce que je mange aujourd'hui ?", "fr-FR"),
    ("ciao, cosa ho in frigo?", "it-IT"),
    ("me comí un mangú con 2 huevos", None),
    ("qué me toca hoy", None),
    ("hola", None),
    ("me tomé 3 Presidente", None),
    ("ya te dije que no me gusta el pescado!!!", None),
    ("", None), (None, None),
])
def test_detecta_el_idioma_del_mensaje(texto, loc):
    from prompts.chat_agent import detect_message_locale
    assert detect_message_locale(texto) == loc


def test_la_directiva_solo_cuando_el_idioma_del_mensaje_no_es_el_de_la_app():
    from prompts.chat_agent import build_message_language_directive, build_language_directive
    d = build_message_language_directive("can you tell me how much protein I have left today?", "es-DO")
    assert d and d == build_language_directive("en-US") and "English" in d
    assert build_message_language_directive("hi! what's for dinner today?", "en-US") == ""     # ya la manda el locale
    assert build_message_language_directive("qué me toca hoy", "es-DO") == ""
    assert build_message_language_directive(None, "es-DO") == ""


def test_los_cuatro_call_sites_y_el_marcador():
    a = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert a.count("system_prompt += build_message_language_directive(prompt, _coach_locale)") == 4
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 81
    assert "P1-PLAN-LOTE-81" in (_BACKEND / "docs" / "coach_bateria_2026_09_15.md").read_text(encoding="utf-8")
