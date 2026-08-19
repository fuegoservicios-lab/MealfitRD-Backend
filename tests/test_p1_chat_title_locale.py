"""[P1-CHAT-TITLE-LOCALE · 2026-08-19] El título dinámico del chat sale en el idioma del
usuario — el follow-up que T3 dejó documentado («si el dueño lo pide») y el dueño pidió:
con el chat ya en inglés (directiva nativa + recency), el título del sidebar era lo único
que seguía saliendo español para un usuario en-US.

Diseño: `generate_chat_title_background` lee el perfil UNA vez (el título se genera una
vez por sesión) y apendea la MISMA directiva SSOT (`build_language_directive`) al prompt
del título — prosa en el idioma del usuario, nombres de platos en español (un título
«Guiso de Habichuelas» debe seguir matcheando el plan). Guests/es-DO ⇒ directiva vacía ⇒
byte-idéntico.

Parser-based sobre agent.py (patrón del repo para wiring de prompts).
"""
import os
import re

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _fn_body(src: str, header_startswith: str) -> str:
    i = src.index(header_startswith)
    nxt = src.find("\ndef ", i + 1)
    return src[i:nxt if nxt != -1 else len(src)]


def test_titulo_lee_locale_y_apendea_la_directiva_de_titulo():
    # [round 2] La directiva del título es la ESPECÍFICA (build_title_language_directive),
    # no la conversacional: los ejemplos españoles del template vencían a la genérica —
    # «Estado del día» se generó a las 07:10 POST-restart 07:09 con la genérica apendeada.
    src = _read("agent.py")
    body = _fn_body(src, "def generate_chat_title_background(")
    assert "build_title_language_directive" in body, (
        "el generador de títulos debe apendear la directiva ESPECÍFICA del título — la "
        "conversacional no vence a los ejemplos españoles del template (round 2 de "
        "P1-CHAT-TITLE-LOCALE)"
    )
    assert re.search(r"get_user_profile", body), (
        "el locale del título sale del perfil (una lectura por sesión, costo nulo)"
    )
    # La directiva se CONCATENA al prompt del título (no reemplaza el prompt).
    assert re.search(
        r"TITLE_GENERATION_PROMPT\.format\([^)]*\)\s*\+\s*_title_lang_directive", body
    ), "la directiva debe apendearse al prompt formateado del título"


def test_directiva_de_titulo_es_nativa_con_ejemplos_propios():
    """La lección del round 2, un nivel más profundo que NATIVE: *los ejemplos son
    instrucciones*. Cada variante trae ejemplos de título EN el idioma destino y declara
    que los españoles del template son solo de FORMATO."""
    from prompts.chat_agent import build_title_language_directive
    casos = {
        "en-US": ("Write the title in English", "Morning check-in"),
        "pt-BR": ("Escreva o título em Português", "Primeiro contato"),
        "fr-FR": ("Rédige le titre en Français", "Premier contact"),
        "it-IT": ("Scrivi il titolo in Italiano", "Primo contatto"),
    }
    for locale, (imperativo, ejemplo) in casos.items():
        r = build_title_language_directive(locale)
        assert imperativo in r, f"{locale}: la directiva del título debe ser nativa"
        assert ejemplo in r, f"{locale}: sin ejemplos nativos, los españoles del template ganan"
        assert "FORMAT" in r.upper(), f"{locale}: debe declarar que los ejemplos del template son solo formato"
    assert build_title_language_directive("es-DO") == ""
    assert build_title_language_directive(None) == ""
    assert build_title_language_directive("xx-XX") == ""


def test_guest_y_fallo_caen_a_directiva_vacia():
    src = _read("agent.py")
    body = _fn_body(src, "def generate_chat_title_background(")
    assert 'user_id != "guest"' in body and "user_id != session_id" in body, (
        "guests/sesiones anónimas jamás leen perfil — directiva vacía, título en español "
        "(Addendum §2: guests ⇒ es-DO)"
    )
    assert re.search(r'except Exception:\s*\n\s*_title_lang_directive = ""', body), (
        "el wiring del título es best-effort: un fallo de perfil no puede tumbar la "
        "generación del título (fallback a conducta previa)"
    )
