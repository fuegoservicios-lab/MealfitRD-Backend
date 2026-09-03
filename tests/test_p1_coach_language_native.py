"""[P1-COACH-LANGUAGE-NATIVE · 2026-08-18] La directiva de idioma del coach se escribe
EN EL IDIOMA DESTINO — round 2 del incidente en-US del día del flip.

Round 1 (P1-COACH-LANGUAGE-RECENCY) movió la directiva al final del prompt. El usuario
volvió a probar y el modelo llegó a DELIBERAR en inglés a mitad de la respuesta («I
should not have started with a greeting...») — prueba de que la directiva le LLEGABA —
y aun así escribió la prosa en español. Una instrucción escrita en español pidiendo
otro idioma es la señal más débil posible contra un prompt 100% español + mensaje del
usuario en español: la directiva nativa es a la vez instrucción Y demostración.

Cubre:
  1. Las 4 directivas son NATIVAS (imperativo en el idioma destino) — el detalle vive
     en los tests F3 de test_p1_country_system_f2.py (re-anclados); aquí el ancla
     estructural del P-fix.
  2. La frontera dura sobrevive en las 4: el ejemplo canónico «Guiso de Habichuelas
     Negras» + la mención de español/Spanish/espagnol/spagnolo/espanhol.
  3. es-DO/garbage siguen devolviendo "" (byte-identidad DO intacta).
"""
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from prompts import chat_agent as chat_agent_prompts
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    chat_agent_prompts = None
    _IMPORT_ERR = _e

requires_prompts = pytest.mark.skipif(
    chat_agent_prompts is None,
    reason=f"prompts.chat_agent no importable: {_IMPORT_ERR}",
)

_NATIVE_TOKENS = {
    "en-US": ("ENTIRE reply in English", "Spanish"),
    "pt-BR": ("TODA a sua resposta em Português", "espanhol"),
    "fr-FR": ("TOUTE ta réponse en Français", "espagnol"),
    "it-IT": ("TUTTA la tua risposta in Italiano", "spagnolo"),
}


@requires_prompts
@pytest.mark.parametrize("locale", sorted(_NATIVE_TOKENS))
def test_directiva_nativa_con_frontera_dura(locale):
    r = chat_agent_prompts.build_language_directive(locale)
    imperativo, spanish_word = _NATIVE_TOKENS[locale]
    assert imperativo in r, (
        f"{locale}: la directiva debe escribirse EN el idioma destino — la versión en "
        f"español fue desobedecida por el modelo el mismo día del flip (round 2)"
    )
    assert spanish_word in r and "Guiso de Habichuelas Negras" in r, (
        f"{locale}: la frontera dura (nombres de comida en español, con su ejemplo "
        f"canónico) debe declararse en el idioma destino"
    )


@requires_prompts
def test_es_do_sigue_vacia_byte_identidad():
    assert chat_agent_prompts.build_language_directive("es-DO") == ""
    assert chat_agent_prompts.build_language_directive(None) == ""
    assert chat_agent_prompts.build_language_directive("xx-XX") == ""


def test_marker_presente_en_el_builder():
    with open(os.path.join(BACKEND, "prompts", "chat_agent.py"), encoding="utf-8") as f:
        src = f.read()
    assert "P1-COACH-LANGUAGE-NATIVE" in src
