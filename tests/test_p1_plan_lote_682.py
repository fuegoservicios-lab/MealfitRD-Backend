"""[P1-PLAN-LOTE-682 · 2026-09-28] Modo voz del coach: el prompt de voz es el del stream, no cinco líneas aparte.

El modo voz vuelve con la voz del propio dispositivo (frontend: `ModoVoz`, `useConversacionPorVoz`,
`vozDelCoach`). El backend ya elegía `CHAT_VOICE_MODE_PROMPT` con `is_call_mode`, pero ese prompt SUSTITUÍA
al del stream y perdía el contexto clínico, «no saludes» y los tres bloques compartidos (brevedad y uso de
herramientas, voz/longitud/riesgo, resolver lo que necesita). Estos tests anclan que la cabeza se DERIVA del
prompt del stream (no puede quedarse atrás otra vez) y que las reglas de voz van al final, con la última palabra.
"""
from pathlib import Path

from prompts.chat_agent import (
    CHAT_STREAM_INLINE_PROMPT,
    CHAT_VOICE_MODE_PROMPT,
    _CHAT_BREVITY_RULES,
    _CHAT_CALL_MODE_RULES,
    _CHAT_RESOLVE_RULES,
    _CHAT_VOICE_RULES,
)


def test_la_cabeza_del_prompt_de_voz_es_la_del_stream():
    cabeza = CHAT_STREAM_INLINE_PROMPT.split("REGLAS DE FORMATO VISUAL")[0].rstrip()
    assert CHAT_VOICE_MODE_PROMPT.startswith(cabeza)
    assert "CONTEXTO PROFESIONAL" in CHAT_VOICE_MODE_PROMPT
    assert "NUNCA saludes" in CHAT_VOICE_MODE_PROMPT


def test_lleva_los_tres_bloques_compartidos_y_no_el_formato_visual():
    for bloque in (_CHAT_BREVITY_RULES, _CHAT_VOICE_RULES, _CHAT_RESOLVE_RULES):
        assert bloque in CHAT_VOICE_MODE_PROMPT
    # Se oye, no se lee: nada que pida negritas ni viñetas.
    assert "REGLAS DE FORMATO VISUAL" not in CHAT_VOICE_MODE_PROMPT
    assert "Usa **negritas**" not in CHAT_VOICE_MODE_PROMPT


def test_las_reglas_de_voz_van_al_final_y_mandan():
    assert CHAT_VOICE_MODE_PROMPT.endswith(_CHAT_CALL_MODE_RULES)
    assert "MANDA SOBRE LOS TOPES DE LONGITUD Y EL FORMATO DE ARRIBA" in _CHAT_CALL_MODE_RULES
    for regla in ("V1. SIN FORMATO", "V2. MUY BREVE", "V3. SIN CIFRAS SALVO QUE LAS PIDA", "V4. LO QUE LLEGA VIENE DEL RECONOCIMIENTO",
                  "V5. REGISTRAR HABLANDO", "V6. UNA sola pregunta", "V7. TEMAS DE RIESGO"):
        assert regla in _CHAT_CALL_MODE_RULES, regla


def test_elevenlabs_ya_no_esta_en_el_backend():
    """La voz la pone el dispositivo: el proxy `POST /tts` a ElevenLabs (sin llamadores desde mayo) salió, y con él
    su limitador, su tope de caracteres y su knob. Si vuelve un TTS de pago, tiene que ser una decisión visible
    (y la política de privacidad tiene que volver a nombrar al proveedor)."""
    chat = Path(__file__).resolve().parents[1].joinpath("routers", "chat.py").read_text(encoding="utf-8", errors="replace")
    for resto in ('@router.post("/tts")', "api.elevenlabs.io", "ELEVENLABS_API_KEY", "_CHAT_TTS_LIMITER",
                  "_CHAT_TTS_MAX_TEXT_CHARS", "MEALFIT_TTS_HTTPX_TIMEOUT_S", '"elevenlabs_tts"'):
        assert resto not in chat, resto


def test_el_stream_elige_el_prompt_de_voz_con_is_call_mode():
    src = Path(__file__).resolve().parents[1].joinpath("agent.py").read_text(encoding="utf-8", errors="replace")
    assert "_base_inline = CHAT_VOICE_MODE_PROMPT if is_call_mode else CHAT_STREAM_INLINE_PROMPT" in src
    chat = Path(__file__).resolve().parents[1].joinpath("routers", "chat.py").read_text(encoding="utf-8", errors="replace")
    assert 'is_call_mode = data.get("is_call_mode", False)' in chat
