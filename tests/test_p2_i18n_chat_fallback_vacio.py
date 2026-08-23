"""[P2-I18N-CHAT-FALLBACK-VACIO-SIGUE-EN-ESPANOL · 2026-08-23] Lo que dice el coach cuando
el modelo no devuelve nada (content vacío y sin tool_calls: filtro del provider) era un
párrafo español FIJO en ``agent.py::call_model`` — y ese AIMessage se persiste en la
conversación, así que un usuario en francés lo leía en español y lo volvía a ver cada vez
que abría la sesión.

Cierre: ``prompts.chat_agent.empty_response_fallback(locale)`` (tabla por locale, español
como suelo — mismo patrón que ``push_nudge_title``) y ``call_model`` resuelve el locale del
perfil SÓLO en esa rama. Se mide la CONDUCTA del nodo real con un LLM falso que devuelve
vacío, no el texto del fichero.
"""
from __future__ import annotations

import pytest


# ── el SSOT por idioma ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fallback():
    from prompts.chat_agent import empty_response_fallback
    return empty_response_fallback


@pytest.mark.parametrize("locale,fragmento", [
    ("en-US", "rephrase"),
    ("pt-BR", "reformular"),
    ("fr-FR", "reformuler"),
    ("it-IT", "riformularla"),
])
def test_cada_idioma_tiene_su_fallback(fallback, locale, fragmento):
    texto = fallback(locale)
    assert fragmento in texto, f"{locale}: {texto!r}"
    assert "No pude procesar" not in texto, f"{locale} cae al español"


@pytest.mark.parametrize("locale", ["es-DO", None, "", "xx-YY", 42])
def test_lo_desconocido_cae_al_espanol_sin_lanzar(fallback, locale):
    texto = fallback(locale)
    assert texto.startswith("No pude procesar esa solicitud")


def test_los_cinco_textos_dicen_lo_mismo(fallback):
    """Cada uno trae el ejemplo de registro («X … Y») — es la mitad útil del mensaje."""
    for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        t = fallback(loc)
        assert " X " in t and " Y " in t, f"{loc} perdió el ejemplo de registro: {t!r}"


# ── el nodo real ───────────────────────────────────────────────────────────────────────────

class _LLMVacio:
    """Un `ChatDeepSeek` que siempre devuelve content vacío y sin tool_calls."""

    def __init__(self, *a, **k):
        pass

    def bind_tools(self, *_a, **_k):
        return self

    def invoke(self, _messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="")


class _BreakerAbierto:
    def can_proceed(self):
        return True

    def record_success(self):
        pass

    def record_failure(self, *_a, **_k):
        pass


def _correr_call_model(monkeypatch, *, perfil, user_id="11111111-1111-1111-1111-111111111111"):
    import agent
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(agent, "ChatDeepSeek", _LLMVacio)
    monkeypatch.setattr(agent, "_get_circuit_breaker", lambda *_a, **_k: _BreakerAbierto())
    vistos = []

    def _perfil(uid):
        vistos.append(uid)
        return perfil

    monkeypatch.setattr(agent, "get_user_profile", _perfil)
    # La métrica `chat_llm_empty_response` va a DB dentro de un try; que no toque la red.
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *_a, **_k: None, raising=False)

    state = {
        "messages": [HumanMessage(content="hola")],
        "sys_prompt": "",
        "user_id": user_id,
        "session_id": "sesion-de-prueba",
    }
    out = agent.call_model(state)
    msg = out["messages"][-1] if isinstance(out, dict) else out
    return msg, vistos


def test_el_nodo_responde_en_el_idioma_del_perfil(monkeypatch):
    msg, vistos = _correr_call_model(monkeypatch, perfil={"locale": "fr-FR"})
    assert "reformuler" in str(msg.content), f"el fallback no salió en francés: {msg.content!r}"
    assert "No pude procesar" not in str(msg.content)
    assert vistos, "la rama de fallback no leyó el perfil para saber el idioma"


def test_sin_perfil_o_guest_cae_al_espanol(monkeypatch):
    msg, _ = _correr_call_model(monkeypatch, perfil=None)
    assert str(msg.content).startswith("No pude procesar esa solicitud")

    msg, vistos = _correr_call_model(monkeypatch, perfil={"locale": "fr-FR"}, user_id="guest")
    assert str(msg.content).startswith("No pude procesar esa solicitud"), "un guest no tiene perfil: español"
    assert not vistos, "para guest no debe ir a buscar perfil"


def test_si_el_perfil_falla_no_rompe_el_chat(monkeypatch):
    """La lectura del perfil es best-effort: un error ahí no puede convertir «respuesta vacía»
    en «excepción en el nodo»."""
    import agent

    def _revienta(_uid):
        raise RuntimeError("db caída")

    monkeypatch.setattr(agent, "ChatDeepSeek", _LLMVacio)
    monkeypatch.setattr(agent, "_get_circuit_breaker", lambda *_a, **_k: _BreakerAbierto())
    monkeypatch.setattr(agent, "get_user_profile", _revienta)
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *_a, **_k: None, raising=False)
    from langchain_core.messages import HumanMessage
    out = agent.call_model({
        "messages": [HumanMessage(content="hola")], "sys_prompt": "",
        "user_id": "11111111-1111-1111-1111-111111111111", "session_id": "s",
    })
    msg = out["messages"][-1]
    assert str(msg.content).startswith("No pude procesar esa solicitud")
