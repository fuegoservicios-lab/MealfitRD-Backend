"""[P1-PLAN-LOTE-684 · 2026-09-28] El modo voz del coach contestaba en ~7 s. Medido en el turno del dueño
(«Cómo estás», 28-sep 18:07 UTC, journal del VPS + `llm_usage_events` + checkpoint de LangGraph):

  1,64 s  clasificar la respuesta a un nudge pendiente (LLM, solo analítica), en línea al guardar el mensaje
  1,75 s  router de RAG (LLM) para acabar en SKIP; 1,00 s de clasificador de sentimiento en paralelo
  1,72 s  la primera respuesta, TIRADA por P1-DIARY-CLAIM-VERIFY: «Va, dime qué desayunaste y lo dejo anotado»
          es una oferta, no un registro

Aquí: la oferta no dispara el guard (y las afirmaciones de verdad sí), la charla corta no paga el router, el modo
voz no corre ni el clasificador ni el LLM del router, y el nudge se clasifica en un hilo aparte.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

import agent
from coach_voz import es_charla_corta


# ── 1. P1-DIARY-CLAIM-VERIFY: una oferta condicionada no es un registro ──────────────────────────────────────────

_RESPUESTA_TIRADA = (
    "Bien, aquí atento. Tú sí que me tienes curioso: tres saludos y todavía no me cuentas qué te has comido hoy 😄"
    "\n\nVa, dime qué desayunaste y lo dejo anotado."
)


@pytest.mark.parametrize("texto", [
    _RESPUESTA_TIRADA,
    "Cuando me digas qué cenaste, queda registrado.",
    "Si me pasas la foto, te lo dejo registrado.",
    "Cuéntame qué almorzaste y lo dejo apuntado.",
    "En cuanto me confirmes la porción, la dejo anotada.",
])
def test_una_oferta_condicionada_no_es_un_registro(texto):
    assert agent._reply_claims_diary_write(texto) is False


@pytest.mark.parametrize("texto", [
    "Lo dejo anotado: 450 kcal y 30 g de proteína.",          # performativo, sin condición
    "Dime si quieres cambiarlo, ya quedó registrado.",       # pretérito: afirma
    "Cena registrada. Asumo 2 panes de molde.",               # el incidente original del guard
    "Anotado: 2 huevos revueltos con pan.",
    "Listo, lo registré como tu almuerzo.",
])
def test_las_afirmaciones_de_verdad_siguen_disparando(texto):
    assert agent._reply_claims_diary_write(texto) is True


# ── 2. La charla corta no paga el router de RAG ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("texto", [
    "Cómo estás", "¿Cómo estás?", "hola, ¿qué tal?", "Buenas, ¿cómo vas?", "todo bien coach", "Qué más",
    "Hola coach", "klk", "bien y tú", "buenos días, ¿cómo amaneciste?",
])
def test_es_charla_corta(texto):
    assert es_charla_corta(texto) is True


@pytest.mark.parametrize("texto", [
    "¿Cómo estás? Me comí dos huevos", "¿qué puedo cenar hoy?", "", "coach", "hola, ¿cuánta proteína llevo?",
    "cómo está mi progreso de la semana",
])
def test_no_es_charla_corta(texto):
    assert es_charla_corta(texto) is False


def test_el_router_salta_la_charla_corta_sin_llamar_al_llm(monkeypatch):
    def _no_llames(*_a, **_k):
        raise AssertionError("la charla corta no debe llegar al LLM del router")
    monkeypatch.setattr(agent, "ChatGLM", _no_llames)
    assert agent.rag_query_router("Cómo estás") == {"skip": True}
    assert agent.rag_query_router("ok gracias") == {"skip": True}


def test_rag_sin_router_busca_con_la_frase_tal_cual():
    assert agent._rag_sin_router("Cómo estás") == {"skip": True}
    assert agent._rag_sin_router("¿qué puedo cenar hoy?") == {"skip": False, "query": "¿qué puedo cenar hoy?"}


# ── 3. El modo voz no corre ni el clasificador ni el LLM del router ─────────────────────────────────────────────

class _PromptCapturado(Exception):
    pass


def _turno(monkeypatch, *, is_call_mode, prompt):
    from types import SimpleNamespace
    import db_core
    import db_inventory
    import db_plans
    import nevera_opcional as no
    import shopping_calculator
    from prompts.sentiment import PERSONALITY_PROFILES

    llamadas = {"sentimiento": 0, "router": 0}

    class _Grafo:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_k):
            llamadas["prompt"] = inputs["sys_prompt"]
            raise _PromptCapturado

        stream = invoke

    class _Builder:
        def compile(self, **_k):
            return _Grafo()

    def _sentimiento(_p):
        llamadas["sentimiento"] += 1
        return {**PERSONALITY_PROFILES["neutral"], "sentiment": "neutral"}

    def _router(_p):
        llamadas["router"] += 1
        return {"skip": True}

    monkeypatch.setattr(no, "nevera_activa", lambda uid: False)
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda uid: [])
    monkeypatch.setattr(db_plans, "get_latest_usable_meal_plan_with_id", lambda uid: None)
    monkeypatch.setattr(shopping_calculator, "aggregate_shopping_list", lambda items, **k: list(items))
    monkeypatch.setattr(agent, "build_memory_context", lambda *_a: {"recent_messages": [], "summary_context": ""})
    # [P1-PLAN-LOTE-717] La memoria a largo plazo se lee fail-closed (sin base ⇒ pausada ⇒ sin RAG): el turno de este
    # arnés es el de un usuario con la memoria ENCENDIDA, que es donde el router tiene algo que decidir.
    monkeypatch.setattr(agent, "_memoria_activa_para_chat", lambda _uid: True)
    monkeypatch.setattr(agent, "classify_sentiment", _sentimiento)
    monkeypatch.setattr(agent, "rag_query_router", _router)
    monkeypatch.setattr(agent, "get_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "get_multimodal_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "_emit_chat_stream_total_duration_best_effort", lambda *_a: None)
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)
    with pytest.raises(_PromptCapturado):
        list(agent.chat_with_agent_stream(session_id="sesion-voz", prompt=prompt, user_id="u-voz",
                                          form_data={}, is_call_mode=is_call_mode))
    return llamadas


@pytest.mark.parametrize("prompt", ["¿qué puedo cenar hoy?", "Cómo estás"])
def test_el_modo_voz_no_llama_ni_al_clasificador_ni_al_router(monkeypatch, prompt):
    voz = _turno(monkeypatch, is_call_mode=True, prompt=prompt)
    assert voz["sentimiento"] == 0 and voz["router"] == 0
    from prompts.chat_agent import CHAT_VOICE_MODE_PROMPT
    assert CHAT_VOICE_MODE_PROMPT[:200] in voz["prompt"]


def test_el_chat_escrito_sigue_con_clasificador_y_router(monkeypatch):
    texto = _turno(monkeypatch, is_call_mode=False, prompt="¿qué puedo cenar hoy?")
    assert texto["sentimiento"] == 1 and texto["router"] == 1


# ── 4. La respuesta a un nudge se clasifica en un hilo aparte ───────────────────────────────────────────────────

def test_el_nudge_se_clasifica_fuera_del_turno(monkeypatch):
    import db_chat

    hecho = threading.Event()
    visto = {}

    def _lento(u, c):
        time.sleep(0.4)                         # la clasificación real es una llamada al LLM (~1,6 s)
        visto.update(u=u, c=c, hilo=threading.current_thread().name)
        hecho.set()

    fake_pa = types.ModuleType("proactive_agent")
    fake_pa.handle_nudge_response = _lento
    monkeypatch.setitem(sys.modules, "proactive_agent", fake_pa)
    monkeypatch.delenv("MEALFIT_NUDGE_RESPONSE_ASYNC", raising=False)
    monkeypatch.setattr(db_chat, "_save_message_insert_with_retry", lambda *a: None)
    monkeypatch.setattr(db_chat, "connection_pool", object())

    t0 = time.monotonic()
    db_chat.save_message("s1", "user", "Cómo estás", "u1")
    assert time.monotonic() - t0 < 0.3, "guardar el mensaje no espera a la clasificación del nudge"
    assert hecho.wait(3), "la clasificación sí corre (una vez), en segundo plano"
    assert visto["u"] == "u1" and visto["c"] == "Cómo estás"
    assert visto["hilo"].startswith("nudge-respuesta")


def test_el_knob_devuelve_el_nudge_a_en_linea(monkeypatch):
    import db_chat

    llamadas = []
    fake_pa = types.ModuleType("proactive_agent")
    fake_pa.handle_nudge_response = lambda u, c: llamadas.append(threading.current_thread().name)
    monkeypatch.setitem(sys.modules, "proactive_agent", fake_pa)
    monkeypatch.setenv("MEALFIT_NUDGE_RESPONSE_ASYNC", "0")
    assert db_chat._procesar_respuesta_a_nudge("u1", "hola") is None
    assert llamadas == [threading.current_thread().name]
