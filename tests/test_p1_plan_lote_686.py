"""[P1-PLAN-LOTE-686 · 2026-09-28] El modo voz, conciso y rápido. La prueba del dueño (28-sep, 02:01 UTC): «dura mucho
para responder y sus respuestas son muy largas; debe apuntar veloz, responder conciso y, si necesita algo, preguntar;
no explica calorías ni macros a menos que el usuario lo pida».

Medido: la respuesta llegó con negritas, cuatro cifras de macros y el total del día (las reglas V vivían en medio de
~20.000 tokens, con bloques de cifras detrás y la regla 5 de brevedad exigiendo números), y el turno tardó 10,3 s,
7,8 de ellos en la primera llamada: DeepSeek razonaba ~1.000 tokens invisibles antes de anotar.
"""
from __future__ import annotations

import pytest

import agent
from prompts.chat_agent import CHAT_VOICE_MODE_PROMPT, CHAT_VOICE_MODE_RECORDATORIO, _CHAT_CALL_MODE_RULES


# ── 1. Las reglas: sin cifras salvo que las pida, anotar ya, una pregunta solo si hace falta ────────────────────

def test_las_reglas_de_voz_quitan_las_cifras_y_mandan_anotar_ya():
    assert "V3. SIN CIFRAS SALVO QUE LAS PIDA" in _CHAT_CALL_MODE_RULES
    assert "la 5 de brevedad" in _CHAT_CALL_MODE_RULES, "manda sobre «Brevedad NUNCA significa recortar números»"
    assert "cuánto suma" not in _CHAT_CALL_MODE_RULES, "la confirmación ya no dice cuánto suma"
    assert "DE INMEDIATO, sin escribir ni razonar nada antes" in _CHAT_CALL_MODE_RULES
    # la cantidad: ver test_p1_plan_lote_689 (se pregunta solo cuando varía mucho: pan, arroz)
    assert "V8. SIN CONSEJOS QUE NO PIDIÓ" in _CHAT_CALL_MODE_RULES
    assert "unas 25 palabras" in _CHAT_CALL_MODE_RULES
    assert CHAT_VOICE_MODE_PROMPT.endswith(_CHAT_CALL_MODE_RULES)


def test_el_recordatorio_repite_el_mandato_en_corto():
    r = CHAT_VOICE_MODE_RECORDATORIO
    assert r.startswith("\n\nRECORDATORIO FINAL — MODO VOZ")
    for trozo in ("SIN CIFRAS", "llama a la herramienta de registro de inmediato", "sin razonar por escrito",
                  "una pregunta corta", "Nada de consejos que no pidió"):
        assert trozo in r, trozo
    assert len(r) < 700, "un recordatorio, no otro bloque de reglas"


# ── 2. El recordatorio va al FINAL del prompt del stream (antes del refuerzo de idioma) ─────────────────────────

class _PromptCapturado(Exception):
    pass


def _prompt(monkeypatch, *, is_call_mode):
    from types import SimpleNamespace
    import db_core
    import db_inventory
    import db_plans
    import nevera_opcional as no
    import shopping_calculator

    capturado = {}

    class _Grafo:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_k):
            capturado.update(inputs)
            raise _PromptCapturado

        stream = invoke

    class _Builder:
        def compile(self, **_k):
            return _Grafo()

    monkeypatch.setattr(no, "nevera_activa", lambda uid: False)
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda uid: [])
    monkeypatch.setattr(db_plans, "get_latest_usable_meal_plan_with_id", lambda uid: None)
    monkeypatch.setattr(shopping_calculator, "aggregate_shopping_list", lambda items, **k: list(items))
    monkeypatch.setattr(agent, "build_memory_context", lambda *_a: {"recent_messages": [], "summary_context": ""})
    monkeypatch.setattr(agent, "classify_sentiment", lambda _p: {})
    monkeypatch.setattr(agent, "rag_query_router", lambda _p: {"skip": True})
    monkeypatch.setattr(agent, "get_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "get_multimodal_embedding", lambda _q: None)
    monkeypatch.setattr(agent, "_emit_chat_stream_total_duration_best_effort", lambda *_a: None)
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)
    with pytest.raises(_PromptCapturado):
        list(agent.chat_with_agent_stream(session_id="sesion-686", prompt="me comí dos huevos", user_id="u-686",
                                          form_data={}, is_call_mode=is_call_mode))
    return capturado


def test_en_voz_el_recordatorio_va_detras_de_todo_el_contexto(monkeypatch):
    entradas = _prompt(monkeypatch, is_call_mode=True)
    sp = entradas["sys_prompt"]
    i = sp.index(CHAT_VOICE_MODE_RECORDATORIO)
    assert i > sp.index("DIARIO DE HOY"), "detrás de los bloques con cifras"
    assert len(sp) - (i + len(CHAT_VOICE_MODE_RECORDATORIO)) < 1500, "solo el refuerzo de idioma va después"
    assert entradas["modo_voz"] is True


def test_el_chat_escrito_no_lleva_el_recordatorio(monkeypatch):
    entradas = _prompt(monkeypatch, is_call_mode=False)
    assert CHAT_VOICE_MODE_RECORDATORIO not in entradas["sys_prompt"]
    assert entradas["modo_voz"] is False


# ── 3. Sin razonamiento del proveedor en voz ────────────────────────────────────────────────────────────────────

def test_en_voz_se_apaga_el_razonamiento(monkeypatch):
    monkeypatch.delenv("MEALFIT_COACH_VOZ_SIN_RAZONAMIENTO", raising=False)
    assert agent._kwargs_de_razonamiento_modo_voz({"modo_voz": True}) == {"extra_body": {"thinking": {"type": "disabled"}}}
    assert agent._kwargs_de_razonamiento_modo_voz({"modo_voz": False}) == {}
    assert agent._kwargs_de_razonamiento_modo_voz({}) == {}
    monkeypatch.setenv("MEALFIT_COACH_VOZ_SIN_RAZONAMIENTO", "0")
    assert agent._kwargs_de_razonamiento_modo_voz({"modo_voz": True}) == {}


def test_call_model_pasa_los_kwargs_y_el_estado_declara_el_campo():
    import inspect
    src = inspect.getsource(agent.call_model)
    assert "**_kwargs_de_razonamiento_modo_voz(state)," in src
    assert "modo_voz" in agent.ChatState.__annotations__, "fuera del schema LangGraph descarta la clave en silencio"
    full = inspect.getsource(agent)
    assert '"modo_voz": False,' in full and '"modo_voz": bool(is_call_mode),' in full, "los DOS inputs la fijan"


def test_deepseek_apaga_de_verdad_y_glm_lo_traduce(monkeypatch):
    """El contrato del wrapper del que depende esto: DeepSeek respeta `disabled`; GLM (no se puede apagar) → `low`."""
    import llm_provider
    src = inspect_src(llm_provider)
    assert 'if isinstance(_think, dict) and _think.get("type") == "disabled":' in src
    assert '_extra["thinking"] = {"type": "disabled"}' in src
    assert 'if _think.get("type") == "disabled":\n                    _legacy_eff = "low"' in src


def inspect_src(mod):
    import inspect
    return inspect.getsource(mod)
