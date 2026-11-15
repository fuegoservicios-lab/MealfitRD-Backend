"""[P1-PLAN-LOTE-1016 · A-05 / A-07 / A-17] Tope de gasto cerrado, terra ensayado y bucles de herramientas cortados.

Hechos del 14-nov que fijan estas pruebas:
  · batería con caché v2: «se me acabó el pollo» llamó 277 veces seguidas a `check_hydration_today` (422 s, US$ 0,116);
  · «acabó» no contenía la palabra clave «acabo» (acento): el coach no veía las herramientas de la Nevera;
  · ensayo real de gpt-6.1-terra (Responses API): el `content` llega en BLOQUES y el bloque `function_call` se colaba como texto.
"""
from __future__ import annotations

import os
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ronda(nombre="check_hydration_today", args=None, i=0):
    return [AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": nombre, "args": args or {}}]),
            ToolMessage(content="0 de 9 vasos", tool_call_id=f"c{i}")]


# ── bucles ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_tres_llamadas_identicas_seguidas_cortan_el_turno():
    from coach_tool_loop import motivo_de_corte

    msgs = [HumanMessage(content="se me acabó el pollo")]
    for i in range(2):
        msgs += _ronda(i=i)
    assert motivo_de_corte(msgs) is None
    msgs += _ronda(i=2)
    assert motivo_de_corte(msgs) == "repeticion"


def test_rondas_distintas_solo_se_cortan_al_llegar_al_tope(monkeypatch):
    from coach_tool_loop import motivo_de_corte

    monkeypatch.delenv("MEALFIT_CHAT_MAX_TOOL_ROUNDS", raising=False)
    msgs = [HumanMessage(content="registra mi día")]
    for i in range(5):
        msgs += _ronda("log_consumed_meal", {"meal_name": f"plato {i}"}, i)
    assert motivo_de_corte(msgs) is None
    msgs += _ronda("log_consumed_meal", {"meal_name": "plato 5"}, 5)
    assert motivo_de_corte(msgs) == "max_rondas"
    monkeypatch.setenv("MEALFIT_CHAT_MAX_TOOL_ROUNDS", "12")
    assert motivo_de_corte(msgs) is None


def test_las_rondas_se_cuentan_solo_desde_el_ultimo_mensaje_del_usuario():
    from coach_tool_loop import motivo_de_corte, rondas_de_este_turno

    msgs = [HumanMessage(content="turno viejo")]
    for i in range(5):
        msgs += _ronda(i=i)
    msgs += [AIMessage(content="ok"), HumanMessage(content="turno nuevo")] + _ronda(i=9)
    assert len(rondas_de_este_turno(msgs)) == 1 and motivo_de_corte(msgs) is None


def test_call_model_corta_sin_herramientas_y_lo_avisa():
    from pathlib import Path

    s = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    i = s.index("_corte = _motivo_corte(messages)")
    bloque = s[i:i + 1400]
    assert "llm_with_tools = chat_llm" in bloque                       # sin herramientas
    assert "_INSTR_CORTE" in bloque and 'fail_open("chat_tool_loop_cut"' in bloque
    assert "record_nudge(f\"tool_loop_cut:{_corte}\")" in bloque


def test_la_instruccion_de_corte_no_permite_mentir():
    from coach_tool_loop import INSTRUCCION_DE_CORTE

    assert "sin inventar que quedó hecho" in INSTRUCCION_DE_CORTE and "idioma del usuario" in INSTRUCCION_DE_CORTE


# ── selección de herramientas sin acentos ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("texto,herramienta", [
    ("se me acabó el pollo", "modify_pantry_inventory"),
    ("se me dañó la leche", "modify_pantry_inventory"),
    ("ponme el tema oscuro", "cambiar_ajuste_de_la_app"),
    ("tengo una intolerancia", "check_clinical_profile"),
])
def test_la_seleccion_de_herramientas_no_depende_de_los_acentos(texto, herramienta):
    import agent

    sel = {t.name for t in agent._select_tools_for_turn(agent.agent_tools, {"messages": [HumanMessage(content=texto)]})}
    assert herramienta in sel and "update_form_field" in sel


# ── texto de la Responses API ────────────────────────────────────────────────────────────────────────────────────
def test_un_bloque_function_call_no_se_cuela_como_texto():
    import agent

    m_tool = AIMessage(content=[{"arguments": '{"count_delta":2}', "call_id": "call_x", "name": "log_water_glass",
                                 "type": "function_call", "id": "fc_1"}],
                       tool_calls=[{"id": "call_x", "name": "log_water_glass", "args": {"count_delta": 2}}])
    assert agent._extract_ai_message_text(m_tool) == ""
    m_txt = AIMessage(content=[{"type": "text", "text": "Registré 2 vasos.", "annotations": [], "phase": "final_answer"},
                               {"type": "reasoning", "summary": []}])
    assert agent._extract_ai_message_text(m_txt) == "Registré 2 vasos."
    assert agent._extract_ai_message_text(AIMessage(content="hola")) == "hola"
    assert agent._extract_ai_message_text(AIMessage(content=["a", {"text": "b"}])) == "a\nb"   # formatos de siempre


def test_terra_va_por_canario(monkeypatch):
    import chat_hard_turn as cht

    monkeypatch.setenv("MEALFIT_CHAT_AGENT_MODEL", "gpt-6-luna")
    monkeypatch.delenv("MEALFIT_CHAT_CANARY_TERRA", raising=False)
    monkeypatch.delenv("MEALFIT_CHAT_CANARY_USER_IDS", raising=False)
    estado = {"user_id": "u-x", "form_data": {"allergies": "maní", "medicalConditions": "diabetes"},
              "messages": [HumanMessage(content="¿puedo comer esto con mi alergia y mi diabetes?")]}
    m, info = cht.evaluar_turno_dificil(estado, estado["messages"], "gpt-6-luna", [])
    assert m == "gpt-6-luna" and info["reason"] == "canary_off"


# ── tope diario ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_las_cuentas_de_administracion_no_se_bloquean(monkeypatch):
    import chat_hard_turn as cht

    monkeypatch.setenv("MEALFIT_CHAT_USER_DAILY_BUDGET_USD", "0.25")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", "u-dueno")
    caro = lambda *a, **k: {"spent_micros": 9_000_000}
    assert cht.check_user_daily_chat_budget("U-DUENO", execute_sql_query=caro)["block_turn"] is False
    assert cht.check_user_daily_chat_budget("u-otro", execute_sql_query=caro)["block_turn"] is True
    monkeypatch.setenv("MEALFIT_CHAT_BUDGET_EXEMPT_USER_IDS", "u-otro")
    assert cht.check_user_daily_chat_budget("u-otro", execute_sql_query=caro)["block_turn"] is False


@pytest.mark.parametrize("locale,palabra", [("es-DO", "24 horas"), ("en-US", "24 hours"), ("pt-BR", "24 horas"),
                                            ("fr-FR", "24 heures"), ("it-IT", "24 ore")])
def test_el_aviso_del_tope_habla_el_idioma_y_no_promete_manana(locale, palabra):
    import agent

    t = agent._budget_exceeded_graceful_reply({"locale": locale})
    assert palabra in t
    for prohibido in ("mañana", "tomorrow", "amanhã", "demain", "domani"):
        assert prohibido not in t.lower()
