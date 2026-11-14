# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-999 · 2026-11-13] Pruebas de Fase 0 (desglose/medición/fórmula de coste) y Fase 1 (tiers Hydrogen).

Todas las pruebas corren 100 % offline ($0 gasto IA).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_SCRIPTS_DIR = _BACKEND_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fase 0.1 — prompt_desglose + telemetría
# ---------------------------------------------------------------------------


def test_prompt_desglose_atribucion_exacta_por_bloque(monkeypatch):
    import prompt_desglose as pd_mod

    monkeypatch.delenv("MEALFIT_CHAT_PROMPT_BREAKDOWN", raising=False)
    assert pd_mod.breakdown_activo() is True

    manual = "MANUAL BASE DEL COACH " * 10
    pais = "\nPAIS RD"
    tools_instr = "\nTOOLS INSTR"
    prefijo = manual + pais + tools_instr

    b_temporal = "\n\n🕒 CONTEXTO TEMPORAL ACTUAL: 2026-11-13 18:30"
    b_circadiano = "\n⚠️ RITMO CIRCADIANO: normal"
    b_proactivo = "\n🌟 REGLA DE CONTINUIDAD TEMPORAL PROACTIVA: saluda breve"
    b_rag = "\n--- MEMORIA VECTORIAL (RAG) ---\n• Prefiere huevo hervido"
    b_inv = "\n\n🛒 ESTADO DE LA DESPENSA Y COMPRAS: pollo, arroz"
    b_id = "\n\n--- 👤 PERFIL DEL USUARIO (NO inventar) ---\nNombre: Angel"
    b_clin = "\n\n🛑 PERFIL CLÍNICO DEL USUARIO (BLOQUEO MÉDICO OBLIGATORIO): Sin maní"
    b_meta = "\n\n🎯 META DIARIA: 2100 kcal"
    b_lang1 = "\n\n🌐 RESPONSE LANGUAGE: Spanish"
    b_plan = "\n\nCONTEXTO CRÍTICO: El usuario tiene plan activo."
    b_mem = "\n\n<contexto_evolutivo_historico>\nMeta bajar 2kg\n</contexto_evolutivo_historico>"
    b_diario = "\n\nDIARIO DE HOY: 1 cafe"
    b_falt = "\n\n🎯 LO QUE LE FALTA HOY: 1950 kcal"
    b_nev = "\n\n🧊 NEVERA: pollo, avena"
    b_hoy = "\n\n📅 HOY es Viernes"
    b_past = "\n\n📖 DÍAS QUE YA PASARON:\nDía 1\n\n🍽️ DIARIO REAL DE DÍAS ANTERIORES:\nDía 1 ok"
    b_lang2 = "\n\n🌐 TITLE LANGUAGE: Spanish"

    sys_prompt = (
        prefijo
        + b_temporal
        + b_circadiano
        + b_proactivo
        + b_rag
        + b_inv
        + b_id
        + b_clin
        + b_meta
        + b_lang1
        + b_plan
        + b_mem
        + b_diario
        + b_falt
        + b_nev
        + b_hoy
        + b_past
        + b_lang2
    )

    res = pd_mod.desglosar_prompt(
        sys_prompt,
        manual_chars=len(manual),
        pais_chars=len(pais),
        tools_instr_chars=len(tools_instr),
        prefijo_estatico_chars=len(prefijo),
        lean_ahorro_chars=120,
        stream=True,
        modo_voz=False,
    )
    assert res is not None
    assert res["total_sys_chars"] == len(sys_prompt)
    assert res["prefijo_estatico"] == len(prefijo)
    assert sum(res["bloques"].values()) == len(sys_prompt)
    assert res["bloques"]["manual_estatico"] == len(prefijo)
    assert res["bloques"]["temporal"] == len(b_temporal)
    assert res["bloques"]["clinico"] == len(b_clin)
    assert res["bloques"]["rag"] == len(b_rag)
    assert res["bloques"]["memoria_historica"] == len(b_mem)
    assert res["bloques"]["dias_pasados"] == len(b_past)
    assert res["bloques"]["idioma_inicial"] == len(b_lang1)
    assert res["bloques"]["refuerzo_idioma"] == len(b_lang2)
    assert res["manual_base"] == len(manual)
    assert res["pais_culinaria"] == len(pais)
    assert res["tools_instr"] == len(tools_instr)
    assert res["lean_ahorro"] == 120
    assert res["stream"] is True
    assert "modo_voz" not in res


def test_prompt_desglose_multimodal_no_infla_base64_y_apagado(monkeypatch):
    import prompt_desglose as pd_mod

    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_BREAKDOWN", "true")
    b64 = "A" * 80000
    msgs = [
        SimpleNamespace(type="system", content="SYS" * 100),
        HumanMessage(
            content=[
                {"type": "text", "text": "Almuerzo"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        ),
        AIMessage(
            content="Anotado.",
            tool_calls=[{"id": "tc1", "name": "log_consumed_meal", "args": {"calories": 500}, "type": "tool_call"}],
        ),
        ToolMessage(content="OK", tool_call_id="tc1"),
    ]
    res = pd_mod.resumen_historial(msgs, active_tools=["t1", "t2", "t3"])
    assert res is not None
    assert res["sys_chars_enviados"] == 300
    assert res["historial_msgs"] == 3
    assert res["historial_media"] == 1
    assert res["historial_tool_calls"] == 1
    assert res["tools_n"] == 3
    # Jamás debe contar los 80k bytes de la imagen en base64 dentro de `historial_chars`.
    assert res["historial_chars"] < 1000

    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_BREAKDOWN", "0")
    assert pd_mod.breakdown_activo() is False
    assert pd_mod.desglosar_prompt("hola") is None
    assert pd_mod.resumen_historial(msgs) is None
    # Aunque el desglose esté apagado, `session_id` sigue saliendo para la vista por sesión.
    meta = pd_mod.metadata_para_evento(
        state={"session_id": "sess-123", "prompt_breakdown": {"total_sys_chars": 100}},
        llm_messages=msgs,
    )
    assert meta == {"session_id": "sess-123"}


def test_llm_telemetry_emite_extra_metadata_sin_pisar_metricas_base(monkeypatch):
    import llm_telemetry as lt
    import prompt_desglose as pd_mod

    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_BREAKDOWN", "1")
    captured = {}

    def _fake_log(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("db_profiles.log_llm_usage_event", _fake_log)

    class _DummyLlm:
        model_name = "gemini-4-hydrogen"

    fake_res = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "input_token_details": {"cache_read": 800},
            "output_token_details": {"reasoning": 33},
        },
    )
    extra = pd_mod.metadata_para_evento(
        state={"session_id": "s-999", "prompt_breakdown": {"total_sys_chars": 60000}},
        llm_messages=[SimpleNamespace(type="system", content="S" * 60000), HumanMessage(content="Hola")],
        active_tools=["a", "b"],
    )
    # Simula intento de sobreescribir `duration_s` / `corr`: deben quedar protegidos.
    extra["duration_s"] = 999.0
    extra["corr"] = "falso"

    lt._emit_llm_usage_event_best_effort(
        llm=_DummyLlm(),
        result=fake_res,
        duration_s=1.25,
        node="chat_call_model",
        extra_metadata=extra,
    )
    meta = captured["metadata"]
    assert meta["duration_s"] == 1.25
    assert meta["reasoning_tokens"] == 33
    assert meta["session_id"] == "s-999"
    assert meta["prompt_breakdown"]["total_sys_chars"] == 60000
    assert meta["prompt_breakdown"]["historial_msgs"] == 1
    assert meta["prompt_breakdown"]["tools_n"] == 2
    assert captured["node"] == "chat_call_model"


# ---------------------------------------------------------------------------
# Fase 0.2 — scripts/medir_cache_coach.py
# ---------------------------------------------------------------------------


def test_medir_cache_coach_resumen_diario_turnos_y_gaps(monkeypatch):
    import medir_cache_coach as mc

    rows = [
        # Turno 1 (corr-A, 2 llamadas en sess-1)
        {
            "id": "1",
            "created_at": "2026-11-13T18:00:00+00:00",
            "user_id": "u1",
            "model": "gemini-4-hydrogen",
            "input_tokens": 34000,
            "cached_tokens": 0,
            "output_tokens": 200,
            "cost_usd_micros": 52500,
            "metadata": {
                "corr": "corr-A",
                "session_id": "sess-1",
                "duration_s": 2.9,
                "reasoning_tokens": 110,
                "prompt_breakdown": {
                    "total_sys_chars": 63300,
                    "prefijo_estatico": 59000,
                    "historial_msgs": 4,
                    "historial_chars": 1200,
                    "tools_n": 8,
                },
            },
        },
        {
            "id": "2",
            "created_at": "2026-11-13T18:00:03+00:00",
            "user_id": "u1",
            "model": "gemini-4-hydrogen",
            "input_tokens": 35500,
            "cached_tokens": 33000,
            "output_tokens": 150,
            "cost_usd_micros": 7350,
            "metadata": {"corr": "corr-A", "session_id": "sess-1", "duration_s": 2.1, "reasoning_tokens": 40},
        },
        # Turno 2 (corr-B en sess-1, 27 s tras la llamada anterior de sess-1)
        {
            "id": "3",
            "created_at": "2026-11-13T18:00:30+00:00",
            "user_id": "u1",
            "model": "gemini-4-hydrogen",
            "input_tokens": 35800,
            "cached_tokens": 32220,
            "output_tokens": 120,
            "cost_usd_micros": 8687,
            "metadata": {"corr": "corr-B", "session_id": "sess-1", "duration_s": 1.8},
        },
        # Turno 3 (corr-C en sess-1, 400 s tras corr-B -> >300s)
        {
            "id": "4",
            "created_at": "2026-11-13T18:07:10+00:00",
            "user_id": "u1",
            "model": "gemini-4-hydrogen",
            "input_tokens": 36000,
            "cached_tokens": 0,
            "output_tokens": 180,
            "cost_usd_micros": 55350,
            "metadata": {"corr": "corr-C", "session_id": "sess-1", "duration_s": 3.0},
        },
    ]

    diario = mc.resumir_diario_modelos(rows)
    assert len(diario) == 1
    assert diario[0]["llamadas"] == 4
    assert diario[0]["pct_cache"] == pytest.approx(46.1, abs=0.2)
    assert diario[0]["usd"] == pytest.approx(0.1239, abs=0.0002)

    turnos = mc.resumir_por_turno(rows)
    assert turnos["turnos_totales"] == 3
    por_m = turnos["por_modelo"][0]
    assert por_m["turnos"] == 3
    assert por_m["turnos_con_2_llamadas"] == 1
    corr_a = next(t for t in turnos["turnos"] if t["corr"] == "corr-A")
    assert corr_a["reasoning_tokens"] == 150
    assert corr_a["call1"]["pct_cache"] == 0.0
    assert corr_a["call2"]["pct_cache"] == pytest.approx(93.0, abs=0.2)
    assert "sys=63.3k" in corr_a["desglose"]
    assert "tools=8" in corr_a["desglose"]

    gaps = mc.resumir_gap_primera_llamada(rows)
    by_b = {g["bucket"]: g for g in gaps}
    # Sólo 3 primeras llamadas (corr-A 2ª llamada queda fuera de esta tabla):
    assert sum(g["llamadas"] for g in gaps) == 3
    assert by_b["sin_previa"]["llamadas"] == 1
    assert by_b["<=30s"]["llamadas"] == 1
    assert by_b["<=30s"]["pct_cache"] == pytest.approx(90.0, abs=0.2)
    assert by_b[">300s"]["llamadas"] == 1
    assert by_b[">300s"]["pct_cache"] == 0.0

    # Prioriza Neon y jamás cae a la antigua DATABASE_URL.
    monkeypatch.setenv("DATABASE_URL", "postgres://legacy-supabase/db")
    monkeypatch.setenv("NEON_DATABASE_URL_POOLED", "postgres://neon-pooled/db")
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    assert mc._resolve_neon_db_url() == "postgres://neon-pooled/db"


# ---------------------------------------------------------------------------
# Fase 0.3 — Auditoría de fórmula de coste en db_profiles.py
# ---------------------------------------------------------------------------


def test_coste_hydrogen_cobra_cached_a_0_075_usd_por_1m():
    from db_profiles import compute_llm_cost_micros
    import gemini_hydrogen as gh

    m = gh.GEMINI_HYDROGEN
    # 1M cached -> $0.075 = 75_000 micros; 1M uncached -> $1.50 = 1_500_000; 1M out -> $7.50 = 7_500_000
    assert compute_llm_cost_micros(m, 1_000_000, 0, 1_000_000) == 75_000
    assert compute_llm_cost_micros(m, 1_000_000, 0, 0) == 1_500_000
    assert compute_llm_cost_micros(m, 0, 1_000_000, 0) == 7_500_000
    # Fila real de producción 13-nov (2.ª llamada del turno corr='7dea29f0'):
    # 35,788 in, 33,155 cached, 150 out -> 2,633*1.5 + 33,155*0.075 + 150*7.5 = 7,561 micros
    assert compute_llm_cost_micros(m, 35_788, 150, 33_155) == 7_561


# ---------------------------------------------------------------------------
# Fase 1 — Enrutado por tier + historial mixto Luna <-> Hydrogen
# ---------------------------------------------------------------------------


def test_fase1_hydrogen_tiers_routing_y_knobs(monkeypatch):
    import agent
    import gemini_hydrogen as gh
    import knobs

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("MEALFIT_CHAT_AGENT_MODEL", raising=False)

    # Default ("todos"): todos los tiers y guest van a Hydrogen.
    monkeypatch.delenv("MEALFIT_CHAT_HYDROGEN_TIERS", raising=False)
    monkeypatch.setattr("llm_provider.get_user_tier", lambda uid: "gratis")
    assert gh.chat_hydrogen_tiers() == "todos"
    assert agent._chat_agent_model_name("u-free") == gh.GEMINI_HYDROGEN
    assert agent._chat_agent_model_name(None) == gh.GEMINI_HYDROGEN

    # Restringido a planes pagos + admin: `gratis`, `free`, guest y error de BD caen al router (`gpt-6-luna`).
    monkeypatch.setenv("MEALFIT_CHAT_HYDROGEN_TIERS", "basic,plus,ultra,admin")
    monkeypatch.setattr("llm_provider.get_user_tier", lambda uid: "gratis")
    monkeypatch.setattr(agent, "resolve_model_for_user", lambda uid: "gpt-6-luna")
    assert agent._chat_agent_model_name("u-free") == "gpt-6-luna"
    assert agent._chat_agent_model_name(None) == "gpt-6-luna"
    assert agent._chat_agent_model_name("guest") == "gpt-6-luna"

    def _boom(_uid):
        raise RuntimeError("db down")

    monkeypatch.setattr("llm_provider.get_user_tier", _boom)
    assert agent._chat_agent_model_name("u-err") == "gpt-6-luna"

    for tier_pago in ("basic", "plus", "ultra", "admin"):
        monkeypatch.setattr("llm_provider.get_user_tier", lambda uid, t=tier_pago: t)
        assert agent._chat_agent_model_name(f"u-{tier_pago}") == gh.GEMINI_HYDROGEN

    # Override explícito `MEALFIT_CHAT_AGENT_MODEL` siempre gana.
    monkeypatch.setenv("MEALFIT_CHAT_AGENT_MODEL", "gpt-6-luna")
    monkeypatch.setenv("MEALFIT_CHAT_HYDROGEN_TIERS", "todos")
    assert agent._chat_agent_model_name("u-vip") == "gpt-6-luna"

    # Sin GEMINI_API_KEY cae al router.
    monkeypatch.delenv("MEALFIT_CHAT_AGENT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert agent._chat_agent_model_name("u-vip") == "gpt-6-luna"

    # Knobs auto-registrados en `_KNOBS_REGISTRY`.
    import prompt_desglose as pd_mod

    pd_mod.breakdown_activo()
    reg = knobs._KNOBS_REGISTRY
    assert "MEALFIT_CHAT_HYDROGEN_TIERS" in reg
    assert reg["MEALFIT_CHAT_HYDROGEN_TIERS"]["default"] == "todos"
    assert "MEALFIT_CHAT_PROMPT_BREAKDOWN" in reg
    assert reg["MEALFIT_CHAT_PROMPT_BREAKDOWN"]["default"] is True


def test_fase1_conversacion_mixta_luna_hydrogen_luna():
    import agent
    import gemini_hydrogen as gh
    from llm_provider import ChatGLM

    # Turno 1: lo atendió Luna (sin firma de pensamiento de Gemini).
    msg_luna = AIMessage(
        content="",
        tool_calls=[{"id": "call_luna_1", "name": "log_consumed_meal", "args": {"calories": 320}, "type": "tool_call"}],
    )
    # Turno 2: lo atendió Hydrogen (capturó `thought_signature` desde la respuesta cruda).
    msg_gem = AIMessage(
        content="",
        tool_calls=[{"id": "call_gem_2", "name": "consultar_dia", "args": {}, "type": "tool_call"}],
    )
    cr_gem = ChatResult(generations=[ChatGeneration(message=msg_gem)])
    raw_gem = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_gem_2",
                            "type": "function",
                            "function": {"name": "consultar_dia", "arguments": "{}"},
                            "extra_content": {"google": {"thought_signature": "sig-real-xyz"}},
                        }
                    ]
                }
            }
        ]
    }
    gh.capturar_de_respuesta(raw_gem, cr_gem)
    assert gh.CLAVE_FIRMAS in msg_gem.additional_kwargs

    historial = [
        HumanMessage(content="Me comí 2 huevos"),
        msg_luna,
        ToolMessage(content="Registrado", tool_call_id="call_luna_1"),
        HumanMessage(content="Qué llevo hoy?"),
        msg_gem,
        ToolMessage(content="320 kcal", tool_call_id="call_gem_2"),
        HumanMessage(content="Y qué ceno?"),
    ]
    llm_msgs = agent._llm_messages_from_state(historial, "SYS PROMPT")

    # Al pasar de vuelta a Hydrogen: el call de Luna recibe `skip_thought_signature_validator`
    # y el call de Hydrogen conserva su firma real.
    payload_gem = ChatGLM(
        model=gh.GEMINI_HYDROGEN, api_key="k-gem", base_url="http://localhost"
    )._get_request_payload(llm_msgs)
    tcalls_gem = [tc for m in payload_gem["messages"] for tc in (m.get("tool_calls") or [])]
    by_id_gem = {tc["id"]: tc for tc in tcalls_gem}
    assert (
        by_id_gem["call_luna_1"]["extra_content"]["google"]["thought_signature"]
        == "skip_thought_signature_validator"
    )
    assert by_id_gem["call_gem_2"]["extra_content"]["google"]["thought_signature"] == "sig-real-xyz"

    # Al volver a `gpt-6-luna`: no se filtra `extra_content` ni `thought_signature` al payload OpenAI.
    payload_luna = ChatGLM(
        model="gpt-6-luna", api_key="k-openai", base_url="http://localhost"
    )._get_request_payload(llm_msgs)
    tcalls_luna = [tc for m in payload_luna["messages"] for tc in (m.get("tool_calls") or [])]
    assert len(tcalls_luna) == 2
    for tc in tcalls_luna:
        assert "extra_content" not in tc
    serialized = json.dumps(payload_luna)
    assert "thought_signature" not in serialized
    assert "gemini_firmas" not in serialized


def test_lote_999_marker_y_wiring_en_agent():
    app_src = (_BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX\s*=\s*"P1-PLAN-LOTE-(\d+)\b', app_src, re.MULTILINE)
    assert m is not None and int(m.group(1)) >= 999

    agent_src = (_BACKEND_DIR / "agent.py").read_text(encoding="utf-8")
    assert agent_src.count('"prompt_breakdown": _prompt_breakdown') == 2
    assert "extra_metadata=_call_extra_meta" in agent_src
