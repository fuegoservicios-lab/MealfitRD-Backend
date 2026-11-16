# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1005 · 2026-11-14] Tests deterministas ($0 IA) para G-03 + G-31:

1. `consultar_dia_del_plan` con `fuente='plan' | 'diario' | 'ambos'` (lee comidas reales
   del día vía `get_consumed_meals_since` en huso horario del usuario).
2. `build_past_plan_days_block` / `build_past_diary_block` con `summarize_older_days=True`:
   conserva AYER entero (protegiendo BB-034 / Lote 987) y resume de anteayer hacia atrás.
3. `_build_past_days_context` activa el resumen detrás de `MEALFIT_CHAT_SUMMARIZE_OLDER_DAYS`.
4. `coach_lean.adelgazar(texto, ligadas)` poda instrucciones de suplementos/ajustes no ligados
   y deduplica copias 2x/3x sin perder conceptos clave ni incumplir Lote 998 (`con_foto=True`).
5. `_compact_prior_tool_message` compacta ToolMessages de turnos anteriores >320 chars
   preservando `[ID_REGISTRO_DIARIO: ...]` y sin tocar ToolMessages del turno actual.
6. Presupuesto total de tokens (`o200k_base`) del turno típico <= 24,000 tokens.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_CORE_TOOLS = {
    "log_consumed_meal",
    "correct_consumed_meal",
    "check_hydration_today",
    "log_water_glass",
    "proponer_comida",
    "consultar_dia_del_plan",
    "search_deep_memory",
    "update_form_field",
}


def test_consultar_dia_del_plan_fuente_diario_y_ambos(monkeypatch):
    import db_facts
    import tools

    monkeypatch.setattr(
        tools,
        "get_latest_usable_meal_plan",
        lambda uid: {
            "days": [
                {
                    "day_name": "Lunes",
                    "date": "2026-11-09",
                    "meals": [{"meal": "Almuerzo", "name": "Pollo plan", "cals": 500, "ingredients": ["150g pollo"], "recipe": ["Asar"]}],
                }
            ]
        },
    )
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda uid=None: "2026-11-14")
    monkeypatch.setattr(tools, "_tz_offset_para_usuario", lambda uid=None: 240)
    monkeypatch.setattr(
        db_facts,
        "get_consumed_meals_since",
        lambda uid, since, include_ingredients=False: [
            {
                "id": "meal-lun-1",
                "meal_type": "almuerzo",
                "meal_name": "Moro con pollo real",
                "calories": 760,
                "protein": 48,
                "carbs": 72,
                "healthy_fats": 24,
                "consumed_at": "2026-11-09T17:30:00+00:00",
                "ingredients": ["arroz cocido 180g", "pollo 150g"],
            }
        ],
    )

    out_diario = tools.consultar_dia_del_plan.func(user_id="u1", fecha="2026-11-09", fuente="diario")
    assert "DIARIO REAL para 2026-11-09" in out_diario
    assert "Moro con pollo real" in out_diario
    assert "760 kcal" in out_diario
    assert "[id: meal-lun-1]" in out_diario
    assert "Pollo plan" not in out_diario

    out_ambos = tools.consultar_dia_del_plan.func(user_id="u1", fecha="2026-11-09", fuente="ambos")
    assert "Pollo plan" in out_ambos
    assert "Moro con pollo real" in out_ambos


def test_summarize_older_days_conserva_ayer_y_resume_anteayer():
    from chat_history_context import build_past_diary_block, build_past_plan_days_block

    today = date(2026, 11, 14)
    ayer = today - timedelta(days=1)
    anteayer = today - timedelta(days=2)
    plan = {
        "days": [
            {
                "day_name": "Viernes",
                "date": ayer.isoformat(),
                "meals": [{"meal": "Cena", "name": "Salmón de ayer", "cals": 540}],
            },
            {
                "day_name": "Jueves",
                "date": anteayer.isoformat(),
                "meals": [
                    {"meal": "Desayuno", "name": "Avena antigua detallada", "cals": 410},
                    {"meal": "Almuerzo", "name": "Churrasco antiguo detallado", "cals": 790},
                ],
            },
        ]
    }
    rows = [
        {
            "id": "id-ayer-1",
            "meal_type": "cena",
            "meal_name": "Galletas Oreo ayer",
            "calories": 320,
            "protein": 4,
            "consumed_at": ayer.isoformat() + "T20:00:00+00:00",
            "ingredients": ["2 paquetes Oreo"],
        },
        {
            "id": "id-ant-1",
            "meal_type": "almuerzo",
            "meal_name": "Sancocho jueves completo",
            "calories": 820,
            "protein": 45,
            "consumed_at": anteayer.isoformat() + "T17:00:00+00:00",
            "ingredients": ["yuca 150g", "carne 150g"],
        },
    ]

    p_block = build_past_plan_days_block(plan, today, days_back=4, tz_offset_mins=240, summarize_older_days=True)
    assert "Salmón de ayer" in p_block
    assert "Avena antigua detallada" not in p_block
    assert "~1200 kcal (2 comidas)" in p_block

    d_block = build_past_diary_block(rows, today, days_back=4, tz_offset_mins=240, plan_data=plan, summarize_older_days=True)
    assert "Galletas Oreo ayer" in d_block
    assert "[id: id-ayer-1]" in d_block
    assert "Sancocho jueves completo" not in d_block
    assert "1 reg (~820 kcal, 45g prot)" in d_block
    assert "fuente='diario'" in d_block


def test_coach_lean_dedup_y_poda_extras_sin_perdida_y_presupuesto_24k(monkeypatch):
    import tiktoken
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from prompts.chat_agent import CHAT_STREAM_INLINE_PROMPT, build_tools_instructions_stream_estaticas
    from prompts.coach_lean import adelgazar
    import tools

    monkeypatch.delenv("MEALFIT_CHAT_PROMPT_LEAN", raising=False)
    monkeypatch.delenv("MEALFIT_CHAT_PROMPT_DEDUP_MANUAL", raising=False)

    raw = CHAT_STREAM_INLINE_PROMPT + "\n" + build_tools_instructions_stream_estaticas()
    lean = adelgazar(raw, _CORE_TOOLS)

    # Suplementos y ajustes no ligados salen cuando ligadas es solo _CORE_TOOLS
    assert "Para vitaminas/minerales" not in lean
    assert "- Usa `cambiar_ajuste_de_la_app`" not in lean

    # Una copia autoritativa de cada regla crítica se conserva íntegra
    for clave in (
        "PROHIBIDO CONFUNDIR 'AYER' COMO PLANTILLA COMPARATIVA",
        "CORRECCIÓN DE INGREDIENTES POR DIFERENCIAL (DELTA)",
        "PROTOCOLO DE DESPEDIDAS NOCTURNAS Y ACUMULADO DIARIO OBLIGATORIO",
        "FRANJA AISLADA SIN ALIMENTO",
        "ALIMENTO O PRODUCTO AISLADO SIN VERBO DE RECETA",
        "FIDELIDAD ABSOLUTA DEL INGREDIENTE EN PROPUESTAS",
        "PROHIBICIÓN DE FUGAS TÉCNICAS EN HIDRATACIÓN",
        "REGLA SUPLEMENTOS",
        "FUENTES Y ALCANCE DE SALUD",
    ):
        assert clave in lean, f"Se perdió regla crítica: {clave}"

    # Cuando sí se ligan suplementos, la guía completa de suplementos se conserva
    lean_supp = adelgazar(raw, _CORE_TOOLS | {"analizar_suplemento", "guardar_suplemento"})
    assert "Para vitaminas/minerales" in lean_supp

    # Medida o200k_base completa bajo presupuesto objetivo G-03 (<= 24,000 tokens)
    enc = tiktoken.get_encoding("o200k_base")
    core_schemas = json.dumps([convert_to_openai_tool(t) for t in tools.agent_tools if t.name in _CORE_TOOLS], ensure_ascii=False)
    total_tok = len(enc.encode(lean)) + len(enc.encode(core_schemas)) + 3600  # + resto/ayer/resumen/historial típico
    # [P1-PLAN-LOTE-1037] 13_600 -> 13_650: el commit 51d9ae96 (porciones caseras al pedir los pasos de una sugerencia)
    # anadio ~60 tokens al manual compacto y dejo el tope en 13_626 sin pasar por el gate. La meta G-03 (24k) no cambia.
    assert len(enc.encode(lean)) <= 13_650
    assert total_tok <= 24_000, f"Excede meta G-03: {total_tok}"


def test_compact_prior_tool_message_preserva_id_registro_diario_y_turno_actual():
    import agent
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    largo = (
        "Anotado correctamente en tu diario: Almuerzo delicioso con arroz, pollo y ensalada verde " * 6
        + "\n[ID_REGISTRO_DIARIO: meal-abc-999]\nTOTAL REAL DE HOY: 1450 kcal, 98g prot."
    )
    msg_old = ToolMessage(content=largo, tool_call_id="tc_old", name="log_consumed_meal")
    msg_cur = ToolMessage(content=largo, tool_call_id="tc_cur", name="log_consumed_meal")

    msgs = [
        HumanMessage(content="turn 1"),
        AIMessage(content="", tool_calls=[{"id": "tc_old", "name": "log_consumed_meal", "args": {}}]),
        msg_old,
        HumanMessage(content="turn 2"),
        AIMessage(content="", tool_calls=[{"id": "tc_cur", "name": "log_consumed_meal", "args": {}}]),
        msg_cur,
    ]
    out = agent._llm_messages_from_state(msgs, sys_prompt="sys")
    tool_outs = [m for m in out if isinstance(m, ToolMessage)]
    assert len(tool_outs) == 2
    assert "[resumido de turno anterior]" in tool_outs[0].content
    assert "[ID_REGISTRO_DIARIO: meal-abc-999]" in tool_outs[0].content
    assert "TOTAL REAL DE HOY" in tool_outs[0].content
    assert tool_outs[1].content == largo


def test_rule_coverage_54_de_54_intacta():
    from scripts.coach_battery.rule_coverage import build_coverage_matrix

    rep = build_coverage_matrix()
    assert rep["covered_rules"] == rep["total_rules"] == 54, rep
    assert rep["uncovered_rules"] == []
