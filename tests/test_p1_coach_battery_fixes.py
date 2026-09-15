"""[P1-PLAN-LOTE-53 · 2026-09-15] Lo que la batería de escritura del coach encontró (encargo del 15-sep).

63 mensajes reales de usuario dominicano contra el coach real, BD en solo lectura y tools de escritura
en dry-run (`scripts/coach_battery/`, rúbrica en `docs/coach_bateria_2026_09_15.md`). Antes: media
9,8/12, 26/63 casos ≥ 11, 2 fallos duros. Causas que se pueden anclar sin LLM:

1. Voz y longitud: el prompt pedía «viñetas SIEMPRE», no ponía tope ni cierre y el bloque de
   hidratación mandaba «recuérdale» en cada respuesta (agua como coletilla en ~la mitad).
2. Las personas de sentimiento: «drill sergeant» con metáforas de guerra («¡Alto ahí, soldado! 🪖»)
   y «modo profesor con tablas» (200-250 palabras a «¿por qué tanto arroz?»).
3. El nudge del diario saltaba con «Quedan anotados 2 de 8 vasos» y «Anotado: lácteos quedó
   registrado en tu perfil»: el usuario veía la respuesta DOS veces en el stream.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import agent as A  # noqa: E402
from prompts import chat_agent as P  # noqa: E402
from prompts.sentiment import PERSONALITY_PROFILES  # noqa: E402

_PROMPTS = {
    "CHAT_SYSTEM_PROMPT_BASE": P.CHAT_SYSTEM_PROMPT_BASE,
    "CHAT_STREAM_SYSTEM_PROMPT_BASE": P.CHAT_STREAM_SYSTEM_PROMPT_BASE,
    "CHAT_AGENT_INLINE_PROMPT": P.CHAT_AGENT_INLINE_PROMPT,
    "CHAT_STREAM_INLINE_PROMPT": P.CHAT_STREAM_INLINE_PROMPT,
}


@pytest.mark.parametrize("nombre", sorted(_PROMPTS))
def test_los_cuatro_prompts_llevan_las_reglas_de_voz(nombre):
    texto = _PROMPTS[nombre]
    for marca in ("VOZ, LONGITUD Y CIERRE", "LO IMPORTANTE PRIMERO", "Máximo 2 emojis",
                  "El agua NO es coletilla", "idioma en que el usuario te escribe ESTE mensaje",
                  "nunca inventes por qué el plan salió", "SÍNTOMAS", "NADA INTERNO", "SIN RELLENO", "TEMAS DE RIESGO",
                  "nunca digas \"superávit de 2100\"", "Nunca des dosis"):
        assert marca in texto, f"{nombre}: falta «{marca}»"
    assert "SIEMPRE para listar" not in texto, "las viñetas obligatorias inflaban cada respuesta"


def test_la_alergia_va_primero_aunque_solo_pida_el_menu():
    """«Después» v2: con las reglas de brevedad, «qué me toca hoy» de un alérgico a maní y mariscos
    listó el menú (con maní y camarones) SIN avisar (FD3). La prioridad va en el bloque clínico."""
    txt = P.build_clinical_guard_context({"allergies": ["Maní", "Mariscos"]})
    assert "AVÍSALO EN LA PRIMERA FRASE" in txt and "también cuando solo te pide el menú" in txt
    assert "nunca cuenta como «dato extra»" in txt
    assert "NUNCA es un dato extra" in P.CHAT_STREAM_INLINE_PROMPT


def test_una_bebida_con_calorias_es_consumo_no_agua():
    """«Después» v2: «me tomé 3 Presidente» se leyó como agua y no se registró."""
    txt = P.build_tools_instructions_stream("u")
    assert "'me tomé 3 Presidente'" in txt and "`log_water_glass` es SOLO para agua" in txt


def test_el_bloque_de_voz_es_una_sola_constante():
    src = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    assert src.count("+ _CHAT_VOICE_RULES") == 4


def test_el_agua_ya_no_es_coletilla():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("def _build_hydration_context(")
    cuerpo = src[i:i + 4000]
    assert "recuérdale amablemente" not in cuerpo
    assert cuerpo.count("coletilla") >= 2


def test_las_personas_no_son_caricatura_ni_tesis():
    todas = " ".join(p["instruction"] for p in PERSONALITY_PROFILES.values())
    for vetado in ("drill sergeant", "GUERRA", "MODO PROFESOR", "Usa tablas", "Yo creo en ti"):
        assert vetado not in todas, vetado
    # La regla dominicana de curiosidad sigue EXACTA (la reemplaza P1-COACH-PERSONA-CURIOSIDAD-DO).
    from prompts.sentiment import _CURIOSITY_DO_RULE
    assert _CURIOSITY_DO_RULE in PERSONALITY_PROFILES["curiosity"]["instruction"]


def test_el_disgusto_y_el_deshacer_estan_en_las_instrucciones_del_stream():
    txt = P.build_tools_instructions_stream("u")
    assert "'no me gusta el pescado' → `dislikes`" in txt
    assert "«Deshacer registro»" in txt


def test_el_tiempo_de_cocina_llega_resumido_y_con_la_respuesta_honesta():
    """K8: «¿por qué platos de 65 minutos?» recibía una causa inventada. El dato real va en claro."""
    plan = {"_fidelity_report": {"issues": [
        {"code": "prep_time_over_budget", "day": 1, "meal": "Almuerzo", "minutes": 30, "budget": 10},
        {"code": "prep_time_over_budget", "day": 3, "meal": "Cena", "minutes": 65, "budget": 10},
        {"code": "culture_share_above", "day": 1},
    ], "slice_hash": "abc"}}
    txt = A._prep_time_context_for_chat(plan)
    assert "10 min" in txt and "2 comida(s)" in txt and "entre 30 y 65 min" in txt
    assert "pocos platos" in txt and "Cambiar Plato" in txt
    assert A._prep_time_context_for_chat({}) == "" and A._prep_time_context_for_chat(None) == ""
    # El reporte crudo (hashes incluidos) ya no viaja en el JSON del plan.
    assert "_fidelity_report" not in A._prune_plan_for_chat(plan)
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert src.count("_prep_time_context_for_chat(plan_vigente)") == 2, "paridad stream / no-stream"


def test_la_meta_diaria_llega_con_su_distancia_al_mantenimiento(monkeypatch):
    """«superávit de ~2100 kcal» salió dos veces (B4 antes, F5 v3): la causa era el contexto."""
    import nutrition_calculator as nc
    monkeypatch.setattr(nc, "get_nutrition_targets", lambda fd: {"tdee": 1750, "target_calories": 2000})
    txt = A._daily_goal_context({}, {"calories": 2100})
    assert "META DIARIA: 2100 kcal" in txt and "unas 350 kcal por encima de su mantenimiento, ~1750 kcal" in txt
    assert "nunca la meta entera" in txt
    assert A._daily_goal_context({}, None).startswith("\n\n🎯 META DIARIA: 2000 kcal")
    monkeypatch.setattr(nc, "get_nutrition_targets", lambda fd: {})
    assert A._daily_goal_context({}, None) == ""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert src.count("_daily_goal_context(form_data, plan_vigente)") == 2, "paridad stream / no-stream"


def _turno(tool: str, texto: str) -> list:
    return [
        HumanMessage(content="x"),
        AIMessage(content="", tool_calls=[{"name": tool, "args": {}, "id": "t1"}]),
        AIMessage(content=texto),
    ]


@pytest.mark.parametrize("tool,texto", [
    ("log_water_glass", "¡Bien! Quedan anotados **2 de 8 vasos** de hoy 💧"),
    ("update_form_field", "Anotado: **lácteos** quedó registrado en tu perfil como alergia."),
    ("modify_pantry_inventory", "Listo, quedó registrado en tu Nevera: 3 lb de pollo."),
])
def test_otra_escritura_del_turno_no_dispara_el_nudge_del_diario(tool, texto):
    assert A.route_tools({"messages": _turno(tool, texto)}) != "nudge_diary_tool"


def test_una_comida_afirmada_sin_tool_de_diario_sigue_disparando():
    # Tras beber agua, afirmar que la CENA quedó registrada sin log_consumed_meal sigue siendo mentira.
    msgs = _turno("log_water_glass", "Listo: sumé tu vaso y tu cena quedó registrada con 520 kcal.")
    assert A.route_tools({"messages": msgs}) == "nudge_diary_tool"
    # Y sin ninguna tool, igual que siempre.
    solo = [HumanMessage(content="cené pan"), AIMessage(content="Cena registrada. Asumo 2 panes.")]
    assert A.route_tools({"messages": solo}) == "nudge_diary_tool"
