# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1002 · 2026-11-14] Completitud en turnos multi-foto ("no detectó el pan").

Reproduce y blinda el incidente de producción (2026-11-14 06:17 UTC):
- Cena ya registrada: «Puré de plátano maduro con pollo desmenuzado en salsa blanca» (600 kcal).
- Usuario envía 2 fotos (Foto 1: 1 pan de agua mediano, 180 kcal; Foto 2: 1 vaso de jugo de lechoza, 110 kcal)
  con el texto «Esto también fue parte de la cena\\nLechoza · Con azúcar».
- Sin este guard, el coach actualizó la cena a 680 kcal (+lechoza) omitiendo el pan de agua y además
  disparó `nudge_photo_to_log` pidiendo otro `log_consumed_meal`.
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

import multiphoto_completeness as mc
from prompts.chat_agent import build_vision_context


_UID = "11111111-1111-1111-1111-111111111111"

_ORIG_CENA = {
    "id": "f959db5f-0000-4444-8888-111111111111",
    "meal_name": "Puré de plátano maduro con pollo desmenuzado en salsa blanca",
    "calories": 600,
    "protein": 31,
    "carbs": 75,
    "healthy_fats": 18,
    "meal_type": "cena",
    "ingredients": ["plátano maduro", "pollo desmenuzado", "salsa blanca"],
    "nutrition": {"attachment_ids": ["att-cena-0000"]},
}

_PLATE_PHOTOS = [
    {
        "desc": "1 pan de agua mediano (parcialmente mordido).",
        "kcal": 180.0,
        "attachment_id": "fe9ad501-f630-4da2-85be-a04e18b55d34",
    },
    {
        "desc": "1 vaso (aprox. 1 taza) de jugo de lechoza fresco.",
        "kcal": 110.0,
        "attachment_id": "bd2e4c2c-ff78-4bd2-b3f2-4656f5dbd038",
    },
]


def test_build_vision_context_reminds_all_plate_photos_on_multi():
    vision = {
        "kind": "multi",
        "has_text": True,
        "respuestas": "Lechoza · Con azúcar",
        "items": [
            {
                "kind": "plato",
                "description": "1 pan de agua mediano (parcialmente mordido). (Estimación: Calorías: 180, Proteína: 6g, Carbohidratos: 36g, Grasas Saludables: 2g)",
            },
            {
                "kind": "plato",
                "description": "1 vaso de jugo de lechoza. DUDAS (pregúntale solo esto): ¿Con o sin azúcar? (opciones: Con azúcar (supuesto) · Sin azúcar) (Estimación: Calorías: 110, Proteína: 1g, Carbohidratos: 26g, Grasas Saludables: 0g)",
            },
        ],
    }
    ctx = build_vision_context(vision)
    assert "MULTI-FOTO OBLIGATORIO (2 fotos de comida:" in ctx
    assert "Foto 1" in ctx and "Foto 2" in ctx
    assert "TODAS LAS 2 FOTOS" in ctx


def test_tokens_alimento_handles_plurals_like_panes_and_tostones():
    toks = mc.tokens_alimento("Puré de plátano con 2 panes de agua y 3 tostones")
    assert "pan" in toks
    assert "toston" in toks
    assert "pure" in toks
    assert "platano" in toks


def test_execute_tools_blocks_fusion_omitting_bread_and_allows_retry_with_both():
    from agent import execute_tools

    incomplete_args = {
        "user_id": "u-test",
        "meal_id": _ORIG_CENA["id"],
        "meal_name": "Puré de plátano maduro con pollo desmenuzado en salsa blanca y jugo de lechoza",
        "calories": 680,
        "protein": 32,
        "carbs": 101,
        "healthy_fats": 18,
        "meal_type": "cena",
        "ingredients": ["plátano maduro", "pollo desmenuzado", "salsa blanca", "jugo de lechoza", "azúcar"],
    }
    human = HumanMessage(content="Esto también fue parte de la cena\nLechoza · Con azúcar")
    ai_incomplete = AIMessage(
        content="",
        tool_calls=[{"id": "tc-1", "name": "correct_consumed_meal", "args": incomplete_args}],
    )
    state = {
        "user_id": _UID,
        "messages": [human, ai_incomplete],
        "turn_plate_photos": _PLATE_PHOTOS,
        "turn_photo_to_log": True,
    }

    with patch("tools._leer_fila_diario", return_value=_ORIG_CENA), \
         patch("agent._buscar_comida_previa_del_slot", return_value=_ORIG_CENA), \
         patch("agent._comida_dentro_de_ttl_fusion", return_value=True), \
         patch("tools.db_update_consumed_meal", return_value=_ORIG_CENA["id"]) as mock_db_correct:
        res = execute_tools(state)

    assert not mock_db_correct.called, "No debe escribir en BD si omitió la Foto 1 (pan de agua)"
    tmsg = res["messages"][0].content
    assert mc.MARKER_BLOQUEO_MULTIFOTO in tmsg
    assert "pan de agua" in tmsg.lower()
    assert "180 kcal" in tmsg

    # Retry completo incluyendo tanto el pan de agua como el jugo de lechoza -> DEBE ejecutar
    complete_args = {
        "user_id": _UID,
        "meal_id": _ORIG_CENA["id"],
        "meal_name": "Puré de plátano maduro con pollo desmenuzado, pan de agua y jugo de lechoza",
        "calories": 890,
        "protein": 38,
        "carbs": 137,
        "healthy_fats": 20,
        "meal_type": "cena",
        "ingredients": ["plátano maduro", "pollo desmenuzado", "pan de agua", "jugo de lechoza", "azúcar"],
    }
    ai_complete = AIMessage(
        content="",
        tool_calls=[{"id": "tc-2", "name": "correct_consumed_meal", "args": complete_args}],
    )
    state_retry = {
        "user_id": _UID,
        "messages": [human, ai_incomplete, res["messages"][0], ai_complete],
        "turn_plate_photos": _PLATE_PHOTOS,
        "turn_photo_to_log": True,
    }

    with patch("tools._leer_fila_diario", return_value=_ORIG_CENA), \
         patch("agent._buscar_comida_previa_del_slot", return_value=_ORIG_CENA), \
         patch("agent._comida_dentro_de_ttl_fusion", return_value=True), \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("db_facts.get_consumed_meals_today", return_value=[_ORIG_CENA]), \
         patch("tools.db_update_consumed_meal", return_value=_ORIG_CENA["id"]) as mock_db_correct:
        res_ok = execute_tools(state_retry)

    assert mock_db_correct.called
    db_kwargs = mock_db_correct.call_args.kwargs
    att_ids = (db_kwargs.get("nutrition") or {}).get("attachment_ids") or []
    assert _PLATE_PHOTOS[0]["attachment_id"] in att_ids
    assert _PLATE_PHOTOS[1]["attachment_id"] in att_ids
    assert _ORIG_CENA["id"] in (res_ok.get("diary_corrected_meal_ids") or [])


def test_execute_tools_blocks_log_merged_into_cena_when_bread_omitted():
    from agent import execute_tools

    only_juice_log = {
        "user_id": _UID,
        "meal_name": "Jugo de lechoza con azúcar",
        "calories": 110,
        "protein": 1,
        "carbs": 26,
        "healthy_fats": 0,
        "meal_type": "cena",
        "ingredients": ["lechoza", "azúcar"],
    }
    human = HumanMessage(content="Esto también fue parte de la cena\nLechoza · Con azúcar")
    ai_call = AIMessage(
        content="",
        tool_calls=[{"id": "tc-log", "name": "log_consumed_meal", "args": only_juice_log}],
    )
    state = {
        "user_id": _UID,
        "messages": [human, ai_call],
        "turn_plate_photos": _PLATE_PHOTOS,
        "turn_photo_to_log": True,
    }
    with patch("tools._leer_fila_diario", return_value=_ORIG_CENA), \
         patch("agent._buscar_comida_previa_del_slot", return_value=_ORIG_CENA), \
         patch("agent._comida_dentro_de_ttl_fusion", return_value=True), \
         patch("tools.db_update_consumed_meal") as mock_db_correct, \
         patch("tools.db_log_consumed_meal") as mock_db_log:
        res = execute_tools(state)

    assert not mock_db_correct.called
    assert not mock_db_log.called
    assert mc.MARKER_BLOQUEO_MULTIFOTO in res["messages"][0].content
    assert "pan de agua" in res["messages"][0].content.lower()


def test_execute_tools_allows_when_user_explicitly_excludes_bread_or_emits_two_calls():
    from agent import execute_tools

    # Caso 1: usuario dice explícitamente "sin el pan"
    human_excl = HumanMessage(content="Esto también fue parte de la cena, sin el pan, solo el jugo")
    only_juice = {
        "user_id": _UID,
        "meal_id": _ORIG_CENA["id"],
        "meal_name": "Puré de plátano maduro con pollo desmenuzado y jugo de lechoza",
        "calories": 690,
        "protein": 32,
        "carbs": 100,
        "healthy_fats": 18,
        "meal_type": "cena",
        "ingredients": ["plátano maduro", "pollo", "jugo de lechoza"],
    }
    state_excl = {
        "user_id": _UID,
        "messages": [human_excl, AIMessage(content="", tool_calls=[{"id": "t1", "name": "correct_consumed_meal", "args": only_juice}])],
        "turn_plate_photos": _PLATE_PHOTOS,
        "turn_photo_to_log": True,
    }
    with patch("tools._leer_fila_diario", return_value=_ORIG_CENA), \
         patch("agent._buscar_comida_previa_del_slot", return_value=_ORIG_CENA), \
         patch("agent._comida_dentro_de_ttl_fusion", return_value=True), \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("db_facts.get_consumed_meals_today", return_value=[_ORIG_CENA]), \
         patch("tools.db_update_consumed_meal", return_value=_ORIG_CENA["id"]) as mock_db_correct:
        res_excl = execute_tools(state_excl)
    assert mock_db_correct.called
    assert mc.MARKER_BLOQUEO_MULTIFOTO not in res_excl["messages"][0].content

    # Caso 2: 2 sibling calls en el mismo AIMessage (pan + jugo)
    human_both = HumanMessage(content="Esto también fue parte de la cena")
    ai_sibling = AIMessage(
        content="",
        tool_calls=[
            {"id": "s1", "name": "log_consumed_meal", "args": {"user_id": "u-test", "meal_name": "1 pan de agua mediano", "calories": 180, "protein": 6, "carbs": 36, "healthy_fats": 2, "meal_type": "cena", "ingredients": ["pan de agua"]}},
            {"id": "s2", "name": "log_consumed_meal", "args": {"user_id": "u-test", "meal_name": "Jugo de lechoza con azúcar", "calories": 110, "protein": 1, "carbs": 26, "healthy_fats": 0, "meal_type": "cena", "ingredients": ["lechoza", "azúcar"]}},
        ],
    )
    calls_args = [tc["args"] for tc in ai_sibling.tool_calls]
    assert mc.fotos_no_cubiertas_en_llamadas(_PLATE_PHOTOS, calls_args) == []


def test_anti_loop_allows_second_attempt_after_one_multiphoto_block():
    human = HumanMessage(content="Esto también fue parte de la cena")
    prior_block = ToolMessage(content=f"{mc.MARKER_BLOQUEO_MULTIFOTO} msg", tool_call_id="prev")
    out = mc.detectar_bloqueo_multifoto_fusion(
        _PLATE_PHOTOS,
        {"meal_name": "Jugo de lechoza", "calories": 680},
        [{"meal_name": "Jugo de lechoza", "calories": 680}],
        [human, prior_block],
        orig_meal=_ORIG_CENA,
    )
    assert out is None


def test_route_tools_ends_cleanly_after_explicit_multiphoto_fusion_without_spurious_nudge():
    from agent import route_tools

    human = HumanMessage(content="Esto también fue parte de la cena\nLechoza · Con azúcar")
    ai_call = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc-2",
            "name": "correct_consumed_meal",
            "args": {
                "meal_id": _ORIG_CENA["id"],
                "meal_name": "Puré de plátano maduro con pollo, pan de agua y jugo de lechoza",
                "calories": 890,
                "ingredients": ["pan de agua", "jugo de lechoza"],
            },
        }],
    )
    tmsg = ToolMessage(content=f"[ID_REGISTRO_DIARIO: {_ORIG_CENA['id']}] Cena actualizada.", tool_call_id="tc-2")
    ai_final = AIMessage(content="La cena quedó actualizada con el pan de agua y la lechoza (~890 kcal).")
    state = {
        "user_id": "u-test",
        "messages": [human, ai_call, tmsg, ai_final],
        "turn_plate_photos": _PLATE_PHOTOS,
        "turn_photo_to_log": True,
        "photo_log_retried": False,
        "diary_claim_retried": False,
        "plate_photo_retried": False,
    }
    assert route_tools(state) == END


def test_nudge_plate_photo_multiphoto_preserves_summed_calories():
    from agent import nudge_plate_photo

    human = HumanMessage(content="Mi cena")
    ai_partial = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc-p",
            "name": "log_consumed_meal",
            "args": {"meal_name": "Jugo de lechoza con azúcar", "calories": 110, "meal_type": "cena"},
        }],
    )
    tmsg = ToolMessage(content="[ID_REGISTRO_DIARIO: abc] Anotado.", tool_call_id="tc-p")
    state = {
        "user_id": "u-test",
        "messages": [human, ai_partial, tmsg, AIMessage(content="Anoté tu jugo de lechoza.")],
        "turn_plate_photos": _PLATE_PHOTOS,
        "turn_photo_to_log": True,
    }
    out = nudge_plate_photo(state)
    note = out["messages"][0].content
    assert "290 kcal" in note
    assert "pan de agua" in note.lower()


def test_multiphoto_breakfast_208_vs_908_context_and_guard():
    from agent import execute_tools, _plate_photos_from_vision

    vision = {
        "kind": "multi",
        "has_text": True,
        "respuestas": "Con miel · 2 huevos · Manteca de cerdo",
        "por_foto": [
            {"index": 0, "respuestas": "Con miel", "ajuste": {"calories": 60, "carbs": 17, "protein": 0, "healthy_fats": 0}},
            {"index": 1, "respuestas": "2 huevos · Manteca de cerdo", "ajuste": {"calories": 0, "carbs": 0, "protein": 0, "healthy_fats": 0}},
        ],
        "items": [
            {
                "kind": "plato",
                "attachment_id": "ad329a31-0000-0000-0000-000000000001",
                "description": (
                    "Bowl de frutas picadas con cubos de lechosa, trozos de piña y uvas rojas. "
                    "DUDAS (pregúntale solo esto): ¿Le echaste algo encima a la fruta? "
                    "(opciones: Solo fruta (supuesto) · Con miel) "
                    "(Estimación: Calorías: 148, Proteína: 2g, Carbohidratos: 37g, Grasas Saludables: 0g)"
                ),
            },
            {
                "kind": "plato",
                "attachment_id": "7a965620-0000-0000-0000-000000000002",
                "description": (
                    "Plato de desayuno con mangú de plátano verde, dos lascas de queso blanco frito y dos huevos fritos. "
                    "DUDAS (pregúntale solo esto): ¿Cuántos huevos fritos son? (opciones: 2 huevos (supuesto) · 3 huevos) "
                    "¿Con qué grasa se preparó el mangú? (opciones: Mantequilla (supuesto) · Manteca de cerdo) "
                    "(Estimación: Calorías: 700, Proteína: 26g, Carbohidratos: 68g, Grasas Saludables: 35g)"
                ),
            },
        ],
    }
    ctx = build_vision_context(vision)
    assert "908 kcal" in ctx
    assert "28g proteína" in ctx
    assert "SUMA todas esas Estimaciones" in ctx

    plate_photos = _plate_photos_from_vision(vision)
    assert len(plate_photos) == 2
    assert plate_photos[0]["kcal"] == 208.0
    assert plate_photos[1]["kcal"] == 700.0

    human = HumanMessage(content="Me desayuné esto\nCon miel · 2 huevos · Manteca de cerdo")
    underestimated = {
        "user_id": _UID,
        "meal_name": "Bowl de frutas con miel y mangú con queso blanco frito y 2 huevos",
        "calories": 208,
        "protein": 7,
        "carbs": 22,
        "healthy_fats": 10,
        "meal_type": "desayuno",
        "ingredients": [
            "1 taza de lechosa picada",
            "1/2 taza de piña",
            "10 uvas rojas",
            "1 cucharada de miel",
            "1 taza de mangú de plátano verde",
            "2 lascas de queso blanco frito",
            "2 huevos fritos",
        ],
        "nutrition": {"attachment_ids": ["ad329a31-0000-0000-0000-000000000001", "7a965620-0000-0000-0000-000000000002"]},
    }
    ai_bad = AIMessage(content="", tool_calls=[{"id": "tc-bad", "name": "log_consumed_meal", "args": underestimated}])
    state = {
        "user_id": _UID,
        "messages": [human, ai_bad],
        "turn_plate_photos": plate_photos,
        "turn_photo_to_log": True,
    }
    with patch("tools.db_log_consumed_meal", return_value="new-meal-id") as mock_db_log:
        res = execute_tools(state)

    assert not mock_db_log.called, "No debe registrar 208 kcal cuando las 2 fotos combinadas suman 908 kcal"
    assert mc.MARKER_BLOQUEO_MULTIFOTO in res["messages"][0].content
    assert "908 kcal" in res["messages"][0].content

    corrected = dict(underestimated, calories=908, protein=28, carbs=105, healthy_fats=35)
    ai_ok = AIMessage(content="", tool_calls=[{"id": "tc-ok", "name": "log_consumed_meal", "args": corrected}])
    state_retry = {
        "user_id": _UID,
        "messages": [human, ai_bad, res["messages"][0], ai_ok],
        "turn_plate_photos": plate_photos,
        "turn_photo_to_log": True,
    }
    with patch("tools.db_log_consumed_meal", return_value="new-meal-id") as mock_db_log_ok, \
         patch("tools._buscar_cena_principal_hoy", return_value=None):
        res_ok = execute_tools(state_retry)
    assert mock_db_log_ok.called
    assert "[ID_REGISTRO_DIARIO:" in res_ok["messages"][0].content
    # Guard contra regresión BB-040/BB-053: `_sanitizar_ingredientes_contra_alucinacion_foto`
    # NO debe aplastar las 908 kcal combinadas hacia plate_photos[0].kcal (208).
    saved_kwargs = mock_db_log_ok.call_args.kwargs
    saved_cal = saved_kwargs.get("calories") if "calories" in saved_kwargs else mock_db_log_ok.call_args.args[2]
    assert saved_cal == 908


def test_multiphoto_sanitizer_uses_matched_photo_sum_not_first_photo():
    """Regresión viva (15-nov 22:52): cuando Vision estima Photo 1=208 y Photo 2=700,
    un registro combinado con 918 kcal NO debe aplastarse a 208 (plate_photos[0]).
    Y dos registros separados conservan la estimación de su foto respectiva."""
    from agent import _sanitizar_ingredientes_contra_alucinacion_foto

    plate_photos = [
        {
            "desc": "Bowl de frutas picadas con cubos de lechosa, piña y uvas rojas.",
            "kcal": 208.0,
            "attachment_id": "att-1",
        },
        {
            "desc": "Plato de mangú de plátano verde, queso blanco frito y 2 huevos fritos.",
            "kcal": 700.0,
            "attachment_id": "att-2",
        },
    ]
    human = HumanMessage(content="Me desayuné esto\nCon miel · 2 huevos · Manteca de cerdo")

    # 1) Registro combinado en 1 llamada (~918 kcal): debe preservar 918 kcal
    combined = {
        "meal_name": "Bowl de frutas con miel y mangú con queso blanco frito y 2 huevos",
        "calories": 918,
        "protein": 29,
        "carbs": 106,
        "healthy_fats": 38,
        "ingredients": ["lechosa", "piña", "uvas", "miel", "mangú de plátano verde", "queso blanco", "huevos", "manteca de cerdo"],
    }
    out_comb = _sanitizar_ingredientes_contra_alucinacion_foto(combined, plate_photos, [human])
    assert out_comb["calories"] == 918

    # 2) Registros separados por foto: Foto 2 (~700 kcal) NO debe aplastarse a Foto 1 (208 kcal)
    only_second = {
        "meal_name": "Mangú de plátano verde con queso blanco frito y 2 huevos",
        "calories": 720,
        "protein": 28,
        "carbs": 70,
        "healthy_fats": 36,
        "ingredients": ["mangú de plátano verde", "queso blanco frito", "2 huevos fritos"],
    }
    out_second = _sanitizar_ingredientes_contra_alucinacion_foto(only_second, plate_photos, [human])
    assert out_second["calories"] == 720

    # 3) Sobreestimación alucinada sin extras ni porciones (~2000 kcal combinando ambas): debe ajustar a la SUMA (908), no a 208
    overblown = {
        "meal_name": "Bowl de frutas con mangú de plátano verde, queso blanco frito y 2 huevos",
        "calories": 2000,
        "protein": 60,
        "carbs": 200,
        "healthy_fats": 80,
        "ingredients": ["lechosa", "piña", "mangú", "queso blanco", "huevos"],
    }
    out_over = _sanitizar_ingredientes_contra_alucinacion_foto(overblown, plate_photos, [HumanMessage(content="Me desayuné esto")])
    assert out_over["calories"] == 908
