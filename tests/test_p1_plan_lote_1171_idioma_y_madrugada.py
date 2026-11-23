# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1171 · 2026-11-23] El día real de un usuario (21-nov): el coach contestó en inglés a un mensaje en
español, y a las 00:29 anotó su bistec como «almuerzo de ayer» sin que él nombrara ningún día."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from biological_date_resolver import corregir_days_ago_madrugada
from idioma_del_turno import idioma_del_mensaje, nota_de_idioma


def test_idioma_de_los_mensajes_reales():
    assert idioma_del_mensaje("agrega al desayuno 4 huevos, hervidos, 4 rodajas de pan con mantequilla de mané y queso cottage") == "es"
    assert idioma_del_mensaje("whats the name of the yogurt I sent u sometimes that has 25 grams of protein") == "en"
    assert idioma_del_mensaje("Solo con leche descremada / ligero · Bistec de res / filete") == "es"
    # sin idioma claro no se dice nada: manda la regla F
    for t in ("ok", "Grilled chicken breast", "Mangú", "", "Add it on yesterday"):
        assert idioma_del_mensaje(t) in (None, "en")
    assert idioma_del_mensaje("ok") is None


def test_la_nota_va_junto_al_ultimo_mensaje(monkeypatch):
    import agent
    msgs = [
        HumanMessage(content="whats the name of the yogurt I sent you"),
        AIMessage(content="It's Ratio Protein yogurt."),
        HumanMessage(content="agrega al desayuno 4 huevos con pan y queso cottage"),
    ]
    out = agent._llm_messages_from_state(msgs, "SISTEMA")
    ultimo = out[-1]
    assert isinstance(ultimo, HumanMessage)
    assert "español" in ultimo.content and "agrega al desayuno" in ultimo.content
    # el primer mensaje (inglés) no se toca
    assert out[1].content == "whats the name of the yogurt I sent you"
    monkeypatch.setenv("MEALFIT_COACH_IDIOMA_DEL_TURNO", "false")
    assert nota_de_idioma(msgs[-1]) is None


def test_madrugada_sin_dia_nombrado_queda_en_la_jornada_abierta():
    msgs = [HumanMessage(content="Solo con leche descremada / ligero · Bistec de res / filete")]
    assert corregir_days_ago_madrugada(msgs, 1, hora_local=0.5, slot_pedido="almuerzo") == 0
    # fuera de la madrugada no se toca
    assert corregir_days_ago_madrugada(msgs, 1, hora_local=18.0, slot_pedido="almuerzo") == 1


def test_madrugada_con_otro_dia_dicho_o_preguntado_se_respeta():
    dicho = [HumanMessage(content="esto fue hace 2 días")]
    assert corregir_days_ago_madrugada(dicho, 2, hora_local=1.0) == 2
    preguntado = [
        HumanMessage(content="me comí un sándwich de pollo"),
        AIMessage(content="¿Fue ayer o hoy?"),
        HumanMessage(content="sí"),
    ]
    assert corregir_days_ago_madrugada(preguntado, 1, hora_local=1.0) == 1


def test_sin_mensajes_humanos_no_rompe():
    assert agent_ok([SystemMessage(content="x")])


def agent_ok(msgs):
    import agent
    return isinstance(agent._llm_messages_from_state(msgs, "S"), list)


def test_el_escaner_usa_el_idioma_del_perfil_y_la_pantalla_solo_sin_eleccion(monkeypatch):
    """Perfil es-DO + pantalla en-US (la app aún no había aplicado el perfil) => «Grilled chicken breast» en su diario."""
    import asyncio
    from routers import diary
    monkeypatch.setattr(diary, "get_user_profile", lambda uid: {"locale": "es-DO"})
    assert asyncio.run(diary._locale_del_escaner("en-US", "u1")) == "es-DO"
    monkeypatch.setattr(diary, "get_user_profile", lambda uid: {"locale": None})
    assert asyncio.run(diary._locale_del_escaner("en-US", "u1")) == "en-US"
    assert asyncio.run(diary._locale_del_escaner(None, None)) == "es-DO"


def test_sumar_a_una_comida_registrada_la_deja_como_parte():
    """La pechuga de ayer + el yogur que pidió «Add it on yesterday»: la ficha («Cada una») los separa."""
    from partes_de_comida import nutrition_con_parte_sumada, partes_de_comida
    fila = {"meal_name": "Pechuga de pollo a la parrilla", "calories": 248, "protein": 46, "carbs": 0,
            "healthy_fats": 5, "ingredients": ["pechuga"], "nutrition": {"attachment_ids": ["a1"]}}
    n = nutrition_con_parte_sumada(fila, "Pechuga de pollo a la parrilla + Yogur Ratio Protein strawberry cheesecake",
                                   418, 71, 15, 7)
    p = partes_de_comida(n)
    assert [x["meal_name"] for x in p] == ["Pechuga de pollo a la parrilla", "Yogur Ratio Protein strawberry cheesecake"]
    assert p[1] == {"meal_name": "Yogur Ratio Protein strawberry cheesecake", "calories": 170, "protein": 25,
                    "carbs": 15, "healthy_fats": 2}
    assert n["attachment_ids"] == ["a1"]
    # una segunda suma conserva las partes previas
    fila2 = {**fila, "meal_name": "Pechuga de pollo a la parrilla + Yogur Ratio Protein strawberry cheesecake",
             "calories": 418, "protein": 71, "carbs": 15, "healthy_fats": 7, "nutrition": n}
    n2 = nutrition_con_parte_sumada(fila2, fila2["meal_name"] + " + Guineo", 523, 72, 42, 7)
    assert [x["meal_name"] for x in partes_de_comida(n2)][-1] == "Guineo" and len(partes_de_comida(n2)) == 3


def test_una_correccion_que_no_es_suma_no_inventa_partes():
    from partes_de_comida import nutrition_con_parte_sumada
    fila = {"meal_name": "Arroz con pollo", "calories": 600, "protein": 30, "carbs": 80, "healthy_fats": 10}
    assert nutrition_con_parte_sumada(fila, "Arroz con pollo", 500, None, None, None) is None      # baja
    assert nutrition_con_parte_sumada(fila, "Moro con pollo", 700, None, None, None) is None       # otro nombre
    assert nutrition_con_parte_sumada(fila, "Arroz con pollo + Aguacate", 650, 29, 85, 20) is None  # una baja
    assert nutrition_con_parte_sumada(fila, "Arroz con pollo + Aguacate", None, None, None, None) is None


def test_la_correccion_la_usa():
    import inspect, tools
    assert "nutrition_con_parte_sumada(_fila, meal_name" in inspect.getsource(tools)
