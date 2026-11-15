"""Regresión: [P1-COACH-RUTINA-GUSTOS] Aprendizaje continuo de rutinas y gustos reales por franja."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import chat_prompt_opt as cpo
import habitos_de_la_foto as hf
import prompt_desglose as pd
import rutina_del_coach as rc


_HISTORIAL_REAL = [
    {
        "id": "m1",
        "meal_type": "Desayuno",
        "meal_name": "2 sándwiches de pan integral con queso cottage, mantequilla de maní y 4 huevos hervidos",
        "calories": 820,
        "protein": 52,
        "ingredients": ["pan integral", "queso cottage", "mantequilla de maní", "huevos"],
        "consumed_at": "2026-11-12T13:00:00Z",
    },
    {
        "id": "m2",
        "meal_type": "Desayuno",
        "meal_name": "Sándwiches con queso cottage, mantequilla de maní y huevos hervidos",
        "calories": 780,
        "protein": 48,
        "ingredients": ["pan", "queso cottage", "mantequilla de maní", "huevos hervidos"],
        "consumed_at": "2026-11-14T13:10:00Z",
    },
    {
        "id": "m3",
        "meal_type": "Desayuno",
        "meal_name": "Bowl de lechoza, sandía y piña",
        "calories": 86,
        "protein": 1,
        "ingredients": ["lechoza", "sandía", "piña"],
        "consumed_at": "2026-11-15T12:00:00Z",
    },
    {
        "id": "m4",
        "meal_type": "Almuerzo",
        "meal_name": "Arroz blanco con cerdo guisado y vegetales",
        "calories": 630,
        "protein": 33,
        "ingredients": ["arroz blanco", "cerdo guisado", "vegetales"],
        "consumed_at": "2026-11-15T18:00:00Z",
    },
    {
        "id": "m5",
        "meal_type": "Cena",
        "meal_name": "4 sándwiches de jamón y queso",
        "calories": 950,
        "protein": 50,
        "nutrition": {
            "base_meal": {"meal_name": "4 sándwiches de jamón y queso", "calories": 950, "protein": 50},
            "addons": [{"meal_name": "1 jugo de mango", "calories": 290, "protein": 1}],
        },
        "consumed_at": "2026-11-15T23:50:00Z",
    },
    {
        "id": "m6",
        "meal_type": "complemento",
        "meal_name": "Té verde sin azúcar",
        "calories": 2,
        "protein": 0,
        "consumed_at": "2026-11-15T14:00:00Z",
    },
]


def test_extraer_rutina_agrupa_habituales_y_excluye_complementos():
    rutina = rc.extraer_rutina_desde_registros(_HISTORIAL_REAL, facts=[{"fact": "Le gusta desayunar sándwiches con queso cottage"}])
    desayunos = rutina["por_franja"]["desayuno"]
    assert desayunos, "debe extraer desayunos reales"
    assert desayunos[0]["count"] == 2
    assert desayunos[0]["habitual"] is True
    assert "cottage" in desayunos[0]["nombre"].lower()

    # El complemento <= 50 kcal nunca debe entrar en merienda/desayuno
    todas_las_franjas = [p["nombre"].lower() for lista in rutina["por_franja"].values() for p in lista]
    assert all("té verde" not in n for n in todas_las_franjas)

    # Los sándwiches con base_meal limpian el addon circunstancial del nombre de la rutina
    cenas = rutina["por_franja"]["cena"]
    assert cenas and "jugo" not in cenas[0]["nombre"].lower()
    assert len(rutina["habitos"]["rellenos_pan"]) >= 1


def test_para_chat_inyecta_reglas_sin_balanza_ni_resta_manana():
    rc.invalidate_rutina_cache("u-rutina")
    with patch("db_facts.get_consumed_meals_since", return_value=_HISTORIAL_REAL), \
         patch("db_facts.get_all_user_facts", return_value=[]):
        bloque = rc.para_chat("u-rutina", consumed_today=[])
    assert "🔄 RUTINA Y GUSTOS APRENDIDOS DEL USUARIO" in bloque
    assert "PRIORIZA SU RUTINA REAL" in bloque
    assert "PORCIONES CASERAS, NO BALANZA" in bloque
    assert "PROHIBIDO RESTAR DEL PRESUPUESTO DE MAÑANA" in bloque
    assert "2x habitual" in bloque


def test_invalidar_habitos_invalida_rutina_tambien():
    with patch("db_facts.get_consumed_meals_since", return_value=_HISTORIAL_REAL) as mock_m, \
         patch("db_facts.get_all_user_facts", return_value=[]):
        rc.invalidate_rutina_cache("u-cache")
        rc.para_chat("u-cache")
        rc.para_chat("u-cache")
        assert mock_m.call_count == 1
        hf.invalidate_habitos_cache("u-cache")
        rc.para_chat("u-cache")
        assert mock_m.call_count == 2


def test_cache_order_y_prompt_desglose_reconocen_rutina_gustos(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_CACHE_ORDER", "1")
    prefijo = "PREFIJO ESTATICO\n" + ("X" * 80)
    cut = len(prefijo)
    sp = (
        prefijo
        + "\n<!--MEALFIT_VOLATILE_SYS_SPLIT-->\nEl user_id actual es: u1"
        + "\n\n🕒 CONTEXTO TEMPORAL ACTUAL: 01:30"
        + "\n\n🛑 PERFIL CLÍNICO DEL USUARIO (PROTEGER CONTRA ALERGIAS): ok"
        + "\n\nDIARIO DE HOY: 2050 kcal"
        + "\n\n🧊 NEVERA: huevos"
        + "\n\n🔄 RUTINA Y GUSTOS APRENDIDOS DEL USUARIO (aprendizaje continuo de su diario real):\n- Desayuno: pan"
        + "\n\n🍽️ DIARIO REAL DE DÍAS ANTERIORES:\n- Ayer: 2000 kcal"
    )
    rebuilt = cpo.reordenar_bloques_cache(sp, prefijo_estatico_chars=cut)
    assert len(rebuilt) == len(sp)
    assert any(name == "rutina_gustos" for name, _ in pd._VOLATILE_BLOCK_SPECS)


def test_plan_context_en_pausa_prohibe_plan_para_manana_sin_decir_activo(monkeypatch):
    monkeypatch.setattr(agent, "_plan_mode_for_chat", lambda _uid: "tracking")
    bloque = agent._plan_context_for_chat("u1", {"days": [{"day": 1, "meals": []}]})
    assert "mañana" in bloque.lower()
    assert "pausa" in bloque.lower()
    assert "activo" not in bloque.lower()


def test_formatear_rutina_para_propuesta_tres_casos():
    solo_fruta = [
        {
            "id": "d-fruta",
            "meal_type": "Desayuno",
            "meal_name": "Bowl de lechoza, sandía y piña",
            "calories": 86,
            "protein": 1,
            "consumed_at": "2026-11-15T12:00:00Z",
        },
        {
            "id": "a-cerdo",
            "meal_type": "Almuerzo",
            "meal_name": "Arroz blanco con cerdo guisado y vegetales",
            "calories": 630,
            "protein": 33,
            "consumed_at": "2026-11-15T18:00:00Z",
        },
        {
            "id": "c-sand",
            "meal_type": "Cena",
            "meal_name": "4 sándwiches de jamón y queso",
            "calories": 950,
            "protein": 50,
            "consumed_at": "2026-11-15T23:50:00Z",
        },
    ]
    rc.invalidate_rutina_cache("u-prop-3casos")
    with patch("db_facts.get_consumed_meals_since", return_value=solo_fruta), \
         patch("db_facts.get_all_user_facts", return_value=[]):
        des_txt = rc.formatear_rutina_para_propuesta(
            "u-prop-3casos", "desayuno", {"kcal": 410, "protein_g": 27}
        )
        cena_txt = rc.formatear_rutina_para_propuesta(
            "u-prop-3casos", "cena", {"kcal": 615, "protein_g": 40}
        )
        alm_txt = rc.formatear_rutina_para_propuesta(
            "u-prop-3casos", "almuerzo", {"kcal": 717, "protein_g": 47}
        )

    assert "Bowl de lechoza, sandía y piña" in des_txt
    assert "poca proteína" in des_txt and "medida casera" in des_txt
    assert "nevera" not in des_txt.lower()

    assert "4 sándwiches de jamón y queso" in cena_txt
    assert "alto en calorías" in cena_txt and "reduciendo la porción en medida casera" in cena_txt
    assert "nevera" not in cena_txt.lower()

    assert "Arroz blanco con cerdo guisado y vegetales" in alm_txt
    assert "encaja bien" in alm_txt
    assert "nevera" not in alm_txt.lower()


def test_proponer_comida_madrugada_con_dia_biologico_planifica_manana_sin_restar(monkeypatch):
    import coach_day_context as cdc
    import db_facts
    import plan_mode
    import tools

    diario_ayer = [
        {"id": "h1", "meal_type": "Desayuno", "meal_name": "Bowl de lechoza, sandía y piña",
         "calories": 86, "protein": 1, "carbs": 21, "healthy_fats": 0},
        {"id": "h2", "meal_type": "Almuerzo", "meal_name": "Arroz blanco con cerdo guisado y vegetales",
         "calories": 630, "protein": 33, "carbs": 68, "healthy_fats": 24},
        {"id": "h3", "meal_type": "Cena", "meal_name": "4 sándwiches de jamón y queso",
         "calories": 1240, "protein": 61, "carbs": 140, "healthy_fats": 45},
    ]

    def _mock_gcmt(uid, date_str=None, tz_offset_mins=None):
        return list(diario_ayer) if date_str == "2026-11-14" else []

    rc.invalidate_rutina_cache("u-madrugada-1")
    monkeypatch.setattr(tools, "get_user_profile", lambda _uid: {
        "health_profile": {"weight": 150, "height": 175, "age": 25, "gender": "male", "mainGoal": "maintain"}
    })
    monkeypatch.setattr(tools, "_hora_local_float", lambda _uid: 1.47)
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda _uid: 240)
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda _uid=None: "2026-11-15")
    monkeypatch.setattr(tools, "_diary_date_str_for_user", lambda _uid=None: "2026-11-14")
    monkeypatch.setattr(tools, "nevera_activa", lambda _uid: False)
    monkeypatch.setattr(plan_mode, "get_plan_mode", lambda _uid: {"plan_mode": "tracking"})
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", _mock_gcmt)
    monkeypatch.setattr(db_facts, "get_consumed_meals_since", lambda *a, **k: list(_HISTORIAL_REAL))
    monkeypatch.setattr(db_facts, "get_all_user_facts", lambda *a, **k: [])
    monkeypatch.setattr(cdc, "metas_del_dia", lambda _fd, _plan: {
        "kcal": 2050, "protein_g": 134, "carbs_g": 230, "fats_g": 68, "fuente": "contador"
    })
    plato_cat = {
        "name": "Queso cottage con casabe, tomate y aguacate",
        "calories": 410,
        "protein": "27g",
        "carbs": "20g",
        "fats": "18g",
        "_minutos": 5,
        "_lineas": [(240.0, "Queso cottage"), (35.0, "Casabe"), (55.0, "Tomate"), (40.0, "Aguacate")],
        "_tiene": [],
        "_falta": [],
        "recipe": ["Servir todo junto."],
    }
    monkeypatch.setattr(cdc, "proponer_comidas", lambda *a, **k: [dict(plato_cat)])

    fn = getattr(tools.proponer_comida, "func", tools.proponer_comida)
    out = fn("u-madrugada-1", meal_type="desayuno")
    assert out.startswith("PROPUESTAS DE DESAYUNO")
    assert "planificación para mañana" in out
    assert "RUTINA REAL DEL USUARIO EN Desayuno" in out
    assert "sándwiches" in out.lower() and "lechoza" in out.lower()
    assert "PRIORIZA primero la rutina/plato real del usuario" in out
    assert "PROHIBIDO decir «si lo comes te quedarían X kcal hoy»" in out
    assert "Si se la come:" not in out
    assert "le quedarían ~1640 kcal" not in out
    assert "hoy le faltan" not in out
    assert "nevera" not in out.lower()

    # Incluso si el modelo omite `meal_type` a las 1:28 AM tras completar el día biológico, apunta al desayuno de mañana
    out_auto = fn("u-madrugada-1")
    assert out_auto.startswith("PROPUESTAS DE DESAYUNO")
    assert "planificación para mañana" in out_auto


def test_linea_casera_convierte_ingredientes_a_medidas_cotidianas():
    import coach_day_context as cdc

    assert "taza" in cdc._linea_casera(240, "Queso cottage").lower()
    assert "240 g" not in cdc._linea_casera(240, "Queso cottage")
    assert "taza" in cdc._linea_casera(40, "Avena").lower()
    assert "1 taza" in cdc._linea_casera(80, "Avena").lower()
    assert "clara" in cdc._linea_casera(150, "Clara de huevo").lower()
    assert "1 cucharada" in cdc._linea_casera(15, "Maní").lower()
    assert "puñado" in cdc._linea_casera(28, "Maní").lower()
    assert "aguacate" in cdc._linea_casera(40, "Aguacate").lower()
    assert "aguacate" in cdc._linea_casera(70, "Aguacate").lower()
    assert "2 huevos" in cdc._linea_casera(110, "Huevo").lower()

    for gramaje, nombre in [
        (240, "Queso cottage"),
        (40, "Avena"),
        (150, "Clara de huevo"),
        (15, "Maní"),
        (28, "Maní"),
        (40, "Aguacate"),
        (70, "Aguacate"),
        (110, "Huevo"),
    ]:
        assert "nevera" not in cdc._linea_casera(gramaje, nombre).lower()


def test_exclusion_semantica_y_gustos_en_rutina_y_catalogo(monkeypatch):
    import coach_day_context as cdc
    import rutina_del_coach

    turno_previo = [
        "¿Qué te parece arrancar mañana con plátano maduro hervido, 2 huevos y queso derretido, uno de tus desayunos habituales?"
    ]
    assert rutina_del_coach.plato_coincide_con_exclusion(
        "Plátano maduro con huevo y queso", turno_previo
    )
    assert not rutina_del_coach.plato_coincide_con_exclusion(
        "Avena con claras y maní", turno_previo
    )

    rutina_fake = {
        "por_franja": {
            "desayuno": [
                {
                    "nombre": "Plátano maduro hervido con 2 huevos y queso derretido",
                    "count": 5,
                    "kcal": 458,
                    "proteina_g": 20,
                    "habitual": True,
                },
                {
                    "nombre": "Avena cocida con claras y mantequilla de maní",
                    "count": 3,
                    "kcal": 420,
                    "proteina_g": 24,
                    "habitual": True,
                },
            ]
        },
        "habitos": {"resumen_prompt": "huevos, plátano maduro, avena"},
        "gustos_facts": [],
    }
    monkeypatch.setattr(
        rutina_del_coach,
        "obtener_rutina_usuario",
        lambda uid, consumed_today=None: rutina_fake,
    )

    obj = {"kcal": 450, "protein_g": 28}
    bloque = rutina_del_coach.formatear_rutina_para_propuesta(
        "u-test",
        "desayuno",
        objetivo=obj,
        excluir=turno_previo,
        pidio_propuesta_nueva=True,
    )
    assert (
        "⛔ YA PROPUESTO EN ESTE CHAT (PROHIBIDO REPETIRLO): "
        "«Plátano maduro hervido con 2 huevos y queso derretido»"
    ) in bloque
    assert "Avena cocida con claras y mantequilla de maní" in bloque
    assert "nevera" not in bloque.lower()

    # Si su único plato habitual ya se propuso en el turno anterior, manda directo al catálogo
    rutina_solo_uno = {
        "por_franja": {"desayuno": [rutina_fake["por_franja"]["desayuno"][0]]},
        "habitos": {},
        "gustos_facts": [],
    }
    monkeypatch.setattr(
        rutina_del_coach,
        "obtener_rutina_usuario",
        lambda uid, consumed_today=None: rutina_solo_uno,
    )
    bloque_solo = rutina_del_coach.formatear_rutina_para_propuesta(
        "u-test",
        "desayuno",
        objetivo=obj,
        excluir=turno_previo,
        pidio_propuesta_nueva=True,
    )
    assert "⛔ YA PROPUESTO EN ESTE CHAT (PROHIBIDO REPETIRLO)" in bloque_solo
    assert "opción nº 1 del catálogo de abajo" in bloque_solo
    assert "nevera" not in bloque_solo.lower()

    propuestas_filtradas = cdc.proponer_comidas(
        {},
        "desayuno",
        obj,
        excluir=turno_previo,
        gustos_tokens={"avena", "claras", "mani", "huevos"},
    )
    assert propuestas_filtradas
    for p in propuestas_filtradas:
        assert not rutina_del_coach.plato_coincide_con_exclusion(p["name"], turno_previo)

    texto_propuestas = cdc.formatear_propuestas(
        propuestas_filtradas,
        "desayuno",
        obj,
        con_nevera=False,
        nevera_activa=False,
        rutina_franja_txt=bloque,
        es_para_manana=True,
        usar_medidas_caseras=True,
        pidio_propuesta_nueva=True,
    )
    assert "Porción casera aprox.:" in texto_propuestas
    assert "⛔ YA PROPUESTO EN ESTE CHAT (PROHIBIDO REPETIRLO)" in texto_propuestas
    assert "nevera" not in texto_propuestas.lower()


def test_dame_algo_tu_propaga_contextvar_y_bloquea_plato_repetido():
    from langchain_core.messages import AIMessage, HumanMessage
    import agent
    import tools

    msgs = [
        HumanMessage(content="Hola"),
        AIMessage(
            content=(
                "¿Qué te parece arrancar mañana con plátano maduro hervido, "
                "2 huevos y queso derretido, uno de tus desayunos habituales?"
            )
        ),
        HumanMessage(content="Dame algo tu"),
    ]

    excluir_ctx, pidio_nueva = tools._extraer_contexto_turno_propuesta(msgs)
    assert pidio_nueva is True
    assert excluir_ctx
    assert any("plátano" in x.lower() for x in excluir_ctx)

    captured = {}

    class _DummyTool:
        def invoke(self, args):
            captured["messages"] = list(tools._CURRENT_CHAT_MESSAGES.get([]))
            captured["args"] = args
            return "ok"

    res = agent._invoke_tool_with_chat_context(
        _DummyTool(),
        "proponer_comida",
        {"meal_type": "desayuno"},
        msgs,
    )
    assert res == "ok"
    assert captured["messages"] == msgs
    assert tools._CURRENT_CHAT_MESSAGES.get([]) == []

    class _FailingTool:
        def invoke(self, args):
            assert tools._CURRENT_CHAT_MESSAGES.get([]) == msgs
            raise RuntimeError("boom")

    try:
        agent._invoke_tool_with_chat_context(_FailingTool(), "dummy", {}, msgs)
        assert False, "Debió propagar RuntimeError"
    except RuntimeError:
        pass
    assert tools._CURRENT_CHAT_MESSAGES.get([]) == []


def test_multiturno_y_nuevos_ingredientes_caseros():
    from langchain_core.messages import AIMessage, HumanMessage
    import coach_day_context as cdc
    import tools

    # Hojas verdes, lácteos cremosos, víveres, pasta y suplementos en medidas naturales
    assert "puñado" in cdc._linea_casera(15, "Espinaca").lower()
    assert "1 taza" in cdc._linea_casera(45, "Espinaca").lower()
    assert "cucharada" in cdc._linea_casera(30, "Queso crema").lower()
    assert "lonja" not in cdc._linea_casera(30, "Queso crema").lower()
    assert "scoop" in cdc._linea_casera(30, "Proteína en polvo whey").lower()
    assert "galleta" in cdc._linea_casera(30, "Galletas de arroz").lower()
    assert "trozo mediano" in cdc._linea_casera(150, "Yuca").lower()
    assert "taza cocida" in cdc._linea_casera(65, "Pasta integral").lower()

    for g, nom in [
        (15, "Espinaca"),
        (45, "Espinaca"),
        (30, "Queso crema"),
        (30, "Proteína en polvo whey"),
        (30, "Galletas de arroz"),
        (150, "Yuca"),
        (65, "Pasta integral"),
    ]:
        assert "nevera" not in cdc._linea_casera(g, nom).lower()

    # Ventana multi-turno (>2 sugerencias previas) y frases naturales vs reportes de consumo
    msgs_multi = [
        HumanMessage(content="Hola"),
        AIMessage(content="Propuesta 1: plátano maduro con huevos y queso"),
        HumanMessage(content="Dame algo tu"),
        AIMessage(content="Propuesta 2: revoltillo de claras con espinaca y casabe"),
        HumanMessage(content="otra"),
        AIMessage(content="Propuesta 3: avena con claras y maní"),
        HumanMessage(content="¿Qué más tienes?"),
        AIMessage(content="Propuesta 4: queso cottage con casabe"),
        HumanMessage(content="algo más"),
    ]
    excluir_ctx, pidio_nueva = tools._extraer_contexto_turno_propuesta(msgs_multi)
    assert pidio_nueva is True
    assert len(excluir_ctx) == 4
    assert any("plátano" in x.lower() for x in excluir_ctx)
    assert any("revoltillo" in x.lower() for x in excluir_ctx)

    for frase in ("otra", "Tienes otra", "no me provoca", "otro desayuno"):
        _, flag = tools._extraer_contexto_turno_propuesta([HumanMessage(content=frase)])
        assert flag is True, f"Debió detectar pedido nuevo en: {frase}"

    _, flag_consumo = tools._extraer_contexto_turno_propuesta(
        [HumanMessage(content="Me comí otra manzana verde")]
    )
    assert flag_consumo is False
