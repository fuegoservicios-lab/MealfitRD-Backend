# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-168 · 2026-09-23] «No dividió las comidas»: foto del desayuno + los tacos por texto.

Caso vivo del dueño (22-sep, 22:27, sesión que venía del 20-sep): foto del desayuno (plátano maduro, huevo, salami,
queso y aguacate; el escáner estimó 670 kcal) + «Este fue el desayuno y me comí 3 tacos rellenos… no tengo foto». El
coach registró SOLO los tacos (1050 kcal, almuerzo) y contestó «el día se te cuadró solo: ~2005 kcal de tus ~2050» —
2005 = los 955 del día ANTERIOR, que él mismo había dicho en ese chat, + los tacos. El diario decía 1050.

Batería con el modelo real (DeepSeek, `run_battery.py --only J5`): código viejo 0 de 4 bien (dos veces un único
desayuno de 1670-1720 kcal, dos veces tacos como un SEGUNDO desayuno con `force=true`); código nuevo 5 de 5 (desayuno
670 + almuerzo, total correcto).

Cinco piezas: (1) la tool devuelve el TOTAL REAL del día; (2) aviso de conversación que viene de días anteriores;
(3) regla «una llamada por comida» + instrucción del plato en la foto; (4) guard determinista de la foto que quedó
fuera (`route_tools` → `nudge_plate_photo`); (5) `force` solo vale como respuesta a un aviso de duplicado previo.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_DESAYUNO = ("Plátano maduro frito (5 rodajas), huevo frito (1 unidad), salami frito (2 rodajas), queso blanco fresco "
             "(1 lasca grande) y aguacate (4 trozos). (Estimación: Calorías: 670, Proteína: 23g, Carbohidratos: 43g, "
             "Grasas Saludables: 46g)")
_TACOS = {"meal_name": "3 tacos grandes con carne molida y pollo desmenuzado", "calories": 1050, "protein": 66,
          "ingredients": ["3 tortillas grandes de harina", "carne molida guisada (~90 g)", "pollo desmenuzado (~90 g)",
                          "vegetales/queso al gusto"], "meal_type": "almuerzo"}


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ── 1. La tool devuelve el TOTAL REAL del día ──────────────────────────────────────────────────────

@pytest.fixture
def diario(monkeypatch):
    import tools
    import db_facts
    estado = {"rows": []}
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: 240)
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda uid=None: "2026-09-22")
    monkeypatch.setattr(db_facts, "get_consumed_meals_today",
                        lambda uid, date_str=None, tz_offset_mins=None: estado.setdefault("pedidos", []).append(date_str)
                        or estado["rows"])
    return tools, estado


def test_la_nota_da_la_suma_real_de_hoy_y_prohibe_sumar_de_memoria(diario):
    tools, estado = diario
    estado["rows"] = [{"calories": 670, "protein": 23, "carbs": 43, "healthy_fats": 46},
                      {"calories": 1050, "protein": 66, "carbs": 90, "healthy_fats": 36}]
    nota = tools._nota_total_del_dia("u1", 0)
    assert "TOTAL REAL DE HOY" in nota
    assert "~1720 kcal, ~89 g de proteína, ~133 g de carbohidratos y ~82 g de grasas (2 registros)" in nota
    assert "NUNCA partas de un total dicho en mensajes anteriores" in nota
    assert estado["pedidos"][-1] == "2026-09-22"   # el «hoy» local del usuario


def test_la_nota_de_otro_dia_nombra_ese_dia_y_cuenta_lo_de_la_bateria(diario):
    tools, estado = diario
    estado["rows"] = [{"calories": 300, "protein": 10}]
    nota = tools._nota_total_del_dia("u1", 1, rows_extra=[{"calories": 200, "protein": 5}])
    assert "el diario de AYER suma ahora ~500 kcal, ~15 g de proteína" in nota
    assert "(2 registros)" in nota
    assert estado["pedidos"][-1] == "2026-09-21"


def test_la_nota_es_best_effort(diario, monkeypatch):
    tools, _ = diario
    import db_facts
    assert tools._nota_total_del_dia("u1", 0) == ""   # diario vacío: nada que sumar
    def revienta(*a, **k):
        raise RuntimeError("bd caída")
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", revienta)
    assert tools._nota_total_del_dia("u1", 0) == ""


def test_log_consumed_meal_devuelve_el_total_tras_registrar(diario, monkeypatch):
    tools, estado = diario
    import db
    import db_inventory
    monkeypatch.setattr(tools, "db_log_consumed_meal", lambda *a, **k: "meal-1")
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda uid, mt, cal, d: mt)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan", lambda uid: {"days": []})
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: None)
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory", lambda *a, **k: None)
    estado["rows"] = [{"calories": 1050, "protein": 66, "carbs": 90, "healthy_fats": 36}]
    fn = tools.log_consumed_meal.func if hasattr(tools.log_consumed_meal, "func") else tools.log_consumed_meal
    out = fn("u1", _TACOS["meal_name"], 1050, 66, 90, 36, meal_type="almuerzo", days_ago=0)
    assert "¡Éxito!" in out and "TOTAL REAL DE HOY en su diario, ya con este registro: ~1050 kcal" in out
    # antes de la nota de lo que falta, y el id de la fila sigue al final
    assert out.index("TOTAL REAL DE HOY") < out.index("[ID_REGISTRO_DIARIO:")


def test_correct_consumed_meal_tambien_devuelve_el_total():
    src = _src("tools.py")
    i = src.index("def correct_consumed_meal(")
    cuerpo = src[i:i + 16000]
    assert "_nota_total = _nota_total_del_dia(user_id, _clamp_days_ago(days_ago))" in cuerpo
    assert "_dias_fila = _dias_atras_de_fila(user_id," in cuerpo
    assert "{_nota_nevera}{_nota_total} [ID_REGISTRO_DIARIO:" in cuerpo


@pytest.mark.parametrize("consumed_at,esperado", [
    ("2026-09-22T16:20:48+00:00", 0),      # 12:20 del 22 en RD
    ("2026-09-23T02:28:13+00:00", 0),      # 22:28 del 22 en RD (ya es 23 en UTC)
    ("2026-09-22T02:19:37+00:00", 1),      # 22:19 del 21 en RD
    ("2026-09-24T16:00:00+00:00", None),   # futuro: no es un día del diario
    (None, None),
])
def test_dias_atras_de_una_fila_en_hora_local(diario, consumed_at, esperado):
    tools, _ = diario
    assert tools._dias_atras_de_fila("u1", consumed_at) == esperado


# ── 2. Conversación que viene de días anteriores ──────────────────────────────────────────────────

@pytest.mark.parametrize("creada,esperado", [
    ("2026-09-20T14:31:01+00:00", "2026-09-20"),   # la sesión del dueño
    ("2026-09-22T16:00:00+00:00", None),           # de hoy
    ("2026-09-23T02:00:00+00:00", None),           # 22:00 del 22 en RD: sigue siendo hoy
    (None, None),
])
def test_aviso_de_conversacion_de_dias_anteriores(monkeypatch, creada, esperado):
    import agent
    monkeypatch.setattr(agent, "_session_created_at", lambda sid: creada)
    nota = agent._conversacion_de_dias_anteriores("s1", "2026-09-22", 240)
    if esperado is None:
        assert nota == ""
    else:
        assert f"ESTA CONVERSACIÓN EMPEZÓ ANTES DE HOY (el {esperado})" in nota
        assert "nunca sumes a partir de un total que dijiste antes" in nota


def test_el_aviso_es_best_effort_y_va_en_los_dos_caminos(monkeypatch):
    import agent
    def revienta(sid):
        raise RuntimeError("bd caída")
    monkeypatch.setattr(agent, "_session_created_at", revienta)
    assert agent._conversacion_de_dias_anteriores("s1", "2026-09-22", 240) == ""
    assert agent._conversacion_de_dias_anteriores("s1", None, 240) == ""
    assert _src("agent.py").count(
        "system_prompt += _conversacion_de_dias_anteriores(session_id, local_date, tz_offset)") == 2


# ── 3. La regla y la instrucción de la foto ───────────────────────────────────────────────────────

def test_la_regla_una_llamada_por_comida_va_en_las_dos_variantes():
    from prompts import chat_agent as ca
    assert "UNA LLAMADA POR COMIDA" in ca._REGLA_UNA_LLAMADA_POR_COMIDA
    assert "«TOTAL REAL DE HOY»" in ca._REGLA_UNA_LLAMADA_POR_COMIDA
    assert ca._REGLA_UNA_LLAMADA_POR_COMIDA in ca.build_tools_instructions("u1")
    assert ca._REGLA_UNA_LLAMADA_POR_COMIDA in ca.build_tools_instructions_stream("u1")


def test_la_foto_de_un_plato_dice_como_registrarla_junto_a_otra_comida():
    from prompts import chat_agent as ca
    plato = ca.build_vision_context({"kind": "multi", "has_text": True, "items": [
        {"kind": "plato", "description": _DESAYUNO}]})
    assert ca._PLATO_INSTRUCCION in plato and "como una comida APARTE" in ca._PLATO_INSTRUCCION
    compra = ca.build_vision_context({"kind": "multi", "has_text": True, "items": [
        {"kind": "items", "description": "6 zanahorias"}]})
    assert ca._PLATO_INSTRUCCION not in compra
    suelta = ca.build_vision_context({"kind": "plato", "has_text": True, "description": _DESAYUNO})
    assert ca._PLATO_INSTRUCCION in suelta


# ── 4. La foto del plato que quedó fuera ──────────────────────────────────────────────────────────

def test_fotos_de_plato_del_turno():
    import agent
    fotos = agent._plate_photos_from_vision({"kind": "multi", "items": [
        {"kind": "plato", "description": _DESAYUNO},
        {"kind": "items", "description": "6 zanahorias"},
        {"kind": "etiqueta", "description": "Whey: 120 kcal por scoop"},
        {"kind": "plato", "description": "Mangú con 3 golpes. Estimado ~780 kcal, 28 g proteína."},
        {"kind": "plato", "description": ""},
    ]})
    assert [f["kcal"] for f in fotos] == [670.0, 780.0]
    assert fotos[0]["desc"].startswith("Plátano maduro frito") and "Estimación" not in fotos[0]["desc"]
    assert fotos[1]["desc"] == "Mangú con 3 golpes."
    assert agent._plate_photos_from_vision(None) == []
    assert agent._plate_photos_from_vision({"kind": "plato", "description": "Pizza ~570 kcal"})[0]["kcal"] == 570.0


def test_un_registro_es_la_foto_por_nombre_o_por_kcal():
    import agent
    foto = agent._plate_photos_from_vision({"kind": "plato", "description": _DESAYUNO})[0]
    assert not agent._registro_es_la_foto(_TACOS, foto)   # «queso al gusto» de los ingredientes NO cuenta
    assert agent._registro_es_la_foto({"meal_name": "Desayuno: plátano maduro con huevo y salami", "calories": 900}, foto)
    assert agent._registro_es_la_foto({"meal_name": "Desayuno típico", "calories": 700}, foto)
    assert not agent._registro_es_la_foto({"meal_name": "Desayuno típico", "calories": 1200}, foto)
    assert not agent._registro_es_la_foto(None, foto)


def _turno(tool_calls, *, texto_final="Quedó como almuerzo: ~1050 kcal.", previos=()):
    msgs = list(previos) + [HumanMessage(content="Este fue el desayuno y me comí 3 tacos rellenos")]
    for n, args in enumerate(tool_calls):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": args, "id": f"t{n}"}]))
        msgs.append(ToolMessage(content="¡Éxito! Se ha registrado…", tool_call_id=f"t{n}"))
    msgs.append(AIMessage(content=texto_final))
    return msgs


def _estado(msgs, fotos, **extra):
    return {"messages": msgs, "user_id": "u1", "turn_plate_photos": fotos, **extra}


def test_route_tools_devuelve_el_turno_cuando_la_foto_quedo_fuera():
    import agent
    fotos = agent._plate_photos_from_vision({"kind": "multi", "items": [{"kind": "plato", "description": _DESAYUNO}]})
    assert agent.route_tools(_estado(_turno([_TACOS]), fotos)) == "nudge_plate_photo"


def test_route_tools_no_insiste_si_la_foto_quedo_o_no_hubo_registro():
    import agent
    fotos = agent._plate_photos_from_vision({"kind": "plato", "description": _DESAYUNO})
    desayuno = {"meal_name": "Plátano maduro frito con huevo, salami, queso y aguacate", "calories": 670}
    assert agent.route_tools(_estado(_turno([desayuno, _TACOS]), fotos)) == "__end__"
    assert agent.route_tools(_estado(_turno([], texto_final="¿Esto cuadra para tu cena?"), fotos)) == "__end__"
    assert agent.route_tools(_estado(_turno([_TACOS]), fotos, plate_photo_retried=True)) == "__end__"
    assert agent.route_tools(_estado(_turno([_TACOS]), [])) == "__end__"
    # un registro de un turno ANTERIOR no cuenta como el de la foto
    previo = _turno([desayuno], texto_final="Anotado el desayuno.")
    assert agent.route_tools(_estado(_turno([_TACOS], previos=previo), fotos)) == "nudge_plate_photo"


def test_el_nudge_nombra_la_foto_y_se_apaga_tras_una_vez():
    import agent
    fotos = agent._plate_photos_from_vision({"kind": "plato", "description": _DESAYUNO})
    out = agent.nudge_plate_photo(_estado(_turno([_TACOS]), fotos))
    assert out["plate_photo_retried"] is True
    nota = out["messages"][0]
    assert isinstance(nota, SystemMessage)
    assert "«Plátano maduro frito" in nota.content and "(~670 kcal según el análisis)" in nota.content
    assert "como un registro APARTE" in nota.content and "no registres nada" in nota.content


def test_el_grafo_y_el_estado_llevan_el_guard():
    import agent
    assert "nudge_plate_photo" in agent.chat_builder.nodes
    anot = agent.ChatState.__annotations__
    assert "turn_plate_photos" in anot and "plate_photo_retried" in anot   # fuera del schema LangGraph las descarta
    src = _src("agent.py")
    i = src.index("existing_state = chat_graph_app.get_state(config)\n    \n    inputs = {")
    assert '"turn_plate_photos": _plate_photos_from_vision(vision),' in src[i:i + 2000]
    assert '"plate_photo_retried": False,' in src[i:i + 2000]
    j = src.index('"new_plan": None,            # Reinicia el plan nuevo en cada ejecución')
    assert '"turn_plate_photos": [],' in src[j:j + 600]


# ── 5. `force` solo como respuesta a un aviso de duplicado ────────────────────────────────────────

_AVISO = ("⚠️ NO REGISTRADO: el usuario YA tiene un desayuno registrado hoy ('Mangú', 520 kcal). Díselo amablemente "
          "y pregúntale si de verdad comió dos desayunos ese día")


def test_aviso_de_duplicado_en_el_turno_anterior():
    import agent
    con_aviso = [HumanMessage(content="me comí un sandwich de desayuno"),
                 AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "a"}]),
                 ToolMessage(content=_AVISO, tool_call_id="a"),
                 AIMessage(content="Ya tienes un desayuno (mangú). ¿Comiste dos?"),
                 HumanMessage(content="sí, dos")]
    assert agent._duplicado_avisado_en_turno_previo(con_aviso)
    assert not agent._duplicado_avisado_en_turno_previo(con_aviso[:1])
    sin_aviso = [HumanMessage(content="hola"), AIMessage(content="¡Hola!"), HumanMessage(content="me comí 3 tacos")]
    assert not agent._duplicado_avisado_en_turno_previo(sin_aviso)
    # el aviso en el turno ACTUAL no cuenta: saltárselo en la misma tanda es justo el fallo
    en_el_mismo = sin_aviso + [AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "b"}]),
                               ToolMessage(content=_AVISO, tool_call_id="b")]
    assert not agent._duplicado_avisado_en_turno_previo(en_el_mismo)


@pytest.fixture
def ejecutar(monkeypatch):
    for mod in ("db", "db_inventory", "db_profiles", "db_plans", "db_facts", "memory_manager", "vision_agent",
                "fact_extractor", "cpu_tasks", "graph_orchestrator", "ai_helpers"):
        if mod not in sys.modules:
            monkeypatch.setitem(sys.modules, mod, MagicMock())
    import agent
    capturado = {}

    class _Tool:
        name = "log_consumed_meal"

        def invoke(self, args):
            capturado["args"] = dict(args)
            return "¡Éxito!"

    monkeypatch.setattr(agent, "agent_tools", [_Tool()])

    def correr(previos, force):
        llamada = AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "id": "x", "args": {
            "meal_name": "Sandwich", "calories": 400, "protein": 20, "meal_type": "desayuno", "force": force}}])
        agent.execute_tools({"messages": list(previos) + [llamada], "user_id": "u1", "session_id": "s1",
                             "form_data": {}, "current_plan": {}, "updated_fields": {}, "new_plan": None,
                             "sys_prompt": ""})
        return capturado["args"]
    return correr


def test_force_sin_aviso_previo_se_apaga_y_con_aviso_se_respeta(ejecutar):
    sin_aviso = [HumanMessage(content="Este fue el desayuno y me comí 3 tacos")]
    assert ejecutar(sin_aviso, True)["force"] is False
    con_aviso = [HumanMessage(content="me comí un sandwich de desayuno"),
                 AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "a"}]),
                 ToolMessage(content=_AVISO, tool_call_id="a"),
                 AIMessage(content="Ya tienes un desayuno. ¿Comiste dos?"),
                 HumanMessage(content="sí, comí dos")]
    assert ejecutar(con_aviso, True)["force"] is True
    assert ejecutar(sin_aviso, False)["force"] is False


# ── La batería en seco comparte las piezas ────────────────────────────────────────────────────────

def test_la_bateria_reproduce_el_caso_y_comparte_la_nota():
    import json
    bat = _src("scripts/coach_battery/run_battery.py")
    assert "tools._nota_total_del_dia(" in bat
    assert "NO REGISTRADO: el usuario YA tiene un" in bat          # el guard de duplicados, con el texto real
    assert "agent._session_created_at = " in bat and "agent.build_memory_context = _memory_with_history" in bat
    casos = {c["id"]: c for c in json.loads(_src("scripts/coach_battery/battery.json"))["casos"]}
    assert casos["J5"]["expect"]["registros"] == 2 and casos["J5"]["historial"]
    assert casos["J6"]["expect"]["no_tools"] == ["log_consumed_meal"]


def test_marcador_del_lote():
    import re
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 168 and m.group(2) >= "2026-09-23"   # la serie sigue; el marker nunca baja
    assert "[P1-PLAN-LOTE-168 · 2026-09-23]" in app
