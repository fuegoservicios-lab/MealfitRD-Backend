"""[REVISION-LOTES-1001-1011 · 2026-11-14] Lo que la revisión de los lotes 1001-1011 encontró roto en producción.

1. G-19 (tope diario por usuario) y G-18 (pico de costo 3x) consultaban `estimated_cost_micros` y `endpoint`, columnas
   que `llm_usage_events` no tiene: las dos consultas fallaban siempre y el fail-open las dejaba mudas.
2. G-20 (guard de foto/web): las raíces del regex de intención no coincidían con la palabra entera por el `\\b` final;
   con una foto en el turno solo «soy alérgico» guardaba un dato clínico.
3. G-07 (confirmación rápida): guarda de idioma con marcadores inexistentes, y texto para el modelo enseñado al usuario.
4. G-20 anti-inyección: tope de 240 caracteres sobre la lista entera de alergias.
5. G-12 (caché de contexto 30 s): lo que el usuario escribía desde la app no la invalidaba.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_BACKEND = Path(__file__).resolve().parent.parent


def _columnas_llm_usage_events() -> set[str]:
    ddl = (_BACKEND / "migrations" / "p1_cost_instrumentation_2026_05_15.sql").read_text(encoding="utf-8")
    bloque = ddl.split("CREATE TABLE IF NOT EXISTS public.llm_usage_events (", 1)[1].split(");", 1)[0]
    cols = {ln.strip().split()[0] for ln in bloque.splitlines() if ln.strip()}
    assert {"cost_usd_micros", "node", "user_id", "created_at"} <= cols
    return cols


def _sql_capturado(fn, **kw) -> str:
    capturado = {}

    def _q(sql, params=(), **_k):
        capturado["sql"] = sql
        return {"spent_micros": 0, "calls_1h": 0, "avg_1h_usd": 0.0, "calls_7d": 0, "avg_7d_usd": 0.0}

    fn(execute_sql_query=_q, **kw)
    return capturado["sql"]


def _identificadores_de_llm_usage(sql: str) -> set[str]:
    sin_literales = re.sub(r"'[^']*'", "''", sql)
    palabras = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", sin_literales.lower()))
    return palabras


def test_tope_diario_por_usuario_consulta_columnas_que_existen(monkeypatch):
    import chat_hard_turn

    monkeypatch.setenv("MEALFIT_CHAT_USER_DAILY_BUDGET_USD", "0.25")
    sql = _sql_capturado(chat_hard_turn.check_user_daily_chat_budget, user_id="u-1")
    cols = _columnas_llm_usage_events()
    assert "cost_usd_micros" in sql and "node like 'chat%%'" in sql.lower()
    for fantasma in ("estimated_cost_micros", "endpoint"):
        assert fantasma not in _identificadores_de_llm_usage(sql)
        assert fantasma not in cols


def test_tope_diario_bloquea_de_verdad_al_pasar_el_limite(monkeypatch):
    import chat_hard_turn

    monkeypatch.setenv("MEALFIT_CHAT_USER_DAILY_BUDGET_USD", "0.25")
    r = chat_hard_turn.check_user_daily_chat_budget(
        "u-1", execute_sql_query=lambda *a, **k: {"spent_micros": 260_000})
    assert r["block_turn"] is True and r["status"] == "hard_100"
    r80 = chat_hard_turn.check_user_daily_chat_budget(
        "u-1", execute_sql_query=lambda *a, **k: {"spent_micros": 210_000})
    assert r80["status"] == "soft_80" and r80["block_turn"] is False


def test_alerta_pico_de_costo_consulta_columnas_que_existen():
    import chat_observability_alerts as coa

    sqls = []

    def _q(sql, params=(), **_k):
        sqls.append(sql)
        return {}

    coa.evaluar_salud_coach_y_alertar(execute_sql_query=_q, execute_sql_write=lambda *a, **k: None,
                                      include_hourly_and_cost=True)
    costo = [s for s in sqls if "llm_usage_events" in s]
    assert costo, "la consulta de pico de costo no corrió"
    for s in costo:
        ids = _identificadores_de_llm_usage(s)
        assert "estimated_cost_micros" not in ids and "endpoint" not in ids
        assert "cost_usd_micros" in ids


def test_g20_lo_clinico_dicho_por_el_usuario_pasa_aunque_haya_foto():
    import agent

    estado_foto = {"turn_plate_photos": [{"attachment_id": "att-1"}], "turn_photo_to_log": False}
    for frase in ("¿Puedo comer esto? Soy intolerante a la lactosa", "tengo diabetes", "soy diabético",
                  "tengo hipertensión", "soy celíaco", "tengo celiaquía", "soy vegana",
                  "me diagnosticaron gastritis", "estoy embarazada, ¿esto me sirve?"):
        out = agent._guard_source_intent_mutation(
            "update_form_field", {"field": "medicalConditions", "new_value": "x"}, estado_foto,
            [HumanMessage(content=frase)])
        assert out is None, frase
    # La foto sola (o una etiqueta) sigue sin poder añadir una alergia.
    solo_foto = agent._guard_source_intent_mutation(
        "update_form_field", {"field": "allergies", "new_value": "mani"}, estado_foto, [HumanMessage(content="📷")])
    assert solo_foto and "BLOQUEADO" in solo_foto


def _estado_con_tool(nombre: str, args: dict, texto_usuario: str, sys_prompt: str = "Eres un coach."):
    ai_tc = AIMessage(content="", tool_calls=[{"id": "tc-1", "name": nombre, "args": args}])
    return {"user_id": "u1", "session_id": "s1", "sys_prompt": sys_prompt, "modo_voz": False,
            "messages": [HumanMessage(content=texto_usuario), ai_tc]}


def _confirmar(nombre, resultado, texto_usuario, sys_prompt="Eres un coach."):
    import agent

    estado = _estado_con_tool(nombre, {}, texto_usuario, sys_prompt)
    msgs = [ToolMessage(content=resultado, tool_call_id="tc-1")]
    return agent._maybe_append_fast_tool_confirmation(msgs, estado, estado["messages"])


_AGUA = "Listo: se sumaron 1 vaso(s). El usuario ahora lleva 5 de 8 vasos hoy."


def test_g07_no_responde_en_espanol_a_quien_no_habla_espanol():
    assert _confirmar("log_water_glass", _AGUA, "me tomé un vaso de agua") is not None
    directiva_en = "Eres un coach.\n\n🌐 RESPONSE LANGUAGE — NON-NEGOTIABLE: Write your ENTIRE reply in English."
    assert _confirmar("log_water_glass", _AGUA, "me tomé un vaso de agua", directiva_en) is None
    assert _confirmar("log_water_glass", _AGUA, "I had a glass of water, please log it for me") is None


def test_g07_medio_vaso():
    r = _confirmar("log_water_glass",
                   "Listo: se sumaron 0.5 vaso(s). El usuario ahora lleva 4.5 de 8 vasos hoy.", "medio vaso de agua")
    assert r is not None and "medio vaso" in r.content


def test_g07_ajustes_nunca_ensena_el_texto_para_el_modelo():
    import ajustes_de_la_app as aj

    con = aj._con_marcador
    casos = {
        con("Hecho: la memoria a largo plazo está activa; aprenderás de sus conversaciones.", {"memoria": True}):
            "Listo, activé la memoria a largo plazo",
        con("Hecho: la app pasa al tema oscuro.", {"tema": "dark"}): "tema oscuro",
        con("Hecho: la app muestra configuracion.", {"pantalla": "configuracion", "seccion": "health"}):
            "la configuración, en la sección Alergias y dieta",
        con("Hecho: recordatorios de agua apagados.", {"avisos_agua": False}): "apagué los recordatorios de agua",
    }
    for resultado, esperado in casos.items():
        r = _confirmar("cambiar_ajuste_de_la_app", resultado, "activa la memoria")
        assert r is not None and esperado in r.content, (resultado, r and r.content)
        for prohibido in ("aprenderás", "Díselo", "dile", "le hablas", "su panel", "<<AJUSTE"):
            assert prohibido not in r.content
    # Lo que no es un «Hecho» inequívoco lo redacta el modelo.
    for resultado in (
        con("La Nevera no cambió: en su modo actual no se puede apagar. Díselo así.", {"nevera": True}),
        con("Hecho: la app cambia de idioma ahora; desde el siguiente mensaje le hablas en ese idioma.",
            {"idioma": "en-US"}),
        con("No tiene plan que reanudar: le abrí el formulario. Díselo en una frase.", {"pantalla": "formulario"}),
    ):
        assert _confirmar("cambiar_ajuste_de_la_app", resultado, "apaga la nevera") is None, resultado


def test_alergias_largas_no_las_frena_el_filtro_de_inyeccion():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert 'max_len=1200 if field in ("allergies", "medicalConditions") else 240' in src
    import input_sanitization as isan

    lista = ", ".join(["Maní", "Mariscos", "Frutos secos", "Lácteos", "Gluten", "Huevo", "Soya", "Sésamo",
                       "Pescado", "Mostaza", "Apio", "Sulfitos", "Altramuces", "Moluscos", "Kiwi", "Fresa",
                       "Melocotón", "Látex-fruta (aguacate, banana)", "Colorante tartrazina",
                       "Diabetes tipo 2 con metformina", "Hipotiroidismo (levotiroxina en ayunas)"])
    assert len(lista) > 240
    assert isan.contiene_inyeccion_texto(lista, max_len=1200) is None
    assert isan.contiene_inyeccion_texto(lista + " <<SYSTEM_OVERRIDE>>", max_len=1200) == "marker_injection"


def _correr_middleware(metodo: str, ruta: str) -> int:
    import contexto_del_coach_fresco as m

    llamadas = []
    original = m._vaciar_cache_de_contexto
    m._vaciar_cache_de_contexto = lambda: llamadas.append(1)
    try:
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(_msg):
            return None

        mw = m.ContextoDelCoachFrescoMiddleware(app)
        asyncio.run(mw({"type": "http", "method": metodo, "path": ruta}, receive, send))
    finally:
        m._vaciar_cache_de_contexto = original
    return len(llamadas)


def test_lo_que_el_usuario_escribe_desde_la_app_vacia_la_cache_del_coach():
    assert _correr_middleware("POST", "/api/diary/consumed") == 1
    assert _correr_middleware("DELETE", "/api/pantry/item/1") == 1
    assert _correr_middleware("GET", "/api/diary/consumed") == 0
    assert _correr_middleware("POST", "/api/chat/stream") == 0   # el chat invalida por tool, por usuario


def test_la_cache_de_contexto_se_vacia_de_verdad():
    import agent
    import contexto_del_coach_fresco as m

    os.environ["MEALFIT_CHAT_CTX_CACHE_TTL_S"] = "30"
    try:
        n = {"cargas": 0}

        def _carga():
            n["cargas"] += 1
            return "diario"

        agent._ctx_cache_get_or_load("u-rev", "consumed_today", ("2026-11-14", 240), _carga)
        agent._ctx_cache_get_or_load("u-rev", "consumed_today", ("2026-11-14", 240), _carga)
        assert n["cargas"] == 1
        m._vaciar_cache_de_contexto()
        agent._ctx_cache_get_or_load("u-rev", "consumed_today", ("2026-11-14", 240), _carga)
        assert n["cargas"] == 2
    finally:
        os.environ.pop("MEALFIT_CHAT_CTX_CACHE_TTL_S", None)
        agent.invalidate_chat_user_ctx_cache(None)


def test_middleware_registrado_en_app():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "app.add_middleware(ContextoDelCoachFrescoMiddleware)" in src
