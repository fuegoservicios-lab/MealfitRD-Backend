"""[P1-NEVERA-OPCIONAL · 2026-09-23] La Nevera opcional en modo contador.

El dueño: «cuando el generador esté desactivado, que la Nevera también se pueda desactivar: hay gente que solo
quiere el contador y el agente», y «encendida como hoy, pero si en 48 h no se usa, que se apague sola». Medido el
23-sep: las 6 cuentas en modo contador tenían la Nevera vacía. LA regla vive en `nevera_opcional.py`; aquí se ancla
que la regla, el apagado automático, el diario, el coach y la API la respeten. Doc: docs/nevera_opcional.md."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import nevera_opcional as no

_BACKEND = Path(__file__).resolve().parents[1]


# ── 1. La regla ──────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("perfil, esperado", [
    ({"plan_mode": "tracking", "nevera_enabled": False}, False),
    ({"plan_mode": "tracking", "nevera_enabled": None}, True),
    ({"plan_mode": "tracking", "nevera_enabled": True}, True),
    ({"plan_mode": "plan", "nevera_enabled": False}, False),      # [P1-PLAN-LOTE-217] también en modo plan
    ({"plan_mode": "tracking"}, True),                            # sin columna todavía: fallo abierto
    ({}, True),
    (None, True),
])
def test_la_regla(perfil, esperado):
    assert no.nevera_activa_de(perfil) is esperado


def test_el_kill_switch_la_deja_activa_para_todos(monkeypatch):
    monkeypatch.setenv("MEALFIT_NEVERA_SWITCH", "false")
    assert no.nevera_activa_de({"plan_mode": "tracking", "nevera_enabled": False}) is True
    assert no.apagar_neveras_sin_uso() == []


def test_nevera_activa_lee_la_fila_y_falla_abierto(monkeypatch):
    monkeypatch.setattr(no, "execute_sql_query", lambda *a, **k: {"plan_mode": "tracking", "nevera_enabled": False})
    assert no.nevera_activa("u1") is False
    assert no.nevera_activa("guest") is True
    assert no.nevera_activa(None) is True

    def _revienta(*a, **k):
        raise RuntimeError("column nevera_enabled does not exist")
    monkeypatch.setattr(no, "execute_sql_query", _revienta)
    assert no.nevera_activa("u1") is True


def test_fijar_filtra_por_id_y_borra_la_marca_automatica(monkeypatch):
    visto = {}

    def _w(sql, params, **k):
        visto["sql"], visto["params"], visto["k"] = sql, params, k
        return [{"id": "u1"}]
    monkeypatch.setattr(no, "execute_sql_write", _w)
    assert no.fijar_nevera("u1", True) is True
    assert "WHERE id = %s" in visto["sql"] and "nevera_auto_off_at = NULL" in visto["sql"]
    assert visto["params"] == (True, "u1") and visto["k"].get("returning") is True


def test_estado_para_configuracion(monkeypatch):
    from datetime import datetime, timezone
    at = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(no, "execute_sql_query", lambda *a, **k: {
        "plan_mode": "tracking", "nevera_enabled": False, "nevera_auto_off_at": at})
    e = no.estado_nevera("u1")
    assert e == {"enabled": False, "activa": False, "auto_off_at": at.isoformat(), "disponible": True}


# ── 2. El apagado automático ─────────────────────────────────────────────────────────────────────────────────
def test_el_sql_del_apagado_solo_toca_contador_automatico_y_vacio():
    sql = no._SQL_APAGAR
    for trozo in ("q.plan_mode = 'tracking'", "q.nevera_enabled IS NULL", "p.nevera_enabled IS NULL",
                  "i.quantity > 0", "i.updated_at >", "nevera_reloj_desde", "plan_mode_changed_at",
                  "SET nevera_enabled = FALSE, nevera_auto_off_at = now()", "RETURNING p.id", "LIMIT %s"):
        assert trozo in sql, trozo


def test_el_update_externo_repite_lo_que_el_usuario_puede_cambiar_entre_medias():
    """[Final fix wave] Entre el SELECT interno (foto del inicio de la sentencia) y el UPDATE, el usuario puede elegir
    (`nevera_enabled`) o encender el generador (`plan_mode`). En READ COMMITTED Postgres re-evalúa sobre la fila NUEVA
    solo las condiciones del alias que se actualiza (`p`); las del subselect (`q`) se quedan con la foto vieja. Sin
    repetir las dos en `p`, una cuenta recién pasada a modo plan saldría con la Nevera apagada y la marca del sistema."""
    sql = no._SQL_APAGAR
    externo = sql[sql.index("LIMIT %s)"):]
    assert "p.nevera_enabled IS NULL" in externo
    assert "p.plan_mode = 'tracking'" in externo


def test_apagar_devuelve_ids_y_pasa_horas_y_limite(monkeypatch):
    visto = {}

    def _w(sql, params, **k):
        visto["params"] = params
        return [{"id": "u1"}, {"id": "u2"}]
    monkeypatch.setattr(no, "execute_sql_write", _w)
    monkeypatch.setenv("MEALFIT_NEVERA_AUTO_OFF_HOURS", "72")
    assert no.apagar_neveras_sin_uso(limite=10) == ["u1", "u2"]
    assert visto["params"] == (72, 72, 10)


def test_apagar_con_el_knob_apagado_no_toca_la_db(monkeypatch):
    monkeypatch.setenv("MEALFIT_NEVERA_AUTO_OFF", "false")
    monkeypatch.setattr(no, "execute_sql_write", lambda *a, **k: pytest.fail("no debía escribir"))
    assert no.apagar_neveras_sin_uso() == []


# ── 3. Migración ─────────────────────────────────────────────────────────────────────────────────────────────
def test_la_migracion_es_idempotente_y_esta_en_las_dos_carpetas():
    nombre = "p1_nevera_opcional_2026_09_23.sql"
    be = (_BACKEND / "migrations" / nombre).read_text(encoding="utf-8")
    raiz = _BACKEND.parent / "migrations" / nombre
    if raiz.exists():
        assert raiz.read_text(encoding="utf-8") == be, "migrations/ y backend/migrations/ deben ser idénticas"
    for col in ("nevera_enabled BOOLEAN", "nevera_auto_off_at TIMESTAMPTZ",
                "nevera_reloj_desde TIMESTAMPTZ NOT NULL DEFAULT now()"):
        assert f"ADD COLUMN IF NOT EXISTS {col}" in be, col
    assert "RAISE EXCEPTION" in be


# ── 4. API y perfil ──────────────────────────────────────────────────────────────────────────────────────────
def test_patch_guarda_y_devuelve_el_estado(monkeypatch):
    from routers import preferences as pref
    llamadas = []
    monkeypatch.setattr(no, "fijar_nevera", lambda uid, en: llamadas.append((uid, en)) or True)
    monkeypatch.setattr(no, "estado_nevera", lambda uid: {"enabled": False, "activa": False,
                                                           "auto_off_at": None, "disponible": True})
    r = asyncio.run(pref.api_set_nevera(body=pref.NeveraPreferenceBody(enabled=False), verified_user_id="u1"))
    assert llamadas == [("u1", False)] and r["activa"] is False


def test_patch_con_el_knob_apagado_es_409(monkeypatch):
    from fastapi import HTTPException
    from routers import preferences as pref
    monkeypatch.setenv("MEALFIT_NEVERA_SWITCH", "false")
    with pytest.raises(HTTPException) as e:
        asyncio.run(pref.api_set_nevera(body=pref.NeveraPreferenceBody(enabled=False), verified_user_id="u1"))
    assert e.value.status_code == 409


def test_el_perfil_trae_la_regla_calculada():
    ud = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    i = ud.index("async def api_get_profile(")
    assert '"nevera_activa": nevera_activa_de(profile)' in ud[i:i + 1500]


@pytest.mark.parametrize("perfil, esperado", [
    ({"id": "u1", "plan_mode": "tracking", "nevera_enabled": False}, False),
    ({"id": "u1", "plan_mode": "plan", "nevera_enabled": False}, False),    # [P1-PLAN-LOTE-217] también en modo plan
    ({"id": "u1", "plan_mode": "tracking"}, True),                          # columnas sin migrar: activa
])
def test_get_profile_sirve_nevera_activa(monkeypatch, perfil, esperado):
    """[Final fix wave] El endpoint REAL, no su fuente: `GET /api/profile` añade `nevera_activa` calculada por LA regla
    (el frontend solo la lee) y devuelve el resto del perfil tal cual."""
    import db
    from routers import user_data
    monkeypatch.setattr(db, "get_user_profile", lambda uid: dict(perfil) if uid == "u1" else None)
    r = asyncio.run(user_data.api_get_profile(verified_user_id="u1"))
    assert r["profile"]["nevera_activa"] is esperado
    assert {k: v for k, v in r["profile"].items() if k != "nevera_activa"} == perfil


# ── 5. Diario y cron ─────────────────────────────────────────────────────────────────────────────────────────
def _persistir(monkeypatch, activa: bool):
    from fastapi import BackgroundTasks
    from routers import diary
    import db_inventory
    llamadas = []
    monkeypatch.setattr(diary, "log_consumed_meal", lambda *a, **k: "meal-1")
    monkeypatch.setattr(diary, "nevera_activa", lambda uid: activa)
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory",
                        lambda *a, **k: llamadas.append(a) or {"succeeded": ["Huevo"]})
    r = diary._persist_consumed_meal(
        user_id="u1", meal_name="Desayuno", meal_type="desayuno", calories=300, protein=20, carbs=10,
        healthy_fats=15, ingredients=["2 huevos"], days_ago=0, background_tasks=BackgroundTasks(), source="photo")
    return r, llamadas


def test_con_la_nevera_apagada_el_diario_guarda_pero_no_descuenta(monkeypatch):
    r, llamadas = _persistir(monkeypatch, activa=False)
    assert r["success"] is True and llamadas == [] and r["deducted"] == []


def test_con_la_nevera_activa_descuenta_como_siempre(monkeypatch):
    r, llamadas = _persistir(monkeypatch, activa=True)
    assert len(llamadas) == 1 and r["deducted"] == ["Huevo"]


def test_el_apagado_automatico_corre_cada_hora():
    ct = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert "def _nevera_auto_off_job(" in ct
    i = ct.index("def register_plan_chunk_scheduler(")
    assert 'id="nevera_auto_off"' in ct[i:], "el job se registra en el SSOT de crons"


def test_el_apagado_roto_es_ruidoso_no_silencioso(monkeypatch, caplog):
    """[P1-NEVERA-OPCIONAL] Un apagado automático que nunca corre debe quedar en el log como ERROR, no como un
    warning que nadie revisa — es la única señal de que 48 h de cuentas vacías no se están apagando."""
    import logging

    def _revienta(*a, **k):
        raise RuntimeError("db caída")
    monkeypatch.setattr(no, "execute_sql_write", _revienta)
    caplog.set_level(logging.ERROR, logger="nevera_opcional")
    assert no.apagar_neveras_sin_uso() == []
    assert any(rec.levelname == "ERROR" for rec in caplog.records)


# ── 6. El coach ──────────────────────────────────────────────────────────────────────────────────────────────
def test_el_prompt_recibe_la_orden_y_no_el_inventario(monkeypatch):
    import agent
    monkeypatch.setattr(no, "nevera_activa", lambda uid: False)
    out = agent._build_pantry_context("u1")
    assert out == no.BLOQUE_PROMPT_NEVERA_APAGADA


def test_la_orden_no_atribuye_el_apagado_al_usuario():
    """[Final fix wave] El apagado automático también la deja en FALSE: «desactivada por el usuario» haría al coach
    afirmar algo que el usuario no hizo («tú la apagaste»). El texto es neutro; las instrucciones, las de siempre."""
    for texto in (no.MENSAJE_NEVERA_APAGADA, no.BLOQUE_PROMPT_NEVERA_APAGADA):
        assert "por el usuario" not in texto, texto
        assert "DESACTIVADA (Configuración → Capacidades)" in texto, texto
        assert "No la menciones" in texto or "no la menciones" in texto, texto
    assert "puede encenderla" in no.MENSAJE_NEVERA_APAGADA and "puede encenderla" in no.BLOQUE_PROMPT_NEVERA_APAGADA


def test_con_la_nevera_activa_el_prompt_no_cambia(monkeypatch):
    import agent
    monkeypatch.setattr(no, "nevera_activa", lambda uid: True)
    monkeypatch.setattr("db.execute_sql_query", lambda *a, **k: [])
    assert "NEVERA FÍSICA AHORA: vacía" in agent._build_pantry_context("u1")


def test_los_dos_caminos_del_chat_no_leen_el_inventario_apagado():
    """[revisión B4] El estado se resuelve UNA vez por turno en cada camino (`_nevera_on`, al tope de la función) y de
    ese dato cuelgan la lectura, el respaldo desde `form_data`, la línea del inventario y el bloque final."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    for ancla in ("get_user_inventory(user_id) if _nevera_on else []",
                  "if not inventory_str and form_data and _nevera_on:",
                  "nevera_activa=_nevera_on,",
                  "system_prompt += _build_pantry_context(user_id, nevera_on=_nevera_on)"):
        assert src.count(ancla) == 2, ancla
    for fn in ("def chat_with_agent(", "def chat_with_agent_stream("):
        i = src.index(fn)
        j = src.find("\ndef ", i + 10)   # tras `chat_with_agent_stream` no hay otra función de módulo
        cuerpo = src[i:j if j > 0 else len(src)]
        assert cuerpo.count("_nevera_on = _nevera_activa_para_chat(user_id)") == 1, fn
        assert cuerpo.index("_nevera_on = _nevera_activa_para_chat(user_id)") < cuerpo.index("system_prompt ="), fn
    stream = src[src.index("def chat_with_agent_stream("):]
    assert stream.count("system_prompt += build_vision_context(vision, nevera_activa=_nevera_on)") == 2


def test_las_tools_de_nevera_responden_desactivada():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    for tool in ("def check_current_pantry(", "def modify_pantry_inventory("):
        i = src.index(tool)
        assert "MENSAJE_NEVERA_APAGADA" in src[i:i + 2500], tool


def test_registrar_y_corregir_no_descuentan_con_la_nevera_apagada():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    for tool in ("def log_consumed_meal(", "def correct_consumed_meal("):
        i = src.index(tool)
        j = src.index("\ndef ", i + 10)
        assert "nevera_activa(" in src[i:j], tool


def test_el_bot_de_ayuda_no_promete_la_nevera_al_contador():
    """«Configuración → Capacidades» ya estaba en el prompt (el interruptor del generador): la frase de la navegación
    del modo contador es la que ancla el cambio — la Nevera sale en la nav solo si el usuario la tiene encendida."""
    hb = (_BACKEND / "prompts" / "help_bot.py").read_text(encoding="utf-8")
    assert "Configuración → Capacidades" in hb
    assert "**Progreso**, **Agente**, **Nevera** e **Historial**" not in hb
    assert "la **Nevera** si el usuario la tiene encendida (se apaga en Configuración → Capacidades)" in hb


# Las tools se invocan como en `test_p1_plan_lote_137.py` (la función detrás del `@tool`). `tools.py` importa
# `nevera_activa` a nivel de módulo —igual que `routers/diary.py` en la sección 5—, así que se parchea donde la tool
# la BUSCA (`tools.nevera_activa`): parchear `no.nevera_activa` no la alcanzaría.
def _fn(tool):
    return getattr(tool, "func", None) or tool


def _prohibido(*a, **k):
    pytest.fail("con la Nevera apagada esta tool no debía tocar la base de datos")


def test_las_tools_de_inventario_apagadas_responden_sin_tocar_la_db(monkeypatch):
    import db
    import db_inventory
    import tools
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: False)
    for helper in ("get_user_inventory", "get_raw_user_inventory", "add_or_update_inventory_item",
                   "find_pantry_rows_for_name"):
        monkeypatch.setattr(db_inventory, helper, _prohibido)
    monkeypatch.setattr(db, "execute_sql_query", _prohibido)
    monkeypatch.setattr(db, "execute_sql_write", _prohibido)
    assert _fn(tools.check_current_pantry)("u1") == no.MENSAJE_NEVERA_APAGADA
    assert _fn(tools.modify_pantry_inventory)(
        "u1", items_to_add=["2 unidades de Manzana"], items_to_remove=["arroz"], items_to_deplete=["leche"],
    ) == no.MENSAJE_NEVERA_APAGADA


def test_con_la_nevera_activa_la_despensa_se_lee_como_siempre(monkeypatch):
    import db_inventory
    import tools
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: True)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda uid: ["2 unidades de Huevo"])
    assert _fn(tools.check_current_pantry)("u1") == (
        "RESULTADO DEL INVENTARIO FÍSICO ACTUAL EN LA DESPENSA:\n- 2 unidades de Huevo")


def _registrar(monkeypatch, activa: bool):
    import db
    import db_inventory
    import tools
    visto = {"log": [], "descuentos": []}
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: activa)
    monkeypatch.setattr(tools, "db_log_consumed_meal", lambda *a, **k: visto["log"].append(k) or "fila-1")
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda uid, mt, kcal, d: mt)
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda *a, **k: "")
    monkeypatch.setattr(tools, "_nota_comidas_sin_registrar", lambda *a, **k: "")
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: None)
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory",
                        lambda *a, **k: visto["descuentos"].append(a) or {"not_in_pantry": ["Huevo"]})
    out = _fn(tools.log_consumed_meal)("u1", "Huevos hervidos", 220, 14, meal_type="merienda",
                                       ingredients=["2 huevos (≈100 g)"])
    return out, visto


def test_registrar_con_la_nevera_apagada_guarda_pero_no_descuenta(monkeypatch):
    out, visto = _registrar(monkeypatch, activa=False)
    assert out.startswith("¡Éxito!") and len(visto["log"]) == 1 and visto["descuentos"] == []
    assert visto["log"][0]["mark_inventory_synced"] is True, "apagada hoy no se vuelve un descuento silencioso mañana"
    assert "nevera" not in out.lower(), out


def test_registrar_con_la_nevera_activa_descuenta_como_siempre(monkeypatch):
    out, visto = _registrar(monkeypatch, activa=True)
    assert len(visto["descuentos"]) == 1 and "NO estaban en la nevera" in out


def _corregir(monkeypatch, activa: bool):
    """El registro original SÍ descontó (con la Nevera encendida: hay rastro en el ledger) y ahora se corrigen sus
    ingredientes."""
    from datetime import datetime, timezone
    import db
    import db_inventory
    import tools
    ahora = datetime.now(timezone.utc)
    visto = {"reverts": [], "descuentos": [], "updates": []}

    def _q(sql, *a, **k):
        if "inventory_consumption_events" in sql:
            return {"c": 2}
        if "SELECT meal_type, consumed_at" in sql:
            return {"meal_type": "almuerzo", "consumed_at": ahora, "inventory_synced_at": ahora}
        return None
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: activa)
    monkeypatch.setattr(db, "execute_sql_query", _q)
    monkeypatch.setattr(tools, "db_update_consumed_meal", lambda uid, mid, **k: visto["updates"].append(k) or mid)
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda *a, **k: "")
    monkeypatch.setattr(tools, "_dias_atras_de_fila", lambda *a, **k: None)
    monkeypatch.setattr(db_inventory, "revert_consumption_events",
                        lambda uid, mid: visto["reverts"].append(mid) or {"reverted": ["1 lb de Pollo"]})
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory",
                        lambda uid, ings, **k: visto["descuentos"].append(ings) or {"not_in_pantry": []})
    out = _fn(tools.correct_consumed_meal)("u1", "m1", ingredients=["200 g de Pollo"])
    return out, visto


def test_corregir_con_la_nevera_apagada_no_toca_el_inventario(monkeypatch):
    """[Final fix wave] Apagada, corregir NO toca el inventario: ni devuelve lo que el registro original descontó ni
    descuenta la lista corregida. Antes devolvía sin volver a descontar, y el inventario oculto quedaba como si la
    comida nunca hubiera ocurrido — la comida SÍ ocurrió, solo cambió de ingredientes."""
    out, visto = _corregir(monkeypatch, activa=False)
    assert out.startswith("¡Corregido!") and len(visto["updates"]) == 1
    assert visto["reverts"] == [] and visto["descuentos"] == [], "apagada, el inventario oculto queda como estaba"
    assert "nevera" not in out.lower(), out


def test_corregir_con_la_nevera_activa_ajusta_como_siempre(monkeypatch):
    out, visto = _corregir(monkeypatch, activa=True)
    assert visto["reverts"] == ["m1"] and visto["descuentos"] == [["200 g de Pollo"]] and "Nevera ajustada" in out


def test_proponer_comida_apagada_no_lee_el_inventario(monkeypatch):
    import db
    import db_facts
    import plan_mode
    import tools
    consultas = []

    def _q(sql, *a, **k):
        consultas.append(sql)
        return [{"ingredient_name": "Avena"}]
    monkeypatch.setattr(db, "execute_sql_query", _q)
    monkeypatch.setattr(tools, "get_user_profile", lambda uid: {"health_profile": {}})
    monkeypatch.setattr(tools, "_hora_local_float", lambda uid: 8.0)
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: 240)
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda uid: "2026-09-23")
    monkeypatch.setattr(plan_mode, "get_plan_mode", lambda uid: {"plan_mode": "tracking"})
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: [])
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: False)
    ctx = tools._contexto_del_dia_para_propuesta("u1")
    assert ctx["nevera"] == [] and ctx["nevera_activa"] is False
    assert not any("user_inventory" in s for s in consultas), consultas
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: True)
    ctx = tools._contexto_del_dia_para_propuesta("u1")
    assert ctx["nevera"] == ["Avena"] and ctx["nevera_activa"] is True


_PLATO = {"name": "Avena con claras", "calories": 500, "protein": "35g", "carbs": "60g", "fats": "12g",
          "_minutos": 10, "_lineas": [(70.0, "Avena"), (150.0, "Clara de huevo")], "_tiene": [],
          "_falta": ["Avena", "Clara de huevo"], "recipe": ["Cocina la avena.", "Añade las claras."]}


def test_el_formateador_no_nombra_la_nevera_apagada():
    import coach_day_context as cdc
    obj = {"kcal": 500, "protein_g": 35}
    textos = {
        "normal": cdc.formatear_propuestas([dict(_PLATO)], "desayuno", obj, con_nevera=False, nevera_activa=False),
        "solo_con_lo_que_tiene": cdc.formatear_propuestas([dict(_PLATO)], "cena", obj, con_nevera=False,
                                                          solo_nevera=True, nevera_activa=False),
        "sin_platos": cdc.formatear_propuestas([], "desayuno", obj, con_nevera=False, nevera_activa=False),
        "contradictorio": cdc.formatear_propuestas([dict(_PLATO)], "desayuno", obj, con_nevera=True,
                                                   nevera_activa=False),
    }
    for caso, txt in textos.items():
        assert "nevera" not in txt.lower(), (caso, txt)
    assert "no sabes qué hay en su casa" in textos["solo_con_lo_que_tiene"], "sin inventario no se finge saberlo"
    assert "No inventes una receta «del catálogo»" in textos["sin_platos"]
    # con la Nevera activa (el default) el texto es el de siempre
    siempre = cdc.formatear_propuestas([dict(_PLATO)], "desayuno", obj, con_nevera=False)
    assert "Su Nevera está vacía EN LA APP" in siempre and "Si le falta algo de la Nevera, dilo." in siempre
    assert siempre == cdc.formatear_propuestas([dict(_PLATO)], "desayuno", obj, con_nevera=False, nevera_activa=True)


def _proponer(monkeypatch, activa: bool):
    import coach_day_context as cdc
    import tools
    visto = {}
    ctx = {"hp": {}, "plan": None, "diario": [], "hora": 8.0, "faltan": [], "n_comidas": 4,
           "nevera": ["Avena"] if activa else [], "nevera_activa": activa}
    monkeypatch.setattr(tools, "_contexto_del_dia_para_propuesta", lambda uid: dict(ctx))
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda uid: "2026-09-23")
    monkeypatch.setattr(cdc, "proponer_comidas", lambda *a, **k: visto.update(k) or [dict(_PLATO)])
    out = _fn(tools.proponer_comida)("u1", meal_type="desayuno", kcal_objetivo=500)
    return out, visto


def test_proponer_comida_con_la_nevera_apagada_no_la_nombra(monkeypatch):
    out, visto = _proponer(monkeypatch, activa=False)
    assert out.startswith("PROPUESTAS DE DESAYUNO") and visto["nevera_nombres"] == []
    assert "nevera" not in out.lower(), out
    out, _ = _proponer(monkeypatch, activa=True)
    assert "   Nevera: tiene 0 de 2; le falta: Avena, Clara de huevo" in out


def test_fui_al_super_con_la_nevera_apagada_no_ofrece_anotarlo_en_ella(monkeypatch):
    import tools
    monkeypatch.setattr(tools, "_usuario_en_modo_contador", lambda uid: True)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan_with_id", _prohibido)
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: False)
    out = _fn(tools.mark_shopping_list_purchased)(user_id="u1")
    assert "NO digas que registraste la compra" in out, "la guarda de honestidad del contador se queda"
    assert no.MENSAJE_NEVERA_APAGADA in out and "modify_pantry_inventory" not in out
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: True)
    assert "modify_pantry_inventory" in _fn(tools.mark_shopping_list_purchased)(user_id="u1")


# ── 6b. Revisión de B4: el respaldo del formulario, la foto de compra y el orden del bloque ──────────────────────
def test_la_orden_no_depende_del_kill_switch_del_snapshot(monkeypatch):
    import agent
    monkeypatch.setenv("MEALFIT_CHAT_PANTRY_SNAPSHOT", "false")
    monkeypatch.setattr(no, "nevera_activa", lambda uid: False)
    assert agent._build_pantry_context("u1") == no.BLOQUE_PROMPT_NEVERA_APAGADA
    monkeypatch.setattr(no, "nevera_activa", lambda uid: True)
    assert agent._build_pantry_context("u1") == "", "encendida, el kill switch del snapshot manda como siempre"
    # el estado que el camino del chat ya resolvió manda: no se vuelve a consultar
    monkeypatch.setattr(no, "nevera_activa", lambda uid: pytest.fail("con nevera_on dado no se consulta"))
    assert agent._build_pantry_context("u1", nevera_on=False) == no.BLOQUE_PROMPT_NEVERA_APAGADA


def test_el_bloque_de_inventario_apagado_no_escribe_la_linea_del_inventario():
    from prompts.chat_agent import build_inventory_context as bic
    apagada = bic("3 unidades de Huevo", "1 lb de Pollo", plan_en_pausa=True, nevera_activa=False)
    assert "INVENTARIO FÍSICO ACTUAL" not in apagada and "Huevo" not in apagada
    assert "[LISTA DEL PLAN EN PAUSA]: 1 lb de Pollo" in apagada, "la parte de compras no cambia"
    assert bic("3 unidades de Huevo", "", sin_plan=True, nevera_activa=False) == "", "ni «Vacío» ni cabecera suelta"
    for args, kw in ((("3 unidades de Huevo", "1 lb de Pollo"), {}), (("", "1 lb de Pollo"), {"plan_en_pausa": True}),
                     (("", ""), {}), (("3 unidades de Huevo", ""), {"sin_plan": True})):
        assert bic(*args, **kw) == bic(*args, **kw, nevera_activa=True)   # encendida: el texto de siempre
    assert "[INVENTARIO FÍSICO ACTUAL]: Vacío." in bic("", "1 lb de Pollo")


_FOTOS_CON_NEVERA = {
    "compra_sin_texto": {"kind": "items", "description": "2 manzanas, 1 lb de pollo", "has_text": False},
    "compra_con_texto": {"kind": "items", "description": "2 manzanas", "has_text": True},
    "etiqueta": {"kind": "etiqueta", "description": "1 scoop (31 g): 120 kcal, 24 g", "has_text": True},
    "varias": {"kind": "multi", "has_text": False, "items": [
        {"kind": "items", "description": "2 manzanas"}, {"kind": "etiqueta", "description": "120 kcal por scoop"},
        {"kind": "plato", "description": "arroz con pollo"}]},
}


def test_la_foto_de_compra_con_la_nevera_apagada_no_la_ofrece():
    from prompts.chat_agent import build_vision_context as bvc
    for caso, foto in _FOTOS_CON_NEVERA.items():
        apagada = bvc(foto, nevera_activa=False)
        assert apagada.startswith("\n\n📷 CONTEXTO DE"), caso
        assert "nevera" not in apagada.lower() and "modify_pantry_inventory" not in apagada, (caso, apagada)
        assert bvc(foto) == bvc(foto, nevera_activa=True), caso
    assert "ofrécele registrar ESE plato" in bvc(_FOTOS_CON_NEVERA["compra_sin_texto"], nevera_activa=False)
    assert "NO registres esto como comida consumida" in bvc(_FOTOS_CON_NEVERA["compra_sin_texto"], nevera_activa=False)
    assert "cifras de la etiqueta × porciones" in bvc(_FOTOS_CON_NEVERA["etiqueta"], nevera_activa=False)
    # encendida (el default): las frases de siempre, al pie de la letra
    assert bvc(_FOTOS_CON_NEVERA["compra_con_texto"]).endswith(
        " Si el usuario quiere, agrégalos a su Nevera con modify_pantry_inventory tras su confirmación. Responde a su "
        "mensaje.")
    assert "y pregúntale si quiere que los agregues a su Nevera. SOLO cuando el usuario confirme, usa la herramienta " \
           "modify_pantry_inventory con items_to_add" in bvc(_FOTOS_CON_NEVERA["compra_sin_texto"])
    assert bvc(_FOTOS_CON_NEVERA["etiqueta"]).endswith(" NO lo ofrezcas para la Nevera salvo que él lo pida.")
    assert (" Para las fotos de compra, ofrece agregarlas a la Nevera y usa modify_pantry_inventory solo después de "
            "confirmación.") in bvc(_FOTOS_CON_NEVERA["varias"])


class _PromptCapturado(RuntimeError):
    pass


def _prompt_del_coach(monkeypatch, path: str, activa: bool, inventario=(), vision=None) -> str:
    """El system prompt ENTERO de un turno autenticado por el camino real (`chat_with_agent` o `_stream`), cortado en el
    grafo — el arnés de `test_p1_coach_country_unnamed.py`. Sin DB (`db_core.connection_pool = None`: toda lectura
    revienta al instante y cae a su fallo abierto, también en un checkout con `.env`) y sin LLM (sentimiento, router
    RAG y grafo falsos). El `form_data` trae el `current_pantry_ingredients` que la última generación dejó en
    `health_profile`; el agregador del respaldo se deja pasar tal cual (sin catálogo lo vaciaría y no se vería nada)."""
    from types import SimpleNamespace
    import agent
    import db_core
    import db_inventory
    import db_plans
    import shopping_calculator
    from prompts.sentiment import PERSONALITY_PROFILES

    capturado = {}

    class _Grafo:
        def get_state(self, _config):
            return SimpleNamespace(values={})

        def invoke(self, inputs, **_k):
            capturado["prompt"] = inputs["sys_prompt"]
            raise _PromptCapturado

        stream = invoke

    class _Builder:
        def compile(self, **_k):
            return _Grafo()

    monkeypatch.setattr(no, "nevera_activa", lambda uid: activa)
    monkeypatch.setattr(db_core, "connection_pool", None)
    monkeypatch.setattr(db_inventory, "get_user_inventory", lambda uid: list(inventario))
    monkeypatch.setattr(db_plans, "get_latest_usable_meal_plan_with_id", lambda uid: None)
    monkeypatch.setattr(shopping_calculator, "aggregate_shopping_list", lambda items, **k: list(items))
    monkeypatch.setattr(agent, "build_memory_context", lambda *_a: {"recent_messages": [], "summary_context": ""})
    monkeypatch.setattr(agent, "classify_sentiment",
                        lambda _p: {**PERSONALITY_PROFILES["neutral"], "sentiment": "neutral"})
    monkeypatch.setattr(agent, "rag_query_router", lambda _p: {"skip": True})
    monkeypatch.setattr(agent, "_emit_chat_stream_total_duration_best_effort", lambda *_a: None)
    monkeypatch.setattr(agent, "chat_builder", _Builder())
    monkeypatch.setattr(agent, "chat_checkpoint_pool", None)
    monkeypatch.setattr(agent, "connection_pool", None)
    kwargs = dict(session_id="sesion-nevera", prompt="¿Qué ceno hoy?", user_id="u-nevera",
                  form_data={"current_pantry_ingredients": ["2 lbs de Salchichón Testigo", "1 lb de Queso Testigo"]})
    with pytest.raises(_PromptCapturado):
        if path == "stream":
            list(agent.chat_with_agent_stream(**kwargs, vision=vision))
        else:
            agent.chat_with_agent(**kwargs)
    return capturado["prompt"]


@pytest.mark.parametrize("path", ("nonstream", "stream"))
def test_con_la_nevera_apagada_el_prompt_no_trae_la_nevera_vieja_del_formulario(monkeypatch, path):
    """[revisión B4] Con la Nevera apagada el inventario queda vacío y el respaldo desde `form_data` corría SIEMPRE:
    la Nevera de la última renovación salía como «[INVENTARIO FÍSICO ACTUAL] … PRIORIZA SIEMPRE recomendar cocinar
    con esto», contra el bloque que ordena no mencionarla."""
    apagada = _prompt_del_coach(monkeypatch, path, activa=False, inventario=["3 unidades de Zarzamora Testigo"])
    for fuga in ("Salchichón Testigo", "Queso Testigo", "Zarzamora Testigo", "INVENTARIO FÍSICO ACTUAL",
                 "NEVERA FÍSICA AHORA"):
        assert fuga not in apagada, fuga
    assert no.BLOQUE_PROMPT_NEVERA_APAGADA in apagada
    # encendida: el arnés SÍ ve el inventario real y, sin él, el respaldo del formulario (si no, el test no probaría nada)
    real = _prompt_del_coach(monkeypatch, path, activa=True, inventario=["3 unidades de Zarzamora Testigo"])
    assert "[INVENTARIO FÍSICO ACTUAL]: 3 unidades de Zarzamora Testigo." in real
    respaldo = _prompt_del_coach(monkeypatch, path, activa=True)
    assert "[INVENTARIO FÍSICO ACTUAL]: 2 lbs de Salchichón Testigo, 1 lb de Queso Testigo." in respaldo
    assert no.BLOQUE_PROMPT_NEVERA_APAGADA not in respaldo


def test_la_foto_de_compra_llega_al_prompt_sin_ofrecer_la_nevera_apagada(monkeypatch):
    foto = _FOTOS_CON_NEVERA["compra_sin_texto"]
    apagada = _prompt_del_coach(monkeypatch, "stream", activa=False, vision=foto)
    assert "ofrécele registrar ESE plato" in apagada
    assert "quiere que los agregues a su Nevera" not in apagada
    encendida = _prompt_del_coach(monkeypatch, "stream", activa=True, vision=foto)
    assert "quiere que los agregues a su Nevera" in encendida


# ── 7. La documentación ──────────────────────────────────────────────────────────────────────────────────────
def test_el_doc_canonico_existe_y_nombra_lo_que_opera():
    doc = (_BACKEND / "docs" / "nevera_opcional.md").read_text(encoding="utf-8")
    for trozo in ("nevera_activa_de", "MEALFIT_NEVERA_SWITCH", "MEALFIT_NEVERA_AUTO_OFF",
                  "MEALFIT_NEVERA_AUTO_OFF_HOURS", "nevera_auto_off", "p1_nevera_opcional_2026_09_23.sql",
                  "/api/user/preferences/nevera", "nevera_reloj_desde"):
        assert trozo in doc, trozo


def test_sin_la_migracion_la_tarjeta_aparece_y_el_patch_da_500(monkeypatch):
    """[Final fix wave] Lo que el doc (§6) afirma de un despliegue SIN la migración, medido: las lecturas fallan
    abiertas (activa, sin error), pero la tarjeta de Configuración sí aparece (`disponible` sale del knob, no de la
    columna) y su PATCH responde 500. El doc decía «nunca un 500»."""
    from fastapi import HTTPException
    from routers import preferences as pref

    def _sin_columna(*a, **k):
        raise RuntimeError('column "nevera_enabled" does not exist')
    monkeypatch.setattr(no, "execute_sql_query", _sin_columna)
    monkeypatch.setattr(no, "execute_sql_write", _sin_columna)
    assert no.nevera_activa("u1") is True
    assert no.estado_nevera("u1") == {"enabled": None, "activa": True, "auto_off_at": None, "disponible": True}
    with pytest.raises(HTTPException) as e:
        asyncio.run(pref.api_set_nevera(body=pref.NeveraPreferenceBody(enabled=False), verified_user_id="u1"))
    assert e.value.status_code == 500
    doc = (_BACKEND / "docs" / "nevera_opcional.md").read_text(encoding="utf-8")
    assert "nunca un 500" not in doc and "responde **500**" in doc


def test_claude_md_apunta_al_doc():
    claude = _BACKEND.parent / "CLAUDE.md"
    if not claude.exists():
        pytest.skip("sin el CLAUDE.md de la raíz al lado")
    txt = claude.read_text(encoding="utf-8")
    assert "[P1-NEVERA-OPCIONAL · 2026-09-23]" in txt and "backend/docs/nevera_opcional.md" in txt


# [Fix round 1] La pasada doc-first de P1-PLAN-MODE (P3-CLAUDEMD-MARGIN-RESTORE) movió su párrafo a un doc propio;
# esto ancla que el destino existe y conserva el knob que gatea la pausa.
def test_el_doc_de_plan_mode_existe_y_conserva_el_knob():
    assert "MEALFIT_PLAN_MODE_SWITCH" in (_BACKEND / "docs" / "plan_mode.md").read_text(encoding="utf-8")
