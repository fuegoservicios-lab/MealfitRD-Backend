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
    ({"plan_mode": "plan", "nevera_enabled": False}, True),       # el plan la necesita: vuelve sola
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


def test_con_la_nevera_activa_el_prompt_no_cambia(monkeypatch):
    import agent
    monkeypatch.setattr(no, "nevera_activa", lambda uid: True)
    monkeypatch.setattr("db.execute_sql_query", lambda *a, **k: [])
    assert "NEVERA FÍSICA AHORA: vacía" in agent._build_pantry_context("u1")


def test_los_dos_caminos_del_chat_no_leen_el_inventario_apagado():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert src.count("get_user_inventory(user_id) if nevera_activa(user_id) else []") == 2


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


def test_corregir_con_la_nevera_apagada_devuelve_lo_anterior_y_no_descuenta(monkeypatch):
    out, visto = _corregir(monkeypatch, activa=False)
    assert out.startswith("¡Corregido!") and len(visto["updates"]) == 1
    assert visto["reverts"] == ["m1"], "lo descontado cuando estaba encendida se devuelve: inventario oculto coherente"
    assert visto["descuentos"] == [] and "nevera" not in out.lower(), out


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
    assert out == no.MENSAJE_NEVERA_APAGADA and "modify_pantry_inventory" not in out
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: True)
    assert "modify_pantry_inventory" in _fn(tools.mark_shopping_list_purchased)(user_id="u1")
