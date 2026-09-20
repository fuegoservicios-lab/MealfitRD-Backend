"""[P1-PLAN-LOTE-137 · 2026-09-20] La app ENTERA con el generador de planes apagado, sin contradicciones.

El dueño, tras el lote 136 (Configuración): «quiero saber si en general con el generador apagado todo está al 100 % listo
para producción». Cinco auditorías en paralelo (rutas y efectos globales, Nevera + Historial, Agente IA, crons y
notificaciones, ciclo de vida de la cuenta), cada hallazgo verificado a mano. Lo del servidor que este fichero ancla:

  1. `POST /generation-runs` ENCIENDE el modo antes de encolar. El gate H1 del pickup (`plan_mode='tracking'` no genera)
     dejaba colgado para siempre el chunk 0 de quien pulsaba «Encender el plan» desde el contador: el reencendido vivía
     solo en el postprocess, que en la cola corre DESPUÉS del pickup.
  2. Encender por esa vía también retira el sello `paused_by_user` (un UPDATE, dos llamadores).
  3. `/restore` vuelve terminales los chunks firmados por la pausa: «Reactivar» en modo contador los dejaba revivibles
     sobre el contenido restaurado.
  4. El coach: macros de HOY del contador (no del plan pausado), sin índice «lo que el plan MANDABA» en pausa, el tercer
     caso «contador SIN plan», sin «su plan actual» a quien no tiene plan, y sin «Día Clásico» sembrado.
  5. Las tools de compras del chat se niegan / reencuadran con el generador apagado.
  6. Tres endpoints que reviven bloques rechazan con 409 en pausa; el escalado de «chunk atascado» ignora a los contadores.
  7. El bot de ayuda conoce el modo contador y cita los créditos reales.
"""
import os
import re
import sys

import pytest

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _src(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


# ───────────────────────── 1 · generar enciende ANTES de encolar ─────────────────────────

def test_generation_run_enciende_el_modo_antes_de_encolar():
    s = _src("routers/plans_generation.py")
    cuerpo = s[s.index("async def api_create_generation_run"):s.index('@router.get("/{run_id}")')]
    i_flag = cuerpo.index("ensure_plan_generation_enabled, user_id")
    i_cola = cuerpo.index("create_placeholder_plan_and_enqueue_initial,\n            user_id=user_id")
    assert i_flag < i_cola, "la bandera va PRIMERO: encolar con el gate H1 puesto deja un chunk 0 que nadie recoge"
    # si encolar falla, el usuario vuelve a SU contador (no se queda en «modo plan sin plan»)
    i_except = cuerpo.index("no se pudo encolar el chunk 0")
    assert "pause_plan_generation" in cuerpo[i_except:i_except + 900]
    assert "_encendido_aqui" in cuerpo[i_except:i_except + 400]


def test_el_gate_h1_sigue_en_el_pickup():
    """El arreglo NO es quitar el freno: es encender antes. El gate del pickup es LA capa que detiene el gasto."""
    import plan_mode
    assert "up.plan_mode = 'tracking'" in plan_mode.PICKUP_GATE_SQL
    assert _src("cron_tasks.py").count("__PLAN_MODE_GATE__") >= 3


# ───────────────────────── 2 · encender retira el sello de la pausa ─────────────────────────

def test_ensure_enabled_retira_el_sello_solo_si_encendio(monkeypatch):
    import plan_mode
    sqls = []

    def _write(sql, params=None, returning=False):
        sqls.append(" ".join(str(sql).split()))
        if "UPDATE user_profiles" in sql:
            return [{"id": "u1"}]
        return []

    monkeypatch.setattr(plan_mode, "execute_sql_write", _write)
    monkeypatch.setattr(plan_mode, "PLAN_MODE_SWITCH_ENABLED", True)
    assert plan_mode.ensure_plan_generation_enabled("u1") is True
    assert any("UPDATE meal_plans" in q and "'paused_by_user'" in q and "_paused_prev_generation_status" in q for q in sqls)
    # la cola NO se revive aquí: el usuario pidió un plan NUEVO (lote 136)
    assert not any("plan_chunk_queue" in q for q in sqls)

    sqls.clear()
    monkeypatch.setattr(plan_mode, "execute_sql_write", lambda *a, **k: [])
    assert plan_mode.ensure_plan_generation_enabled("u1") is False
    assert sqls == []


def test_resume_y_ensure_comparten_el_mismo_update():
    s = _src("plan_mode.py")
    assert s.count("def _restore_paused_plan_status(") == 1
    assert s.count("_restore_paused_plan_status(user_id)") >= 2
    assert s.count("plan_data - '_paused_at' - '_paused_prev_generation_status'") == 1, "un solo UPDATE, dos llamadores"


# ───────────────────────── 3 · /restore y los chunks firmados por la pausa ─────────────────────────

def test_restore_vuelve_terminales_los_chunks_de_la_pausa():
    s = _src("routers/plans.py")
    from routers.plans import api_restore_plan
    import inspect
    src = inspect.getsource(api_restore_plan)
    assert "from plan_mode import PAUSE_CANCEL_REASON as _PAUSA_FIRMA" in src
    # los DOS cancels (target y source) cubren también las filas firmadas por la pausa, y les quitan la firma
    assert src.count("OR (status = 'cancelled' AND dead_letter_reason = %s") == 2
    assert src.count("WHEN status = 'cancelled' THEN %s") == 2
    assert '("restore_overwrite", "restore_overwrite", target_plan_id, _PAUSA_FIRMA)' in src
    assert '("restore_source_archived", "restore_source_archived", source_plan_id, _PAUSA_FIRMA)' in src
    # …sin tocar el SSOT de 5 estados vivos ni añadir una tercera sentencia sobre la cola
    assert src.count("UPDATE plan_chunk_queue") == 2
    assert src.count("dead_lettered_at = COALESCE(") == 2


def test_el_revive_exige_justo_lo_que_restore_quita():
    """`_revive_paused_chunks` busca firma + `dead_lettered_at IS NULL`: una fila con `dead_lettered_at` ya no revive."""
    s = _src("plan_mode.py")
    revive = s[s.index("def _revive_paused_chunks"):s.index("def resume_plan_generation")]
    assert "AND dead_letter_reason = %s" in revive and "AND dead_lettered_at IS NULL" in revive


# ───────────────────────── 4 · el coach ─────────────────────────

_PERFIL = {"weight": 170, "weightUnit": "lb", "height": 175, "age": 30, "gender": "male",
           "activityLevel": "moderate", "mainGoal": "lose_fat"}


def test_macros_de_hoy_salen_del_contador_con_el_plan_en_pausa():
    from agent import _macro_totals_line
    from nutrition_calculator import get_nutrition_targets
    consumido = [{"protein": 40, "carbs": 50, "healthy_fats": 10}]
    macros = (get_nutrition_targets(dict(_PERFIL)) or {}).get("macros") or {}
    prot = round(float(macros.get("protein_g") or 0))
    assert prot > 0
    linea = _macro_totals_line(consumido, None, dict(_PERFIL))      # plan_vigente=None ⇒ en pausa o sin plan
    assert f"(meta {prot}g)" in linea
    assert "gg" not in linea


def test_macros_del_plan_vigente_sin_la_g_doble():
    from agent import _macro_totals_line
    linea = _macro_totals_line([{"protein": 10, "carbs": 20, "healthy_fats": 5}],
                               {"macros": {"protein": "134g", "carbs": "250g", "fats": "60g"}}, None)
    assert "(meta 134g)" in linea and "(meta 250g)" in linea and "(meta 60g)" in linea
    assert "gg" not in linea


def test_los_dos_paths_pasan_el_plan_vigente():
    s = _src("agent.py")
    assert s.count("system_prompt += _macro_totals_line(consumed_today, plan_vigente, form_data)") == 2
    assert "_macro_totals_line(consumed_today, current_plan)" not in s


def test_en_pausa_no_hay_indice_de_lo_que_el_plan_mandaba(monkeypatch):
    import agent
    import db_facts
    llamadas = []
    monkeypatch.setattr(agent, "build_past_plan_days_block", lambda *a, **k: llamadas.append("plan") or "IDX-PLAN")
    diario = {}
    monkeypatch.setattr(agent, "build_past_diary_block",
                        lambda rows, today, **k: diario.update(k) or " DIARIO")
    monkeypatch.setattr(agent, "_build_pending_days_lines_block", lambda *a, **k: "")
    monkeypatch.setattr(db_facts, "get_consumed_meals_since", lambda *a, **k: [])
    monkeypatch.setattr(db_facts, "get_plan_meal_deviations_since", lambda *a, **k: [])

    pausado = {"generation_status": "paused_by_user", "days": [{"meals": []}]}
    out = agent._build_past_days_context("u1", pausado, local_date_str="2026-09-20", tz_offset=240)
    assert llamadas == [] and "IDX-PLAN" not in out
    assert "DIARIO" in out, "el diario multi-día es la memoria del contador: sigue"
    assert diario.get("plan_data") is None

    vivo = {"generation_status": "complete", "days": [{"meals": []}]}
    out = agent._build_past_days_context("u1", vivo, local_date_str="2026-09-20", tz_offset=240)
    assert llamadas == ["plan"] and "IDX-PLAN" in out
    assert diario.get("plan_data") is vivo


def test_contador_sin_plan_es_un_caso_propio(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "_plan_mode_for_chat", lambda uid: "tracking")
    assert agent._contador_sin_plan_para_prompt("u1", None) is True
    assert agent._contador_sin_plan_para_prompt("u1", {"days": [1]}) is False    # con plan manda `plan_en_pausa`
    assert agent._contador_sin_plan_para_prompt("guest", None) is False
    monkeypatch.setattr(agent, "_plan_mode_for_chat", lambda uid: "plan")
    assert agent._contador_sin_plan_para_prompt("u1", None) is False

    def _boom(uid):
        raise RuntimeError("db")
    monkeypatch.setattr(agent, "_plan_mode_for_chat", _boom)
    assert agent._contador_sin_plan_para_prompt("u1", None) is False, "fail-open al comportamiento histórico"


def test_el_bullet_del_contador_nombra_la_puerta_real():
    from prompts import chat_agent as p
    if p._plan_tools_enabled():
        pytest.skip("con las tools de plan encendidas los bullets son otros")
    contador = p._plan_tools_bullets_inline(False, True)
    assert contador == p._plan_tools_bullets_stream(False, True)
    assert "Configuración → Capacidades" in contador and "APAGADA" in contador
    assert "usa los botones de la página Plan" not in contador
    pausa = p._plan_tools_bullets_inline(True, False)
    assert "Historial" in pausa and "Capacidades" in pausa
    normal = p._plan_tools_bullets_inline(False, False)
    assert "página Plan" in normal and "Capacidades" not in normal
    s = _src("agent.py")
    assert s.count("contador_sin_plan=_contador_sin_plan") == 4, "los cuatro call sites de los dos paths"


def test_sin_plan_no_se_afirma_un_plan_actual():
    from prompts.chat_agent import build_inventory_context
    assert "plan actual" not in build_inventory_context("Arroz, Huevo", "", sin_plan=True)
    assert "plan actual" in build_inventory_context("Arroz, Huevo", "")
    assert "plan actual" not in build_inventory_context("Arroz, Huevo", "", plan_en_pausa=True)
    assert _src("agent.py").count("sin_plan=not current_plan") == 2


def test_sin_horario_declarado_no_hay_dia_clasico():
    from prompts.chat_agent import build_circadian_context
    assert "Día Clásico" not in build_circadian_context("")
    assert "NO ha dicho" in build_circadian_context("")
    assert "Día Clásico" in build_circadian_context("standard")
    assert "Turno Nocturno" in build_circadian_context("night_shift")
    s = _src("agent.py")
    assert 'form_data.get("scheduleType", "standard")' not in s
    assert s.count('(form_data.get("scheduleType") or "")') == 2


def test_la_alerta_de_la_tarde_no_manda_a_una_tool_que_no_existe():
    import agent
    out = agent._build_today_remaining_context(None, [], 2000, 1500.0)
    assert "ALERTA DE MICRO-ADAPTACIÓN" in out
    assert "modify_single_meal" not in out and "déficit" not in out


# ───────────────────────── 5 · tools de compras ─────────────────────────

def _fn(tool):
    return getattr(tool, "func", None) or tool


def test_fui_al_super_no_mete_la_lista_del_plan_pausado(monkeypatch):
    import tools

    def _prohibido(*a, **k):
        raise AssertionError("en modo contador no se lee el plan ni se toca la Nevera")
    monkeypatch.setattr(tools, "_usuario_en_modo_contador", lambda uid: True)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan_with_id", _prohibido)
    out = _fn(tools.mark_shopping_list_purchased)(user_id="u1")
    assert "APAGADA" in out and "modify_pantry_inventory" in out and "NO digas que registraste" in out


def test_lista_de_compras_en_contador_va_con_su_encuadre(monkeypatch):
    import tools
    monkeypatch.setattr(tools, "_usuario_en_modo_contador", lambda uid: True)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan", lambda uid: None)
    out = _fn(tools.check_shopping_list)(user_id="u1")
    assert "Capacidades" in out and "no tiene ningún plan" in out
    monkeypatch.setattr(tools, "_usuario_en_modo_contador", lambda uid: False)
    assert "no tiene un plan de comidas activo" in _fn(tools.check_shopping_list)(user_id="u1")
    assert "LISTA DEL PLAN EN PAUSA" in _src("tools.py")


def test_el_helper_de_modo_falla_abierto(monkeypatch):
    import plan_mode
    import tools
    assert tools._usuario_en_modo_contador("guest") is False
    assert tools._usuario_en_modo_contador("") is False
    monkeypatch.setattr(plan_mode, "get_plan_mode", lambda uid: {"plan_mode": "tracking"})
    assert tools._usuario_en_modo_contador("u1") is True

    def _boom(uid):
        raise RuntimeError("db")
    monkeypatch.setattr(plan_mode, "get_plan_mode", _boom)
    assert tools._usuario_en_modo_contador("u1") is False


# ───────────────────────── 6 · endpoints y crons ─────────────────────────

def test_los_tres_endpoints_que_reviven_bloques_se_niegan_en_pausa(monkeypatch):
    import plan_mode
    from fastapi import HTTPException
    from routers import plans
    monkeypatch.setattr(plan_mode, "get_plan_mode", lambda uid: {"plan_mode": "tracking"})
    with pytest.raises(HTTPException) as e:
        plans._rechazar_si_generador_apagado("u1")
    assert e.value.status_code == 409 and "Capacidades" in str(e.value.detail)
    monkeypatch.setattr(plan_mode, "get_plan_mode", lambda uid: {"plan_mode": "plan"})
    assert plans._rechazar_si_generador_apagado("u1") is None

    def _boom(uid):
        raise RuntimeError("db")
    monkeypatch.setattr(plan_mode, "get_plan_mode", _boom)
    assert plans._rechazar_si_generador_apagado("u1") is None, "modo ilegible ⇒ el endpoint sigue como siempre"

    s = _src("routers/plans.py")
    assert s.count("_rechazar_si_generador_apagado(verified_user_id)") == 3
    for ancla in ("def api_retry_chunk(", "def api_regenerate_dead_lettered_simplified(", "def api_regen_degraded_chunks("):
        cuerpo = s[s.index(ancla):s.index(ancla) + 5200]
        assert "_rechazar_si_generador_apagado(verified_user_id)" in cuerpo, ancla
    # la guarda va ANTES de la primera escritura del retry
    retry = s[s.index("def api_retry_chunk("):]
    assert retry.index("_rechazar_si_generador_apagado(") < retry.index("UPDATE plan_chunk_queue")


def test_el_escalado_de_atascados_ignora_a_los_contadores():
    s = _src("cron_tasks.py")
    i = s.index("# 1. Detectar stuck (>24h sin pickup)")
    sql = s[i:s.index('""", fetch_all=True', i)]
    assert "status IN ('pending', 'stale')" in sql and "escalated_at" in sql
    assert "up.plan_mode = 'tracking'" in sql and "NOT EXISTS" in sql


# ───────────────────────── 7 · bot de ayuda ─────────────────────────

def test_el_bot_de_ayuda_conoce_el_contador_y_los_creditos_reales():
    from prompts.help_bot import help_bot_system_prompt
    texto = help_bot_system_prompt("es-DO")
    assert "Modo contador" in texto and "Configuración → Capacidades" in texto and "**Progreso**" in texto
    auth = _src("auth.py")
    for tier, etiqueta in (("GRATIS", "Gratuito"), ("BASIC", "Básico"), ("PLUS", "Plus"), ("ULTRA", "Max")):
        m = re.search(r'_env_int\("MEALFIT_TIER_LIMIT_%s", (\d+)\)' % tier, auth)
        assert m, tier
        linea = next(ln for ln in texto.splitlines() if ln.startswith(f"- **{etiqueta}**"))
        assert f"{m.group(1)} créditos de IA al mes" in linea, (tier, linea)
    assert "15 usos" not in texto and "ilimitado" not in texto


# ───────────────────────── marcador ─────────────────────────

def test_marker_del_lote():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 137
