# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-223 · 2026-09-24] Los recordatorios de comida, a la hora que elige la persona.

El dueño, a la 1:18 p. m.: «hoy nada más me llegó la notificación del desayuno… son la 1 de la tarde y todavía tiene
la notificación del desayuno». Dos defectos, uno sobre otro:

  1. La hora salía de lo REGISTRADO. `consumed_at` es la hora del registro y él anota después de comer (el 23-sep,
     desayuno y almuerzo juntos a la 1:36 p. m.): su almuerzo se programaba hacia las 2:15 (el techo del lote 151).
     Ahora cada comida tiene su interruptor y su hora en Configuración; sin tocar nada, las normales: 8:45, 12:45,
     15:45 y 19:15. El cálculo por historial queda detrás de `MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY` (apagado).
  2. El chat llegaba DESPUÉS que el teléfono. El cron corría a y media y escribía en el tick de la hora del aviso:
     con un aviso a las 2:15 el teléfono sonaba a las 2:15 y el mensaje llegaba a las 2:30, así que al tocar la
     notificación el chat seguía en el desayuno. Con la cena por defecto (19:15) le pasaba a todos. Ahora el cron
     corre cada 15 min y escribe en el último tick ANTES de que suene el teléfono.
"""
from __future__ import annotations

import re
from datetime import datetime as _dt_real, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"

UID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
SID = "11111111-2222-4333-8444-555555555555"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _rd(hora: int, minuto: int, dia_mes: int = 24) -> _dt_real:
    """Instante UTC de las `hora:minuto` en RD (UTC-4), con los 5 s de un tick real."""
    return _dt_real(2026, 9, dia_mes, hora + 4, minuto, 5, tzinfo=timezone.utc) if hora + 4 < 24 else \
        _dt_real(2026, 9, dia_mes + 1, hora + 4 - 24, minuto, 5, tzinfo=timezone.utc)


# ───────────────────────── 1. La hora: elegida, o la normal ─────────────────────────

@pytest.mark.parametrize("comida,esperada", [("Desayuno", 8.75), ("Almuerzo", 12.75), ("Merienda", 15.75), ("Cena", 19.25)])
def test_sin_tocar_nada_suenan_las_horas_normales_y_no_se_consulta_lo_registrado(comida, esperada, monkeypatch):
    import db_facts
    import proactive_agent as pa
    monkeypatch.delenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", raising=False)

    def _no_debe_leerse(*_a, **_k):
        raise AssertionError("la hora por defecto no sale de lo registrado")

    monkeypatch.setattr(db_facts, "get_avg_meal_hour", _no_debe_leerse)
    def_hour = pa.HORAS_POR_DEFECTO_DE_COMIDA[comida]
    assert pa.hora_del_aviso("u", comida, def_hour, {}) == pytest.approx(esperada)
    assert pa.hora_del_aviso("u", comida, def_hour, None) == pytest.approx(esperada), "sin perfil, igual"


def test_el_caso_del_duenio_su_almuerzo_ya_no_se_corre_a_las_2(monkeypatch):
    """Anotaba el almuerzo a la 1:36 y a las 3:20: con el historial, el techo lo dejaba en las 2:15."""
    import db_facts
    import proactive_agent as pa
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 14.9)
    monkeypatch.delenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", raising=False)
    assert pa.hora_del_aviso("u", "Almuerzo", 13.0, {}) == pytest.approx(12.75), "12:45, no 2:15"
    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", "1")
    assert pa.hora_del_aviso("u", "Almuerzo", 13.0, {}) == pytest.approx(14.25), "el knob devuelve la conducta anterior"


def test_la_hora_elegida_gana_siempre_incluso_con_el_historial_encendido(monkeypatch):
    import db_facts
    import proactive_agent as pa
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 14.9)
    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", "1")
    health = {"avisos_por_comida": {"almuerzo": {"activo": True, "hora": "12:30"}}}
    assert pa.hora_del_aviso("u", "Almuerzo", 13.0, health) == pytest.approx(12.5)
    assert pa.hora_elegida(health, "Almuerzo") == pytest.approx(12.5)
    assert pa.hora_elegida(health, "Cena") is None, "las demás siguen sin elegir"


@pytest.mark.parametrize("valor", ["25:00", "12:60", "7:30", "12h30", "", None, 1230, "12:30:00"])
def test_una_hora_ilegible_no_se_inventa(valor):
    import proactive_agent as pa
    assert pa.hora_hhmm_a_float(valor) is None
    assert pa.hora_elegida({"avisos_por_comida": {"cena": {"hora": valor}}}, "Cena") is None


def test_el_interruptor_de_cada_comida_ausente_es_encendido():
    import proactive_agent as pa
    assert pa.comida_con_aviso({}, "Merienda") is True
    assert pa.comida_con_aviso({"avisos_por_comida": {"merienda": {"hora": "16:00"}}}, "Merienda") is True
    assert pa.comida_con_aviso({"avisos_por_comida": {"merienda": {"activo": False}}}, "Merienda") is False
    assert pa.comida_con_aviso({"avisos_por_comida": "basura"}, "Merienda") is True, "lo que no se entiende no apaga"


# ───────────────────────── 2. Lo que se guarda: se valida, no se corrige ─────────────────────────

@pytest.mark.parametrize("valor", [
    None,
    {},
    {"almuerzo": {"activo": True, "hora": "12:45"}},
    {"desayuno": {"activo": False}, "cena": {"hora": "20:00"}, "merienda": {"hora": None}},
])
def test_formas_validas(valor):
    import proactive_agent as pa
    assert pa.error_en_avisos_por_comida(valor) is None


@pytest.mark.parametrize("valor,trozo", [
    ("12:45", "objeto"),
    ({"brunch": {"hora": "11:00"}}, "desconocida"),
    ({"almuerzo": "12:45"}, "solo admite"),
    ({"almuerzo": {"activo": True, "color": "rojo"}}, "solo admite"),
    ({"almuerzo": {"activo": "sí"}}, "true o false"),
    ({"almuerzo": {"hora": "1:45 PM"}}, "HH:MM"),
    ({"desayuno": {"hora": "05:30"}}, "silencio"),
    ({"cena": {"hora": "23:10"}}, "resumen del día"),
    ({"cena": {"hora": "00:15"}}, "silencio"),
])
def test_formas_invalidas_dicen_por_que(valor, trozo, monkeypatch):
    import proactive_agent as pa
    monkeypatch.delenv("MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR", raising=False)
    error = pa.error_en_avisos_por_comida(valor)
    assert error and trozo in error


def test_solo_se_eligen_horas_en_las_que_suenan_el_telefono_y_el_chat(monkeypatch):
    """De la hora de silencio (6:00) a antes de la del resumen (23:00): a las 23:10 el cron solo manda el Resumen
    del día, así que el teléfono sonaría y el chat no escribiría — justo lo que vio el dueño con el desayuno."""
    import proactive_agent as pa
    monkeypatch.delenv("MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR", raising=False)
    assert pa.HORA_DEL_RESUMEN == 23
    for ok in ("06:00", "12:45", "22:55", "22:59"):
        assert pa.error_en_avisos_por_comida({"cena": {"hora": ok}}) is None, ok
    for fuera in ("05:59", "23:00", "23:55"):
        assert pa.error_en_avisos_por_comida({"cena": {"hora": fuera}}), fuera
    # el silencio es un knob: el límite de abajo lo sigue
    monkeypatch.setenv("MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR", "7")
    assert pa.error_en_avisos_por_comida({"desayuno": {"hora": "06:30"}})
    assert pa.error_en_avisos_por_comida({"desayuno": {"hora": "07:00"}}) is None
    # el cron usa la MISMA constante para su hora del resumen
    assert "if now_ast.hour == HORA_DEL_RESUMEN:" in _src("proactive_agent.py")


def test_el_patch_del_perfil_rechaza_con_400():
    src = _src("routers/user_data.py")
    i = src.index('if body.health_profile and "avisos_por_comida" in body.health_profile:')
    tramo = src[i:i + 500]
    assert "error_en_avisos_por_comida(body.health_profile[\"avisos_por_comida\"])" in tramo
    assert "status_code=400" in tramo


# ───────────────────────── 3. El teléfono y Configuración ─────────────────────────

def test_el_telefono_no_programa_lo_apagado_y_usa_la_hora_elegida(monkeypatch):
    import meal_reminders as mr
    monkeypatch.delenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", raising=False)
    health = {"avisos_por_comida": {"almuerzo": {"activo": True, "hora": "12:30"}, "merienda": {"activo": False}}}
    out = mr.horario_de_avisos("u", locale="es-DO", consumed_today=[], health=health)
    assert [(r["meal"], r["hour"], r["minute"]) for r in out] == [
        ("desayuno", 8, 45), ("almuerzo", 12, 30), ("cena", 19, 15)]


def test_configuracion_recibe_las_cuatro_con_su_hora_normal(monkeypatch):
    import meal_reminders as mr
    monkeypatch.delenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", raising=False)
    health = {"avisos_por_comida": {"almuerzo": {"hora": "12:30"}, "merienda": {"activo": False}}}
    comidas = {c["meal"]: c for c in mr.comidas_para_configuracion("u", health)}
    assert list(comidas) == ["desayuno", "almuerzo", "merienda", "cena"]
    assert comidas["almuerzo"] == {"meal": "almuerzo", "active": True, "hour": 12, "minute": 30, "chosen": True,
                                   "default_hour": 12, "default_minute": 45}
    assert comidas["merienda"]["active"] is False and comidas["merienda"]["hour"] == 15, "la apagada también sale"
    assert comidas["cena"] == {"meal": "cena", "active": True, "hour": 19, "minute": 15, "chosen": False,
                               "default_hour": 19, "default_minute": 15}


def test_el_endpoint_manda_las_comidas_aunque_los_avisos_esten_apagados(monkeypatch):
    import db
    from routers import notifications as rn
    perfil = {"health_profile": {"avisos_comida": False, "avisos_por_comida": {"cena": {"hora": "20:00"}}}, "locale": "es-DO"}
    monkeypatch.setattr(db, "get_user_profile", lambda _u: perfil)
    monkeypatch.setattr(db, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(db, "get_consumed_meals_today", lambda _u, date_str=None, tz_offset_mins=None: [])
    monkeypatch.delenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", raising=False)
    r = rn._meal_reminders_sync("u")
    assert r["enabled"] is False and r["reminders"] == []
    assert [c["meal"] for c in r["comidas"]] == ["desayuno", "almuerzo", "merienda", "cena"]
    assert r["comidas"][3]["hour"] == 20 and r["comidas"][3]["chosen"] is True
    # hasta dónde deja elegir Configuración: lo dice el servidor, no una segunda copia del 23 en la app
    assert r["reminders_before_hour"] == 23 and r["quiet_until_hour"] == 6


# ───────────────────────── 4. El cron: el chat antes que el teléfono ─────────────────────────

@pytest.fixture
def dia(monkeypatch):
    """Un día con todas las dependencias de `run_proactive_checks` simuladas; el bucle y la hora son el código real."""
    import proactive_agent as pa
    import utils_push

    estado = {
        "ahora": _rd(12, 30),
        "health": {"dietType": "balanceada", "mainGoal": "ganar músculo"},
        "comidas": [],
        "avisos_hoy": [],
        "sesion": SID,
        "avisos_nuevos": [],
        "guardados": [],
        "push": [],
        "prompts": [],
    }

    class _Reloj(_dt_real):
        @classmethod
        def now(cls, tz=None):
            return estado["ahora"] if tz else estado["ahora"].replace(tzinfo=None)

    def _sql_pa(query, params=None, fetch_all=False, fetch_one=False):
        if "nudge_outcomes" in query:
            return [{"nudge_type": t} for t in estado["avisos_hoy"]]
        return []

    class _LLM:
        def __init__(self, *a, **k):
            pass

        def invoke(self, prompt):
            estado["prompts"].append(prompt)

            class _R:
                content = "Recordatorio"
            return _R()

    def _log(user_id, nudge_type, **k):
        estado["avisos_nuevos"].append(nudge_type)
        estado["avisos_hoy"].append(nudge_type)

    monkeypatch.setattr(pa, "datetime", _Reloj)
    monkeypatch.setattr(pa, "get_active_users_for_proactive", lambda: [{"id": estado["sesion"], "user_id": UID}])
    monkeypatch.setattr(pa, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(pa, "get_daily_nudge_count", lambda _u: len(estado["avisos_hoy"]))
    monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, nudge_type=None: (1.0, 0))
    monkeypatch.setattr(pa, "get_recent_messages", lambda _s, limit=5: [])
    monkeypatch.setattr(pa, "get_user_profile", lambda _u: {"health_profile": estado["health"], "locale": "es-DO"})
    monkeypatch.setattr(pa, "get_consumed_meals_today", lambda _u, date_str=None, tz_offset_mins=None: list(estado["comidas"]))
    monkeypatch.setattr(pa, "_usuario_en_modo_contador", lambda _u: True)
    monkeypatch.setattr(pa, "_sesion_del_dia_para_aviso", lambda s, _u, _a, _t: s)
    monkeypatch.setattr(pa, "get_embedding", lambda _t: None)
    monkeypatch.setattr(pa, "get_best_nudge_style", lambda _u: "directo")
    monkeypatch.setattr(pa, "execute_sql_query", _sql_pa)
    monkeypatch.setattr(pa, "ChatGLM", _LLM)
    monkeypatch.setattr(pa, "save_message", lambda s, r, c: estado["guardados"].append((s, r, c)))
    monkeypatch.setattr(pa, "log_nudge_outcome", _log)
    monkeypatch.setattr(utils_push, "send_push_notification", lambda **k: estado["push"].append(k) or True)
    for knob in ("MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY", "MEALFIT_PROACTIVE_NUDGE_RETRY_HOURS",
                 "MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", "MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR"):
        monkeypatch.delenv(knob, raising=False)

    def _correr(hora, minuto, dia_mes=24):
        estado["ahora"] = _rd(hora, minuto, dia_mes)
        antes = len(estado["avisos_nuevos"])
        pa.run_proactive_checks()
        return estado["avisos_nuevos"][antes:]

    estado["correr"] = _correr
    return estado


def test_el_almuerzo_de_las_1245_se_escribe_a_las_1230_antes_de_que_suene(dia):
    assert dia["correr"](12, 15) == [], "a las 12:15 aún no toca: faltan 30 min"
    assert dia["correr"](12, 30) == ["Almuerzo"], "último tick antes de las 12:45"
    assert "12:45 PM" in dia["prompts"][-1], "el mensaje dice la hora a la que suena el teléfono"
    assert dia["correr"](12, 45) == [], "y no se repite en el tick siguiente"


def test_un_aviso_a_las_215_ya_no_llega_al_chat_a_las_230(dia):
    """El caso del dueño: con el tick a y media, el teléfono sonaba a las 2:15 y el chat escribía a las 2:30."""
    dia["health"]["avisos_por_comida"] = {"almuerzo": {"activo": True, "hora": "14:15"}}
    assert dia["correr"](13, 45) == []
    assert dia["correr"](14, 0) == ["Almuerzo"], "a las 2:00, antes de que suene"
    assert "2:15 PM" in dia["prompts"][-1]


def test_la_cena_por_defecto_llega_al_chat_antes_de_las_715(dia):
    """19:15 caía en el tick de las 19:30: a TODOS les llegaba el teléfono antes que el mensaje."""
    dia["avisos_hoy"] = ["Desayuno", "Almuerzo", "Merienda"]
    assert dia["correr"](18, 45) == []
    assert dia["correr"](19, 0) == ["Cena"]
    assert "7:15 PM" in dia["prompts"][-1]


def test_una_comida_apagada_no_se_recuerda_en_el_chat(dia):
    dia["health"]["avisos_por_comida"] = {"merienda": {"activo": False}}
    dia["avisos_hoy"] = ["Desayuno", "Almuerzo"]
    assert dia["correr"](15, 30) == [], "la merienda está apagada"
    assert dia["correr"](19, 0) == ["Cena"], "las demás siguen"


def test_un_aviso_que_no_pudo_salir_se_reintenta_y_no_se_repite(dia, monkeypatch):
    import proactive_agent as pa
    dia["avisos_hoy"] = ["Desayuno"]
    # a las 12:30 el anti-spam lo frena (el coach acaba de contestar), a las 12:45 ya puede salir
    llamadas = {"n": 0}

    def _recientes(_s, limit=5):
        llamadas["n"] += 1
        return [{"role": "model", "created_at": "2026-09-24 16:20:00+00"}] if llamadas["n"] == 1 else []

    monkeypatch.setattr(pa, "get_recent_messages", _recientes)
    assert dia["correr"](12, 30) == []
    assert dia["correr"](12, 45) == ["Almuerzo"]
    assert dia["correr"](13, 0) == []


def test_el_resumen_de_las_23_sale_una_sola_vez_aunque_haya_cuatro_ticks(dia):
    """Al suscriptor sin chat (sin anti-spam que lo frene) le llegaban cuatro resúmenes: uno por tick de la hora 23."""
    dia["sesion"] = None
    dia["avisos_hoy"] = []
    salidos = []
    for minuto in (0, 15, 30, 45):
        salidos += dia["correr"](23, minuto)
    assert salidos == ["Resumen del día"]
    assert len(dia["push"]) == 1


# ───────────────────────── 5. Anclas ─────────────────────────

def test_el_cron_corre_cada_15_minutos():
    import proactive_agent as pa
    assert pa.MINUTOS_ENTRE_TICKS == 15
    assert 'minute=f"*/{MINUTOS_ENTRE_TICKS}"' in _src("app.py")
    assert "minute=30,\n                          id=\"proactive_meal_reminders\"" not in _src("app.py")


def test_el_prompt_ya_no_dice_que_son_pasadas_ni_hora_habitual():
    prompt = _src("prompts/proactive.py")
    assert "son pasadas las" not in prompt
    assert "No la llames «tu hora habitual»" in prompt


def test_el_knob_del_historial_esta_apagado_y_documentado():
    import proactive_agent as pa
    assert pa._avisos_desde_historial() is False
    doc = _src("docs/recordatorios_de_comida.md")
    for trozo in ("P1-PLAN-LOTE-223", "MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", "avisos_por_comida"):
        assert trozo in doc, trozo


def test_el_marcador_va_con_su_lote():
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"), re.M)
    assert m and int(m.group(1)) >= 223


def test_configuracion_manda_la_forma_que_el_servidor_valida():
    """El panel del frontend (lote 223) guarda `{meal: {activo, hora: "HH:MM" | null}}` de las CUATRO comidas: la forma
    que `error_en_avisos_por_comida` acepta. Y el formulario no la manda (su dueño es el panel)."""
    util = _FRONT / "src" / "utils" / "recordatoriosPorComida.js"
    comp = _FRONT / "src" / "components" / "settings" / "RecordatoriosPorComida.jsx"
    sfs = _FRONT / "src" / "config" / "secureFormStorage.js"
    if not (util.exists() and comp.exists() and sfs.exists()):
        pytest.skip("sin el repo del frontend al lado")
    u = util.read_text(encoding="utf-8")
    assert "export const CLAVE_AVISOS_POR_COMIDA = 'avisos_por_comida';" in u
    assert "out[c.meal] = { activo: c.active !== false, hora: c.chosen ? aHHMM(c.hour, c.minute) : null };" in u
    c = comp.read_text(encoding="utf-8")
    assert "fetchWithAuth('/api/notifications/meal-reminders')" in c
    assert "body: JSON.stringify({ health_profile: { [CLAVE_AVISOS_POR_COMIDA]: configParaGuardar(lote) } })," in c
    m = re.search(r"export const CLAVES_CON_CONTROL_PROPIO = Object\.freeze\(\[([^\]]*)\]\);", sfs.read_text(encoding="utf-8"))
    assert m and "avisos_por_comida" in re.findall(r"'([^']+)'", m.group(1))
    # lo que ese código produce, el servidor lo acepta
    import proactive_agent as pa
    assert pa.error_en_avisos_por_comida({
        "desayuno": {"activo": True, "hora": None}, "almuerzo": {"activo": True, "hora": "12:00"},
        "merienda": {"activo": False, "hora": None}, "cena": {"activo": True, "hora": "21:05"},
    }) is None
