# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-133 · 2026-09-20] «Alertas Inteligentes» lista para producción.

El dueño, con la captura del interruptor y su insignia BETA: «revisa a profundidad el sistema de notificaciones ya que quiero
que le avise al usuario si no ha desayunado, almorzado, merendado etc, deja eso al 100 % listo para producción, y quítale
ese beta».

MEDIDO EN PRODUCCIÓN antes de tocar nada (solo lectura, 20-sep):
  · el motor del servidor funciona: 38 avisos en 5 días, a las 10:30 / 14:30 / 17:30 / 21:00 de RD;
  · `push_subscriptions` tiene UNA fila (una PWA de iOS del 4-sep), y sus envíos salen «exitosos» en el journal: la Web
    Push y sus llaves están bien. Lo que falla es la ENTREGA: casi nadie tiene cómo recibirlos;
  · la app nativa de iOS es un WKWebView sin Service Worker ni PushManager: ahí el interruptor no podía funcionar (tocarlo
    daba «Tu navegador no soporta notificaciones Push») y el dueño —que usa esa app— solo veía sus avisos al abrir el chat.

Lo que cambia en el servidor:
  1. `proactive_agent.hora_de_aviso`: la hora del recordatorio de una comida, fuera del bucle del cron. SSOT de los dos
     canales — el chat/Web Push y los avisos LOCALES que la app nativa programa en el teléfono.
  2. `GET /api/notifications/meal-reminders` (+ `meal_reminders.py`): hora local, texto corto en los 5 idiomas, etiqueta y
     qué comidas ya están registradas hoy. Misma puerta que el cron: turno nocturno o rotativo ⇒ nada.
  3. Quien ENCENDIÓ las alertas las recibe: fuera el `send_push = False` que apagaba la pantalla para siempre a los 5 avisos
     «ignorados» — y «ignorado» deja de medirse solo con respuestas en el chat en 60 min: cuenta también REGISTRAR una comida
     tras el aviso, y la ventana es un knob (`MEALFIT_PROACTIVE_RESPONSE_WINDOW_MIN`, 180).
  4. Quien encendió las alertas y lleva más de 3 días sin abrir el chat dejaba de existir para el cron (la lista salía solo
     de `agent_sessions`): ahora recibe el aviso corto y fijo, sin LLM y sin escribir en un chat viejo.
  5. La notificación lleva etiqueta por comida (la nueva sustituye a la vieja en vez de apilarse) y el cron tiene `id`.
Contrato del cliente: `frontend/src/__tests__/lote133.test.js`."""
from __future__ import annotations

import ast
import re
from datetime import datetime as _dt_real, timezone
from pathlib import Path

import pytest

from test_p1_plan_lote_72 import dia, SID, UID  # noqa: F401  (el día simulado del cron de avisos: reloj, SQL y LLM falsos)

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


# ───────────────────────── 1. La hora del aviso, una sola cuenta ─────────────────────────

def test_la_hora_del_aviso_es_la_habitual_MENOS_la_antelacion(monkeypatch):
    """[P1-PLAN-LOTE-150] Era «+ la espera» (1,5 h, y hasta 2,5 a quien lo ignoraba). El dueño: «solo avisa al rato
    después del horario». Ahora el aviso se ADELANTA 15 min a tu hora habitual, para que puedas hacer algo con él.

    [P1-PLAN-LOTE-220] La hora habitual ya no es el camino por defecto (la eligen en Configuración; si no, la normal):
    vive detrás de `MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY`, y esto sigue midiendo que ESE camino hace su cuenta bien."""
    import db_facts
    import proactive_agent as pa
    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", "1")
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 8.5)
    monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, _m=None: (1.0, 0))
    assert pa.hora_de_aviso("u", "Desayuno", 9.0)[0] == pytest.approx(8.25), "desayuna a las 8:30 → aviso 8:15"
    # La tasa de respuesta YA NO mueve la hora: retrasar el aviso de quien lo ignora lo hacía aún más inútil.
    for tasa, total in ((0.9, 5), (0.1, 5)):
        monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, _m=None, _t=(tasa, total): _t)
        assert pa.hora_de_aviso("u", "Desayuno", 9.0)[0] == pytest.approx(8.25), "responda o no, el aviso es a su hora"
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: None)
    monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, _m=None: (1.0, 0))
    assert pa.hora_de_aviso("u", "Cena", 19.5)[0] == pytest.approx(19.25), "sin historial, la hora por defecto"
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 0.1)
    assert pa.hora_de_aviso("u", "Cena", 19.5)[0] == pytest.approx(20.75), \
        "[P1-PLAN-LOTE-151] cenar a las 00:06 ya no manda el aviso a las 23:51: la media se acota al tope de su " \
        "franja (19:30 + 1,5 h). El `% 24` sigue siendo load-bearing, ahora para medir la distancia en el RELOJ — " \
        "restar diría que las 00:06 están 19 h antes de las 19:30, cuando están 4,6 h después"
    monkeypatch.setenv("MEALFIT_PROACTIVE_NUDGE_LEAD_H", "0")
    monkeypatch.setattr(db_facts, "get_avg_meal_hour", lambda _u, _m, ventana=None: 8.5)
    assert pa.hora_de_aviso("u", "Desayuno", 9.0)[0] == pytest.approx(8.5), "con la antelación en 0, a la hora exacta"


def test_el_cron_usa_esa_misma_cuenta_y_no_otra_copia():
    src = _src("proactive_agent.py")
    arbol = ast.parse(src)
    asignaciones = [n for n in ast.walk(arbol) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "nudge_hour" for t in n.targets)]
    assert len(asignaciones) == 1, "UNA asignación a `nudge_hour` (la de `hora_del_aviso`): dos copias acaban avisando a horas distintas"
    # [P1-PLAN-LOTE-220] el cron y el teléfono piden la hora a la MISMA función, con la configuración de la persona
    assert "_hora_aviso = hora_del_aviso(user_id, meal, def_hour, _health)" in src
    assert "pa.hora_del_aviso(user_id, meal, def_hour, health)" in _src("meal_reminders.py")
    import proactive_agent as pa
    assert pa.HORAS_POR_DEFECTO_DE_COMIDA == {"Desayuno": 9.0, "Almuerzo": 13.0, "Merienda": 16.0, "Cena": 19.5}


# ───────────────────────── 2. Los recordatorios como dato ─────────────────────────

def test_el_horario_que_programa_el_telefono(monkeypatch):
    import meal_reminders as mr
    import proactive_agent as pa
    horas = {"Desayuno": 10.8, "Almuerzo": 14.5, "Merienda": 17.5, "Cena": 1.0}   # quien cena a las 23:30
    monkeypatch.setattr(pa, "hora_del_aviso", lambda _u, meal, _d, _h=None: horas[meal])
    monkeypatch.delenv("MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR", raising=False)
    out = mr.horario_de_avisos("u", locale="es-DO", consumed_today=[{"meal_type": "desayuno", "meal_name": "Mangú"}])
    assert [(r["meal"], r["hour"], r["minute"]) for r in out] == [("desayuno", 10, 48), ("almuerzo", 14, 30), ("merienda", 17, 30)], \
        "[P1-PLAN-LOTE-150] el minuto SALE de la hora calculada; clavarlo en :35 mandaba el aviso de una cena de " \
        "las 19:30 a las 19:35, es decir DESPUÉS de la cena"
    assert out[0]["logged_today"] is True and out[1]["logged_today"] is False
    assert "cena" not in [r["meal"] for r in out], "un aviso a la 1:00 cae en las horas de silencio: el cron tampoco lo manda"
    assert out[1]["tag"] == "comida-almuerzo" and out[1]["title"] == "Bioboros"
    assert out[1]["body"] == "Es tu hora de almorzar. Cuando comas, cuéntamelo y lo anoto."


def test_los_textos_estan_en_los_cinco_idiomas_y_caen_al_espanol():
    import meal_reminders as mr
    for comida in (*mr.COMIDAS, "Resumen del día"):
        cuerpos = {mr.texto_del_aviso(comida, loc)[1] for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT")}
        assert len(cuerpos) == 5, f"{comida}: falta una traducción"
        assert all(len(c) <= 110 for c in cuerpos), "los lee una pantalla de bloqueo: cortos"
    assert mr.texto_del_aviso("Almuerzo", "xx-YY") == mr.texto_del_aviso("Almuerzo", "es-DO")
    assert mr.texto_del_aviso("Almuerzo", None) == mr.texto_del_aviso("Almuerzo", "es-DO")
    assert mr.texto_del_aviso("Merienda", "en-US")[1].startswith("Time for a snack.")
    # [P1-PLAN-LOTE-150] Llega ANTES de comer: ninguno puede dar por hecho que ya comiste.
    for comida in mr.COMIDAS:
        assert "Ya " not in mr.texto_del_aviso(comida, "es-DO")[1], f"{comida}: el aviso pregunta en vez de animar"


def test_el_endpoint_usa_la_misma_puerta_que_el_cron(monkeypatch):
    import db
    import meal_reminders as mr
    from routers import notifications as rn
    perfil = {"health_profile": {"scheduleType": "night_shift"}, "locale": "en-US"}
    monkeypatch.setattr(db, "get_user_profile", lambda _u: perfil)
    monkeypatch.setattr(db, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(db, "get_consumed_meals_today", lambda _u, date_str=None, tz_offset_mins=None: [])
    monkeypatch.setattr(mr, "horario_de_avisos",
                        lambda _u, locale=None, consumed_today=None, health=None: [{"meal": "cena", "locale": locale}])
    r = rn._meal_reminders_sync("u")
    assert r["enabled"] is False and r["reason"] == "schedule" and r["reminders"] == [], \
        "con turno nocturno el cron no avisa: el teléfono tampoco"
    perfil["health_profile"]["scheduleType"] = "standard"
    r = rn._meal_reminders_sync("u")
    assert r["enabled"] is True and r["reminders"] == [{"meal": "cena", "locale": "en-US"}]
    assert r["tz_offset_min"] == 240 and r["url"] == "/dashboard/agent" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["local_date"])
    src = _src("routers/notifications.py")
    assert '@router.get("/meal-reminders")' in src and "Depends(_MEAL_REMINDERS_LIMITER)" in src


# ───────────────────────── 3. Quien encendió las alertas las recibe ─────────────────────────

def test_ignorar_avisos_cambia_el_tono_pero_no_apaga_la_pantalla(dia, monkeypatch):
    import proactive_agent as pa
    monkeypatch.setattr(pa, "get_nudge_response_rate", lambda _u, nudge_type=None: (0.0, 9))   # «ignoró» nueve
    # [P1-PLAN-LOTE-150] 13:30 en RD: el almuerzo (13:00 − 15 min). A las 15:30 la pendiente ya sería la merienda.
    dia["ahora"] = _dt_real(2026, 9, 16, 17, 30, 5, tzinfo=timezone.utc)
    dia["mensajes"] = []
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Almuerzo"]
    assert len(dia["push"]) == 1, "el interruptor es un consentimiento explícito: se apaga en Configuración, no solo"
    assert dia["push"][0]["tag"] == "comida-almuerzo"
    assert "ignorando notificaciones" in dia["prompts"][0] or "ignora o abandona" in dia["prompts"][0], "el tono sí cambia"
    assert not re.search(r"^\s*send_push = False", _src("proactive_agent.py"), re.M), "ni una línea de CÓDIGO lo apaga"


def test_registrar_la_comida_tras_el_aviso_cuenta_como_respuesta(monkeypatch):
    import proactive_agent as pa
    vistos = []

    def _sql(query, params=None, fetch_one=False, fetch_all=False):
        vistos.append((query, params))
        return {"total": 4, "responded_count": 3}

    monkeypatch.setattr(pa, "execute_sql_query", _sql)
    monkeypatch.delenv("MEALFIT_PROACTIVE_RESPONSE_WINDOW_MIN", raising=False)
    assert pa.get_nudge_response_rate("u", "Almuerzo") == (0.75, 4)
    q, p = vistos[0]
    assert "FROM consumed_meals c" in q and "c.created_at >= n.sent_at" in q and "n.responded OR EXISTS" in q
    assert p == (180, "u", "Almuerzo"), "la ventana es el knob (3 h), no 60 min fijos"
    assert pa._ventana_de_respuesta_min() == 180
    assert "INTERVAL '60 minutes'" not in _src("proactive_agent.py")


def test_quien_encendio_las_alertas_y_no_abre_el_chat_sigue_existiendo(monkeypatch):
    import proactive_agent as pa

    def _sql(query, params=None, fetch_one=False, fetch_all=False):
        if "agent_sessions" in query:
            return [{"id": "s1", "user_id": "con-chat"}]
        if "push_subscriptions" in query:
            return [{"user_id": "con-chat"}, {"user_id": "solo-push"}]
        return []

    monkeypatch.setattr(pa, "execute_sql_query", _sql)
    monkeypatch.setattr(pa, "connection_pool", object())
    assert pa.get_active_users_for_proactive() == [{"id": "s1", "user_id": "con-chat"}, {"id": None, "user_id": "solo-push"}]


def test_sin_chat_reciente_el_aviso_es_corto_fijo_y_sin_llm(dia, monkeypatch):
    import proactive_agent as pa
    monkeypatch.setattr(pa, "get_active_users_for_proactive", lambda: [{"id": None, "user_id": UID}])
    logs = []
    monkeypatch.setattr(pa, "log_nudge_outcome", lambda user_id, nudge_type, **k: logs.append((nudge_type, k)))
    dia["correr"]()
    assert dia["prompts"] == [], "sin LLM"
    assert dia["guardados"] == [], "sin escribir un mensaje en un chat que nadie va a abrir"
    assert logs == [("Almuerzo", {"nudge_content": "Es tu hora de almorzar. Cuando comas, cuéntamelo y lo anoto.",
                                  "nudge_style": "fijo"})], "queda anotado: el tope diario y el «no repetir» siguen valiendo"
    assert dia["push"] == [{"user_id": UID, "title": "Bioboros",
                            "body": "Es tu hora de almorzar. Cuando comas, cuéntamelo y lo anoto.",
                            "url": "/dashboard/agent", "tag": "comida-almuerzo"}]


# ───────────────────────── 4. El transporte ─────────────────────────

def test_la_etiqueta_viaja_en_el_payload_y_el_service_worker_la_usa():
    up = _src("utils_push.py")
    assert 'def send_push_notification(user_id: str, title: str, body: str, url: str = "/dashboard", tag: str = None) -> bool:' in up
    assert '_payload["tag"] = str(tag)[:64]' in up
    sw = _FRONT / "src" / "custom-sw.js"
    if sw.exists():
        txt = sw.read_text(encoding="utf-8")
        assert "notificationOptions.tag = data.tag.slice(0, 64);" in txt and "return mismaApp.navigate(urlToOpen)" in txt


def test_el_cron_de_avisos_tiene_nombre():
    app = _src("app.py")
    # [P1-PLAN-LOTE-220] cada `MINUTOS_ENTRE_TICKS` (15) en vez de a y media: el chat sale antes que el teléfono
    assert re.search(r'_add_job_jittered\(scheduler, run_proactive_checks, "cron", minute=f"\*/\{MINUTOS_ENTRE_TICKS\}",'
                     r'\s+id="proactive_meal_reminders", replace_existing=True\)', app)


def test_el_interruptor_ya_no_dice_beta():
    st = _FRONT / "src" / "pages" / "Settings.jsx"
    if not st.exists():
        pytest.skip("sin el repo del frontend al lado")
    src = st.read_text(encoding="utf-8").replace("\r\n", "\n")
    k = src.index("{t('Alertas Inteligentes')}")
    tarjeta = src[k:src.index("{/* SECCIÓN PREFERENCIAS", k)]
    assert not re.search(r">\s*Beta\s*<", tarjeta) and "t('Beta')" not in tarjeta
    assert "from '../utils/avisosDeComida';" in src


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', _src("app.py"))
    assert m and int(m.group(1)) >= 133
