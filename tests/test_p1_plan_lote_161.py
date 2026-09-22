"""[P1-PLAN-LOTE-161 · 2026-09-22] Auditoría de la beta en modo contador — lo que se cierra en el backend.

EL ENCARGO. El dueño iba a repartir la app a dos testers de Android (más él en iPhone) en modo contador, con el
generador apagado, y pidió un análisis «para dejarlo al 100 %». Seis revisiones en paralelo + verificación propia
contra el código, la base (solo lectura), el VPS y la API de DeepSeek. Lo que cae aquí:

1. **El aviso de la mañana caía en un chat que la app no abría** (regresión del lote 159, desplegado ese mismo día).
   `_sesion_del_dia_para_aviso` corría ANTES de todos los filtros: el primer tick del día —las 00:30, en pleno
   silencio— abría un chat por usuario aunque no fuera a recibir nada. A las 8:30 el aviso se escribía ahí, y el
   servidor marcaba esa sesión (solo mensajes del modelo) como `empty`, que es justo lo que el cliente descarta al
   elegir «tu chat de hoy»: aviso invisible y un «Nuevo chat» suelto en Recientes. Y como «activo» se medía por
   `agent_sessions.created_at`, el chat que abría el cron mantenía activo PARA SIEMPRE a quien abandonaba la app.
2. **Los avisos no conocían la dieta ni las alergias**: el cron leía `dietTypes`/`goals`, que el formulario no
   guarda (`dietType`/`mainGoal`), y el prompt no recibía alergias. Un tercio de las veces el estilo sorteado pide
   «aportar opciones»: es el único texto del coach sin filtro determinista detrás.
3. **Sentry del backend estaba APAGADO en producción.** El VPS no pasa el `.env` por systemd; lo carga `db_core`,
   que `app.py` importa ~150 líneas después de leer `SENTRY_DSN`. Reproducido en el VPS con el código desplegado.
4. **Borrar la cuenta no olvidaba la identidad cacheada**: los tokens de la cuenta borrada seguían entrando en
   este proceso hasta reiniciar (P1-AUTH-CUENTA-BORRADA reabierto por «Eliminar cuenta»).
5. **Dos IDOR que exigen conocer el UUID de la víctima**: un invitado con `session_id = <id de otra cuenta>`
   usaba las herramientas del coach sobre esa cuenta, y `/api/auth/migrate` movía sus datos a quien llamaba.

Tooltip-anchor: P1-PLAN-LOTE-161
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import proactive_agent

_BACKEND = Path(__file__).resolve().parents[1]
_SRC_PROACTIVO = (_BACKEND / "proactive_agent.py").read_text(encoding="utf-8")
_SRC_APP = (_BACKEND / "app.py").read_text(encoding="utf-8")

_RD = 240   # UTC-4, convención getTimezoneOffset


# ── 1 · el aviso y su chat ──────────────────────────────────────────────────────────────────────────────────

class _LLMFalso:
    """Sustituye a ChatGLM: guarda el prompt y contesta un aviso fijo."""
    prompts: list = []

    def __init__(self, *a, **k):
        pass

    def invoke(self, prompt):
        _LLMFalso.prompts.append(prompt)

        class _R:
            content = "Es tu hora de almorzar."
        return _R()


def _reloj(instante_utc: datetime):
    class _Reloj(datetime):
        @classmethod
        def now(cls, tz=None):
            return instante_utc if tz is None else instante_utc.astimezone(tz)
    return _Reloj


def _hora_de_aviso_falsa(hora_del_almuerzo: float):
    """Solo el almuerzo toca; las demás comidas quedan lejos (fuera de su ventana de reintento)."""
    def _f(user_id, meal, def_hour):
        return (hora_del_almuerzo if meal == "Almuerzo" else 3.0), 0.5, 4
    return _f


def _correr_cron(ahora_utc, health=None, estilo="directo"):
    """Corre `run_proactive_checks` para UN usuario cuyo último chat es de AYER. Devuelve (sesiones creadas,
    mensajes guardados, prompts enviados a la IA)."""
    creadas, guardados = [], []
    _LLMFalso.prompts = []
    ayer = ahora_utc - timedelta(hours=20)
    perfil = {"health_profile": health if health is not None else {"allergies": ["Ninguna"]}, "locale": "es-DO"}
    with patch.object(proactive_agent, "datetime", _reloj(ahora_utc)), \
         patch.object(proactive_agent, "get_active_users_for_proactive",
                      return_value=[{"id": "ses-ayer", "user_id": "u-1"}]), \
         patch.object(proactive_agent, "user_tz_offset_min", return_value=_RD), \
         patch.object(proactive_agent, "get_daily_nudge_count", return_value=0), \
         patch.object(proactive_agent, "get_nudge_response_rate", return_value=(0.5, 2)), \
         patch.object(proactive_agent, "hora_de_aviso", _hora_de_aviso_falsa(12.75)), \
         patch.object(proactive_agent, "_comidas_avisadas_hoy", return_value=set()), \
         patch.object(proactive_agent, "get_recent_messages", return_value=[]), \
         patch.object(proactive_agent, "get_user_profile", return_value=perfil), \
         patch.object(proactive_agent, "get_consumed_meals_today", return_value=[]), \
         patch.object(proactive_agent, "get_embedding", return_value=None), \
         patch.object(proactive_agent, "get_best_nudge_style", return_value=estilo), \
         patch.object(proactive_agent, "ChatGLM", _LLMFalso), \
         patch.object(proactive_agent, "log_nudge_outcome", return_value=None), \
         patch.object(proactive_agent, "execute_sql_query", return_value={"ultima": ayer}), \
         patch.object(proactive_agent, "save_message",
                      side_effect=lambda sid, role, content: guardados.append((sid, role, content))), \
         patch.object(proactive_agent, "get_or_create_session",
                      side_effect=lambda sid, user_id=None: creadas.append((sid, user_id))), \
         patch("utils_push.send_push_notification", return_value=None):
        proactive_agent.run_proactive_checks()
    return creadas, guardados, list(_LLMFalso.prompts)


def test_en_horas_de_silencio_no_se_abre_ningun_chat():
    """EL DEFECTO: a las 00:30 locales (silencio hasta las 6:00) el lote 159 ya había abierto un chat nuevo por
    usuario, antes de mirar si iba a mandarle algo. Un chat vacío cada día y, peor, un usuario «activo» para siempre."""
    creadas, guardados, prompts = _correr_cron(datetime(2026, 9, 23, 4, 30, tzinfo=timezone.utc))
    assert creadas == [], (
        "El cron abrió un chat en horas de silencio: la sesión del día se decide antes de los filtros."
    )
    assert guardados == [] and prompts == []


def test_el_aviso_se_escribe_en_el_chat_de_hoy_y_solo_cuando_hay_aviso():
    """A las 12:50 locales toca el almuerzo: se abre el chat de HOY (el último era de ayer) y el aviso cae ahí."""
    creadas, guardados, prompts = _correr_cron(datetime(2026, 9, 23, 16, 50, tzinfo=timezone.utc))
    assert len(prompts) == 1, "tenía que salir exactamente un aviso (el del almuerzo)"
    assert len(creadas) == 1 and creadas[0][1] == "u-1", "el chat del día se abre a nombre del usuario"
    assert guardados and guardados[0][0] == creadas[0][0] != "ses-ayer", (
        "El aviso tiene que ir al chat NUEVO de hoy, no al de ayer."
    )


def test_el_prompt_del_aviso_lleva_la_dieta_y_las_alergias_reales():
    """Perfil vegano y alérgico al huevo, con el estilo que pide opciones: el prompt tiene que decirlo y pedir que no
    se nombre comida."""
    health = {"dietType": "vegana", "mainGoal": "gain_muscle", "allergies": ["Huevo"], "medicalConditions": ["Ninguna"]}
    _c, _g, prompts = _correr_cron(datetime(2026, 9, 23, 16, 50, tzinfo=timezone.utc), health=health, estilo="sugestivo")
    assert prompts, "no salió el aviso"
    p = prompts[0]
    assert "Huevo" in p and "vegana" in p, "las restricciones reales no llegaron al prompt del aviso"
    assert "gain_muscle" in p, "el objetivo se sigue leyendo de una clave que el formulario no guarda"
    assert "sin nombrar alimentos concretos" in p, "con restricciones, el estilo sugestivo no puede pedir opciones"
    assert "aportando opciones" not in p


# ── 1b · qué es «activo» y qué ve el cliente ────────────────────────────────────────────────────────────────

def test_activo_lo_decide_una_sesion_que_abrio_una_persona():
    """La consulta no puede volver a contar los chats que abre el propio cron (solo mensajes del modelo)."""
    i = _SRC_PROACTIVO.index("def get_active_users_for_proactive")
    cuerpo = _SRC_PROACTIVO[i:i + 4000]
    assert "P1-PLAN-LOTE-161-ACTIVO-POR-PERSONA" in cuerpo
    assert "m.role = 'user'" in cuerpo
    assert "NOT EXISTS (SELECT 1 FROM agent_messages m WHERE m.session_id = s.id)" in cuerpo
    # la sesión devuelta sigue siendo la MÁS RECIENTE del usuario (también la del aviso de hoy)
    assert "ORDER BY s.user_id, s.created_at DESC" in cuerpo


def test_una_sesion_con_solo_el_aviso_no_es_empty():
    """El cliente descarta `empty` al elegir el chat de hoy. Una sesión con el aviso del coach NO está vacía."""
    import db_chat
    sesiones = [
        {"id": "11111111-1111-1111-1111-111111111111", "created_at": "2026-09-23 12:30:00+00"},
        {"id": "22222222-2222-2222-2222-222222222222", "created_at": "2026-09-23 07:00:00+00"},
    ]
    mensajes = [
        {"session_id": "11111111-1111-1111-1111-111111111111", "content": "Es tu hora de desayunar.",
         "created_at": "2026-09-23 12:30:05+00", "role": "model"},
    ]
    with patch.object(db_chat, "execute_sql_query", return_value=mensajes):
        out = {s["id"]: s for s in db_chat._process_and_sort_sessions([dict(s) for s in sesiones])}
    assert out["11111111-1111-1111-1111-111111111111"]["title_key"] == "coach"
    assert out["22222222-2222-2222-2222-222222222222"]["title_key"] == "empty", "sin mensajes sigue siendo `empty`"


# ── 2 · el contexto del aviso ───────────────────────────────────────────────────────────────────────────────

def test_contexto_del_aviso_lee_las_claves_del_formulario():
    ctx = proactive_agent.contexto_del_aviso({"dietType": "vegetariana", "mainGoal": "lose_fat",
                                              "allergies": ["Maní"], "otherAllergies": "kiwi"})
    assert ctx["dieta"] == "vegetariana" and ctx["objetivo"] == "lose_fat"
    assert "Maní" in ctx["bloque"] and "kiwi" in ctx["bloque"] and ctx["restringido"] is True


def test_contexto_del_aviso_sin_nada_declarado_no_inventa_restricciones():
    ctx = proactive_agent.contexto_del_aviso({"dietType": "balanceada", "allergies": ["Ninguna"],
                                              "medicalConditions": ["Ninguna"]})
    assert ctx["bloque"] == "" and ctx["restringido"] is False


def test_contexto_del_aviso_respeta_las_claves_viejas_como_respaldo():
    ctx = proactive_agent.contexto_del_aviso({"dietTypes": ["keto"], "goals": ["perder grasa"]})
    assert ctx["dieta"] == "keto" and ctx["objetivo"] == "perder grasa"


# ── 3 · Sentry ──────────────────────────────────────────────────────────────────────────────────────────────

def test_el_env_se_carga_antes_de_leer_la_configuracion_de_sentry():
    carga = _SRC_APP.index("_load_dotenv_antes_de_sentry()")
    assert "P1-PLAN-LOTE-161-SENTRY-DOTENV" in _SRC_APP
    assert carga < _SRC_APP.index("_SENTRY_TRACES_SAMPLE_RATE = _knob_env_float(")
    assert carga < _SRC_APP.index("_SENTRY_DSN = (os.environ.get(\"SENTRY_DSN\")")
    assert carga < _SRC_APP.index("\nsentry_sdk.init(")


def test_la_suite_sigue_sin_hablar_con_sentry():
    """Cargar el `.env` antes no puede encender Sentry en los tests: el conftest fija el DSN vacío antes."""
    import os
    assert os.environ.get("SENTRY_DSN", "") == ""


# ── 4 · borrar la cuenta ────────────────────────────────────────────────────────────────────────────────────

def test_borrar_la_cuenta_olvida_la_identidad_cacheada():
    import db_profiles
    uid = "33333333-4444-5555-6666-777777777777"
    db_profiles._AUTH_ROW_ALIVE_IDS.add(uid)
    with patch.object(db_profiles, "connection_pool", object()), \
         patch.object(db_profiles, "execute_sql_write", return_value=[]), \
         patch.object(db_profiles, "_purge_visual_diary_storage", return_value=0):
        db_profiles.delete_account_data(uid, True)
    assert uid not in db_profiles._AUTH_ROW_ALIVE_IDS, (
        "Tras borrar la cuenta, el positivo cacheado sigue ahí: sus tokens en otros dispositivos siguen entrando."
    )


# ── 5 · IDOR del invitado y de la migración ────────────────────────────────────────────────────────────────

VICTIMA = "11111111-2222-3333-4444-555555555555"
INVITADO = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
NUEVO = "99999999-8888-7777-6666-555555555555"


def test_un_invitado_no_puede_usar_como_sesion_el_id_de_una_cuenta():
    from routers.chat import _resolve_chat_identity
    with patch("db.uuid_es_de_una_cuenta", return_value=True):
        with pytest.raises(HTTPException) as exc:
            _resolve_chat_identity(None, VICTIMA, None)
    assert exc.value.status_code == 403


@pytest.mark.parametrize("veredicto", [False, None])
def test_el_invitado_normal_sigue_siendo_invitado(veredicto):
    """Una sesión de invitado de verdad (o una comprobación que no pudo hacerse) sigue funcionando."""
    from routers.chat import _resolve_chat_identity
    with patch("db.uuid_es_de_una_cuenta", return_value=veredicto):
        assert _resolve_chat_identity(None, INVITADO, None) == "guest"


def test_con_token_no_se_consulta_nada():
    from routers.chat import _resolve_chat_identity
    with patch("db.uuid_es_de_una_cuenta", side_effect=AssertionError("no debía consultarse")):
        assert _resolve_chat_identity(None, VICTIMA, NUEVO) == NUEVO


def _migrar(ids, es_cuenta):
    import db_profiles
    sentencias = []
    with patch.object(db_profiles, "connection_pool", object()), \
         patch.object(db_profiles, "uuid_es_de_una_cuenta", side_effect=es_cuenta), \
         patch.object(db_profiles, "execute_sql_transaction", side_effect=lambda s: sentencias.extend(s)), \
         patch("db_plans.get_ingredient_frequencies_from_plans", return_value=[]):
        ok = db_profiles.migrate_guest_data(ids, NUEVO)
    return ok, sentencias


def test_la_migracion_no_mueve_datos_de_otra_cuenta():
    ok, sentencias = _migrar([INVITADO, VICTIMA], lambda i: i == VICTIMA)
    assert ok is True and sentencias
    for sql, params in sentencias:
        ids = params[1]
        assert VICTIMA not in ids, f"la migración sigue tocando la cuenta ajena: {sql}"
        assert INVITADO in ids
    sesiones = [sql for sql, _p in sentencias if sql.startswith("UPDATE agent_sessions")]
    assert sesiones and "user_id IS NULL" in sesiones[0], "solo se adoptan las sesiones SIN dueño"


def test_la_migracion_sin_ids_de_invitado_no_hace_nada():
    ok, sentencias = _migrar([VICTIMA], lambda i: True)
    assert ok is False and sentencias == []


def test_si_no_se_puede_comprobar_un_id_no_se_migra():
    """Perder la migración de un invitado se recupera; mover datos ajenos, no."""
    ok, sentencias = _migrar([INVITADO], lambda i: None)
    assert ok is False and sentencias == []


def test_uuid_es_de_una_cuenta():
    import db_profiles
    assert db_profiles.uuid_es_de_una_cuenta("no-es-un-uuid") is False
    assert db_profiles.uuid_es_de_una_cuenta("") is False
    with patch.object(db_profiles, "connection_pool", None):
        assert db_profiles.uuid_es_de_una_cuenta(VICTIMA) is None
    with patch.object(db_profiles, "connection_pool", object()):
        with patch.object(db_profiles, "execute_sql_query", return_value={"hay": 1}):
            assert db_profiles.uuid_es_de_una_cuenta(VICTIMA) is True
        with patch.object(db_profiles, "execute_sql_query", return_value=None):
            assert db_profiles.uuid_es_de_una_cuenta(VICTIMA) is False
        with patch.object(db_profiles, "execute_sql_query", side_effect=RuntimeError("caída")):
            assert db_profiles.uuid_es_de_una_cuenta(VICTIMA) is None


def test_el_marcador_esta_al_dia():
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', _SRC_APP, re.M)
    assert m and int(m.group(1)) >= 161
