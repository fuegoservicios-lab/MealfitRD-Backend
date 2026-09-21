# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-135 · 2026-09-20] Avisos de hidratación — y la hidratación que se apaga sola si nadie la usa.

El encargo del dueño: «si el usuario tiene la hidratación encendida y no agrega vasos de agua, o vasos suficientes en el
día, le enviará notificaciones; pero si llegan a pasar 48 horas y el usuario ignora las notificaciones y no agrega vasos
de agua nunca, significa que no le interesa estar agregando vasos manualmente, así que debe desactivarse automáticamente».

TRES PUNTOS DE CONTROL al día (hora LOCAL del usuario), cada uno con la fracción de la meta que ya debería llevar:
11 h → 25 %, 15 h → 55 %, 19 h → 80 %. Si va por debajo, sale el aviso; si va al día, silencio. Los textos son fijos y
cortos (pantalla de bloqueo), sin LLM y sin escribir nada en el chat: el agua no es una conversación.

DOS CANALES, UNA CUENTA (igual que los recordatorios de comida, lote 133):
  · navegador / PWA: Web Push desde el cron horario `run_hydration_checks`;
  · app nativa: notificaciones LOCALES que el teléfono programa con `horario_de_avisos_de_agua` (viaja dentro de
    `GET /api/notifications/meal-reminders`). Solo HOY y los dos días siguientes: si el usuario no vuelve a abrir la
    app, a las 48 h la hidratación se apaga y el teléfono ya no tiene nada más programado.

EL APAGADO AUTOMÁTICO exige las tres cosas a la vez: (1) el usuario es ALCANZABLE (tiene suscripción push o su teléfono
sincronizó avisos locales en las últimas 72 h — a quien no le llegó nada no se le puede acusar de ignorarlo), (2) el
primer aviso sin respuesta tiene ≥ `MEALFIT_HYDRATION_AUTO_OFF_HOURS` (48) y ya van ≥ `…_MIN_NUDGES` (3), y (3) no hay
NI UN vaso anotado desde ese primer aviso (por el botón o por el coach: se mira `water_intake_log`, no quién escribió).
Un solo vaso reinicia la cuenta. Volver a encenderla en Configuración la deja a cero (`al_encender`).

EL ESTADO vive en `app_kv_store` (`hydration_state:<user_id>`): sin DDL nuevo, con barrido por TTL
(`cron_tasks._KV_SWEEP_PREFIXES`) y borrado explícito en la purga de la cuenta (`db_profiles._USER_SCOPED_KV_PREFIXES`).
NO se usa `nudge_outcomes`: `get_daily_nudge_count` cuenta TODAS sus filas contra el tope de 4 avisos de comida al día,
y tres avisos de agua habrían dejado al usuario sin el recordatorio de la cena.
tooltip-anchor: run_hydration_checks, evaluar_apagado, horario_de_avisos_de_agua (test_p1_plan_lote_135.py)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from knobs import _env_bool, _env_int

logger = logging.getLogger(__name__)

# hora local → fracción de la meta diaria que ya debería estar anotada
PUNTOS_DE_CONTROL = ((11, 0.25), (15, 0.55), (19, 0.80))
MINUTO_DEL_AVISO_LOCAL = 35          # el mismo «a y 35» de los recordatorios de comida (5 min tras el tick del cron)
DIAS_EN_EL_TELEFONO = 3              # hoy + 2: lo que tarda en apagarse sola
HORAS_DE_MARGEN = 2                  # un punto de control sigue tocando 2 h (deploy a y media, DB caída un tick)
PREFIJO_ESTADO = "hydration_state:"
PREFIJO_CANAL_LOCAL = "avisos_locales:"
HORAS_DE_ALCANCE_LOCAL = 72
ETIQUETA = "agua"
RUTA = "/dashboard"

_TITULO = {"es-DO": "Hidratación", "en-US": "Hydration", "pt-BR": "Hidratação", "fr-FR": "Hydratation", "it-IT": "Idratazione"}
_CUERPO_CERO = {
    "es-DO": "Hoy no has anotado agua. ¿Empezamos con un vaso?",
    "en-US": "You haven't logged any water today. Start with a glass?",
    "pt-BR": "Hoje você ainda não anotou água. Começamos com um copo?",
    "fr-FR": "Tu n'as pas encore noté d'eau aujourd'hui. On commence par un verre ?",
    "it-IT": "Oggi non hai ancora segnato acqua. Iniziamo con un bicchiere?",
}
_CUERPO_PARCIAL = {
    "es-DO": "Llevas {n} de {meta} vasos hoy. ¿Un vaso de agua ahora?",
    "en-US": "You're at {n} of {meta} glasses today. A glass of water now?",
    "pt-BR": "Você está em {n} de {meta} copos hoje. Um copo de água agora?",
    "fr-FR": "Tu en es à {n} verres sur {meta} aujourd'hui. Un verre d'eau maintenant ?",
    "it-IT": "Sei a {n} bicchieri su {meta} oggi. Un bicchiere d'acqua adesso?",
}
_CUERPO_GENERICO = {
    "es-DO": "¿Cómo va tu agua de hoy? Anota tus vasos para llevar la cuenta.",
    "en-US": "How's your water today? Log your glasses to keep count.",
    "pt-BR": "Como vai sua água hoje? Anote seus copos para manter a conta.",
    "fr-FR": "Où en es-tu avec l'eau aujourd'hui ? Note tes verres pour garder le compte.",
    "it-IT": "Come va l'acqua oggi? Segna i tuoi bicchieri per tenere il conto.",
}
_CUERPO_APAGADO = {
    "es-DO": "Pausamos la hidratación: llevas 2 días sin anotar agua. Puedes volver a encenderla en Configuración.",
    "en-US": "We paused hydration: 2 days without logging water. You can turn it back on in Settings.",
    "pt-BR": "Pausamos a hidratação: 2 dias sem anotar água. Você pode reativá-la em Configurações.",
    "fr-FR": "Hydratation en pause : 2 jours sans noter d'eau. Tu peux la réactiver dans les Réglages.",
    "it-IT": "Idratazione in pausa: 2 giorni senza segnare acqua. Puoi riattivarla nelle Impostazioni.",
}


def _encendidos() -> bool:
    """Kill switch de los avisos de agua Y del apagado automático (van juntos: sin avisos no hay «ignorados»)."""
    return _env_bool("MEALFIT_HYDRATION_REMINDERS", True)


def _horas_para_apagar() -> int:
    return _env_int("MEALFIT_HYDRATION_AUTO_OFF_HOURS", 48, validator=lambda v: 24 <= v <= 336)


def _avisos_minimos_para_apagar() -> int:
    return _env_int("MEALFIT_HYDRATION_AUTO_OFF_MIN_NUDGES", 3, validator=lambda v: 1 <= v <= 20)


def _loc(locale: Optional[str]) -> str:
    return locale if isinstance(locale, str) and locale in _TITULO else "es-DO"


def _vasos_legibles(v: float) -> str:
    """2.0 → «2», 2.5 → «2.5»: el contador admite medios vasos."""
    f = float(v or 0)
    return str(int(f)) if f == int(f) else f"{f:.1f}"


def vasos_esperados(meta: int, fraccion: float) -> int:
    """Vasos que ya deberían estar anotados en ese punto del día. Nunca menos de 1: a las 11 con meta 6 también toca."""
    return max(1, int(math.floor(float(meta or 0) * float(fraccion) + 1e-9)))


def punto_de_control(hora_local: float) -> Optional[tuple]:
    """El punto de control que toca a esta hora local (con su margen), o None. Devuelve `(hora, fracción)`."""
    h = int(math.floor(float(hora_local))) % 24
    for hora, fraccion in reversed(PUNTOS_DE_CONTROL):
        if hora <= h < hora + HORAS_DE_MARGEN:
            return hora, fraccion
    return None


def texto_del_aviso_de_agua(locale: Optional[str], vasos: float, meta: int) -> tuple:
    loc = _loc(locale)
    if float(vasos or 0) <= 0:
        return _TITULO[loc], _CUERPO_CERO[loc]
    return _TITULO[loc], _CUERPO_PARCIAL[loc].format(n=_vasos_legibles(vasos), meta=int(meta))


def texto_del_apagado(locale: Optional[str]) -> tuple:
    loc = _loc(locale)
    return _TITULO[loc], _CUERPO_APAGADO[loc]


def horario_de_avisos_de_agua(locale: Optional[str], vasos_hoy: float, meta: int) -> list:
    """Lo que el TELÉFONO programa: un aviso por punto de control. `met_today` = hoy ya va al día en ese punto (no se
    programa). `body` lleva la cuenta de HOY; `body_generic` es el de los días siguientes, cuya cuenta nadie conoce."""
    loc = _loc(locale)
    titulo, cuerpo = texto_del_aviso_de_agua(loc, vasos_hoy, meta)
    out = []
    for hora, fraccion in PUNTOS_DE_CONTROL:
        esperados = vasos_esperados(meta, fraccion)
        out.append({
            "kind": "water", "hour": hora, "minute": MINUTO_DEL_AVISO_LOCAL, "title": titulo, "body": cuerpo,
            "body_generic": _CUERPO_GENERICO[loc], "tag": ETIQUETA, "expected_glasses": esperados,
            "met_today": float(vasos_hoy or 0) >= esperados,
        })
    return out


def evaluar_apagado(estado: Optional[dict], ahora: datetime, hubo_agua: bool) -> bool:
    """¿Toca apagar la hidratación? Pura. `hubo_agua` = hay algún vaso anotado desde `ignored_since`."""
    if not estado or hubo_agua:
        return False
    desde = _parse_iso(estado.get("ignored_since"))
    if desde is None:
        return False
    if int(estado.get("nudges") or 0) < _avisos_minimos_para_apagar():
        return False
    return (ahora - desde) >= timedelta(hours=_horas_para_apagar())


def _parse_iso(valor) -> Optional[datetime]:
    if not valor or not isinstance(valor, str):
        return None
    try:
        dt = datetime.fromisoformat(valor)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ── estado en app_kv_store ─────────────────────────────────────────────────────────────────────────────────────────────
def _leer_estado(user_id: str) -> dict:
    from db_core import execute_sql_query
    try:
        fila = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (PREFIJO_ESTADO + str(user_id),), fetch_one=True)
        valor = (fila or {}).get("value") if isinstance(fila, dict) else None
        if isinstance(valor, str):
            valor = json.loads(valor)
        return dict(valor) if isinstance(valor, dict) else {}
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-135] estado de hidratación de {user_id} no leído: {e}")
        return {}


def _guardar_estado(user_id: str, estado: dict) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
        (PREFIJO_ESTADO + str(user_id), json.dumps(estado, ensure_ascii=False)),
    )


def al_encender(user_id: str) -> None:
    """El usuario volvió a encender la hidratación: la cuenta de «ignorados» empieza de cero. Sin esto, quien la
    reactiva tras un apagado automático la vería apagarse de nuevo en el siguiente tick. Best-effort."""
    from db_core import execute_sql_write
    try:
        execute_sql_write("DELETE FROM app_kv_store WHERE key = %s", (PREFIJO_ESTADO + str(user_id),))
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-135] estado de hidratación de {user_id} no reiniciado: {e}")


def marcar_canal_local(user_id: str) -> None:
    """El teléfono acaba de pedir su horario de avisos locales: este usuario es ALCANZABLE por 72 h. Best-effort."""
    from db_core import execute_sql_write
    try:
        execute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
            (PREFIJO_CANAL_LOCAL + str(user_id), json.dumps({"canal": "local"})),
        )
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-135] canal local de {user_id} no anotado: {e}")


def apagado_automatico_de(user_id: str) -> Optional[str]:
    """ISO del último apagado automático (o None): el dashboard lo dice UNA vez («Pausamos la hidratación…»)."""
    return _leer_estado(user_id).get("auto_off_at") or None


def _hubo_agua_desde(user_id: str, desde: datetime) -> bool:
    from db_core import execute_sql_query
    fila = execute_sql_query(
        "SELECT 1 AS ok FROM water_intake_log WHERE user_id = %s AND glasses > 0 AND updated_at >= %s LIMIT 1",
        (user_id, desde), fetch_one=True,
    )
    return bool(fila)


def usuarios_alcanzables_con_agua() -> list:
    """Usuarios con la hidratación encendida a los que un aviso SÍ les llega: suscripción push o teléfono sincronizado."""
    from db_core import execute_sql_query
    filas = execute_sql_query(
        "SELECT p.id::text AS user_id, p.locale AS locale, p.health_profile->>'scheduleType' AS schedule, "
        "       EXISTS (SELECT 1 FROM push_subscriptions s WHERE s.user_id::text = p.id::text) AS con_push "
        "FROM user_profiles p "
        # [P1-PLAN-LOTE-150] El interruptor de Configuración, en la MISMA consulta: quien lo apaga deja de estar
        # entre los alcanzables y el cron ni lo mira. `COALESCE` ⇒ ausente es activo.
        "WHERE COALESCE(p.water_tracker_enabled, TRUE) = TRUE "
        "  AND COALESCE((p.health_profile->>'avisos_agua')::boolean, TRUE) = TRUE "
        "  AND (EXISTS (SELECT 1 FROM push_subscriptions s WHERE s.user_id::text = p.id::text) "
        "       OR EXISTS (SELECT 1 FROM app_kv_store k WHERE k.key = %s || p.id::text "
        "                  AND k.updated_at >= NOW() - make_interval(hours => %s)))",
        (PREFIJO_CANAL_LOCAL, HORAS_DE_ALCANCE_LOCAL), fetch_all=True,
    )
    return list(filas or [])


def revisar_usuario(user_id: str, locale: Optional[str], con_push: bool, ahora: Optional[datetime] = None) -> dict:
    """Un tick para un usuario. Devuelve `{accion: 'nada'|'aviso'|'apagado'|'al_dia'|'reinicio', ...}`."""
    from db import get_water_intake_glasses_today, update_water_tracker_enabled, user_tz_offset_min
    import proactive_agent as pa

    ahora = ahora or datetime.now(timezone.utc)
    try:
        tz_off = int(user_tz_offset_min(user_id))
    except Exception:
        tz_off = pa._proactive_tz_offset_min()
    local = ahora - timedelta(minutes=tz_off)
    fecha_local = local.strftime("%Y-%m-%d")
    hora_local = local.hour + local.minute / 60.0

    estado = _leer_estado(user_id)
    desde = _parse_iso(estado.get("ignored_since"))
    if desde is not None and _hubo_agua_desde(user_id, desde):
        estado = {"fecha": estado.get("fecha"), "horas": estado.get("horas") or []}   # anotó agua: cuenta a cero
        _guardar_estado(user_id, estado)
        desde = None

    if evaluar_apagado(estado, ahora, hubo_agua=False):
        update_water_tracker_enabled(user_id, False)
        _guardar_estado(user_id, {"auto_off_at": ahora.isoformat(), "nudges_ignorados": int(estado.get("nudges") or 0)})
        if con_push:
            from utils_push import send_push_notification
            titulo, cuerpo = texto_del_apagado(locale)
            send_push_notification(user_id, titulo, cuerpo, url="/dashboard/settings", tag=ETIQUETA)
        logger.info(f"💧 [P1-PLAN-LOTE-135] hidratación apagada sola para {user_id}: "
                    f"{estado.get('nudges')} avisos sin un vaso desde {estado.get('ignored_since')}")
        return {"accion": "apagado"}

    pc = punto_de_control(hora_local)
    if pc is None:
        return {"accion": "nada"}
    hora_pc, fraccion = pc
    horas_hoy = list(estado.get("horas") or []) if estado.get("fecha") == fecha_local else []
    if hora_pc in horas_hoy:
        return {"accion": "nada"}

    from routers.plans import _compute_water_goal
    meta = int((_compute_water_goal(user_id) or {}).get("goal") or 8)
    vasos = float(get_water_intake_glasses_today(user_id, fecha_local) or 0)
    horas_hoy.append(hora_pc)
    if vasos >= vasos_esperados(meta, fraccion):
        _guardar_estado(user_id, {**estado, "fecha": fecha_local, "horas": horas_hoy})
        return {"accion": "al_dia", "vasos": vasos, "meta": meta}

    titulo, cuerpo = texto_del_aviso_de_agua(locale, vasos, meta)
    if con_push:
        from utils_push import send_push_notification
        send_push_notification(user_id, titulo, cuerpo, url=RUTA, tag=ETIQUETA)
    _guardar_estado(user_id, {
        **estado, "fecha": fecha_local, "horas": horas_hoy,
        "nudges": int(estado.get("nudges") or 0) + 1,
        "ignored_since": estado.get("ignored_since") or ahora.isoformat(),
        "last_nudge_at": ahora.isoformat(),
    })
    return {"accion": "aviso", "vasos": vasos, "meta": meta, "hora": hora_pc}


def run_hydration_checks() -> dict:
    """Cron horario (a y 32). Por usuario, best-effort: el fallo de uno no corta a los demás."""
    cuenta = {"usuarios": 0, "aviso": 0, "apagado": 0, "al_dia": 0, "errores": 0}
    if not _encendidos():
        return cuenta
    try:
        usuarios = usuarios_alcanzables_con_agua()
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-135] usuarios con hidratación no leídos: {e}")
        return cuenta
    for u in usuarios:
        uid = str(u.get("user_id") or "")
        if not uid:
            continue
        # la misma puerta que los recordatorios de comida: con turno nocturno o rotativo «las 11» no significa nada
        if str(u.get("schedule") or "standard") in ("night_shift", "variable"):
            continue
        cuenta["usuarios"] += 1
        try:
            r = revisar_usuario(uid, u.get("locale"), bool(u.get("con_push")))
            if r.get("accion") in cuenta:
                cuenta[r["accion"]] += 1
        except Exception as e:
            cuenta["errores"] += 1
            logger.warning(f"[P1-PLAN-LOTE-135] hidratación de {uid}: {e!r}")
    if cuenta["aviso"] or cuenta["apagado"] or cuenta["errores"]:
        logger.info(f"💧 [P1-PLAN-LOTE-135] hidratación: {cuenta}")
    return cuenta
