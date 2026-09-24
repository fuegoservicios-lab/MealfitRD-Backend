import os
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
# [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] Gemini → GLM.
from llm_provider import ChatGLM, GLM_FLASH

from db_core import connection_pool, execute_sql_query, execute_sql_write
import uuid
# [P1-PLAN-LOTE-159] `get_or_create_session`: el aviso de hoy abre el chat de hoy cuando la
# última conversación es de un día anterior.
from db_chat import save_message, get_recent_messages, get_or_create_session
from db import get_consumed_meals_today, get_user_profile, user_tz_offset_min
from fact_extractor import get_embedding
from knobs import _env_bool, _env_int, _env_float

logger = logging.getLogger(__name__)

from prompts.proactive import PROACTIVE_PROMPT
# [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] Addendum del dueño §2: el nudge proactivo es
# la MISMA voz que el coach del chat — mismo directive, mismo builder SSOT (no reimplementa
# el texto). Ver `run_proactive_checks` para los 2 call sites.
from prompts.chat_agent import build_language_directive


# [P3-PREVIEW-MODEL-KNOB · 2026-05-12] Knob para overridear el modelo LLM
# usado por las 2 callsites del proactive agent sin redeploy:
#   - `classify_nudge_sentiment` (analiza respuestas del usuario al nudge).
#   - `_compose_proactive_message` (genera el texto del nudge).
#
# [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] Default GLM-5.3 Flash: nudges y
# clasificación de sentiment son tareas aux baratas — mismo modelo para
# todos los tiers. Swap sin redeploy:
# `MEALFIT_PROACTIVE_SENTIMENT_MODEL=glm-5.3` + restart del worker.
def _proactive_model_name() -> str:
    return os.environ.get(
        "MEALFIT_PROACTIVE_SENTIMENT_MODEL",
        GLM_FLASH,
    )


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Timeout per-invoke de los 2 constructores
# `ChatGoogleGenerativeAI` del proactive agent: `classify_nudge_sentiment` (148)
# y el compose-nudge del cron `run_proactive_checks` (457). Pre-fix: sin
# `timeout=`. `run_proactive_checks` es SÍNCRONO y corre en el threadpool de
# APScheduler con `max_instances=1`: si Gemini cuelga un socket, el invoke
# bloquea el thread del cron indefinidamente → el slot del job queda tomado y el
# nudge cron NUNCA vuelve a correr (no dispara MISSED ni ERROR — está "running",
# no errored → ningún watchdog lo ve). El budget wall-clock (_max_runtime_s) solo
# se chequea al tope de cada iteración de usuario, no puede abortar un invoke en
# vuelo. El `timeout=` propaga al deadline gRPC → DeadlineExceeded, capturado por
# los `except Exception` existentes (compose por-usuario contenido). Default 20s
# (flash-lite, prompt corto); clamp (0, 120]. Knob auto-registrado.
# Tooltip-anchor: P2-LLM-TIMEOUT-SWEEP.
def _proactive_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_PROACTIVE_LLM_TIMEOUT_S",
        20.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )


# [P1-PROACTIVE-TZ · 2026-05-30] Offset (en minutos, UTC→local sumando) de la
# zona horaria dominicana (AST = UTC-4, sin DST). El cron computa `now_ast`
# con `-4h` hardcodeado; este knob mantiene la MISMA constante para el filtro
# de comidas consumidas y la convierte en operacional sin redeploy si DR
# adoptara DST. Convención `getTimezoneOffset()` de JS = +240 para UTC-4 (lo
# que `get_consumed_meals_today` suma a la fecha local para ir a UTC).
# Tooltip-anchor: P1-PROACTIVE-TZ.
def _proactive_tz_offset_min() -> int:
    # [P1-NUDGE-TZ-PER-USER · 2026-08-21] El clamp era `0 <= v <= 720`, que RECHAZA
    # estructuralmente cualquier offset negativo: Europa era inexpresable incluso como override
    # manual. Un knob que no puede representar el caso que debería mitigar no es una mitigación.
    # Ahora ±840 min (±14 h), el rango real de husos IANA. Sigue siendo sólo el FALLBACK: el
    # disparo por usuario lo decide `user_tz_offset_min`.
    return _env_int(
        "MEALFIT_PROACTIVE_TZ_OFFSET_MIN",
        240,
        validator=lambda v: -840 <= v <= 840,
    )


# [P1-PLAN-LOTE-72 · 2026-09-16] Franja local (desde, hasta) en la que un registro dice cuándo se COME esa comida.
# `consumed_at` es la hora del registro: el dueño anotó su desayuno a las 12:58 y, promediado, su recordatorio del
# desayuno pasó a las 14:30 —encima del del almuerzo, que se perdió—. Fuera de la franja, el registro es tardío (o
# adelantado) y no mueve el aviso. `desde > hasta` = la franja cruza la medianoche (cenas tardías). Holgadas a
# propósito: caben el almuerzo español de las 15:00 y su cena de las 23:00.
FRANJA_DE_COMIDA = {
    "Desayuno": (4.0, 12.0),
    "Almuerzo": (10.5, 17.0),
    "Merienda": (14.0, 20.0),
    "Cena": (17.0, 3.0),
}


# [P1-PLAN-LOTE-73 · 2026-09-16] El verbo de cada comida para el aviso. El dueño recibió «¿Ya cenaste tu
# merienda de la tarde?»: el prompt solo nombraba la comida y el modelo tomó el verbo de otra.
# [P1-PLAN-LOTE-150] El aviso llega ANTES de la comida, así que el prompt necesita el infinitivo («es tu hora de
# desayunar») además del pretérito. Se mantienen los dos: el verbo de cada comida sigue siendo suyo (lote 73).
INFINITIVO_DE_COMIDA = {
    "Desayuno": "desayunar", "Almuerzo": "almorzar", "Merienda": "merendar", "Cena": "cenar",
}
VERBO_DE_COMIDA = {
    "Desayuno": "desayunaste",
    "Almuerzo": "almorzaste",
    "Merienda": "merendaste",
    "Cena": "cenaste",
}


# [P1-PLAN-LOTE-133 · 2026-09-20] Las horas por defecto y la hora del aviso, fuera del bucle del cron: la app nativa
# programa los MISMOS recordatorios en el teléfono (`GET /api/notifications/meal-reminders`) y dos copias de esta cuenta
# habrían acabado avisando a horas distintas en el chat y en la pantalla.
HORAS_POR_DEFECTO_DE_COMIDA = {"Desayuno": 9.0, "Almuerzo": 13.0, "Merienda": 16.0, "Cena": 19.5}


# [P1-PLAN-LOTE-216 · 2026-09-24] LA HORA DE CADA AVISO LA ELIGE LA PERSONA.
#
# El dueño, a la 1:18 p. m.: «hoy nada más me llegó la notificación del desayuno… son la 1 de la tarde». Su almuerzo
# estaba programado hacia las 2:15: la «hora habitual» salía de `consumed_at`, que es la hora del REGISTRO, y él anota
# después de comer (el 23-sep, desayuno y almuerzo juntos a la 1:36 p. m.). Una hora de registro es siempre POSTERIOR a
# la comida, así que promediarla empuja el aviso tarde justo a quien anota tarde, y el aviso quiere llegar ANTES de
# comer (lote 150). Los lotes 72, 83 y 151 fueron parches sobre esa misma señal (franja, días pasados, techo): el dato
# no dice lo que se le pedía.
#
# Ahora cada comida tiene en Configuración su interruptor y su hora (`health_profile.avisos_por_comida`). Sin tocar
# nada: las horas normales menos la antelación → 8:45, 12:45, 15:45 y 19:15. El cálculo por historial queda detrás de
# `MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY` (apagado), por si hubiera que volver a él sin desplegar; una hora elegida
# gana siempre. tooltip-anchor: P1-PLAN-LOTE-216-HORA-ELEGIDA
CLAVE_AVISOS_POR_COMIDA = "avisos_por_comida"
# La clave de cada comida dentro de `avisos_por_comida` (la misma, en minúscula, que `meal_reminders` manda al teléfono).
CLAVE_DE_COMIDA = {"Desayuno": "desayuno", "Almuerzo": "almuerzo", "Merienda": "merienda", "Cena": "cena"}
_HORA_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
# La hora local del «Resumen del día». En esa hora el cron no recuerda comidas (solo manda el resumen), así que una
# hora elegida tiene que quedar ANTES: a las 23:10 el teléfono sonaría y el chat no escribiría nada.
HORA_DEL_RESUMEN = 23


def minuto_del_dia(hora: float) -> int:
    """La hora del aviso en minutos del día (0..1439), REDONDEANDO. [P1-PLAN-LOTE-150] El coach truncaba («6:49») y el
    teléfono redondeaba («6:50»): el mismo aviso con dos horas. [P1-PLAN-LOTE-216] Una sola función para las dos vías
    (el cron y `meal_reminders`), en vez de dos expresiones que un test comparaba letra a letra."""
    return int(round(float(hora) * 60)) % (24 * 60)


def hora_hhmm_a_float(valor) -> Optional[float]:
    """`"12:45"` → 12.75. `None` si no es una hora «HH:MM» de 00:00 a 23:59 (lo que no se entiende no se inventa)."""
    m = _HORA_HHMM.match(valor) if isinstance(valor, str) else None
    return int(m.group(1)) + int(m.group(2)) / 60.0 if m else None


def _config_de_la_comida(health: dict, meal: str) -> dict:
    todas = (health or {}).get(CLAVE_AVISOS_POR_COMIDA) if isinstance(health, dict) else None
    propia = todas.get(CLAVE_DE_COMIDA.get(meal, "")) if isinstance(todas, dict) else None
    return propia if isinstance(propia, dict) else {}


def error_en_avisos_por_comida(valor) -> Optional[str]:
    """Por qué `valor` no vale como `avisos_por_comida`, o `None` si vale. Lo usa `PATCH /api/profile`.

    Forma: `{"almuerzo": {"activo": true, "hora": "12:45"}, ...}` con cualquier subconjunto de las cuatro comidas;
    `null` borra la configuración (vuelven las horas normales). Se rechaza en vez de corregir: una hora que no se
    entiende guardada tal cual haría que el lector la ignorara en silencio y el usuario no sabría por qué no le llega."""
    if valor is None:
        return None
    if not isinstance(valor, dict):
        return "avisos_por_comida debe ser un objeto por comida."
    for clave, cfg in valor.items():
        if clave not in CLAVE_DE_COMIDA.values():
            return f"Comida desconocida en avisos_por_comida: {clave!r}. Permitidas: {sorted(CLAVE_DE_COMIDA.values())}."
        if not isinstance(cfg, dict) or set(cfg) - {"activo", "hora"}:
            return f"avisos_por_comida.{clave} solo admite «activo» y «hora»."
        if "activo" in cfg and not isinstance(cfg["activo"], bool):
            return f"avisos_por_comida.{clave}.activo debe ser true o false."
        if cfg.get("hora") is None:
            continue
        hora = hora_hhmm_a_float(cfg["hora"])
        if hora is None:
            return f"avisos_por_comida.{clave}.hora debe ser «HH:MM» (00:00 a 23:59)."
        # Solo horas en las que suenan LOS DOS: en el silencio de la madrugada no sale ninguno, y desde la hora del
        # resumen el cron solo manda el resumen — el teléfono sonaría sin mensaje en el chat.
        desde = _hora_de_silencio()
        if not desde <= hora < HORA_DEL_RESUMEN:
            return (f"avisos_por_comida.{clave}.hora debe estar entre las {desde:02d}:00 y las "
                    f"{HORA_DEL_RESUMEN - 1:02d}:59: antes son horas de silencio y a las {HORA_DEL_RESUMEN:02d}:00 "
                    f"llega el resumen del día.")
    return None


def comida_con_aviso(health: dict, meal: str) -> bool:
    """¿Quiere el recordatorio de ESTA comida? Ausente ⇒ sí, como `avisos_comida`: solo lo apaga un `False` explícito."""
    return _config_de_la_comida(health, meal).get("activo") is not False


def hora_elegida(health: dict, meal: str) -> Optional[float]:
    """La hora que la persona eligió para el aviso de `meal` (0..23,99), o `None` si no eligió ninguna."""
    return hora_hhmm_a_float(_config_de_la_comida(health, meal).get("hora"))


def _avisos_desde_historial() -> bool:
    """Knob de vuelta atrás: `True` devuelve la hora calculada con lo registrado a quien no eligió ninguna."""
    return _env_bool("MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY", False)


def hora_del_aviso(user_id: str, meal: str, def_hour: float, health: Optional[dict] = None) -> float:
    """La hora local (0..23,99) a la que suena el recordatorio de `meal`. La MISMA para el teléfono y el chat.

    1. La que la persona eligió en Configuración, si eligió una.
    2. Si no, la hora normal de esa comida menos la antelación (`MEALFIT_PROACTIVE_NUDGE_LEAD_H`, 15 min): 8:45,
       12:45, 15:45, 19:15 — el aviso llega cuando aún puedes comer.
    3. Solo con `MEALFIT_PROACTIVE_NUDGE_FROM_HISTORY`, la conducta de los lotes 72-151: hora habitual (media circular
       de 14 días dentro de su franja, acotada por arriba) menos la antelación.

    [P1-PLAN-LOTE-150 · 2026-09-21] Antes era una ESPERA de 1,5 h y el dueño lo dijo claro: «solo avisa al rato
    después del horario… quiero que a las 8:45 me anime a desayunarme si todavía no he registrado ningún desayuno».
    Con el desayuno a las 9:00 el aviso caía a las 10:35, cuando o ya comiste —y es ruido— o ya vas tarde —y es un
    reproche—; en ninguno de los dos casos puedes hacer nada con él. **Se MUEVE, no se añade otro**: dos avisos por
    comida serían ocho al día con los del agua, y en iOS quien se harta no apaga un ajuste, apaga TODAS las
    notificaciones de la app y se lleva por delante las que importan (plan listo, semana nueva). Eso no se revierte
    desde aquí."""
    elegida = hora_elegida(health, meal)
    if elegida is not None:
        return elegida % 24
    if not _avisos_desde_historial():
        return (float(def_hour) - _antelacion_del_aviso_h()) % 24

    from db_facts import get_avg_meal_hour
    avg_hr = get_avg_meal_hour(user_id, meal, ventana=FRANJA_DE_COMIDA.get(meal))
    if avg_hr is None:
        avg_hr = def_hour
    avg_hr = _acotar_a_su_franja(avg_hr, def_hour, meal)
    delay_hours = -_antelacion_del_aviso_h()

    # Nudge dinámico ajustado según historial de adherencia específica.
    #
    # [P3-AVG-MEAL-HOUR-CIRCULAR · 2026-08-23] El `% 24` es LOAD-BEARING y va
    # ANTES que el arreglo de la media, no después. `avg_hr` es una hora de
    # reloj (0..23,99) y `delay_hours` llega hasta 2,5: la suma se sale del
    # reloj en cuanto la comida es tardía. `current_hour_float` sólo vale
    # 0..23, así que un `nudge_hour` de 24,0 NUNCA iguala a nada y el nudge
    # de esa comida no se envía JAMÁS — silenciosamente, sin log ni error.
    #
    # Es un defecto con sesgo de país: una cena dominicana (~19:00) + 1,5 h
    # cae en 20,5 y no se nota; una cena española (21:00-23:00, con picoteo)
    # cae en 22,5-25,5 y se cae del reloj. Y ES PRERREQUISITO de la media
    # circular: la media circular de [21,22,23,0] es ~22,5 (correcta), pero
    # sumarle el delay sin `% 24` la empuja fuera del reloj más a menudo que
    # la media aritmética rota, que tendía al centro del día. Arreglar la
    # media sin arreglar esto EMPEORA el caso español.
    #
    # El nudge cruza la medianoche a propósito: si comes a las 23:30, el
    # recordatorio de esa comida es a la 1:00 del día siguiente, no "nunca".
    # [P1-PLAN-LOTE-150] Con la antelación el `% 24` sigue siendo load-bearing, ahora por el otro lado: un desayuno
    # a las 00:10 menos 15 min da -0,08, y en Python el módulo de un negativo con divisor positivo vuelve al reloj
    # (23,92). El cruce de medianoche se conserva en las DOS direcciones.
    # tooltip-anchor: P3-AVG-MEAL-HOUR-CIRCULAR
    nudge_hour = (avg_hr + delay_hours) % 24
    return nudge_hour


def hora_de_aviso(user_id: str, meal: str, def_hour: float, health: Optional[dict] = None):
    """`(hora, meal_rate, meal_total)`: la hora de `hora_del_aviso` y la tasa de respuesta a los avisos de esa comida.

    La tasa ya NO mueve la hora (lote 150: a quien ignoraba el aviso se le avisaba MÁS TARDE, con lo que era aún menos
    útil); sigue sirviendo donde sí ayuda: el TONO del mensaje del cron."""
    meal_rate, meal_total = get_nudge_response_rate(user_id, meal)
    return hora_del_aviso(user_id, meal, def_hour, health), meal_rate, meal_total


def _banda_del_aviso_h() -> float:
    """[P1-PLAN-LOTE-151] Cuánto puede alejarse el aviso de la hora normal de esa comida, en horas."""
    return _env_float("MEALFIT_PROACTIVE_NUDGE_BAND_H", 1.5, validator=lambda v: 0.5 <= v <= 12.0)


def _acotar_a_su_franja(avg_hr: float, def_hour: float, meal: str) -> float:
    """[P1-PLAN-LOTE-151 · 2026-09-21] La hora habitual, acotada por ARRIBA a la franja de su comida.

    Medido en la cuenta del dueño el 21-sep, con los avisos ya encendidos en su teléfono: el del almuerzo salía a
    las **15:06** y el de la merienda a las 15:45 — dos avisos a 39 minutos, y el del almuerzo a una hora que ya no
    es de almorzar. No es un fallo de la cuenta: la hora sale de la MEDIA de lo que registra, y con 4-5 muestras una
    comida tardía arrastra la media fuera de cualquier rango sensato. El dueño eligió acotar («la 1»).

    **Solo se acota el lado TARDÍO, y es deliberado.** Acotar por abajo rompería lo que el lote 150 vino a
    conseguir: a quien de verdad desayuna a las 7:05 le movería el aviso de las 6:50 a las 7:15, o sea DESPUÉS de
    su desayuno — el defecto original, reintroducido por el arreglo. Con una sola dirección, el acotado nunca puede
    retrasar un aviso: o lo adelanta o lo deja igual.

    La distancia se mide en el reloj (circular), no restando: para una cena de las 19:30, una media de las 00:06 no
    está «19 horas antes», está 4,6 h DESPUÉS. Restar daría lo contrario y adelantaría el aviso a media tarde.
    """
    banda = _banda_del_aviso_h()
    distancia = (float(avg_hr) - float(def_hour) + 12.0) % 24.0 - 12.0
    if distancia > banda:
        return (float(def_hour) + banda) % 24.0
    return float(avg_hr) % 24.0


def avisos_de_comida_activos(health: dict) -> bool:
    """[P1-PLAN-LOTE-150 · 2026-09-21] ¿Quiere esta persona los recordatorios de COMIDA?

    El dueño: «quiero que se pueda desactivar y activar en configuraciones esa opción, ya que con eso tendría la
    opción de tener más tranquilidad al tener menos notificaciones». Son DOS interruptores y no uno global porque
    «menos notificaciones» casi siempre significa «unas sí y otras no».

    Vive en `user_profiles.health_profile`, que ya es jsonb libre y ya tiene su endpoint de merge
    (`PATCH /api/profile`): ni columna nueva ni migración. **Ausente ⇒ ACTIVO**: quien ya los tenía no se queda sin
    ellos por un despliegue, y solo apagarlo explícitamente los quita. Lo consultan las DOS vías —el horario que
    programa el teléfono y el cron que manda el aviso— para que apagarlo signifique lo mismo en las dos."""
    return (health or {}).get("avisos_comida") is not False


def avisos_de_agua_activos(health: dict) -> bool:
    """[P1-PLAN-LOTE-150] El gemelo del de arriba para la hidratación. Ausente ⇒ activo."""
    return (health or {}).get("avisos_agua") is not False


# [P1-PLAN-LOTE-161] Los centinelas de «nada declarado» del formulario: no son datos clínicos.
_SIN_DATO_CLINICO = {"", "ninguna", "ninguno", "no", "n/a", "na", "nada", "none"}


def _lista_del_perfil(health: dict, *claves) -> list:
    """Los valores declarados bajo `claves` (lista o texto separado por comas), sin centinelas ni repetidos."""
    out = []
    for clave in claves:
        v = (health or {}).get(clave)
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",")]
        if not isinstance(v, list):
            continue
        for x in v:
            s = str(x or "").strip()
            if s and s.lower() not in _SIN_DATO_CLINICO and s not in out:
                out.append(s)
    return out


def contexto_del_aviso(health: dict) -> dict:
    """[P1-PLAN-LOTE-161 · 2026-09-22] Lo que el aviso de comida tiene que saber de la persona.

    El cron leía `dietTypes` y `goals`, que el formulario NO guarda (guarda `dietType` y `mainGoal`): todos los
    avisos salían con «balanceada / mantener de manera saludable». Y no recibía ni alergias ni condiciones: un tercio
    de las veces el estilo sorteado pide «aportar opciones», así que un vegano o un alérgico al huevo podía leer
    «¿qué tal unos huevos revueltos?». Es el único texto del coach sin filtro determinista detrás.

    Devuelve `dieta` y `objetivo` para la plantilla (con los nombres viejos como respaldo, por si algún perfil antiguo
    los tiene), el `bloque` de restricciones que se AÑADE al prompt ("" si no hay nada que decir: un bloque vacío que
    dice «ninguna alergia» da una certeza que no se tiene) y `restringido`, que apaga el estilo que nombra comida.
    tooltip-anchor: P1-PLAN-LOTE-161-AVISO-CON-RESTRICCIONES
    """
    h = health if isinstance(health, dict) else {}
    dieta = h.get("dietType")
    if not isinstance(dieta, str) or not dieta.strip():
        _viejas = h.get("dietTypes")
        dieta = str(_viejas[0]) if isinstance(_viejas, list) and _viejas else "balanceada"
    objetivo = h.get("mainGoal")
    if not isinstance(objetivo, str) or not objetivo.strip():
        _viejos = h.get("goals")
        objetivo = ", ".join(str(x) for x in _viejos) if isinstance(_viejos, list) and _viejos else "mantener de manera saludable"

    alergias = _lista_del_perfil(h, "allergies", "otherAllergies")
    condiciones = _lista_del_perfil(h, "medicalConditions", "otherConditions")
    no_le_gusta = _lista_del_perfil(h, "dislikes", "otherDislikes")
    try:
        from constants import canonicalize_diet_type
        dieta_restrictiva = canonicalize_diet_type(dieta) != "balanced"
    except Exception:
        dieta_restrictiva = False

    lineas = []
    if alergias:
        lineas.append(f"- ALERGIAS / INTOLERANCIAS: {', '.join(alergias)}.")
    if dieta_restrictiva:
        lineas.append(f"- DIETA: {dieta}.")
    if condiciones:
        lineas.append(f"- CONDICIONES MÉDICAS: {', '.join(condiciones)}.")
    if no_le_gusta:
        lineas.append(f"- NO LE GUSTA: {', '.join(no_le_gusta)}.")
    bloque = ""
    if lineas:
        bloque = (
            "\n\n🛑 RESTRICCIONES DEL PACIENTE — PRIORIDAD 1, POR ENCIMA DEL ESTILO Y DEL TONO:\n"
            + "\n".join(lineas)
            + "\nNUNCA nombres un alimento que choque con esto, ni siquiera como ejemplo. Si dudas de que un alimento sea "
              "compatible, no lo nombres: anima sin sugerir platos.\n"
        )
    return {
        "dieta": dieta,
        "objetivo": objetivo,
        "bloque": bloque,
        "restringido": bool(alergias or dieta_restrictiva or condiciones),
    }


def _antelacion_del_aviso_h() -> float:
    """[P1-PLAN-LOTE-150] Cuánto ANTES de tu hora habitual llega el recordatorio de esa comida. 0 = a la hora exacta.
    El tope de 2 h evita que el aviso de una comida se solape con el de la anterior."""
    return _env_float("MEALFIT_PROACTIVE_NUDGE_LEAD_H", 0.25, validator=lambda v: 0.0 <= v <= 2.0)


def _max_avisos_por_dia() -> int:
    """[P1-PLAN-LOTE-72] Tope anti-fatiga de recordatorios por día. Era un `2` fijo, y con cuatro comidas dos
    avisos dejaban sin recordatorio a la merienda y la cena. El dueño lo subió a 4 (16-sep): uno por comida. El
    Resumen de las 23:00 comparte el tope, así que a quien no registró nada y ya recibió los cuatro no le llega."""
    return _env_int("MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY", 4, validator=lambda v: 0 <= v <= 8)


def _ventana_de_respuesta_min() -> int:
    """[P1-PLAN-LOTE-133 · 2026-09-20] Minutos durante los que un mensaje del usuario cuenta como RESPUESTA al último
    aviso. Eran 60 fijos: el dueño contestó el del desayuno a las 2 h 27 min y quedó como «ignorado» (abierto desde el
    lote 72). Nadie contesta una notificación en menos de una hora por obligación."""
    return _env_int("MEALFIT_PROACTIVE_RESPONSE_WINDOW_MIN", 180, validator=lambda v: 15 <= v <= 720)


# [P1-PLAN-LOTE-216 · 2026-09-24] Cada cuántos minutos corre el cron (app.py lo registra con este mismo número).
#
# Corría a y media y escribía el mensaje del chat en el tick de la HORA del aviso, mientras el teléfono sonaba al
# minuto exacto (lote 150). Con un aviso a las 2:15 el teléfono sonaba a las 2:15 y el coach escribía a las 2:30: al
# tocar la notificación el chat seguía mostrando el mensaje del desayuno — lo que vio el dueño. Y con la cena por
# defecto (19:15) le pasaba a todos. Ahora el mensaje sale en el último tick ANTES de que suene el teléfono: nunca
# después, y como mucho 15 min antes (que es además cuando llega la Web Push, que sale con él).
MINUTOS_ENTRE_TICKS = 15


def _horas_de_reintento() -> int:
    """[P1-PLAN-LOTE-72] Horas durante las que un aviso sigue tocando desde su hora. El cron corre una vez por hora:
    con 1 (la conducta de antes) un aviso que no pudo salir en su hora —el coach respondió hace menos de una hora,
    la IA no contestó, un despliegue a y media— se perdía para todo el día."""
    return _env_int("MEALFIT_PROACTIVE_NUDGE_RETRY_HOURS", 3, validator=lambda v: 1 <= v <= 6)


def _hora_de_silencio() -> int:
    """[P1-PLAN-LOTE-83 · 2026-09-17] Hora local hasta la que NO sale ningún recordatorio de comida. El dueño recibió
    «Aún no veo registrada tu cena; ¿ya cenaste?» a las 2:30 de la madrugada: había anotado a las 00:38 la cena de
    AYER, el registro quedó como una cena a las 00:38 (dentro de la franja 17→3) y la hora media la arrastró. La media
    ya no cuenta registros de días pasados, y además a esas horas no se pregunta nada: quien cena de verdad a las
    23:30 no recibe el aviso de la 1:00 (P3-AVG-MEAL-HOUR-CIRCULAR lo quería «nunca» ≠ «a la 1:00»; el Resumen de
    las 23:00 ya cubre ese día). 0 = sin silencio."""
    return _env_int("MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR", 6, validator=lambda v: 0 <= v <= 12)


def _comidas_avisadas_hoy(user_id: str):
    """[P1-PLAN-LOTE-72] Comidas que YA recibieron recordatorio hoy (día local del usuario), para que el reintento no
    repita un aviso. `None` si no se puede saber: el llamador vuelve entonces a la hora exacta, sin reintentos."""
    try:
        _tz_off = user_tz_offset_min(user_id)
        filas = execute_sql_query(
            "SELECT DISTINCT nudge_type FROM nudge_outcomes "
            "WHERE user_id = %s "
            "AND (sent_at - make_interval(mins => %s))::date "
            "= (NOW() - make_interval(mins => %s))::date",
            (user_id, _tz_off, _tz_off), fetch_all=True,
        )
        return {str(f.get("nudge_type") or "") for f in (filas or [])}
    except Exception as e:
        logger.error(f"[P1-PLAN-LOTE-72] No se pudieron leer los avisos de hoy de {user_id}: {e}")
        return None


def _comida_ya_registrada(consumed, meal: str) -> bool:
    """¿Hay hoy un registro de esa comida (por tipo o por nombre)? Misma regla que tenía el bucle, sin reventar con
    un `meal_type` nulo."""
    objetivo = str(meal).lower()
    for m in consumed or []:
        mt = str(m.get("meal_type") or "").lower()
        mn = str(m.get("meal_name") or "").lower()
        if objetivo in mt or objetivo in mn:
            return True
    return False


def _local_hour_float_for_offset(now_utc, tz_offset_min) -> float:
    """[P1-NUDGE-TZ-PER-USER · 2026-08-21] Hora local del usuario como float (13.5 = 13:30).

    `tz_offset_min` sigue la convención de `getTimezoneOffset()` de JS que usa todo el sistema:
    minutos que hay que SUMAR a la hora local para llegar a UTC (RD = +240, España verano = −120).
    Por eso se RESTA aquí.

    Existe para que el reloj del cron viva DENTRO del bucle por usuario. Antes
    `run_proactive_checks` calculaba `now_ast = datetime.now(timezone(timedelta(hours=-4)))` una
    sola vez, fuera del bucle, y con eso decidía el disparo de TODAS las sesiones activas: a un
    español el «Resumen del día» —que manda una notificación push real— le llegaba a las 05:00.

    tooltip-anchor: _local_hour_float_for_offset (test_p1_nudge_tz_per_user.py)"""
    from datetime import timedelta
    try:
        _off = int(tz_offset_min)
    except Exception:
        _off = 240
    local = now_utc - timedelta(minutes=_off)
    return local.hour + local.minute / 60.0

def _usuario_en_modo_contador(user_id) -> bool:
    """[P2-CHAT-PLAN-TOOLS-PAUSE · 2026-08-15] ¿Este usuario apagó la generación?

    El coach proactivo era el único camino que no pasa por `_plan_context_for_chat`,
    así que necesita su propia lectura del modo. Reusa `plan_mode.get_plan_mode`
    —la MISMA fuente que el chat— en vez de consultar `user_profiles` por su cuenta:
    dos lecturas del mismo campo drift ean en cuanto una de las dos aprende algo.

    Fail-open a False (asumir 'plan'): un fallo de DB no puede cambiar el mensaje
    que reciben todos los usuarios normales, que son la inmensa mayoría.
    """
    try:
        from plan_mode import get_plan_mode
        return str((get_plan_mode(user_id) or {}).get("plan_mode") or "plan") == "tracking"
    except Exception as e:
        logger.warning(f"[P2-CHAT-PLAN-TOOLS-PAUSE] plan_mode ilegible para {user_id}: {e}")
        return False


def _sesion_del_dia_para_aviso(session_id, user_id, ahora_utc, tz_offset_min):
    """[P1-PLAN-LOTE-159 · 2026-09-22] La sesión donde escribir el aviso de HOY.

    Devuelve `session_id` si esa conversación ya es la del día del usuario, y una sesión NUEVA
    si la última actividad es de un día anterior. La frontera es el día LOCAL de cada usuario
    (`tz_offset_min`), el mismo reloj con el que se decide la hora del aviso dos líneas arriba.

    Tres decisiones que no son obvias:

    · **Por el último mensaje, no por `created_at`.** Una sesión abierta anoche en la que se
      sigue hablando a las 00:30 es la de hoy; mirar su nacimiento la partiría en dos justo
      cuando el usuario está escribiendo. Es el criterio del cliente (`debeRenovarse`).
    · **Una sesión sin mensajes se respeta.** Es una recién creada —por el propio cliente, que
      la abre vacía al renovar— y fabricar otra dejaría dos chats vacíos el mismo día.
    · **Fail-open.** Si la consulta falla, se devuelve la sesión que había: un aviso en el chat
      de ayer es peor que uno de hoy, pero mucho mejor que ninguno.

    `session_id` a `None` (suscriptor sin chat reciente, P1-PLAN-LOTE-133) se devuelve tal cual:
    esa rama manda el aviso a la pantalla A PROPÓSITO, sin escribir en ningún chat.
    """
    if not session_id or not user_id:
        return session_id
    try:
        fila = execute_sql_query(
            "SELECT MAX(created_at) AS ultima FROM public.agent_messages WHERE session_id = %s",
            (str(session_id),),
            fetch_one=True,
        )
        ultima = (fila or {}).get("ultima")
        if ultima is None:
            return session_id          # sesión aún vacía: es la que el cliente acaba de abrir
        hoy_local = (ahora_utc - timedelta(minutes=tz_offset_min)).date()
        dia_de_la_sesion = (ultima - timedelta(minutes=tz_offset_min)).date()
        if dia_de_la_sesion >= hoy_local:
            return session_id          # ya es la conversación de hoy
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-LOTE-159] no se pudo fechar la sesión {session_id}: {e}. "
            f"Se usa la que había (conducta previa)."
        )
        return session_id

    nueva = str(uuid.uuid4())
    try:
        get_or_create_session(nueva, user_id=user_id)
    except Exception as e:
        logger.error(
            f"❌ [P1-PLAN-LOTE-159] no se pudo abrir el chat del día para {user_id}: {e}. "
            f"El aviso va a la sesión anterior."
        )
        return session_id
    logger.info(
        f"🗓️ [P1-PLAN-LOTE-159] chat del día para {user_id}: {nueva} "
        f"(la anterior, {session_id}, es de {dia_de_la_sesion})"
    )
    return nueva


def get_active_users_for_proactive() -> list:
    """Busca session_ids que pertenezcan a usuarios registrados con actividad reciente."""
    try:
        # Obtenemos sesiones que sí tienen un user_id y han estado activas en los últimos 3 días (72 hrs)
        #
        # [P1-PLAN-LOTE-161 · 2026-09-22] «Activo» lo decide una sesión que abrió una PERSONA en los últimos 3
        # días: la que tiene algún mensaje suyo, o la vacía que abre el cliente al renovar el chat del día. NO
        # cuenta la que abre este mismo cron para escribir un aviso (solo mensajes del modelo): desde el lote 159
        # el cron crea una sesión al día, y medido solo por `created_at` quien abandonaba la app seguía «activo»
        # para siempre — cuatro avisos diarios con IA escritos en chats que nadie abre. La sesión DEVUELTA sigue
        # siendo la más reciente del usuario (también la del aviso de hoy), para que el segundo aviso del día caiga
        # en el mismo chat que el primero y no abra otro. tooltip-anchor: P1-PLAN-LOTE-161-ACTIVO-POR-PERSONA
        if not connection_pool: return []
        query = (
            "WITH activos AS ("
            " SELECT DISTINCT s.user_id FROM agent_sessions s"
            " WHERE s.user_id IS NOT NULL"
            " AND s.user_id::text != 'guest'"
            " AND s.created_at >= NOW() - INTERVAL '3 days'"
            " AND (EXISTS (SELECT 1 FROM agent_messages m WHERE m.session_id = s.id AND m.role = 'user')"
            " OR NOT EXISTS (SELECT 1 FROM agent_messages m WHERE m.session_id = s.id))"
            ") "
            "SELECT DISTINCT ON (s.user_id) s.id, s.user_id "
            "FROM agent_sessions s JOIN activos a ON a.user_id = s.user_id "
            "ORDER BY s.user_id, s.created_at DESC"
        )
        res = execute_sql_query(query, fetch_all=True)
        res = list(res) if res else []
        # [P1-PLAN-LOTE-133 · 2026-09-20] Quien encendió «Alertas Inteligentes» y lleva más de 3 días sin abrir el chat
        # es justo a quien más falta le hace el recordatorio, y aquí dejaba de existir: la lista salía solo de
        # `agent_sessions`. Entra con `id = None`: recibe el aviso corto en su pantalla, sin LLM y sin escribir un
        # mensaje en un chat viejo que nadie va a abrir.
        try:
            _con_chat = {str(r.get("user_id")) for r in res}
            _subs = execute_sql_query("SELECT DISTINCT user_id FROM push_subscriptions WHERE user_id IS NOT NULL",
                                      fetch_all=True) or []
            for _r in _subs:
                _uid = str(_r.get("user_id"))
                if _uid and _uid not in _con_chat:
                    res.append({"id": None, "user_id": _uid})
                    _con_chat.add(_uid)
        except Exception as _e_subs:
            logger.warning(f"[P1-PLAN-LOTE-133] suscriptores sin chat reciente no leídos: {_e_subs}")
        return res
    except Exception as e:
        logger.error(f"Error fetching active sessions for proactive check: {e}")
        return []

def get_best_nudge_style(user_id: str) -> str:
    """Implementa A/B testing (Epsilon-Greedy) para formatos de mensajes."""
    import random
    styles = ["directo", "sugestivo", "gamificado"]
    try:
        query = """
            SELECT nudge_style, 
                   COUNT(*) as total, 
                   SUM(CASE WHEN meal_logged THEN 1 ELSE 0 END) as successes
            FROM nudge_outcomes 
            WHERE user_id = %s AND nudge_style IS NOT NULL
            GROUP BY nudge_style
        """
        stats = execute_sql_query(query, (user_id,), fetch_all=True)
        
        total_nudges = sum(s['total'] for s in stats) if stats else 0
        
        if total_nudges < 10:
            return random.choice(styles)
            
        best_style = None
        best_rate = -1.0
        for s in stats:
            rate = s['successes'] / float(s['total'])
            if rate > best_rate:
                best_rate = rate
                best_style = s['nudge_style']
                
        if random.random() < 0.1 or not best_style:
            return random.choice(styles)
            
        return best_style
    except Exception as e:
        logger.error(f"Error calculando mejor nudge style: {e}")
        return random.choice(styles)

def log_nudge_outcome(user_id, nudge_type, context_embedding=None, context_summary=None, nudge_content=None, nudge_style=None):
    try:
        if context_embedding and context_summary and nudge_content:
            emb_str = f"[{','.join(map(str, context_embedding))}]"
            query = """INSERT INTO nudge_outcomes 
                       (user_id, nudge_type, sent_at, responded, meal_logged, context_embedding, context_summary, nudge_content, nudge_style) 
                       VALUES (%s, %s, NOW(), false, false, %s, %s, %s, %s)"""
            execute_sql_write(query, (user_id, nudge_type, emb_str, context_summary, nudge_content, nudge_style))
        else:
            query = "INSERT INTO nudge_outcomes (user_id, nudge_type, sent_at, responded, meal_logged, nudge_style) VALUES (%s, %s, NOW(), false, false, %s)"
            execute_sql_write(query, (user_id, nudge_type, nudge_style))
    except Exception as e:
        logger.error(f"Error logging nudge outcome: {e}")

# [P1-PLAN-LOTE-133 · 2026-09-20] «Respondió» = contestó en el chat O registró una comida en la ventana tras el aviso.
# Solo contaba lo primero: quien toca la notificación y anota su almuerzo desde el Dashboard —lo que el aviso PIDE—
# figuraba como que lo ignoró, y esa tasa decide el tono («has estado ignorando…») y la espera del siguiente aviso.
_SQL_TASA_DE_RESPUESTA = (
    "SELECT COUNT(*) as total, SUM(CASE WHEN n.responded OR EXISTS ("
    "SELECT 1 FROM consumed_meals c WHERE c.user_id::text = n.user_id::text AND c.created_at >= n.sent_at "
    "AND c.created_at < n.sent_at + make_interval(mins => %s)) THEN 1 ELSE 0 END) as responded_count "
    "FROM nudge_outcomes n WHERE n.user_id = %s"
)


def get_nudge_response_rate(user_id: str, nudge_type: str = None):
    try:
        if nudge_type:
            res = execute_sql_query(_SQL_TASA_DE_RESPUESTA + " AND n.nudge_type = %s",
                                    (_ventana_de_respuesta_min(), user_id, nudge_type), fetch_one=True)
        else:
            res = execute_sql_query(_SQL_TASA_DE_RESPUESTA, (_ventana_de_respuesta_min(), user_id), fetch_one=True)
            
        if res and res.get("total", 0) > 0:
            return float(res["responded_count"] or 0) / res["total"], res["total"]
    except Exception as e:
        logger.error(f"Error getting nudge response rate: {e}")
    return 1.0, 0

def get_daily_nudge_count(user_id: str) -> int:
    try:
        # [P2-PROACTIVE-NUDGE-BUDGET-TZ · 2026-05-30] Contar contra el día
        # CALENDARIO AST, no el UTC. Pre-fix: `DATE(sent_at) = CURRENT_DATE`
        # con DB en TimeZone=UTC contaba el día UTC, que rota a las 20:00 AST
        # (=00:00 UTC). Como los nudges se agendan en reloj AST y abarcan un día
        # AST que cruza el límite UTC a las 20:00, hasta 2 nudges diurnos (día
        # UTC D) + 2 vespertinos/Resumen 20:00-23:00 AST (día UTC D+1) = 4 en un
        # mismo día AST, el DOBLE del cap anti-fatiga (>=2). Convertir a AST
        # alinea el conteo con el reloj de agendado.
        # Tooltip-anchor: P2-PROACTIVE-NUDGE-BUDGET-TZ.
        #
        # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T5)] El hardcode 'America/Santo_Domingo' pasa a
        # offset por usuario (`db_facts.user_tz_offset_min`, fail-safe 240=RD). `sent_at`/`NOW()`
        # son timestamptz: `(col AT TIME ZONE 'zona')::date` == `(col - make_interval(mins =>
        # offset_oeste))::date` — equivalencia algebraica exacta, offset=240 reproduce el
        # hardcode previo byte a byte (verificado contra Neon 2026-08-16). Este sitio SIEMPRE fue
        # el hop simple/correcto (a diferencia de `db_facts.get_avg_meal_hour`, que preserva un
        # signo '+' heredado de un bug pre-existente — ver su comentario). Sin DST en
        # America/Santo_Domingo: 240 vale los 365 días.
        _tz_off = user_tz_offset_min(user_id)
        res = execute_sql_query(
            "SELECT COUNT(*) as total FROM nudge_outcomes "
            "WHERE user_id = %s "
            "AND (sent_at - make_interval(mins => %s))::date "
            "= (NOW() - make_interval(mins => %s))::date",
            (user_id, _tz_off, _tz_off), fetch_one=True,
        )
        return res.get("total", 0) if res else 0
    except Exception as e:
        logger.error(f"Error getting daily nudge count: {e}")
        return 0


def classify_nudge_sentiment(user_reply: str) -> dict:
    import json
    
    prompt = f"""Analiza la siguiente respuesta de un usuario a un recordatorio (nudge) para registrar su comida.
Debes determinar tres cosas:
1. sentiment: El sentimiento principal de la respuesta. Selecciona SOLO UNO de: positive, neutral, annoyed, guilt, motivation, curiosity, frustration, sadness.
2. meal_logged: Booleano (true/false) que indica si en este mensaje el usuario está efectivamente reportando lo que comió o confirmando que ya comió.
3. causal_reason: Si NO comió lo planeado (abandonó la comida), clasifica la razón principal. Selecciona SOLO UNO de: no_time, ate_out, no_ingredients, not_hungry, didnt_like, null (si no aplica o sí comió).

Respuesta del usuario: "{user_reply}"

Devuelve ÚNICAMENTE un JSON válido con las claves "sentiment", "meal_logged" y "causal_reason". No uses bloques markdown."""

    try:
        chat_llm = ChatGLM(
            model=_proactive_model_name(),
            temperature=0.1,
            timeout=_proactive_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
        )
        res = chat_llm.invoke(prompt)
        text = str(res.content).strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()
            
        data = json.loads(text)
        return {
            "sentiment": data.get("sentiment", "neutral"),
            "meal_logged": data.get("meal_logged", False),
            "causal_reason": data.get("causal_reason")
        }
    except Exception as e:
        logger.error(f"Error clasificando sentimiento de nudge: {e}")
        return {"sentiment": "neutral", "meal_logged": False, "causal_reason": None}

def handle_nudge_response(user_id: str, content: str):
    try:
        pending = execute_sql_query(
            "SELECT id, nudge_type FROM nudge_outcomes WHERE user_id = %s AND responded = false "
            "AND sent_at >= NOW() - make_interval(mins => %s) ORDER BY sent_at DESC LIMIT 1",
            (user_id, _ventana_de_respuesta_min()), fetch_one=True
        )
        if pending:
            nudge_id = pending['id']
            nudge_type = pending.get('nudge_type', 'Desconocido')
            classification = classify_nudge_sentiment(content)
            sentiment = classification['sentiment']
            meal_logged = classification['meal_logged']
            causal_reason = classification.get('causal_reason')
            
            try:
                execute_sql_write(
                    "UPDATE nudge_outcomes SET responded = true, response_sentiment = %s, meal_logged = %s WHERE id = %s",
                    (sentiment, meal_logged, nudge_id)
                )
            except Exception as e:
                logger.error(f"Error actualizando nudge_outcomes id={nudge_id}: {e}")
            
            logger.info(f"✅ Nudge {nudge_id} respondido por {user_id}. Sentiment: {sentiment}, Logged: {meal_logged}, Causal: {causal_reason}")

            # Persist causal reason if meal was abandoned
            if not meal_logged and causal_reason and causal_reason != "null":
                try:
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (user_id, nudge_type, causal_reason)
                    )
                    logger.info(f"🧠 Causal reason '{causal_reason}' saved for {nudge_type} (User: {user_id})")
                except Exception as e:
                    logger.error(f"Error persisting causal reason: {e}")
                    
    except Exception as e:
        logger.error(f"Error procesando respuesta al nudge: {e}")

def run_proactive_checks():
    """Esta función será llamada por apscheduler (cron job)."""
    logger.info("⏱️ [CRON] Iniciando verificación proactiva de comidas.")
    
    # --- PHASE 3: JIT Rolling Window Trigger ---
    # check_and_trigger_jit_rolling_windows() # Desactivado: El paso a Micro-Batching usa triggers interactivos vía UI ("Actualizar Platos")

    
    # 1. El instante UTC del tick. La hora LOCAL se calcula por usuario dentro del bucle.
    # [P1-NUDGE-TZ-PER-USER · 2026-08-21] Aquí vivía `now_ast = datetime.now(timezone(
    # timedelta(hours=-4)))`: un reloj dominicano literal con el que se decidía el disparo de
    # TODAS las sesiones activas. Fase 1 T5 parametrizó `get_daily_nudge_count` por usuario y
    # dejó el reloj. Traducido a hora local con los horarios por defecto, un español recibía
    # «¿desayunaste?» a las 15:00, «¿cenaste?» a la 01:30 y el Resumen del día —que dispara una
    # notificación push REAL— a las 05:00.
    _now_utc = datetime.now(timezone.utc)

    sessions = get_active_users_for_proactive()
    logger.info(f"🔍 [CRON] Encontradas {len(sessions)} sesiones activas para verificar (Proactividad Inteligente).")

    # [P1-PROACTIVE-BUDGET · 2026-05-28] Cota de escala del cron de nudges.
    # El loop hace ~10 queries/usuario (N+1: daily_nudge_count + global rate +
    # 4×(avg_meal_hour + per-meal rate) + perfil/consumed/embedding) y 1 LLM
    # invoke SERIAL cuando dispara nudge. A 1k-10k usuarios activos esto explota
    # a 10k-150k queries + miles de invokes serial por tick, solapándose con el
    # siguiente tick y saturando el pool de DB/threads. Sin cambiar la lógica de
    # decisión, acotamos por tick: usuarios procesados, wall-clock total, y nudges
    # (=invokes LLM). El excedente se atiende en ticks subsecuentes. Knobs con
    # clamps; runtime/nudges = 0 desactiva esa cota. Tooltip-anchor: P1-PROACTIVE-BUDGET.
    _max_users = max(1, min(_env_int("MEALFIT_PROACTIVE_MAX_USERS_PER_TICK", 250), 100000))
    _max_runtime_s = max(0, min(_env_int("MEALFIT_PROACTIVE_MAX_RUNTIME_S", 240), 3600))
    _max_nudges = max(0, min(_env_int("MEALFIT_PROACTIVE_MAX_NUDGES_PER_TICK", 150), 100000))
    _t_start = datetime.now(timezone.utc)
    _nudges_sent = 0
    if len(sessions) > _max_users:
        logger.warning(
            f"⚠️ [P1-PROACTIVE-BUDGET] {len(sessions)} sesiones activas > cap "
            f"{_max_users}/tick. Procesando las primeras {_max_users}; el resto "
            f"se atenderá en ticks subsecuentes."
        )
        sessions = sessions[:_max_users]

    for s in sessions:
        # [P1-PROACTIVE-BUDGET] cotas de wall-clock y de nudges (gasto LLM) por tick
        if _max_runtime_s and (datetime.now(timezone.utc) - _t_start).total_seconds() > _max_runtime_s:
            logger.warning(
                f"⚠️ [P1-PROACTIVE-BUDGET] Wall-clock > {_max_runtime_s}s "
                f"(enviados={_nudges_sent}); abortando resto del tick."
            )
            break
        if _max_nudges and _nudges_sent >= _max_nudges:
            logger.warning(
                f"⚠️ [P1-PROACTIVE-BUDGET] Cap de nudges/tick ({_max_nudges}) "
                f"alcanzado; abortando resto del tick."
            )
            break
        session_id = str(s.get("id")) if s.get("id") else None   # [P1-PLAN-LOTE-133] None = suscriptor sin chat reciente
        user_id = str(s.get("user_id"))
        # [P1-NUDGE-TZ-PER-USER · 2026-08-21] El reloj, DENTRO del bucle. `user_tz_offset_min` ya
        # estaba importado en este mismo archivo y se usaba 100 líneas más arriba: la maquinaria
        # existía y este call site no la llamaba. El knob global queda sólo de fallback.
        try:
            _user_tz_off = user_tz_offset_min(user_id)
        except Exception:
            _user_tz_off = _proactive_tz_offset_min()
        now_ast = _now_utc - timedelta(minutes=_user_tz_off)
        current_hour_float = _local_hour_float_for_offset(_now_utc, _user_tz_off)
        # [P1-PLAN-LOTE-159 · 2026-09-22] El aviso de HOY va al chat de HOY.
        #
        # `get_active_users_for_proactive` elige la sesión MÁS RECIENTE de los últimos 3 días,
        # sin mirar de qué día es. Así que a las 10:00 de hoy el aviso caía dentro de la
        # conversación de AYER — y el cliente, que sí sabe de días, dibujaba su separador «HOY»
        # en medio: dos días en el mismo chat, que es justo lo que el dueño no quiere.
        #
        # Y el daño no se deshacía solo: la regla del cliente (`debeRenovarse`) no renueva un
        # chat cuyo último mensaje ES de hoy. Al escribir ahí, el servidor convertía la
        # conversación de ayer en «la de hoy» y la dejaba pegada para siempre.
        #
        #   *Cuando dos lados comparten una regla —«un chat por día»— y solo uno la conoce, el
        #   que no la conoce no es neutral: la rompe para los dos.*
        #
        # El corte es por el ÚLTIMO MENSAJE, no por `created_at`: una sesión abierta anoche en
        # la que se sigue hablando a las 00:30 es la de hoy, y mirar su nacimiento la partiría
        # en dos. Es el mismo criterio que usa el cliente.
        #
        # [P1-PLAN-LOTE-161 · 2026-09-22] La llamada a `_sesion_del_dia_para_aviso` ya NO va aquí:
        # iba antes de TODOS los filtros (tope diario, horas de silencio, interruptor, comida ya
        # registrada), así que el primer tick de cada día —las 00:30, en pleno silencio— abría un
        # chat nuevo para cada usuario aunque no fuera a recibir nada. Y como «activo» se medía por
        # `agent_sessions.created_at`, ese chat lo mantenía activo para siempre. Ahora la sesión del
        # día se decide justo antes de escribir el aviso (más abajo, junto a `save_message`).
        # GAP 3: Nudge Budget (tope diario anti-fatiga)
        # [P1-PLAN-LOTE-72] El tope es el knob `MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY`: 4 por defecto (era un 2 fijo).
        _tope_diario = _max_avisos_por_dia()
        daily_nudges = get_daily_nudge_count(user_id)
        if daily_nudges >= _tope_diario:
            logger.info(f"🛑 [CRON] Usuario {user_id} ya agotó su presupuesto de nudges hoy ({daily_nudges}/{_tope_diario}). Saltando.")
            continue
            
        # Global stats para el tono base
        global_rate, global_total = get_nudge_response_rate(user_id)
        
        base_tone_instruction = "Usa un tono amistoso y motivacional, nunca de regaño, ni parezcas un robot asustadizo."
        send_push = True
        
        if global_total >= 5:
            if global_rate < 0.20:
                logger.info(f"📉 [CRON] Usuario {user_id} tiene response rate muy bajo ({global_rate:.0%}). Cambiando a tono empático.")
                # [P1-PLAN-LOTE-133 · 2026-09-20] Aquí había `send_push = False`: con 5 avisos «ignorados» el usuario
                # dejaba de recibir la notificación PARA SIEMPRE, sin señal en la interfaz y con el interruptor
                # encendido. Y «ignorado» se mide con respuestas en el chat dentro de una ventana corta: quien toca la
                # notificación y registra su comida desde el Dashboard cuenta como que ignoró. El interruptor es un
                # consentimiento explícito: la tasa cambia el TONO, no apaga la pantalla. Se apaga en Configuración.
                base_tone_instruction = "El usuario ha estado ignorando notificaciones recientemente. Usa un tono empático, pregúntale si hay algún obstáculo, estrés o falta de tiempo que le impida registrar sus comidas. NO asumas que se le olvidó, asume que podría estar ocupado o desmotivado. Sé muy breve y sin presiones."
            elif global_rate > 0.70:
                logger.info(f"🌟 [CRON] Usuario {user_id} tiene response rate alto ({global_rate:.0%}). Usando tono de refuerzo positivo.")
                base_tone_instruction = "El usuario tiene excelente disciplina. Usa un tono de celebración y refuerzo positivo animándolo a mantener la racha."
        
        meal_to_check = None
        trigger_time_str = ""
        final_tone_instruction = base_tone_instruction
        # [P1-PLAN-LOTE-72] Las comidas cuyo aviso toca en este tick, la más reciente primero.
        candidatas = []
        # [P1-PLAN-LOTE-216] El perfil se lee UNA vez por usuario y tick: la rama de las comidas lo necesita antes
        # (interruptor y hora de cada comida) y las puertas de más abajo reutilizan esa misma lectura.
        _perfil = None

        # Resumen del día siempre a las 11 PM
        if now_ast.hour == HORA_DEL_RESUMEN:
            # [P1-PLAN-LOTE-216] Con el cron cada 15 min, la hora 23 tiene cuatro ticks: el resumen sale en el primero
            # que pueda y no se repite (el anti-spam solo lo frenaba a quien tiene chat, no al suscriptor sin chat).
            _avisadas = _comidas_avisadas_hoy(user_id)
            if (_avisadas is None and now_ast.minute >= MINUTOS_ENTRE_TICKS) or (
                    _avisadas and "Resumen del día" in _avisadas):
                continue
            meal_to_check = "Resumen del día"
            trigger_time_str = "11:00 PM"
        else:
            # [P1-PLAN-LOTE-83] Horas de silencio: de madrugada no se recuerda ninguna comida.
            _silencio_hasta = _hora_de_silencio()
            if current_hour_float < _silencio_hasta:
                logger.info(f"🌙 [P1-PLAN-LOTE-83] Usuario {user_id}: {current_hour_float:.1f} h local, horas de "
                            f"silencio hasta las {_silencio_hasta}:00. Saltando.")
                continue

            # Horarios default (9AM, 1PM, 4PM, 7:30PM)
            defaults = {
                "Desayuno": 9.0,
                "Almuerzo": 13.0,
                "Merienda": 16.0,
                "Cena": 19.5
            }
            _reintento_min = _horas_de_reintento() * 60
            _ahora_min = current_hour_float * 60.0
            _avisadas = _comidas_avisadas_hoy(user_id)
            try:
                _perfil = get_user_profile(user_id)
            except Exception as e:
                logger.warning(f"[P1-PLAN-LOTE-216] perfil de {user_id} ilegible ({e}); avisos a las horas normales.")
            _health = ((_perfil or {}).get("health_profile") or {}) if isinstance(_perfil, dict) else {}

            for _orden, (meal, def_hour) in enumerate(defaults.items()):
                # [P1-PLAN-LOTE-216] El interruptor de ESTA comida (Configuración → Recordatorios de comida).
                if not comida_con_aviso(_health, meal):
                    continue
                # [P1-PLAN-LOTE-133] la cuenta vive en `hora_del_aviso`: el teléfono programa los mismos recordatorios
                _hora_aviso = hora_del_aviso(user_id, meal, def_hour, _health)
                _aviso_min = minuto_del_dia(_hora_aviso)

                # [P1-PLAN-LOTE-72 · 2026-09-16] Antes el aviso tocaba SOLO en la hora exacta y el bucle se quedaba con
                # la PRIMERA comida que coincidía: si esa ya estaba registrada, el `continue` de más abajo saltaba la
                # hora entera. El dueño registró el desayuno a las 12:58, su aviso cayó en la hora del almuerzo y el
                # del almuerzo no salió. Ahora toca desde su hora y durante `_reintento_min` (sin cruzar la
                # medianoche: el atraso se mide sin módulo), salvo que ya se haya enviado hoy; y se elige después,
                # con lo registrado delante, la primera que falte.
                # [P1-PLAN-LOTE-216] «Su hora» es el último tick ANTES de que suene el teléfono (ver
                # `MINUTOS_ENTRE_TICKS`): al tocar la notificación, el mensaje de esa comida ya está en el chat.
                _desde_min = max(0, _aviso_min - MINUTOS_ENTRE_TICKS)
                _atraso = _ahora_min - _desde_min
                if not (0 <= _atraso < _reintento_min):
                    continue
                if _avisadas is None and _atraso >= MINUTOS_ENTRE_TICKS:
                    continue  # sin saber qué salió hoy, solo su primer tick: sin repetidos
                if _avisadas and meal in _avisadas:
                    continue
                # La tasa de respuesta (solo para el TONO) se pide para las comidas que tocan, no para las cuatro.
                meal_rate, meal_total = get_nudge_response_rate(user_id, meal)
                # [P1-PLAN-LOTE-150] Los minutos se REDONDEAN, igual que en `meal_reminders`: truncando, el mensaje
                # del coach decía «6:49» y la notificación del teléfono sonaba a las 6:50 — el mismo aviso con dos
                # horas distintas según por dónde llegara.
                hours, mins = divmod(_aviso_min, 60)
                am_pm = "AM" if hours < 12 else "PM"
                display_hr = hours if hours <= 12 else hours - 12
                if display_hr == 0: display_hr = 12
                _tono = base_tone_instruction
                if meal_rate < 0.30 and meal_total >= 3:
                    _tono = "El usuario frecuentemente ignora o abandona esta comida específica. Pregúntale qué está fallando particularmente con esta comida (ej. tiempo, no le gusta, está fuera de casa) sin sonar acusador."
                candidatas.append({
                    "meal": meal, "hora": f"{display_hr}:{mins:02d} {am_pm}", "tono": _tono,
                    "meal_rate": meal_rate, "meal_total": meal_total, "atraso": _atraso, "orden": _orden,
                })

            # La más reciente primero; a igual hora, la más tardía del día: a la 1:30 PM se pregunta por el
            # almuerzo antes que por el desayuno.
            candidatas.sort(key=lambda c: (c["atraso"], -c["orden"]))
            if candidatas:
                meal_to_check = candidatas[0]["meal"]
                trigger_time_str = candidatas[0]["hora"]
                final_tone_instruction = candidatas[0]["tono"]

        tone_instruction = final_tone_instruction
        
        if not meal_to_check:
            continue
            
        logger.info(f"🔍 [CRON] Verificando {meal_to_check} para el usuario {user_id} (Nudge Dinámico: {trigger_time_str}).")
        
        try:
            # Regla Anti-Spam: Solo bloquear si ya enviamos un mensaje PROACTIVO (model) en la última hora.
            # Los mensajes del usuario NO bloquean recordatorios — chatear no impide recibir nudges.
            recent = get_recent_messages(session_id, limit=5) if session_id else None
            spam_blocked = False
            if recent:
                for msg in recent:
                    if msg.get("role") != "model":
                        continue  # Solo nos importan mensajes del modelo
                    last_msg_time_str = msg.get("created_at")
                    if last_msg_time_str:
                        if last_msg_time_str.endswith("Z"):
                            last_msg_time_str = last_msg_time_str[:-1] + "+00:00"
                        last_time = datetime.fromisoformat(last_msg_time_str)
                        diff_hours = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
                        if diff_hours < 1:
                            spam_blocked = True
                            logger.info(f"🚫 [CRON] Anti-spam: Usuario {user_id} ya recibió mensaje del modelo hace {diff_hours:.1f}h. Saltando.")
                            break
            
            if spam_blocked:
                continue
                
            # La evaluación de send_push ya se hizo al inicio del bucle por la Mejora 3
            
            # Vemos perfil para checar scheduleType (turno nocturno)
            # [P1-PLAN-LOTE-216] el que ya leyó la rama de las comidas; el Resumen del día lo lee aquí
            profile = _perfil if _perfil is not None else get_user_profile(user_id)
            if not profile:
                logger.info(f"🚫 [CRON] Usuario {user_id}: sin perfil. Saltando.")
                continue
            
            health = profile.get("health_profile", {})
            # [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] `locale` del MISMO `profile` ya
            # leído arriba (get_user_profile) — cero round-trips extra. Ausente/falsy ⇒
            # fallback explícito 'es-DO'; `build_language_directive` colapsa cualquier valor
            # no reconocido a "" (byte-idéntico a hoy).
            _nudge_locale = profile.get("locale") or "es-DO"
            schedule = health.get("scheduleType", "standard")
            if schedule == "night_shift" or schedule == "variable":
                logger.info(f"🚫 [CRON] Usuario {user_id}: turno {schedule}. Saltando.")
                continue
            # [P1-PLAN-LOTE-150] El interruptor de Configuración. Va aquí, junto a la puerta del turno, para que
            # apagarlo pare el aviso ANTES de gastar nada: ni consulta de lo registrado ni llamada a la IA.
            if not avisos_de_comida_activos(health):
                logger.info(f"🔕 [CRON] Usuario {user_id}: recordatorios de comida apagados. Saltando.")
                continue
                
            # Validar el consumo de HOY
            # [P1-PROACTIVE-TZ · 2026-05-30] PASAR tz_offset_mins. Pre-fix se
            # pasaba `date_str` SIN offset → `get_consumed_meals_today` caía a
            # su rama `else` (UTC), descartando `date_str` y construyendo la
            # ventana con el día UTC. El "Resumen del día" dispara a
            # now_ast.hour==23 (= 03:00 UTC del día siguiente): a esa hora la
            # ventana UTC `[00:00Z..23:59Z]` del día ya rotado = `[20:00 AST hoy
            # .. 19:59 AST mañana]`, EXCLUYENDO desayuno/almuerzo/cena
            # registrados antes de las 20:00 AST. Un usuario cumplidor caía a
            # `consumed==[]` y recibía el nudge indulgente "no registraste nada,
            # ¿descuento todo de tu nevera?" — falso-positivo NOCTURNO para cada
            # usuario standard. Con el offset, la rama AST-aware usa el día AST
            # correcto. Tooltip-anchor: P1-PROACTIVE-TZ.
            # [P1-NUDGE-TZ-PER-USER · 2026-08-21] El filtro de consumo usa el huso DEL USUARIO,
            # no el knob global. Sin esto el Resumen del día de un español evaluaba una ventana
            # que era mayoritariamente su día ANTERIOR — el mismo falso-positivo nocturno que
            # P1-PROACTIVE-TZ cerró para RD, reabierto para todo el que no viva en RD.
            consumed = get_consumed_meals_today(
                user_id,
                date_str=now_ast.strftime("%Y-%m-%d"),
                tz_offset_mins=_user_tz_off,
            )
            
            if meal_to_check == "Resumen del día":
                if consumed:
                    logger.info(f"✅ [CRON] Usuario {user_id}: registró comidas hoy. Todo ok para el resumen.")
                    continue
                else:
                    # Enviar mensaje especial: No comió nada
                    logger.info(f"⚠️ [CRON] Usuario {user_id} ({session_id}) no registró NADA. Generando nudge indulgente...")
                    # [P2-CHAT-PLAN-TOOLS-PAUSE · 2026-08-15] Este era el UNICO camino del
                    # coach que no pasa por `_plan_context_for_chat`, y su oferta —«restamos
                    # lo de hoy de tu nevera como si lo hubieras cocinado»— presupone un plan
                    # que prescribio algo. En modo contador no existe «lo de hoy»: restar de
                    # la Nevera «como si lo hubieras cocinado» no tiene referente.
                    #
                    # El nudge NO se apaga: recordar que registres es exactamente para lo que
                    # sirve el contador. Lo que cae es la oferta anclada al plan.
                    _en_pausa = _usuario_en_modo_contador(user_id)
                    _cierre = ("\"Veo que no registraste nada hoy, ¿se te paso anotar o comiste fuera?\""
                               if _en_pausa else
                               "\"Veo que no registraste nada hoy, ¿restamos lo de hoy de tu nevera "
                               "como si lo hubieras cocinado o comiste fuera?\"")
                    prompt = f"""
Eres tu nutricionista IA. Son las {trigger_time_str} de la noche.
He notado que el paciente no ha registrado NINGUNA comida en todo el día en su diario de Bioboros.
Escríbele un mensaje corto (máximo 2 líneas) muy amistoso e indulgente al estilo WhatsApp preguntándole:
{_cierre}
No uses demasiados emojis. Sé directo, breve y empático.
"""
                    # [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] Addendum §2: este nudge es
                    # prosa LLM user-facing (chat + body de la Web Push) — misma frontera que
                    # el coach: solo mueve el IDIOMA de la prosa, comida/nombres siguen español.
                    prompt += build_language_directive(_nudge_locale)
            else:
                # [P1-PLAN-LOTE-72] De las comidas que tocan, la primera que falte. Antes se miraba solo la primera
                # que coincidía con la hora y, si ya estaba registrada, se abandonaba el tick.
                elegida = None
                for cand in candidatas:
                    if _comida_ya_registrada(consumed, cand["meal"]):
                        logger.info(f"✅ [CRON] Usuario {user_id}: ya registró {cand['meal']}. Todo ok.")
                        continue
                    elegida = cand
                    break
                if elegida is None:
                    continue
                meal_to_check = elegida["meal"]
                trigger_time_str = elegida["hora"]
                tone_instruction = elegida["tono"]
                meal_rate, meal_total = elegida["meal_rate"], elegida["meal_total"]

                # ESTADO: olvido registrar. Generar mensaje proactivo.
                logger.info(f"⚠️ [CRON] Usuario {user_id} ({session_id}) no registró {meal_to_check}. Generando mensaje...")
                
                # [P1-PLAN-LOTE-161] Dieta, objetivo y restricciones REALES del perfil (ver `contexto_del_aviso`).
                _ctx_aviso = contexto_del_aviso(health)
                diet_type = _ctx_aviso["dieta"]
                goals = _ctx_aviso["objetivo"]
                
                # --- GAP 3: Embedding-based Nudge Personalization ---
                context_summary = f"Usuario ignoró {meal_to_check} {int((1-meal_rate)*meal_total)} veces de {meal_total} registradas. Tono base: {tone_instruction}"
                context_embedding = None
                proven_strategies_text = ""
                try:
                    context_embedding = get_embedding(context_summary)
                    if context_embedding:
                        emb_str = f"[{','.join(map(str, context_embedding))}]"
                        query = "SELECT * FROM match_successful_nudges(query_embedding => %s, match_threshold => 0.85, match_count => 2)"
                        successful_nudges = execute_sql_query(query, (emb_str,), fetch_all=True)
                        if successful_nudges:
                            proven_strategies_text = "Estrategias previas que funcionaron con usuarios en situaciones emocionales idénticas:\n"
                            for sn in successful_nudges:
                                n_content = sn.get("nudge_content", "")
                                n_sentiment = sn.get("response_sentiment", "motivado")
                                proven_strategies_text += f"- (Éxito: sintió {n_sentiment}): \"{n_content}\"\n"
                            proven_strategies_text += "\nInspírate en estas estrategias para tu respuesta (NO las copies exactas).\n"
                except Exception as emb_err:
                    logger.error(f"Error en embedding de nudge: {emb_err}")
                
                final_tone = (tone_instruction or "") + "\n" + proven_strategies_text
                
                # --- GAP 4: A/B Testing de Formato ---
                nudge_style = get_best_nudge_style(user_id)
                style_instruction = ""
                if nudge_style == "directo":
                    style_instruction = "Haz una pregunta directa y al grano sin rodeos."
                elif nudge_style == "sugestivo":
                    style_instruction = "Haz una sugerencia suave y comprensiva, aportando opciones."
                    # [P1-PLAN-LOTE-161] «Aportando opciones» es pedirle al modelo que nombre comida. Con alergias, una
                    # dieta restrictiva o una condición declaradas, el aviso anima sin nombrar platos: es el único
                    # texto del coach que no pasa por ningún filtro determinista, así que la regla es no darle ocasión.
                    if _ctx_aviso["restringido"]:
                        style_instruction = "Haz una sugerencia suave y comprensiva, sin nombrar alimentos concretos."
                elif nudge_style == "gamificado":
                    style_instruction = "Usa un tono de reto amistoso, motivando como si fuera un logro a desbloquear."
                
                prompt = PROACTIVE_PROMPT.format(
                    missing_meal=meal_to_check,
                    verbo=VERBO_DE_COMIDA.get(meal_to_check, f"tomaste tu {meal_to_check.lower()}"),
                    infinitivo=INFINITIVO_DE_COMIDA.get(meal_to_check, f"tomar tu {meal_to_check.lower()}"),
                    trigger_time=trigger_time_str,
                    diet_type=diet_type,
                    goals=goals,
                    tone_instruction=final_tone,
                    style_instruction=style_instruction
                )
                # [P1-PLAN-LOTE-161] Las restricciones van DESPUÉS de la plantilla (no como hueco): un placeholder nuevo
                # obligaría a cada `.format()` que la arma a conocerlo, y la cadena vacía no añade nada si no hay nada.
                prompt += _ctx_aviso["bloque"]
                # [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] Mismo directive que el bloque
                # "Resumen del día" arriba — ver esa nota para el contrato completo.
                prompt += build_language_directive(_nudge_locale)
                
            if not session_id:
                # [P1-PLAN-LOTE-133] Sin chat reciente: el aviso corto y fijo en su idioma, directo a la pantalla.
                from utils_push import send_push_notification
                from meal_reminders import texto_del_aviso, etiqueta_del_aviso
                _t_fijo, _b_fijo = texto_del_aviso(meal_to_check, _nudge_locale)
                log_nudge_outcome(user_id, meal_to_check, nudge_content=_b_fijo, nudge_style="fijo")
                send_push_notification(user_id=user_id, title=_t_fijo, body=_b_fijo, url="/dashboard/agent",
                                       tag=etiqueta_del_aviso(meal_to_check))
                logger.info(f"✅ [CRON] Aviso fijo de {meal_to_check} a {user_id} (suscriptor sin chat reciente)")
                continue

            chat_llm = ChatGLM(
                model=_proactive_model_name(),
                temperature=0.8,
                timeout=_proactive_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
            )
            response = chat_llm.invoke(prompt)
            _nudges_sent += 1  # [P1-PROACTIVE-BUDGET] cuenta el gasto LLM real del tick
            raw_content = response.content
            if isinstance(raw_content, list):
                content = " ".join([b.get("text", "") for b in raw_content if isinstance(b, dict) and "text" in b]).strip()
            else:
                content = str(raw_content).strip()
            
            if content:
                # [P1-PLAN-LOTE-161] El chat de HOY se elige aquí, cuando ya hay un aviso que escribir: si no toca
                # avisar, no se abre ninguna conversación.
                session_id = _sesion_del_dia_para_aviso(session_id, user_id, _now_utc, _user_tz_off)
                # Enviar a la base de datos con rol de modelo
                save_message(session_id, "model", content)
                logger.info(f"✅ [CRON] Mensaje proactivo enviado a {session_id} -> '{content[:40]}...'")
                
                # Intentar pasar embedding si se generó, para GAP 3 y style para GAP 4
                try:
                    log_nudge_outcome(user_id, meal_to_check, context_embedding=locals().get("context_embedding"), context_summary=locals().get("context_summary"), nudge_content=content, nudge_style=locals().get("nudge_style"))
                except Exception as log_err:
                    log_nudge_outcome(user_id, meal_to_check, nudge_style=locals().get("nudge_style"))
                
                if send_push:
                    # ---------------------------------------------
                    # NUEVO: Enviar Web Push Notification a todos los dispositivos del paciente
                    # ---------------------------------------------
                    from utils_push import send_push_notification
                    from prompts.chat_agent import push_nudge_title
                    # [P2-I18N-PUSH-SIN-LOCALE · 2026-08-21] El cuerpo YA seguia el idioma
                    # del usuario (`build_language_directive(_nudge_locale)`, arriba); el
                    # titulo era un literal espanol pegado aqui. La notificacion llegaba
                    # BILINGUE, y en la pantalla de bloqueo el titulo es lo unico que se
                    # lee de un vistazo — o sea, la mitad que no se traducia era la que
                    # decide si el usuario abre.
                    send_push_notification(
                        user_id=user_id,
                        title=push_nudge_title(_nudge_locale),
                        body=content,
                        url=f"/dashboard/agent?session_id={session_id}",
                        tag=f"comida-{str(meal_to_check).lower().split()[0]}",   # [P1-PLAN-LOTE-133]
                    )
                
        except Exception as e:
            logger.error(f"Error procesando proactividad para {session_id}: {e}")

# ============================================================
# [P3-COLDSTART-E2E · 2026-05-29] ⚰️ CÓDIGO MUERTO — NO REACTIVAR SIN MIGRAR
# ------------------------------------------------------------
# Las DOS funciones JIT Rolling Window de abajo
# (`_trigger_week2_background_generation` + `check_and_trigger_jit_rolling_windows`)
# están DESACTIVADAS. Su única invocación está comentada en
# `run_proactive_checks` (línea ~213: "# check_and_trigger_jit_rolling_windows()
# # Desactivado: El paso a Micro-Batching usa triggers interactivos") y NINGÚN
# cron las registra (`register_plan_chunk_scheduler` en cron_tasks.py no las lista).
#
# El disparo REAL del próximo chunk vive 100% en `plan_chunk_queue` +
# `process_plan_chunk_queue` (cron_tasks.py:23122): todos los chunks 2..N se
# encolan al CREAR el plan con un `execute_after` de calendario y el worker
# (cada 1 min, server-side) los levanta cuando llega su fecha. Modelo mental
# único: "todo se dispara por plan_chunk_queue".
#
# Por qué NO se borran: el helper de append (`_apply_week2_append` →
# `update_plan_data_atomic`) tiene cobertura de regresión lost-update activa en
# `test_p1_audit_1_update_meal_plan_data_lostupdate.py`. Borrar las funciones perdería
# ese path de test. Se MARCAN como DEAD para que un mantenedor no las reactive
# por error: reactivarlas junto al worker = DOBLE generación del mismo bloque, y
# el cuerpo exige `len(days) == 7` exacto (incompatible con chunks de 3 días del
# micro-batching actual).
# Tooltip-anchor: P3-COLDSTART-E2E-JIT-DEAD.
# ============================================================
def _trigger_week2_background_generation(user_id, plan_id, existing_plan_data):
    """[⚰️ DEAD CODE — ver banner P3-COLDSTART-E2E-JIT-DEAD arriba] Generates the
    next 7 days in the background and appends to the existing plan.

    NO invocada en producción (micro-batching usa plan_chunk_queue). Conservada
    solo por la cobertura lost-update de `update_plan_data_atomic`."""
    from graph_orchestrator import run_plan_pipeline
    from db_profiles import get_user_profile
    from db_plans import update_plan_data_atomic
    from db import get_user_likes, get_active_rejections
    from agent import analyze_preferences_agent
    import threading

    def _bg_task():
        try:
            logger.info(f"🔄 [JIT BG TASK] Arrancando generación asíncrona de Semana 2 para {user_id}...")
            profile = get_user_profile(user_id) or {}
            health_profile = profile.get("health_profile", {})
            form_data = health_profile.copy()
            form_data["user_id"] = user_id
            # [P1-NEW-9 · 2026-05-11] Atribución del caller para que
            # `_emit_plan_quality_degraded_alert` (graph_orchestrator
            # should_retry, las 5 ramas "end") correlacione el alert
            # al plan_id correcto. Sin estos kwargs, el alert antes
            # quedaba con alert_key="plan_quality_degraded:<user>:no_plan_id"
            # porque `plan_result` de la extensión week-2 no trae `id`
            # (el meal_plan ya existe en DB, este pipeline NO inserta).
            # Resultado pre-fix: SRE veía la alert pero NO sabía cuál
            # plan se extendió degradado, y los alerts colapsaban con
            # alert_key={user_id}:no_plan_id usados para /generate-plan.
            form_data["_caller_target_plan_id"] = plan_id
            form_data["_caller_context"] = "jit_week2"

            likes = get_user_likes(user_id)
            rejections = get_active_rejections(user_id)
            rej_names = [r["meal_name"] for r in rejections] if rejections else []
            taste_profile = analyze_preferences_agent(likes, [], active_rejections=rej_names)

            result = run_plan_pipeline(form_data, [], taste_profile, memory_context="")

            new_days = result.get("days", [])
            if not new_days:
                logger.warning(
                    f"[JIT BG TASK] run_plan_pipeline retornó 0 días para "
                    f"user={user_id} plan={plan_id}. Skip persistencia."
                )
                return

            # [P1-AUDIT-1 · 2026-05-15] Append + persistencia atómica bajo
            # FOR UPDATE row lock (vía `update_plan_data_atomic`). Cierre del
            # follow-up natural documentado en P1-RECALC-LOSTUPDATE
            # (2026-05-14):
            #
            # Pre-fix flow:
            #   t=0  El handler `proactive_agent` recibe `existing_plan_data`
            #        leído por el cron `check_and_trigger_jit_rolling_windows`
            #        (línea ~524, SELECT plano sin lock).
            #   t=1  Mutación in-memory: `existing_plan_data["days"] = existing
            #        _days + new_days` con re-numeración de `day`.
            #   t=2  acquire advisory lock + UPDATE full-overwrite via
            #        `update_meal_plan_data` (P1-NEXT-1).
            #
            # Ventana lost-update entre t=0 y t=2 (puede ser HORAS si
            # `run_plan_pipeline` es lento — la generación LLM puede tomar
            # 30-180s, multiplicando el tamaño de la ventana): si un endpoint
            # hermano muta `plan_data` quirúrgico entre el cron read y
            # nuestro UPDATE, esa mutación se pierde. JIT week-2 es el caso
            # con la ventana MÁS LARGA del sistema (chunk_worker, swap-meal,
            # /recipe/expand son sub-segundo).
            #
            # Fix: `update_plan_data_atomic` re-SELECTea plan_data FRESH bajo
            # FOR UPDATE row lock al momento de persistir, así el callback
            # appendea los new_days a `plan_data_fresh.days` (post-merge de
            # cualquier mutación que ocurrió en la ventana de 30-180s) — el
            # número de días base puede haber cambiado, el re-numerado se
            # hace contra el fresh.
            #
            # Tooltip-anchor: P1-AUDIT-1-JIT-WEEK2-START |
            # test_p1_audit_1_proactive_week2_lostupdate
            def _apply_week2_append(plan_data_fresh: dict) -> dict | bool:
                """Appendea `new_days` a `plan_data_fresh.days` re-numerando
                contra la longitud fresh. Si days no es lista, retorna False
                para abortar UPDATE.
                """
                if not isinstance(plan_data_fresh, dict):
                    return False
                existing_days_fresh = plan_data_fresh.get("days") or []
                if not isinstance(existing_days_fresh, list):
                    existing_days_fresh = []
                start_idx = len(existing_days_fresh) + 1
                for i, d in enumerate(new_days):
                    if isinstance(d, dict):
                        d["day"] = start_idx + i
                plan_data_fresh["days"] = existing_days_fresh + new_days
                return plan_data_fresh

            # [P2-OPEN-1] user_id pasado al helper para que el SELECT/UPDATE
            # filtren `AND user_id = %s` defense-in-depth. plan_id viene del
            # cron `check_and_trigger_jit_rolling_windows` que ya tiene
            # ownership (SELECT con user_id en el row).
            merged = update_plan_data_atomic(
                plan_id, _apply_week2_append, user_id=user_id
            )
            if not merged:
                logger.warning(
                    f"⚠️ [JIT BG TASK] update_plan_data_atomic retornó vacío "
                    f"para plan={plan_id} user={user_id}: el plan desapareció "
                    f"o el filtro user_id no matchea. Week-2 NO persistida."
                )
                return
            logger.info(f"✅ [JIT BG TASK] Semana 2 añadida exitosamente al plan {plan_id} de {user_id}")
        except Exception as e:
            logger.error(f"❌ [JIT BG TASK] Error en generación de semana 2 para {user_id}: {e}")

    threading.Thread(target=_bg_task, daemon=True).start()

def check_and_trigger_jit_rolling_windows():
    """[⚰️ DEAD CODE — ver banner P3-COLDSTART-E2E-JIT-DEAD arriba]
    JIT Rolling Windows Trigger:
    Detects users who are on Day 5 or 6 (i.e. plan generated 4-6 days ago)
    and if their plan has only 7 days, generates Week 2.

    DESACTIVADA: su único callsite está comentado en `run_proactive_checks`.
    El disparo del próximo chunk vive en `process_plan_chunk_queue`
    (cron_tasks.py:23122). No reactivar sin migrar — ver banner.
    """
    logger.info("⏱️ [CRON JIT] Chequeando ventanas JIT (Rolling Windows) para Semana 2...")
    if not connection_pool: return
    
    try:
        from db_core import execute_sql_query
        query = """
            SELECT id, user_id, plan_data
            FROM meal_plans
            WHERE created_at >= NOW() - INTERVAL '6 days'
            AND created_at <= NOW() - INTERVAL '4 days'
            ORDER BY created_at DESC
        """
        res = execute_sql_query(query, fetch_all=True)
        if not res: return
        
        seen_users = set()
        for row in res:
            uid = row.get("user_id")
            if uid in seen_users: continue
            seen_users.add(uid)
            
            plan_data = row.get("plan_data")
            if not isinstance(plan_data, dict): continue
            
            days = plan_data.get("days", [])
            if len(days) == 7:
                logger.info(f"🔄 [JIT TRIGGER] Usuario {uid} está en Día 5-6. Disparando Fase 3 (Semana 2)...")
                _trigger_week2_background_generation(uid, str(row.get("id")), plan_data)
                
    except Exception as e:
        logger.error(f"⚠️ [JIT CRON] Error en check_and_trigger_jit_rolling_windows: {e}")
