# backend/ajustes_cuenta.py
"""[P1-PLAN-LOTE-837 · 2026-09-29] Los ajustes de cada cuenta: qué tiene encendido, cuándo lo cambió y quién.

Spec docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md §13.1, §13.2, §13.3 y §13.7 (raíz del
workspace). Un ajuste es MODO DE USO, no contenido: se ve en la ficha de CUALQUIER cuenta. El perfil de salud (peso, edad,
sexo, alergias, dieta, condiciones, medicamentos, súper personalización, perfil clínico) NO es un ajuste: de los dos
paneles solo sale si están rellenos y cuándo se guardaron, nunca lo que dicen.

`REGISTRO` es el SSOT que leen la ficha (`ajustes_de`), el resumen de Métricas (`resumen`) y el CSV (`columnas_csv`,
`fila_csv`). Cada descriptor dice de dónde sale el ajuste (`fuente`): una columna de `user_profiles`, una clave de
`health_profile`, una tabla, una clave de `app_kv_store` o el informe del dispositivo. Una clave de `health_profile` que
empieza por `avisos_` y no está en el registro sale igual en «Otros ajustes» con su valor tal cual: un aviso nuevo aparece
sin tocar el panel (y `test_p1_plan_lote_837` pide darle su sitio y su etiqueta).

EL HISTORIAL (`public.ajustes_cambios`) lo llena un TRIGGER de Postgres (`trg_ajustes_cambios`, migración
`p1_plan_lote_837_ajustes_cambios_2026_09_29.sql`) sobre las columnas y claves vigiladas (`COLUMNAS_VIGILADAS`,
`CLAVES_PERFIL_VIGILADAS`): cubre a TODOS los escritores (Configuración, el coach, el sistema y los que vengan) sin
tocarlos. Lo que el trigger no puede saber solo es QUIÉN escribe; eso lo dice el `origen`:

  · `app` (por defecto): la persona, desde la app.
  · `coach`: la herramienta `cambiar_ajuste_de_la_app` (`ajustes_de_la_app.cambiar_ajuste` corre dentro de
    `origen_de_ajustes("coach")`). Las demás herramientas del coach no corren en ese bloque: si una toca un ajuste
    vigilado (`update_form_field` con el país o el presupuesto), su cambio queda como `app`. Tampoco el IDIOMA que la
    persona le pide al coach: la herramienta solo devuelve un marcador `{"idioma": …}` y lo GUARDA el cliente (la
    pantalla lo aplica y escribe `locale` como cualquier cambio de Configuración), así que también queda como `app`.
    Límite conocido: el panel etiqueta esos cambios como «la persona».
  · `sistema`: los apagados automáticos (Nevera vacía 48 h, hidratación sin un vaso 48 h) y el encendido automático de
    una Nevera que el sistema había apagado.

EL MECANISMO DEL ORIGEN, y por qué funciona. El escritor fija el origen EN LA MISMA SENTENCIA:

    UPDATE user_profiles SET … FROM (SELECT set_config('mealfit.origen_ajuste', 'coach', true)) AS _origen WHERE …

(`sql_con_origen` lo inserta delante del WHERE de nivel cero). `execute_sql_write` corre cada sentencia en AUTOCOMMIT
(`db_core`: `autocommit=True` en el pool): la transacción ES la sentencia. `set_config(…, true)` vale hasta el final de
esa transacción, el trigger `AFTER … FOR EACH ROW` se dispara al final de la sentencia DENTRO de ella —después de que el
FROM haya producido la fila con la que se actualiza cada registro— y lo lee con `current_setting(…, true)`; al terminar,
la variable desaparece, así que el pool no arrastra el origen a la sentencia siguiente (tampoco a través de PgBouncer en
modo transacción). Por eso NO sirve un `SELECT set_config(…)` suelto antes del UPDATE: en autocommit serían dos
transacciones y el trigger vería el origen vacío. `set_config` es VOLATILE: Postgres no aplana esa subconsulta ni la poda
aunque nadie lea su columna. Una sentencia que ya corre dentro de una transacción explícita (la fusión atómica del perfil,
`update_user_health_profile_atomic`) lo lleva igual: el valor dura hasta su COMMIT. El origen va como LITERAL de una lista
cerrada (`ORIGENES`), no como parámetro: así no se mueve ningún `%s` de la sentencia original.

Quién decide el origen: el argumento explícito de `sql_con_origen` y, si no hay, el bloque `origen_de_ajustes(...)` en
curso (un `ContextVar`). Los escritores compartidos (`db_profiles.update_water_tracker_enabled`, `…long_term_memory…`,
`update_user_health_profile_atomic`, `nevera_opcional.fijar_nevera`, `plan_mode.pause/resume_plan_generation`) llaman
siempre a `sql_con_origen(SQL)`: fuera de un bloque devuelve la MISMA cadena (sin cambio de conducta ni de SQL), dentro
lleva el origen del bloque. Así la tool del coach no cambia la firma de nadie.

LOS AJUSTES DEL DISPOSITIVO (§13.3): el tema, el permiso de notificaciones, la barra plegada… viven solo en el teléfono.
La app los informa (`PUT /api/profile/ajustes-dispositivo` → `guardar_dispositivo`) con una LISTA CERRADA de claves: lo
desconocido se descarta. Se guardan por plataforma en `user_profiles.ajustes_dispositivo` (`{"web": {…, "at"}, …}`), que
el trigger NO vigila. Con el interruptor maestro `MEALFIT_ADMIN_TEST_ACCOUNTS` apagado no se escribe nada.

La purga (`purgar_cambios_antiguos`, cron diario `purge_ajustes_cambios`) borra lo que pasa de
`MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS` (730, acotado a [90, 3650]). El historial se exporta con la cuenta y se va con
ella (ON DELETE CASCADE). La misma pasada diaria deja en el log un CANARIO (`count(*)` y `max(at)` de la tabla): el
trigger que la llena avisa de sus fallos con un WARNING de Postgres, que NO llega a los logs de la app; una tabla que
dejó de llenarse se ve porque ese `max(at)` no avanza mientras la app se usa.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator, Optional

from db import execute_sql_query, execute_sql_write
from knobs import _env_int

logger = logging.getLogger(__name__)

GRUPOS = ("Uso", "Avisos", "Capacidades", "Privacidad", "Plan", "Dispositivo")
GRUPO_OTROS = "Otros ajustes"
ESTADOS = ("encendido", "apagado", "automatico", "sin_elegir", "valor")
ORIGENES = ("app", "coach", "sistema")
GUC_ORIGEN = "mealfit.origen_ajuste"
MAX_HISTORIAL = 500
MAX_TEXTO = 120
_JSON_MAX = 2000

# Lo que el trigger vigila (debe coincidir con la migración: lo compara `test_p1_plan_lote_837`).
COLUMNAS_VIGILADAS = ("plan_mode", "logging_preference", "long_term_memory_enabled", "water_tracker_enabled",
                      "nevera_enabled", "locale", "analytics_consent", "ai_training_consent", "ai_consent_version",
                      "ai_consent_revoked_at")
CLAVES_PERFIL_VIGILADAS = ("avisos_comida", "avisos_agua", "avisos_por_comida", "country", "groceryDuration", "budget",
                           "budgetCurrency", "weightUnit")

# El informe del dispositivo: la lista CERRADA de claves y sus valores (bool = true/false).
PLATAFORMAS = ("web", "ios", "android")
CLAVES_DISPOSITIVO = {
    "tema": ("system", "light", "dark"),
    "notificaciones_permiso": ("granted", "denied", "default", "unsupported"),
    "alertas_activadas": bool,
    "analitica_vetada": bool,
    "barra_plegada": bool,
    "unidad_altura": ("cm", "ft"),
    "avatar_elegido": bool,
}
_RE_BUILD = re.compile(r"[0-9A-Za-z._+\-() ]{1,64}")
_RE_HORA = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _horas_canal_local() -> int:
    """Cuánto vale un teléfono sincronizado: `hydration_reminders.HORAS_DE_ALCANCE_LOCAL` (SSOT; antes una copia a mano
    de 72 que se habría desfasado en silencio). Import perezoso: el módulo del agua no se carga hasta que hace falta."""
    from hydration_reminders import HORAS_DE_ALCANCE_LOCAL
    return HORAS_DE_ALCANCE_LOCAL


# ─────────────────────────────────────────────────────────────────────────────────────────────── el origen del cambio
_ORIGEN_ACTUAL: contextvars.ContextVar = contextvars.ContextVar("mealfit_origen_ajuste", default=None)


def _validar_origen(origen) -> str:
    if origen not in ORIGENES:
        raise ValueError(f"origen de ajuste desconocido: {origen!r} (valen {ORIGENES})")
    return origen


def origen_actual() -> Optional[str]:
    """El origen del bloque `origen_de_ajustes` en curso, o None (la persona, desde la app)."""
    return _ORIGEN_ACTUAL.get()


@contextlib.contextmanager
def origen_de_ajustes(origen: str) -> Iterator[None]:
    """Todo UPDATE de `user_profiles` que pase por `sql_con_origen` dentro del bloque lleva este origen. Se restaura al
    salir, también con error (un origen que se quedara pegado mentiría en todos los cambios siguientes del hilo)."""
    token = _ORIGEN_ACTUAL.set(_validar_origen(origen))
    try:
        yield
    finally:
        _ORIGEN_ACTUAL.reset(token)


def _palabras_de_nivel_cero(sql: str) -> list:
    """(posición, PALABRA) de cada palabra fuera de paréntesis, literales, identificadores entre comillas y comentarios.
    Basta para las sentencias UPDATE del repo (sin literales con `$$` ni cadenas `E'…'` con `\\'`)."""
    palabras, i, n, nivel = [], 0, len(sql), 0
    while i < n:
        c = sql[i]
        if c == "'":
            i += 1
            while i < n and not (sql[i] == "'" and not sql.startswith("''", i)):
                i += 2 if sql.startswith("''", i) else 1
            i += 1
        elif c == '"':
            j = sql.find('"', i + 1)
            i = n if j < 0 else j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c == "(":
            nivel, i = nivel + 1, i + 1
        elif c == ")":
            nivel, i = nivel - 1, i + 1
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            if nivel == 0:
                palabras.append((i, sql[i:j].upper()))
            i = j
        else:
            i += 1
    return palabras


def sql_con_origen(sql_update: str, origen: Optional[str] = None) -> str:
    """El UPDATE con la marca de origen en su propio FROM (ver la cabecera del módulo). `origen` explícito manda sobre el
    bloque en curso; sin ninguno de los dos devuelve `sql_update` TAL CUAL (el mismo objeto). Solo `UPDATE … SET …
    WHERE …`: otra forma lanza ValueError (un escritor nuevo lo descubre en su test, no en producción)."""
    origen = origen if origen is not None else _ORIGEN_ACTUAL.get()
    if origen is None:
        return sql_update
    _validar_origen(origen)
    palabras = _palabras_de_nivel_cero(sql_update)
    nombres = [p for _, p in palabras]
    if not nombres or nombres[0] != "UPDATE" or "SET" not in nombres:
        raise ValueError("sql_con_origen: solo sentencias UPDATE … SET … WHERE …")
    i_set = nombres.index("SET")
    if "WHERE" not in nombres[i_set:]:
        raise ValueError("sql_con_origen: el UPDATE no tiene WHERE (toda escritura de user_profiles filtra por id)")
    i_where = nombres.index("WHERE", i_set)
    pos = palabras[i_where][0]
    fuente = f"(SELECT set_config('{GUC_ORIGEN}', '{origen}', true)) AS _origen "
    enlace = (", " if "FROM" in nombres[i_set:i_where] else "FROM ") + fuente
    return sql_update[:pos] + enlace + sql_update[pos:]


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────── el registro
@dataclass(frozen=True)
class Ajuste:
    clave: str
    etiqueta: str
    grupo: str
    # ("columna", nombre) | ("perfil", clave_hp) | ("tabla", nombre) | ("kv", prefijo) | ("dispositivo", clave)
    fuente: tuple
    tipo: str
    por_defecto: Any = None


# Cómo se lee cada tipo (ver `_evaluar`):
#   bool          True/False → encendido/apagado; ausente → `por_defecto` (None ⇒ sin_elegir).
#   tri           NULL → automatico (la Nevera: la decide el sistema), TRUE/FALSE → encendido/apagado.
#   modo          un valor de dos (`_MODOS`) → encendido/apagado; ausente → `por_defecto`.
#   enum          estado `valor` con su valor; ausente → `por_defecto` (None ⇒ sin_elegir).
#   aviso_comida  el interruptor y la hora de UNA comida en `avisos_por_comida` (ausente ⇒ encendido, hora normal).
#   relleno       `valor` «relleno» o sin_elegir, y la fecha en que se guardó; jamás el contenido.
#   conteo        cuántos (valor n) o sin_elegir.
#   existe        encendido si hay al menos una fila (valor: cuántas).
#   hecho         valor «sí»/«no» (un hecho, no un interruptor).
#   plataformas_push, canal_local, apagado_solo, invitacion, consentimiento_ia, plataformas: ver `_evaluar`.
TIPOS = ("bool", "tri", "modo", "enum", "aviso_comida", "relleno", "conteo", "existe", "hecho", "plataformas_push",
         "canal_local", "apagado_solo", "invitacion", "consentimiento_ia", "plataformas")
_MODOS = {"plan_mode": {"plan": "encendido", "tracking": "apagado"},
          "logging_preference": {"auto_proxy": "encendido", "manual": "apagado"}}
# `hydration_reminders.PREFIJO_ESTADO`: el estado del apagado automático de la hidratación (`auto_off_at`).
_PREFIJO_HIDRATACION = "hydration_state:"
_COMIDAS = (("desayuno", "Desayuno"), ("almuerzo", "Almuerzo"), ("merienda", "Merienda"), ("cena", "Cena"))

REGISTRO: tuple = (
    # ── Uso
    Ajuste("plan_mode", "Generación de planes", "Uso", ("columna", "plan_mode"), "modo", "plan"),
    Ajuste("locale", "Idioma", "Uso", ("columna", "locale"), "enum", "es-DO"),
    Ajuste("country", "País", "Uso", ("perfil", "country"), "enum"),
    Ajuste("weightUnit", "Unidad de peso", "Uso", ("perfil", "weightUnit"), "enum", "lb"),
    Ajuste("entro_con_apple", "Sesión con Apple", "Uso", ("tabla", "apple_signin_tokens"), "hecho"),
    # ── Avisos
    Ajuste("avisos_comida", "Recordatorios de comida", "Avisos", ("perfil", "avisos_comida"), "bool", True),
    Ajuste("avisos_agua", "Recordatorios de agua", "Avisos", ("perfil", "avisos_agua"), "bool", True),
    *(Ajuste(f"avisos_por_comida.{c}", f"Recordatorio: {nombre}", "Avisos", ("perfil", f"avisos_por_comida.{c}"),
             "aviso_comida", True) for c, nombre in _COMIDAS),
    Ajuste("push_web", "Notificaciones del navegador", "Avisos", ("tabla", "push_subscriptions"), "existe"),
    Ajuste("push_app", "Notificaciones de la app", "Avisos", ("tabla", "device_push_tokens"), "plataformas_push"),
    Ajuste("avisos_locales", "Avisos programados en el teléfono", "Avisos", ("kv", "avisos_locales:"), "canal_local"),
    # ── Capacidades
    Ajuste("logging_preference", "Modo automático", "Capacidades", ("columna", "logging_preference"), "modo", "manual"),
    Ajuste("long_term_memory_enabled", "Memoria a Largo Plazo", "Capacidades", ("columna", "long_term_memory_enabled"),
           "bool", True),
    Ajuste("water_tracker_enabled", "Hidratación", "Capacidades", ("columna", "water_tracker_enabled"), "bool", True),
    Ajuste("hidratacion_apagada_sola", "Hidratación apagada por el sistema", "Capacidades", ("kv", _PREFIJO_HIDRATACION),
           "apagado_solo"),
    Ajuste("nevera_enabled", "Nevera", "Capacidades", ("columna", "nevera_enabled"), "tri"),
    Ajuste("suplementos", "Suplementos en la Alacena", "Capacidades", ("tabla", "user_inventory"), "conteo"),
    # ── Privacidad
    Ajuste("ai_consent", "Permiso para la IA de terceros", "Privacidad", ("columna", "ai_consent_version"),
           "consentimiento_ia"),
    Ajuste("analytics_consent", "Ayuda a mejorar Bioboros", "Privacidad", ("columna", "analytics_consent"), "bool"),
    Ajuste("ai_training_consent", "Entrenamiento de modelos de IA", "Privacidad", ("columna", "ai_training_consent"),
           "bool", False),
    # ── Plan
    Ajuste("groceryDuration", "Frecuencia de compras", "Plan", ("perfil", "groceryDuration"), "enum"),
    Ajuste("budget", "Presupuesto", "Plan", ("perfil", "budget"), "enum"),
    Ajuste("budgetCurrency", "Moneda del presupuesto", "Plan", ("perfil", "budgetCurrency"), "enum"),
    Ajuste("super_personalization", "Súper Personalización", "Plan", ("perfil", "super_personalization"), "relleno"),
    Ajuste("clinical_profile", "Perfil Clínico Avanzado", "Plan", ("perfil", "clinical_profile"), "relleno"),
    Ajuste("staple_foods", "Mis básicos", "Plan", ("perfil", "staple_foods"), "conteo"),
    Ajuste("marcas_elegidas", "Marcas elegidas", "Plan", ("tabla", "user_brand_preferences"), "conteo"),
    Ajuste("invitacion_al_plan", "Invitación al plan («Ahora no»)", "Plan", ("kv", "plan_invite:"), "invitacion"),
    # ── Dispositivo (el informe de la app, §13.3)
    Ajuste("plataformas", "Plataformas", "Dispositivo", ("dispositivo", "plataforma"), "plataformas"),
    Ajuste("pwa", "App instalada (PWA)", "Dispositivo", ("dispositivo", "pwa"), "bool"),
    Ajuste("app_build", "Versión de la app", "Dispositivo", ("dispositivo", "app_build"), "enum"),
    Ajuste("tema", "Tema de la aplicación", "Dispositivo", ("dispositivo", "tema"), "enum"),
    Ajuste("notificaciones_permiso", "Permiso de notificaciones del sistema", "Dispositivo",
           ("dispositivo", "notificaciones_permiso"), "enum"),
    Ajuste("alertas_activadas", "Alertas del dispositivo", "Dispositivo", ("dispositivo", "alertas_activadas"), "bool"),
    Ajuste("analitica_vetada", "Analítica vetada en el dispositivo", "Dispositivo", ("dispositivo", "analitica_vetada"),
           "bool"),
    Ajuste("barra_plegada", "Barra de navegación plegada", "Dispositivo", ("dispositivo", "barra_plegada"), "bool"),
    Ajuste("unidad_altura", "Unidad de altura", "Dispositivo", ("dispositivo", "unidad_altura"), "enum"),
    Ajuste("avatar_elegido", "Avatar elegido", "Dispositivo", ("dispositivo", "avatar_elegido"), "bool"),
)

# Las claves de `health_profile` que la consulta proyecta tal cual (las de «relleno» y «conteo» solo se resumen en SQL).
_HP_VALORES = tuple(dict.fromkeys(a.fuente[1].split(".")[0] for a in REGISTRO
                                  if a.fuente[0] == "perfil" and a.tipo not in ("relleno", "conteo")))
_CLAVES_AVISOS_REGISTRADAS = frozenset(k for k in _HP_VALORES if k.startswith("avisos_"))
_COLUMNAS_PERFIL = ("plan_mode", "plan_mode_changed_at", "logging_preference", "long_term_memory_enabled",
                    "water_tracker_enabled", "nevera_enabled", "nevera_auto_off_at", "locale", "analytics_consent",
                    "ai_training_consent", "ai_consent_version", "ai_consent_at", "ai_consent_revoked_at",
                    "ai_cn_transfer_at", "ajustes_dispositivo")

# Lo que el historial llama a cada clave vigilada que no es la fuente directa de un ajuste del registro.
_ETIQUETA_DE_CAMBIO = {
    "avisos_por_comida": "Recordatorio de cada comida (interruptor y hora)",
    "ai_consent_version": "Permiso para la IA de terceros",
    "ai_consent_revoked_at": "Permiso para la IA de terceros (retirado)",
}


def _clave_de_cambio(a: Ajuste) -> Optional[str]:
    """La `clave` con la que el trigger anota los cambios de este ajuste (None: el trigger no lo vigila)."""
    if a.fuente[0] == "columna":
        return a.fuente[1]
    if a.fuente[0] == "perfil":
        return a.fuente[1].split(".")[0]
    return None


def _claves_de_cambio(a: Ajuste) -> tuple:
    if a.tipo == "consentimiento_ia":
        return ("ai_consent_version", "ai_consent_revoked_at")
    clave = _clave_de_cambio(a)
    return (clave,) if clave else ()


def etiqueta_de_cambio(clave) -> str:
    """El nombre legible de una clave del historial; una desconocida se enseña tal cual."""
    if clave in _ETIQUETA_DE_CAMBIO:
        return _ETIQUETA_DE_CAMBIO[clave]
    for a in REGISTRO:
        if _clave_de_cambio(a) == clave:
            return a.etiqueta
    return str(clave)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────── utilidades
def _uuid(v) -> Optional[str]:
    """El id en forma canónica, o None si no es un uuid (un invitado no lo es): no se consulta la base con basura."""
    try:
        return str(uuid.UUID(str(v).strip()))
    except (ValueError, AttributeError, TypeError):
        return None


def _iso(v) -> Optional[str]:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v) if v not in (None, "") else None


def _fecha(v) -> Optional[datetime]:
    """Un instante comparable (UTC si viene sin zona), o None."""
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str) and v:
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _mas_reciente(*fechas):
    """La fecha más reciente de las dadas (en su forma original), o None."""
    mejores = [(f, _fecha(f)) for f in fechas if _fecha(f) is not None]
    return max(mejores, key=lambda x: x[1])[0] if mejores else None


def _seguro(v):
    """Un valor apto para enseñar: textos recortados, colecciones grandes resumidas, fechas en ISO."""
    if isinstance(v, str):
        return v if len(v) <= MAX_TEXTO else v[:MAX_TEXTO] + "…"
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (dict, list)):
        texto = json.dumps(v, ensure_ascii=False, default=str)
        return v if len(texto) <= _JSON_MAX else texto[:MAX_TEXTO] + "…"
    return v


def _json(v):
    """jsonb que llega como texto (algunos drivers o filas viejas) → objeto; lo demás tal cual."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _acotar(v, minimo: int, maximo: int, defecto: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return defecto
    return max(minimo, min(maximo, n))


def _origen(v) -> str:
    return v if v in ORIGENES else "app"


# ───────────────────────────────────────────────────────────────────────────────────────────────────── las consultas
def _sql_panel(clave: str) -> str:
    """`{relleno, updatedAt}` de un panel de `health_profile`, calculado EN la base: su contenido nunca sale de ella.
    Relleno = alguna clave, además de `updatedAt`, con valor no vacío (null, false, "", [] y {} cuentan como vacío)."""
    panel = f"p.health_profile -> '{clave}'"
    return (
        "jsonb_build_object("
        "'relleno', COALESCE((SELECT bool_or(r.value NOT IN ('null'::jsonb, 'false'::jsonb, '[]'::jsonb, "
        "'{}'::jsonb, to_jsonb(''::text))) "
        f"FROM jsonb_each(CASE WHEN jsonb_typeof({panel}) = 'object' THEN {panel} ELSE '{{}}'::jsonb END) AS r "
        "WHERE r.key <> 'updatedAt'), false), "
        f"'updatedAt', {panel} ->> 'updatedAt')"
    )


def _sql_perfiles(filtro: str) -> str:
    """Las columnas de ajustes y la PROYECCIÓN de `health_profile`: solo las claves de ajustes (`_HP_VALORES`, primer
    parámetro) y las `avisos_*`; de los paneles, si están rellenos y cuándo se guardaron; de `staple_foods`, cuántos."""
    columnas = ", ".join(f"p.{c}" for c in _COLUMNAS_PERFIL)
    return (
        f"SELECT p.id::text AS id, {columnas}, "
        "(SELECT COALESCE(jsonb_object_agg(e.key, e.value), '{}'::jsonb) "
        "FROM jsonb_each(CASE WHEN jsonb_typeof(p.health_profile) = 'object' THEN p.health_profile "
        "ELSE '{}'::jsonb END) AS e "
        "WHERE e.key = ANY(%s::text[]) OR left(e.key, 7) = 'avisos_') AS hp, "
        "jsonb_build_object("
        f"'super_personalization', {_sql_panel('super_personalization')}, "
        f"'clinical_profile', {_sql_panel('clinical_profile')}, "
        "'staple_foods', CASE WHEN jsonb_typeof(p.health_profile -> 'staple_foods') = 'array' "
        "THEN jsonb_array_length(p.health_profile -> 'staple_foods') END"
        ") AS hp_resumen "
        f"FROM public.user_profiles p WHERE {filtro}"
    )


# Una consulta por tabla y un solo resultado por cuenta (`id`, `v`). Nunca se lee el contenido: ni el token de Apple, ni
# la suscripción push, ni qué suplementos o marcas; solo si hay y cuántos.
_SQL_TABLAS = {
    "device_push_tokens": ("SELECT user_id::text AS id, array_agg(DISTINCT platform ORDER BY platform) AS v "
                           "FROM public.device_push_tokens WHERE user_id = ANY(%s::uuid[]) GROUP BY user_id"),
    "push_subscriptions": ("SELECT user_id::text AS id, count(*) AS v "
                           "FROM public.push_subscriptions WHERE user_id = ANY(%s::uuid[]) GROUP BY user_id"),
    "user_brand_preferences": ("SELECT user_id::text AS id, count(*) AS v "
                               "FROM public.user_brand_preferences WHERE user_id = ANY(%s::uuid[]) GROUP BY user_id"),
    # [SUPLEMENTOS-OK: aquí se cuentan SOLO los suplementos (el ajuste «Suplementos en la Alacena»)]
    "user_inventory": ("SELECT user_id::text AS id, count(*) AS v "
                       "FROM public.user_inventory WHERE kind = 'supplement' AND user_id = ANY(%s::uuid[]) "
                       "GROUP BY user_id"),
    "apple_signin_tokens": ("SELECT user_id::text AS id, true AS v "
                            "FROM public.apple_signin_tokens WHERE user_id = ANY(%s::uuid[])"),
}
_SQL_KV = "SELECT key, value, updated_at FROM public.app_kv_store WHERE key = ANY(%s::text[])"
_SQL_ULTIMOS_CAMBIOS = ("SELECT clave, antes, despues, origen, at FROM public.ajustes_cambios "
                        "WHERE user_id = %s ORDER BY at DESC, id DESC LIMIT %s")
_PREFIJOS_KV = tuple(a.fuente[1] for a in REGISTRO if a.fuente[0] == "kv")


def _cargar(filtro: str, param) -> dict:
    """{uid: fila} con todo lo que necesitan los ajustes de esas cuentas: una consulta por fuente, no una por cuenta.
    El perfil es obligatorio (si la base falla, LANZA: quien llama decide); las fuentes auxiliares son best-effort y la
    que falla queda en `fallidas`, así sus ajustes se omiten en vez de inventar un estado."""
    filas = {}
    for f in execute_sql_query(_sql_perfiles(filtro), (list(_HP_VALORES), param), fetch_all=True) or []:
        uid = str(f.get("id"))
        hp = _json(f.get("hp"))
        resumen = _json(f.get("hp_resumen"))
        disp = _json(f.get("ajustes_dispositivo"))
        filas[uid] = {
            "col": {c: f.get(c) for c in _COLUMNAS_PERFIL},
            "hp": hp if isinstance(hp, dict) else {},
            "hp_resumen": resumen if isinstance(resumen, dict) else {},
            "tablas": {},
            "kv": {},
            "dispositivo": disp if isinstance(disp, dict) else {},
            "fallidas": set(),
        }
    uids = list(filas)
    if not uids:
        return filas
    for tabla, sql in _SQL_TABLAS.items():
        try:
            for r in execute_sql_query(sql, (uids,), fetch_all=True) or []:
                if str(r.get("id")) in filas:
                    filas[str(r.get("id"))]["tablas"][tabla] = r.get("v")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ [P1-PLAN-LOTE-837] ajustes: {tabla} ilegible ({e!r}); sus ajustes no se muestran")
            for fila in filas.values():
                fila["fallidas"].add(("tabla", tabla))
    try:
        claves = [f"{pref}{uid}" for pref in _PREFIJOS_KV for uid in uids]
        for r in execute_sql_query(_SQL_KV, (claves,), fetch_all=True) or []:
            clave = str(r.get("key") or "")
            for pref in _PREFIJOS_KV:
                if clave.startswith(pref) and clave[len(pref):] in filas:
                    filas[clave[len(pref):]]["kv"][pref] = {"value": _json(r.get("value")),
                                                            "updated_at": r.get("updated_at")}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-837] ajustes: app_kv_store ilegible ({e!r}); sus ajustes no se muestran")
        for fila in filas.values():
            fila["fallidas"].add(("kv", "*"))
    return filas


def _ultimos_cambios(uid: str) -> list:
    """Los últimos cambios de la cuenta (más nuevo primero), para `cambiado_at`/`origen` de la ficha. Best-effort."""
    try:
        return list(execute_sql_query(_SQL_ULTIMOS_CAMBIOS, (uid, MAX_HISTORIAL), fetch_all=True) or [])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-837] historial de ajustes ilegible para {uid[:8]} ({e!r})")
        return []


# ──────────────────────────────────────────────────────────────────────────────────────────── el estado de cada ajuste
def _fila_estado(estado: str, valor=None, cambiado_at=None, origen=None) -> dict:
    return {"estado": estado, "valor": _seguro(valor), "cambiado_at": _iso(cambiado_at), "origen": origen}


def _de_bool(crudo, por_defecto) -> dict:
    if crudo is None:
        if por_defecto is None:
            return _fila_estado("sin_elegir")
        return _fila_estado("encendido" if por_defecto else "apagado")
    if isinstance(crudo, bool):
        return _fila_estado("encendido" if crudo else "apagado", crudo)
    return _fila_estado("valor", crudo)


def _de_enum(crudo, por_defecto) -> dict:
    if crudo is None or crudo == "":
        return _fila_estado("valor", por_defecto) if por_defecto is not None else _fila_estado("sin_elegir")
    return _fila_estado("valor", crudo)


def _de_conteo(n) -> dict:
    n = n if isinstance(n, int) and not isinstance(n, bool) else 0
    return _fila_estado("valor", n) if n > 0 else _fila_estado("sin_elegir", 0)


def _dispositivo_limpio(disp) -> dict:
    """Solo plataformas y claves de la lista cerrada (lo que la base traiga de más no sale)."""
    salida = {}
    if not isinstance(disp, dict):
        return salida
    for plataforma in PLATAFORMAS:
        datos = disp.get(plataforma)
        if not isinstance(datos, dict):
            continue
        limpio = {}
        for k, regla in CLAVES_DISPOSITIVO.items():
            v = datos.get(k)
            if (regla is bool and isinstance(v, bool)) or (regla is not bool and isinstance(v, str) and v in regla):
                limpio[k] = v
        if isinstance(datos.get("pwa"), bool):
            limpio["pwa"] = datos["pwa"]
        if isinstance(datos.get("app_build"), str) and _RE_BUILD.fullmatch(datos["app_build"]):
            limpio["app_build"] = datos["app_build"]
        if _fecha(datos.get("at")) is not None:
            limpio["at"] = datos["at"]
        salida[plataforma] = limpio
    return salida


def _aviso_de_comida(hp: dict, comida: str) -> dict:
    todas = hp.get("avisos_por_comida")
    if todas is None:
        return _fila_estado("encendido")
    if not isinstance(todas, dict):
        return _fila_estado("valor", todas)
    propia = todas.get(comida)
    if propia is None:
        return _fila_estado("encendido")
    if not isinstance(propia, dict):
        return _fila_estado("valor", propia)
    hora = propia.get("hora") if isinstance(propia.get("hora"), str) and _RE_HORA.match(propia["hora"]) else None
    activo = propia.get("activo")
    if activo is None or isinstance(activo, bool):
        return _fila_estado("apagado" if activo is False else "encendido", hora)
    return _fila_estado("valor", propia)


def _consentimiento_ia(col: dict) -> dict:
    from consentimientos import estado_de_fila
    version, revocado = col.get("ai_consent_version"), col.get("ai_consent_revoked_at")
    cuando = _mas_reciente(col.get("ai_consent_at"), revocado)
    if estado_de_fila(col).get("vigente"):
        return _fila_estado("encendido", version, cuando)
    if revocado is not None:
        return _fila_estado("apagado", version, cuando)
    if version:
        return _fila_estado("valor", version, cuando)      # un permiso de una versión vieja: hay que volver a pedirlo
    return _fila_estado("sin_elegir")


def _evaluar(a: Ajuste, fila: dict) -> Optional[dict]:
    """`{estado, valor, cambiado_at, origen}` de un ajuste en una cuenta, sin el historial (lo añade `_lista`). None si su
    fuente no se pudo leer: sin dato no se inventa un estado."""
    tipo, (fuente, nombre) = a.tipo, a.fuente
    if (fuente, nombre) in fila["fallidas"] or (fuente == "kv" and ("kv", "*") in fila["fallidas"]):
        return None
    col, hp, tablas, kv = fila["col"], fila["hp"], fila["tablas"], fila["kv"]
    if tipo == "aviso_comida":
        return _aviso_de_comida(hp, nombre.split(".", 1)[1])
    if tipo == "relleno":
        panel = fila["hp_resumen"].get(nombre)
        panel = panel if isinstance(panel, dict) else {}
        return _fila_estado("valor" if panel.get("relleno") is True else "sin_elegir",
                            "relleno" if panel.get("relleno") is True else None, panel.get("updatedAt"))
    if tipo == "consentimiento_ia":
        return _consentimiento_ia(col)
    if fuente == "dispositivo":
        return _del_dispositivo(a, fila["dispositivo"])
    crudo = col.get(nombre) if fuente == "columna" else hp.get(nombre) if fuente == "perfil" else None
    if tipo == "bool":
        base = _de_bool(crudo, a.por_defecto)
        if a.clave == "water_tracker_enabled" and crudo is False:
            estado = (kv.get(_PREFIJO_HIDRATACION) or {}).get("value")
            if isinstance(estado, dict) and estado.get("auto_off_at"):   # la apagó el sistema y sigue apagada
                base.update(cambiado_at=_iso(estado["auto_off_at"]), origen="sistema")
        return base
    if tipo == "tri":
        if crudo is None:
            base = _fila_estado("automatico")
        elif isinstance(crudo, bool):
            base = _fila_estado("encendido" if crudo else "apagado", crudo)
        else:
            base = _fila_estado("valor", crudo)
        if crudo is False and col.get("nevera_auto_off_at"):     # la apagó el sistema (y nadie la tocó después)
            base.update(cambiado_at=_iso(col["nevera_auto_off_at"]), origen="sistema")
        return base
    if tipo == "modo":
        valor = crudo if crudo not in (None, "") else a.por_defecto
        estado = _MODOS[a.clave].get(valor) if isinstance(valor, str) else None
        base = _fila_estado(estado, valor) if estado else _fila_estado("valor", valor)
        if a.clave == "plan_mode" and col.get("plan_mode_changed_at"):
            base["cambiado_at"] = _iso(col["plan_mode_changed_at"])
        return base
    if tipo == "enum":
        return _de_enum(crudo, a.por_defecto)
    if tipo == "conteo":
        if fuente == "perfil":
            return _de_conteo(fila["hp_resumen"].get(nombre))
        return _de_conteo(tablas.get(nombre))
    if tipo == "existe":
        n = tablas.get(nombre)
        n = n if isinstance(n, int) and not isinstance(n, bool) else 0
        return _fila_estado("encendido", n) if n > 0 else _fila_estado("apagado", 0)
    if tipo == "hecho":
        return _fila_estado("valor", "sí" if tablas.get(nombre) else "no")
    if tipo == "plataformas_push":
        plataformas = sorted({str(p) for p in (tablas.get(nombre) or []) if p})
        return _fila_estado("encendido", ",".join(plataformas)) if plataformas else _fila_estado("apagado")
    if tipo == "canal_local":
        entrada = kv.get(nombre)
        if not entrada:
            return _fila_estado("sin_elegir")
        cuando = _fecha(entrada.get("updated_at"))
        vivo = cuando is not None and cuando >= datetime.now(timezone.utc) - timedelta(hours=_horas_canal_local())
        return _fila_estado("encendido" if vivo else "apagado", None, entrada.get("updated_at"))
    if tipo == "apagado_solo":
        valor = (kv.get(nombre) or {}).get("value")
        cuando = valor.get("auto_off_at") if isinstance(valor, dict) else None
        if cuando:
            return _fila_estado("valor", "sí", cuando, "sistema")
        return _fila_estado("valor", "no")
    if tipo == "invitacion":
        valor = (kv.get(nombre) or {}).get("value")
        valor = valor if isinstance(valor, dict) else {}
        if valor.get("dismissed_at"):
            return _fila_estado("valor", "descartada", valor["dismissed_at"])
        if valor.get("shown_at"):
            return _fila_estado("valor", "vista", valor["shown_at"])
        return _fila_estado("sin_elegir")
    raise ValueError(f"tipo de ajuste sin lector: {tipo}")


def _del_dispositivo(a: Ajuste, disp_crudo) -> dict:
    """Del informe más reciente que TRAE la clave (una persona puede usar la web y el teléfono)."""
    disp = _dispositivo_limpio(disp_crudo)
    clave = a.fuente[1]
    if clave == "plataforma":
        return _fila_estado("valor", ",".join(sorted(disp))) if disp else _fila_estado("sin_elegir")
    minimo = datetime.min.replace(tzinfo=timezone.utc)
    informes = sorted((d for d in disp.values() if clave in d), key=lambda d: _fecha(d.get("at")) or minimo,
                      reverse=True)
    if not informes:
        return _fila_estado("sin_elegir")
    valor, cuando = informes[0][clave], informes[0].get("at")
    if a.tipo == "bool":
        base = _de_bool(valor, None)
    else:
        base = _de_enum(valor, None)
    base["cambiado_at"] = _iso(cuando)
    return base


def _config_comida(v, comida: str):
    return v.get(comida) if isinstance(v, dict) else v


def _ultimo_cambio(a: Ajuste, cambios: list) -> Optional[dict]:
    claves = _claves_de_cambio(a)
    if not claves:
        return None
    comida = a.fuente[1].split(".", 1)[1] if a.tipo == "aviso_comida" else None
    for c in cambios:
        if c.get("clave") not in claves:
            continue
        if comida and _config_comida(c.get("antes"), comida) == _config_comida(c.get("despues"), comida):
            continue          # ese cambio de `avisos_por_comida` no tocó esta comida
        return c
    return None


def _otros(fila: dict) -> list:
    """Las `avisos_*` que la app guarda y el registro aún no conoce: salen con su valor tal cual."""
    return [{"clave": k, "etiqueta": k, "grupo": GRUPO_OTROS, "estado": "valor", "valor": _seguro(v),
             "cambiado_at": None, "origen": None}
            for k, v in sorted(fila["hp"].items()) if k.startswith("avisos_") and k not in _CLAVES_AVISOS_REGISTRADAS]


def _lista(fila: dict, cambios: Optional[list] = None, con_otros: bool = True) -> list:
    salida = []
    for a in REGISTRO:
        base = _evaluar(a, fila)
        if base is None:
            continue
        if cambios:
            ultimo = _ultimo_cambio(a, cambios)
            if ultimo:
                base["cambiado_at"], base["origen"] = _iso(ultimo.get("at")), _origen(ultimo.get("origen"))
        salida.append({"clave": a.clave, "etiqueta": a.etiqueta, "grupo": a.grupo, **base})
    if con_otros:
        salida.extend(_otros(fila))
    return salida


# ─────────────────────────────────────────────────────────────────────────────────────────────────── la API del módulo
def ajustes_de(user_id) -> dict:
    """`{"ajustes": [{clave, etiqueta, grupo, estado, valor, cambiado_at, origen}], "ajustes_dispositivo": {…}}` de UNA
    cuenta (contrato 3). Sin perfil (o id que no es un uuid): listas vacías. LANZA si la base no devuelve el perfil."""
    uid = _uuid(user_id)
    if not uid:
        return {"ajustes": [], "ajustes_dispositivo": {}}
    fila = _cargar("p.id = ANY(%s::uuid[])", [uid]).get(uid)
    if fila is None:
        return {"ajustes": [], "ajustes_dispositivo": {}}
    return {"ajustes": _lista(fila, _ultimos_cambios(uid)),
            "ajustes_dispositivo": _dispositivo_limpio(fila["dispositivo"])}


def ajustes_de_varias(user_ids) -> dict:
    """{uid: {"ajustes", "ajustes_dispositivo"}} de varias cuentas con UNA consulta por fuente (el CSV de la lista). Sin
    el historial: `cambiado_at` solo cuando la fuente lo trae (la marca de un apagado automático, `updatedAt`…)."""
    uids = list(dict.fromkeys(u for u in (_uuid(x) for x in (user_ids or [])) if u))
    if not uids:
        return {}
    return {uid: {"ajustes": _lista(fila), "ajustes_dispositivo": _dispositivo_limpio(fila["dispositivo"])}
            for uid, fila in _cargar("p.id = ANY(%s::uuid[])", uids).items()}


def historial(user_id, dias: int = 90) -> list:
    """Los cambios de ajustes de la cuenta en los últimos `dias` (1..365), el más nuevo primero, como mucho
    `MAX_HISTORIAL` (contrato 7). LANZA si la base falla (el router responde sin datos)."""
    uid = _uuid(user_id)
    if not uid:
        return []
    filas = execute_sql_query(
        "SELECT clave, antes, despues, origen, at FROM public.ajustes_cambios "
        "WHERE user_id = %s AND at >= now() - make_interval(days => %s) ORDER BY at DESC, id DESC LIMIT %s",
        (uid, _acotar(dias, 1, 365, 90), MAX_HISTORIAL), fetch_all=True) or []
    return [{"at": _iso(f.get("at")), "clave": f.get("clave"), "etiqueta": etiqueta_de_cambio(f.get("clave")),
             "antes": _seguro(f.get("antes")), "despues": _seguro(f.get("despues")),
             "origen": _origen(f.get("origen"))} for f in filas]


def _clave_de_valor(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)[:MAX_TEXTO]


def _cambios_por_origen(dias: int, fuera: list) -> list:
    try:
        filas = execute_sql_query(
            "SELECT clave, origen, count(*) AS n FROM public.ajustes_cambios "
            "WHERE at >= now() - make_interval(days => %s) AND user_id::text <> ALL(%s::text[]) GROUP BY clave, origen",
            (dias, fuera), fetch_all=True) or []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-837] resumen: historial de ajustes ilegible ({e!r})")
        return []
    por_clave: dict = {}
    for f in filas:
        clave = str(f.get("clave"))
        cuenta = por_clave.setdefault(clave, {o: 0 for o in ORIGENES})
        cuenta[_origen(f.get("origen"))] += int(f.get("n") or 0)
    orden = sorted(por_clave.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    return [{"clave": c, "etiqueta": etiqueta_de_cambio(c), "por_origen": o} for c, o in orden]


def resumen(dias: int = 30) -> dict:
    """«Ajustes de la gente» (contrato 8): por ajuste del registro, cuántas cuentas lo tienen encendido, apagado,
    automático, sin elegir o con cada valor; los cambios del periodo (1..90 días) por origen; y el dispositivo (tema,
    permiso de notificaciones y plataformas, contando un informe por cuenta y plataforma). Sin las cuentas admin."""
    dias = _acotar(dias, 1, 90, 30)
    from admin_metricas import _ids_fuera
    fuera = list(_ids_fuera())
    filas = _cargar("p.id::text <> ALL(%s::text[])", fuera)
    conteo = {a.clave: {"encendido": 0, "apagado": 0, "automatico": 0, "sin_elegir": 0, "valores": {}}
              for a in REGISTRO}
    dispositivo = {"tema": {}, "notificaciones_permiso": {}, "plataformas": {}}
    for fila in filas.values():
        for item in _lista(fila, con_otros=False):
            c = conteo[item["clave"]]
            if item["estado"] == "valor":
                k = _clave_de_valor(item["valor"])
                c["valores"][k] = c["valores"].get(k, 0) + 1
            else:
                c[item["estado"]] += 1
        for plataforma, datos in _dispositivo_limpio(fila["dispositivo"]).items():
            plats = dispositivo["plataformas"]
            plats[plataforma] = plats.get(plataforma, 0) + 1
            if datos.get("pwa") is True:
                plats["pwa"] = plats.get("pwa", 0) + 1
            for k in ("tema", "notificaciones_permiso"):
                if k in datos:
                    dispositivo[k][datos[k]] = dispositivo[k].get(datos[k], 0) + 1
    return {
        "cuentas": len(filas),
        "ajustes": [{"clave": a.clave, "etiqueta": a.etiqueta, "grupo": a.grupo, "conteo": conteo[a.clave]}
                    for a in REGISTRO],
        "cambios": _cambios_por_origen(dias, fuera),
        "dispositivo": dispositivo,
    }


def columnas_csv() -> list:
    """Una columna por ajuste del registro, en su orden (`ajuste.<clave>`)."""
    return [f"ajuste.{a.clave}" for a in REGISTRO]


def _texto(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


def fila_csv(ajustes) -> list:
    """Un valor por columna de `columnas_csv`: el estado, o el valor si el estado es `valor`. Lo que empieza como una
    fórmula de hoja de cálculo (`=`, `+`, `-`, `@`) va precedido de un apóstrofo: abrir el CSV no ejecuta nada."""
    por_clave = {a.get("clave"): a for a in (ajustes or []) if isinstance(a, dict)}
    salida = []
    for aj in REGISTRO:
        item = por_clave.get(aj.clave)
        if not item:
            salida.append("")
            continue
        texto = _texto(item.get("valor")) if item.get("estado") == "valor" else str(item.get("estado") or "")
        salida.append("'" + texto if texto[:1] in ("=", "+", "-", "@", "\t", "\r") else texto)
    return salida


# ───────────────────────────────────────────────────────────────────────────────────────── los ajustes del dispositivo
def limpiar_dispositivo(cuerpo) -> Optional[tuple]:
    """`(plataforma, datos)` con solo lo que admite la lista cerrada, o None si la plataforma no es de la lista."""
    if not isinstance(cuerpo, dict):
        return None
    plataforma = cuerpo.get("plataforma")
    if not isinstance(plataforma, str) or plataforma not in PLATAFORMAS:
        return None
    datos: dict = {}
    if isinstance(cuerpo.get("pwa"), bool):
        datos["pwa"] = cuerpo["pwa"]
    build = cuerpo.get("app_build")
    if isinstance(build, str) and _RE_BUILD.fullmatch(build.strip()):
        datos["app_build"] = build.strip()
    ajustes = cuerpo.get("ajustes")
    if isinstance(ajustes, dict):
        for k, regla in CLAVES_DISPOSITIVO.items():
            v = ajustes.get(k)
            if (regla is bool and isinstance(v, bool)) or (regla is not bool and isinstance(v, str) and v in regla):
                datos[k] = v
    return plataforma, datos


def guardar_dispositivo(user_id, cuerpo) -> bool:
    """Guarda el informe de UNA plataforma en `user_profiles.ajustes_dispositivo` (con su `at`), sin tocar las demás.
    False sin escribir con el interruptor maestro apagado, un id que no es de cuenta o una plataforma fuera de lista.
    Nunca lanza: el informe es un extra, la app no espera por él."""
    import cuentas_prueba
    if not cuentas_prueba.activo():
        return False
    uid = _uuid(user_id)
    informe = limpiar_dispositivo(cuerpo) if uid else None
    if informe is None:
        return False
    plataforma, datos = informe
    datos["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        filas = execute_sql_write(
            "UPDATE public.user_profiles SET ajustes_dispositivo = jsonb_set("
            "CASE WHEN jsonb_typeof(ajustes_dispositivo) = 'object' THEN ajustes_dispositivo ELSE '{}'::jsonb END, "
            "%s::text[], %s::jsonb, true) WHERE id = %s RETURNING id",
            ([plataforma], json.dumps(datos, ensure_ascii=False), uid), returning=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-837] ajustes del dispositivo no guardados ({uid[:8]}, {plataforma}): {e!r}")
        return False
    return bool(filas)


# ───────────────────────────────────────────────────────────────────────────────────────────────────────── la purga
def dias_de_historial() -> int:
    """Plazo del historial de ajustes. Fuera de [90, 3650] (o ilegible) vale el defecto."""
    return _env_int("MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS", 730, validator=lambda v: 90 <= v <= 3650)


def _canario_del_historial() -> None:
    """[P1-PLAN-LOTE-837 · 2026-09-29] Una línea al día con cuántos cambios hay y cuándo fue el último (`count(*)` y
    `max(at)`). El trigger que llena la tabla avisa de sus propios fallos con un WARNING de POSTGRES, que no llega a los
    logs de la app: una tabla que dejó de llenarse (trigger caído, migración a medias) pasaría meses sin que nada lo
    dijera. Con la app en uso, un `max(at)` que no avanza es la señal. Nunca lanza."""
    try:
        fila = execute_sql_query(
            "SELECT count(*) AS n, max(at) AS ultimo FROM public.ajustes_cambios", fetch_one=True) or {}
        n = fila.get("n")
        n = n if isinstance(n, int) and not isinstance(n, bool) else 0
        logger.info(f"[P1-PLAN-LOTE-837] canario del historial de ajustes: {n} cambios, el último: "
                    f"{_iso(fila.get('ultimo')) or 'ninguno'}.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-837] canario del historial de ajustes sin leer: {e!r}")


def purgar_cambios_antiguos() -> int:
    """Borra los cambios con más de `dias_de_historial()` días y deja el canario del día (`_canario_del_historial`,
    también si la purga falla). Cron diario; nunca lanza (un fallo se reintenta mañana)."""
    dias = dias_de_historial()
    try:
        r = execute_sql_write(
            "DELETE FROM public.ajustes_cambios WHERE at < now() - make_interval(days => %s) RETURNING id",
            (dias,), returning=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-837] no se pudo purgar el historial de ajustes: {e!r}")
        _canario_del_historial()
        return 0
    n = len(r) if isinstance(r, list) else 0
    if n:
        logger.info(f"[P1-PLAN-LOTE-837] historial de ajustes: {n} cambios con más de {dias} días purgados.")
    _canario_del_historial()
    return n
