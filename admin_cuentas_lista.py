# backend/admin_cuentas_lista.py
"""[P1-PLAN-LOTE-831 · 2026-09-29] Panel admin · Cuentas: la lista con la actividad EN NÚMEROS, el CSV y la ficha
ampliada (actividad, ajustes y cuenta de prueba).

Spec docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md §4.1-§4.3, §13.5 y §13.6 (raíz del
workspace). Todo va detrás del interruptor maestro `MEALFIT_ADMIN_TEST_ACCOUNTS`: con él apagado el router
(`routers/admin.py`) responde 404 a lo nuevo y no llama a este módulo. La ficha de siempre (`admin_cuentas.ficha`,
lote 774) no cambia —la usan también los regalos al escribir—: `ampliar_ficha` le añade los bloques nuevos solo en las
vistas del panel.

Solo números y ajustes, NUNCA contenido: ni platos, ni mensajes, ni el perfil de salud (eso vive en el detalle de una
cuenta de prueba, lote 832). Las definiciones son UNA para la lista, el CSV y la ficha (§4.1):

  · comidas: filas de `consumed_meals`, con su `consumed_at` (la fecha que usa `admin_metricas._actividad_sql`);
  · planes: filas de `meal_plans`;
  · mensajes al coach: `agent_messages` con `role = 'user'` en los HILOS de la cuenta, que decide el SSOT
    `USER_CHAT_THREAD_IDS_SQL` (el mismo conjunto que exportan y borran «Mis datos» y «Eliminar cuenta»);
  · escaneos: `llm_usage_events` con `node = 'vision_scan'`;
  · gasto de IA: la suma de `llm_usage_events.cost_usd_micros` de la cuenta en 30 días, en US$ con 2 decimales;
  · día activo: un día UTC con una comida o un mensaje al coach;
  · última actividad: lo más reciente de todo lo anterior y de cualquier uso de IA.

Una consulta por página: cada cuenta con sus subconsultas (LATERAL), el filtro y el orden encima, y el total en la misma
consulta (`count(*) OVER ()`). Los hilos de cada fila son el SSOT con `p.id` en sus tres parámetros (los tres son el
uid: ver `db_profiles`). El orden y el filtro salen de una lista CERRADA (el router ya respondió 422 a lo demás) y el
texto que busca el admin va como parámetro, con sus comodines escapados: nada del cliente se interpola en el SQL.

La lista enseña también las cuentas de administración, con su etiqueta (`es_admin`: tier `admin` o la lista del .env
del panel); los agregados las excluyen (`ajustes_cuenta.resumen`). Si falla la consulta principal, las funciones de la
lista y del CSV LANZAN (el router responde 503 sin datos); en la ficha, cada bloque nuevo que falla sale vacío y la
ficha carga igual.
"""
from __future__ import annotations

import csv
import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import ajustes_cuenta
import cuentas_prueba
import regalos_cuenta as rc
from admin_acceso import admin_ids
from db import USER_CHAT_THREAD_IDS_SQL, execute_sql_query

logger = logging.getLogger(__name__)

POR_PAGINA = 50
MAX_CSV = 5000
MAX_BUSQUEDA = 254          # un correo entero cabe
PLATAFORMAS = ("web", "ios", "android")

# Listas CERRADAS: la clave es el valor del contrato, el valor es SQL sobre las columnas de `filas` (ver `_sql_filas`).
ORDENES = {
    "actividad": "ultima DESC NULLS LAST, alta DESC, user_id",
    "alta": "alta DESC, user_id",
    "comidas": "comidas_total DESC, ultima DESC NULLS LAST, user_id",
    "gasto": "micros_30d DESC, ultima DESC NULLS LAST, user_id",
}
FILTROS = {
    "todas": "TRUE",
    "prueba": "prueba_desde IS NOT NULL",
    "sin_marcar": "prueba_desde IS NULL",
    "activas_7d": "ultima >= now() - interval '7 days'",
    # Sin actividad en 14 días; la que nunca hizo nada cuenta desde su alta (la cuenta de ayer no está «inactiva»).
    "inactivas_14d": "COALESCE(ultima, alta) < now() - interval '14 days'",
    "con_plan": "planes > 0",
    "seguimiento": "plan_mode = 'tracking'",
}

_ACTIVIDAD = ("ultima", "comidas_total", "comidas_30d", "planes", "mensajes_coach", "escaneos", "gasto_ia_30d_usd",
              "dias_activos_30d")
# El CSV: la fila del contrato aplanada, en su orden (después van las columnas de `ajustes_cuenta.columnas_csv`).
COLUMNAS_FILA = ("user_id", "email", "nombre", "alta", "plan_pagado", "plan_efectivo", "es_admin", "prueba.estado",
                 "prueba.desde", *(f"actividad.{k}" for k in _ACTIVIDAD), "modo", "idioma", "pais")
_INICIO_DE_FORMULA = ("=", "+", "-", "@", "\t", "\r")

_TREINTA_DIAS = "now() - interval '30 days'"
# Los hilos de la fila `p` (lista y ficha): el SSOT con `p.id` en sus tres parámetros.
_EN_SUS_HILOS = f"(m.user_id = p.id OR m.session_id::text IN ({USER_CHAT_THREAD_IDS_SQL.replace('%s', 'p.id')}))"
# Los de UNA cuenta con parámetros (extras de la ficha): cada `%s` es el uid.
_EN_SUS_HILOS_UID = f"(m.user_id = %s OR m.session_id::text IN ({USER_CHAT_THREAD_IDS_SQL}))"


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────── utilidades
def _uuid(v) -> Optional[str]:
    """El id en forma canónica, o None si no es un uuid (un invitado no lo es): no se consulta la base con basura."""
    try:
        return str(uuid.UUID(str(v).strip()))
    except (ValueError, AttributeError, TypeError):
        return None


def _n(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _usd(micros) -> float:
    return round(_n(micros) / 1_000_000, 2)


def _instante(v) -> Optional[datetime]:
    if not isinstance(v, datetime):
        return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def _el_primero(*fechas):
    """La fecha más temprana de las dadas (en su forma original), o None."""
    validas = [f for f in fechas if _instante(f) is not None]
    return min(validas, key=_instante) if validas else None


def _validar(orden: str, filtro: str) -> None:
    if orden not in ORDENES:
        raise ValueError(f"orden desconocido: {orden!r}")
    if filtro not in FILTROS:
        raise ValueError(f"filtro desconocido: {filtro!r}")


def patron_de_busqueda(buscar) -> Optional[str]:
    """El patrón ILIKE de `buscar` con `\\`, `%` y `_` escapados (van con `ESCAPE '\\'`): «50%» busca «50%», no todo.
    None si no hay texto."""
    texto = str(buscar or "").strip()[:MAX_BUSQUEDA]
    if not texto:
        return None
    return "%" + texto.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _donde(buscar) -> tuple:
    patron = patron_de_busqueda(buscar)
    if patron is None:
        return "TRUE", ()
    return "(p.email ILIKE %s ESCAPE '\\' OR p.full_name ILIKE %s ESCAPE '\\')", (patron, patron)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────── el SQL
def _sql_filas(donde: str) -> str:
    """`WITH base …, filas …`: una fila por cuenta con sus números (subconsultas por cuenta). `donde` filtra
    `user_profiles p` (la búsqueda, o el id de la ficha)."""
    return (
        "WITH base AS ("
        "SELECT p.id::text AS user_id, p.email, p.full_name AS nombre, p.created_at AS alta, "
        "COALESCE(p.plan_tier, 'gratis') AS plan_pagado, p.plan_mode, p.locale, "
        "CASE WHEN jsonb_typeof(p.health_profile) = 'object' THEN p.health_profile ->> 'country' END AS pais, "
        "pr.marcada_at AS prueba_desde, pr.aviso_visto_at AS prueba_aviso_visto_at, "
        "cm.total AS comidas_total, cm.d30 AS comidas_30d, cm.ultima AS comidas_ultima, "
        "mp.total AS planes, mp.ultima AS planes_ultima, "
        "msg.total AS mensajes_coach, msg.ultima AS mensajes_ultima, "
        "ia.escaneos, ia.micros_30d, ia.ultima AS ia_ultima, "
        "(SELECT count(*) FROM ("
        "SELECT (c.consumed_at AT TIME ZONE 'UTC')::date FROM public.consumed_meals c "
        f"WHERE c.user_id = p.id AND c.consumed_at >= {_TREINTA_DIAS} "
        "UNION SELECT (m.created_at AT TIME ZONE 'UTC')::date FROM public.agent_messages m "
        f"WHERE m.role = 'user' AND {_EN_SUS_HILOS} AND m.created_at >= {_TREINTA_DIAS}"
        ") AS dias) AS dias_activos_30d "
        "FROM public.user_profiles p "
        "LEFT JOIN LATERAL (SELECT t.marcada_at, t.aviso_visto_at FROM public.cuentas_de_prueba t "
        "WHERE t.user_id = p.id AND t.quitada_at IS NULL LIMIT 1) pr ON true "
        "LEFT JOIN LATERAL (SELECT count(*) AS total, "
        f"count(*) FILTER (WHERE c.consumed_at >= {_TREINTA_DIAS}) AS d30, max(c.consumed_at) AS ultima "
        "FROM public.consumed_meals c WHERE c.user_id = p.id) cm ON true "
        "LEFT JOIN LATERAL (SELECT count(*) AS total, max(x.created_at) AS ultima "
        "FROM public.meal_plans x WHERE x.user_id = p.id) mp ON true "
        "LEFT JOIN LATERAL (SELECT count(*) AS total, max(m.created_at) AS ultima "
        f"FROM public.agent_messages m WHERE m.role = 'user' AND {_EN_SUS_HILOS}) msg ON true "
        "LEFT JOIN LATERAL (SELECT count(*) FILTER (WHERE e.node = 'vision_scan') AS escaneos, "
        f"COALESCE(sum(e.cost_usd_micros) FILTER (WHERE e.created_at >= {_TREINTA_DIAS}), 0) AS micros_30d, "
        "max(e.created_at) AS ultima FROM public.llm_usage_events e WHERE e.user_id = p.id) ia ON true "
        f"WHERE {donde}), "
        "filas AS (SELECT base.*, GREATEST(comidas_ultima, planes_ultima, mensajes_ultima, ia_ultima) AS ultima "
        "FROM base) "
    )


def _sql_pagina(donde: str, filtro: str, orden: str) -> str:
    return (_sql_filas(donde) + "SELECT filas.*, count(*) OVER () AS total FROM filas "
            f"WHERE {FILTROS[filtro]} ORDER BY {ORDENES[orden]} LIMIT %s OFFSET %s")


def _sql_total(donde: str, filtro: str) -> str:
    return _sql_filas(donde) + f"SELECT count(*) AS n FROM filas WHERE {FILTROS[filtro]}"


# Los números de la ficha que la lista no trae (UNA cuenta; todo `%s` es su uid). Nunca contenido: ni el texto de un
# mensaje, ni un plato, ni los vasos o el peso, solo cuántos.
_SQL_EXTRAS = (
    "SELECT "
    "(SELECT count(*) FROM public.water_intake_log w WHERE w.user_id = %s AND w.glasses > 0 "
    f"AND w.log_date >= ({_TREINTA_DIAS})::date) AS dias_con_agua_30d, "
    "(SELECT count(*) FROM public.weight_log wl WHERE wl.user_id = %s) AS registros_peso, "
    "(SELECT count(*) FROM public.plan_chunk_queue q WHERE q.user_id = %s AND q.status = 'failed' "
    f"AND COALESCE(q.dead_lettered_at, q.updated_at) >= {_TREINTA_DIAS}) AS bloques_fallidos_30d, "
    f"(SELECT count(*) FROM public.agent_messages m WHERE m.feedback = 'down' AND {_EN_SUS_HILOS_UID}) "
    "AS pulgares_abajo, "
    f"(SELECT min(m.created_at) FROM public.agent_messages m WHERE m.role = 'user' AND {_EN_SUS_HILOS_UID}) "
    "AS primer_mensaje, "
    "(SELECT min(x.created_at) FROM public.meal_plans x WHERE x.user_id = %s) AS primer_plan, "
    "(SELECT min(c.consumed_at) FROM public.consumed_meals c WHERE c.user_id = %s) AS primera_comida, "
    "(SELECT min(e.created_at) FROM public.llm_usage_events e WHERE e.user_id = %s AND e.node = 'vision_scan') "
    "AS primer_escaneo, "
    "(SELECT min(pm.created_at) FROM public.pipeline_metrics pm WHERE pm.user_id = %s "
    "AND pm.node = 'wizard_funnel' AND pm.metadata ->> 'event' = 'wizard_submit') AS formulario_enviado, "
    "(SELECT count(*) FROM public.system_alerts a WHERE a.resolved_at IS NULL AND strpos(a.alert_key, %s) > 0) "
    "AS avisos_abiertos, "
    "(SELECT array_agg(DISTINCT d.platform) FROM public.device_push_tokens d WHERE d.user_id = %s) "
    "AS plataformas_push, "
    "(SELECT count(*) FROM public.push_subscriptions s WHERE s.user_id = %s) AS suscripciones_web, "
    "(SELECT array_agg(DISTINCT uc.platform) FROM public.user_consents uc WHERE uc.user_id = %s) "
    "AS plataformas_consentimiento, "
    "(SELECT ARRAY(SELECT jsonb_object_keys(p.ajustes_dispositivo)) FROM public.user_profiles p "
    "WHERE p.id = %s AND jsonb_typeof(p.ajustes_dispositivo) = 'object') AS plataformas_dispositivo"
)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────── las filas
def _cortesias(uids: list) -> dict:
    """{uid: plan} de la cortesía vigente más reciente de cada cuenta (la regla de `regalos_cuenta.cortesia_de`), en
    UNA consulta. Fail-open hacia lo pagado, como `regalos_cuenta`: apagados o ilegibles, cada cuenta queda con lo que
    paga."""
    if not uids or not rc.activo():
        return {}
    try:
        # `rc._VIGENTE`: la condición de «regalo vigente» de `regalos_cuenta`, UNA sola definición para los dos.
        filas = execute_sql_query(
            "SELECT DISTINCT ON (g.user_id) g.user_id::text AS user_id, g.plan FROM public.account_grants g "
            f"WHERE g.user_id = ANY(%s::uuid[]) AND g.kind = 'plan' AND g.plan = ANY(%s::text[]) AND {rc._VIGENTE} "
            "ORDER BY g.user_id, g.created_at DESC",
            (list(uids), list(rc.PLANES_REGALABLES)), fetch_all=True) or []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-831] cortesías ilegibles en la lista ({e!r}): queda lo pagado")
        return {}
    return {str(f.get("user_id")): f.get("plan") for f in filas if f.get("plan") in rc.PLANES_REGALABLES}


def _modo(v) -> str:
    return "tracking" if v == "tracking" else "plan"


def _actividad(r: dict) -> dict:
    return {"ultima": rc.iso(r.get("ultima")), "comidas_total": _n(r.get("comidas_total")),
            "comidas_30d": _n(r.get("comidas_30d")), "planes": _n(r.get("planes")),
            "mensajes_coach": _n(r.get("mensajes_coach")), "escaneos": _n(r.get("escaneos")),
            "gasto_ia_30d_usd": _usd(r.get("micros_30d")), "dias_activos_30d": _n(r.get("dias_activos_30d"))}


def _fila_cuenta(r: dict, cortesias: dict, admins) -> dict:
    uid = str(r.get("user_id"))
    pagado = r.get("plan_pagado") or "gratis"
    cortesia = {"plan": cortesias[uid]} if uid in cortesias else None
    marca = ({"marcada_at": r.get("prueba_desde"), "aviso_visto_at": r.get("prueba_aviso_visto_at")}
             if r.get("prueba_desde") is not None else None)
    return {
        "user_id": uid, "email": r.get("email"), "nombre": r.get("nombre"), "alta": rc.iso(r.get("alta")),
        "plan_pagado": pagado, "plan_efectivo": rc.plan_efectivo(pagado, cortesia) or "gratis",
        "es_admin": pagado == "admin" or uid.lower() in admins,
        "prueba": ({"estado": cuentas_prueba.estado_de(marca), "desde": rc.iso(marca["marcada_at"])}
                   if marca else None),
        "actividad": _actividad(r),
        "modo": _modo(r.get("plan_mode")), "idioma": r.get("locale") or None, "pais": r.get("pais") or None,
    }


def _cuentas_de(filas: list) -> list:
    cortesias = _cortesias([str(f.get("user_id")) for f in filas])
    admins = admin_ids()
    return [_fila_cuenta(f, cortesias, admins) for f in filas]


# ─────────────────────────────────────────────────────────────────────────────────────────── la lista y el CSV
def listar(buscar="", orden: str = "actividad", filtro: str = "todas", pagina: int = 1) -> dict:
    """Contrato 1: `{"cuentas": [FilaCuenta], "total", "pagina", "por_pagina": 50}`. LANZA si la base falla."""
    _validar(orden, filtro)
    pagina = max(1, _n(pagina))
    donde, params = _donde(buscar)
    filas = execute_sql_query(_sql_pagina(donde, filtro, orden), (*params, POR_PAGINA, (pagina - 1) * POR_PAGINA),
                              fetch_all=True) or []
    if filas:
        total = _n(filas[0].get("total"))
    elif pagina > 1:     # más allá del final no hay fila que traiga el total: se cuenta aparte
        total = _n((execute_sql_query(_sql_total(donde, filtro), params, fetch_one=True) or {}).get("n"))
    else:
        total = 0
    return {"cuentas": _cuentas_de(filas), "total": total, "pagina": pagina, "por_pagina": POR_PAGINA}


def _valor(cuenta: dict, columna: str):
    if "." in columna:
        bloque, clave = columna.split(".", 1)
        return (cuenta.get(bloque) or {}).get(clave)
    return cuenta.get(columna)


def _celda(v) -> str:
    """El texto de una celda. Lo que empieza como una fórmula de hoja de cálculo va precedido de un apóstrofo (el
    nombre y el correo los escribe la persona): abrir el CSV no ejecuta nada. Igual que `ajustes_cuenta.fila_csv`."""
    if v is None:
        texto = ""
    elif isinstance(v, bool):
        texto = "sí" if v else "no"
    elif isinstance(v, float):
        texto = f"{v:.2f}"
    else:
        texto = str(v)
    return "'" + texto if texto[:1] in _INICIO_DE_FORMULA else texto


def exportar_csv(buscar="", orden: str = "actividad", filtro: str = "todas") -> tuple:
    """Contrato 2: `(texto, n)`. El texto lleva BOM (Excel lo abre en UTF-8), la cabecera y una fila por cuenta
    (`MAX_CSV` como mucho): la fila del contrato aplanada + un ajuste por columna. Nunca contenido. LANZA si la base
    falla, también si fallan los ajustes (un CSV sin sus columnas de ajustes diría «sin elegir» de todo)."""
    _validar(orden, filtro)
    donde, params = _donde(buscar)
    filas = execute_sql_query(_sql_pagina(donde, filtro, orden), (*params, MAX_CSV, 0), fetch_all=True) or []
    cuentas = _cuentas_de(filas)
    ajustes = ajustes_cuenta.ajustes_de_varias([c["user_id"] for c in cuentas]) if cuentas else {}
    salida = io.StringIO()
    escritor = csv.writer(salida)
    escritor.writerow([*COLUMNAS_FILA, *ajustes_cuenta.columnas_csv()])
    for c in cuentas:
        escritor.writerow([*(_celda(_valor(c, col)) for col in COLUMNAS_FILA),
                           *ajustes_cuenta.fila_csv((ajustes.get(c["user_id"]) or {}).get("ajustes"))])
    return "\ufeff" + salida.getvalue(), len(cuentas)


def csv_de(buscar="", orden: str = "actividad", filtro: str = "todas") -> str:
    """El texto del CSV (ver `exportar_csv`, que devuelve además cuántas cuentas lleva)."""
    return exportar_csv(buscar, orden, filtro)[0]


# ─────────────────────────────────────────────────────────────────────────────────────────────────── la ficha
def _plataformas(extras: dict) -> list:
    """Las plataformas en las que se ha visto la cuenta: sus tokens push nativos, `web` si tiene suscripción push del
    navegador, desde dónde decidió sus permisos y de dónde llegó el informe del dispositivo. Solo las tres conocidas."""
    vistas = set()
    for clave in ("plataformas_push", "plataformas_consentimiento", "plataformas_dispositivo"):
        vistas |= {str(x) for x in (extras.get(clave) or []) if x}
    if _n(extras.get("suscripciones_web")) > 0:
        vistas.add("web")
    return [p for p in PLATAFORMAS if p in vistas]


def actividad_de(user_id) -> Optional[dict]:
    """La actividad de la ficha (contrato 3): la de la fila de la lista —la MISMA consulta, filtrada por su id— más
    los números que solo enseña la ficha y el embudo. None si la cuenta no existe. LANZA si la base falla."""
    uid = _uuid(user_id)
    if not uid:
        return None
    filas = execute_sql_query(_sql_pagina("p.id = %s", "todas", "alta"), (uid, 1, 0), fetch_all=True) or []
    if not filas:
        return None
    r = filas[0]
    extras = execute_sql_query(_SQL_EXTRAS, (uid,) * _SQL_EXTRAS.count("%s"), fetch_one=True) or {}
    base = _actividad(r)
    dias = base["dias_activos_30d"]
    primer_plan = extras.get("primer_plan")
    return {
        **base,
        "comidas_por_dia_activo": round(base["comidas_30d"] / dias, 1) if dias else 0.0,
        "dias_con_agua_30d": _n(extras.get("dias_con_agua_30d")),
        "registros_peso": _n(extras.get("registros_peso")),
        "bloques_fallidos_30d": _n(extras.get("bloques_fallidos_30d")),
        "pulgares_abajo": _n(extras.get("pulgares_abajo")),
        "plataformas": _plataformas(extras),
        "avisos_abiertos": _n(extras.get("avisos_abiertos")),
        # El formulario: su primer envío (telemetría del wizard) o, si fue antes o no quedó registrado (lo rellenó
        # como invitado), el primer plan, que solo nace de un formulario enviado.
        "embudo": {"alta": rc.iso(r.get("alta")),
                   "formulario": rc.iso(_el_primero(extras.get("formulario_enviado"), primer_plan)),
                   "primer_plan": rc.iso(primer_plan), "primera_comida": rc.iso(extras.get("primera_comida")),
                   "primer_mensaje": rc.iso(extras.get("primer_mensaje")),
                   "primer_escaneo": rc.iso(extras.get("primer_escaneo"))},
        "modo": _modo(r.get("plan_mode")), "idioma": r.get("locale") or None, "pais": r.get("pais") or None,
    }


def _correos(ids) -> dict:
    """{uuid: correo} del personal que marcó. Best-effort: sin correo, el panel enseña la marca igual."""
    uids = sorted({u for u in (_uuid(x) for x in ids if x) if u})
    if not uids:
        return {}
    try:
        filas = execute_sql_query("SELECT id::text AS id, email FROM public.user_profiles WHERE id = ANY(%s::uuid[])",
                                  (uids,), fetch_all=True) or []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-831] correos del personal ilegibles ({e!r}): la marca sale sin ellos")
        return {}
    return {str(f.get("id")).lower(): f.get("email") for f in filas}


def prueba_de(user_id) -> Optional[dict]:
    """El bloque `prueba` de la ficha (contrato 3), o None sin marca viva. `marcada_por` es el CORREO del admin (su id
    no sale del servidor, como en la exportación); el historial trae todas las marcas, la viva primero, y dice cuándo
    salió la propia persona."""
    marca = cuentas_prueba.marca_viva(user_id)
    if not marca:
        return None
    historial = cuentas_prueba.historial(user_id)
    correos = _correos([marca.get("marcada_por"), *(h.get("marcada_por") for h in historial)])

    def correo(v):
        return correos.get(_uuid(v) or "")
    return {
        "estado": cuentas_prueba.estado_de(marca),
        "desde": rc.iso(marca.get("marcada_at")),
        "motivo": marca.get("motivo"),
        "marcada_por": correo(marca.get("marcada_por")),
        "aviso_visto_at": rc.iso(marca.get("aviso_visto_at")),
        "historial": [{"desde": rc.iso(h.get("marcada_at")), "hasta": rc.iso(h.get("quitada_at")),
                       "motivo": h.get("motivo"), "motivo_quitar": h.get("motivo_quitar"),
                       "quitada_por_la_persona": bool(h.get("quitada_por_la_persona")),
                       "marcada_por": correo(h.get("marcada_por"))} for h in historial],
    }


def ampliar_ficha(ficha: Optional[dict]) -> Optional[dict]:
    """La ficha del lote 774 con `actividad`, `ajustes`, `ajustes_dispositivo` y `prueba` (contrato 3), en un dict
    nuevo. Cada bloque que no se puede leer (una migración sin aplicar, la base que falla) sale vacío —`None`, o `{}`
    en `ajustes_dispositivo`— con su aviso en el log: la ficha carga igual."""
    if not ficha:
        return ficha
    uid = ficha.get("user_id")
    salida = dict(ficha)
    try:
        salida["actividad"] = actividad_de(uid)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-831] actividad ilegible para {str(uid)[:8]} ({e!r}): la ficha sale sin ella")
        salida["actividad"] = None
    try:
        ajustes = ajustes_cuenta.ajustes_de(uid)
        salida["ajustes"], salida["ajustes_dispositivo"] = ajustes["ajustes"], ajustes["ajustes_dispositivo"]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-831] ajustes ilegibles para {str(uid)[:8]} ({e!r}): la ficha sale sin ellos")
        salida["ajustes"], salida["ajustes_dispositivo"] = None, {}
    try:
        salida["prueba"] = prueba_de(uid)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-831] marca de prueba ilegible para {str(uid)[:8]} ({e!r})")
        salida["prueba"] = None
    return salida
