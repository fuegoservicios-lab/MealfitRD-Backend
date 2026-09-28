# backend/admin_metricas.py
"""[P1-PLAN-LOTE-576 · 2026-09-27] Métricas del panel de administración, capa 1 (spec §2): SOLO agregados, ya
redactados para pintar.

El frontend es un pintor genérico (spec §6): cada bloque trae su título y sus filas con la etiqueta y el valor ya
formateados, así que una métrica nueva es solo backend. Jamás sale un id, un correo, un nombre, texto de un mensaje ni
nombre de un plato. Un bloque que falla no tumba a los demás: sale como `error`.

[P1-PLAN-LOTE-637 · 2026-09-28] El dueño: «se ve feo y poco entendible». Lo que fallaba no era solo el aspecto:
(1) las cifras de uso contaban las cuentas de administración (el dueño probando se leía como un usuario activo);
(2) ningún número decía contra qué compararse; (3) «11 alertas abiertas» en grande eran 6 notas `info` y 5 avisos, y
«Bloques en cola» eran los CREADOS en el periodo, no los que esperan; (4) el Escáner llenaba media pantalla con
porcentajes de 2 fotos. Ahora: bloques con `seccion` (Resumen → Requiere atención → Usuarios → Producto → Costes →
Calidad), tipos nuevos `resumen` (cifra + cambio frente al periodo anterior + qué significa), `avisos` (alertas por
TIPO en español llano, nunca la clave con su id), `serie` (barras por día o semana) y `embudo` (qué hicieron las
cuentas nuevas). Las cifras de personas excluyen a los admin; el gasto no (el dinero sale igual).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from admin_acceso import admin_ids
from db import execute_sql_query

logger = logging.getLogger(__name__)

_VENTANA = "now() - make_interval(days => %s)"
# [P1-PLAN-LOTE-637] Los días del panel son los del reloj del dueño (RD): una barra «27 sep» que empezara a las 8 p. m.
# por UTC partiría la noche en dos.
_ZONA = "America/Santo_Domingo"
_MES = ("", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")

# Revisión final: `generation_status` vive dentro de plan_data, que `/restore-local` deja escribir al cliente — un correo
# o un mensaje acabaría como fila del panel. Solo se pintan los estados que escribe el backend; el resto, «otro».
_ESTADOS_PLAN = frozenset({
    "complete", "complete_partial", "partial", "partial_no_shopping", "active", "generating", "generating_next",
    "in_progress", "paused_by_user", "failed", "abandoned", "degraded_pending_engagement", "expired_pending_pantry",
    "sin estado",
})
# Las etiquetas de código (estado de la cola, función del gasto, tipo de alerta) las escribe el servidor, pero se
# pintan igual de acotadas: solo [a-z0-9_].
_ETIQUETA_DE_CODIGO = re.compile(r"^[a-z0-9_]{1,64}$")


# [P1-PLAN-LOTE-620 · 2026-09-27] Los códigos se pintan en español. Un código seguro sin traducción sale tal cual (una
# función nueva del gasto se ve igual, solo que sin nombre bonito); el inseguro ya llegó aquí como «otro».
_ETIQUETA_ESTADO_PLAN = {
    "complete": "Completos", "complete_partial": "Completos con huecos", "partial": "Parciales",
    "partial_no_shopping": "Parciales sin lista", "active": "Activos", "generating": "Generándose",
    "generating_next": "Generando el siguiente bloque", "in_progress": "En curso", "paused_by_user": "Pausados",
    "failed": "Fallidos", "abandoned": "Abandonados", "degraded_pending_engagement": "Degradados (esperan uso)",
    "expired_pending_pantry": "Caducados (esperan la Nevera)", "sin estado": "Sin estado", "otro": "Otro estado",
}
_ETIQUETA_COLA = {
    "pending": "Pendientes", "processing": "Procesándose", "completed": "Completados", "failed": "Fallidos",
    "cancelled": "Cancelados", "pending_user_action": "Esperan al usuario", "otro": "Otro",
}
_ETIQUETA_FUNCION = {
    "day_generator": "Generación de días", "vision_scan": "Escáner de fotos", "planner": "Planificador",
    "reviewer": "Revisor clínico", "chat_call_model": "Coach (chat)", "plan_display_i18n": "Traducción del plan",
    "self_critique": "Autocrítica", "self_critique_correction": "Corrección tras la autocrítica",
    "culinary_judge": "Juez culinario", "surgical_marker": "Corrección puntual", "compressor": "Resumen de la memoria",
    "fact_extractor_extract_facts": "Extractor de hechos",
    "fact_extractor_contradiction_merge": "Contradicciones de hechos",
    "fact_extractor_router": "Clasificador de hechos", "diary_plate_estimate": "Estimación de platos (diario)",
    "diary_freetext_estimate": "Estimación de texto libre (diario)",
    # [P1-PLAN-LOTE-637] los que salían con su código en la tabla de 90 días (medido en producción el 28-sep)
    "swap_meal": "Cambio de plato", "judge": "Juez de calidad", "scan_doubt_adjust": "Dudas de la foto",
    "tool_analyze_preferences": "Análisis de preferencias (coach)", "meta_learning": "Aprendizaje entre planes",
    "vision_scan_display_i18n": "Traducción del escáner", "otro": "Otro",
}
_ETIQUETA_TIER = {"gratis": "Gratis", "basic": "Basic", "plus": "Plus", "ultra": "Ultra", "otro": "Otro plan"}

# [P1-PLAN-LOTE-637] Qué significa cada tipo de alerta, en llano. La clave de una alerta lleva ids detrás de «:»
# (`plan_quality_degraded:<usuario>:<plan>`): se agrupa por el prefijo y el prefijo pasa por la lista blanca de código.
# Un tipo sin traducir sale con su código legible; nunca con la clave entera.
_ALERTA = {
    "country_beta_first_plan": ("Primer plan en un país beta",
                                "Alguien generó su primer plan fuera de RD. Conviene revisar que salió bien."),
    "temporal_gate_proactive": ("Bloque de plan aplazado",
                                "Un bloque esperó varias veces a que terminaran los días anteriores; se avisó al "
                                "usuario. Informativo."),
    "plan_quality_degraded": ("Plan entregado sin aprobar la revisión",
                              "La revisión clínica no lo aprobó del todo y se entregó igual para no dejar al usuario "
                              "sin plan."),
    "review_failed_delivered_rate_high": ("Muchos planes sin aprobar la revisión",
                                          "La proporción de planes entregados sin aprobar supera el umbral."),
    "dream_contradiction": ("Contradicción en la memoria del coach",
                            "Dos datos del mismo usuario se contradicen (p. ej., dos pesos distintos)."),
    "registry_dishes_unused": ("Platos del catálogo que no salen",
                               "Los planes casi no usan los platos del catálogo cuando deberían."),
    "llm_circuit_breaker_open": ("Un modelo de IA está fallando",
                                 "Varias llamadas seguidas fallaron y se pausó ese modelo unos segundos."),
    "llm_provider_balance_exhausted": ("Sin saldo en un proveedor de IA",
                                       "Hay que recargar la cuenta del proveedor o la IA dejará de responder."),
    "gemini_spend_cap_exceeded": ("Se pasó el tope de gasto de IA", "El gasto superó el límite configurado."),
    "chunks_stuck_processing": ("Bloques de plan atascados", "Llevan demasiado tiempo generándose."),
    "chunk_overdue": ("Bloques de plan atrasados", "Debían haber empezado y no lo hicieron."),
    "plan_chunk_zombie": ("Bloques de plan atascados", "Llevan demasiado tiempo generándose."),
    "dead_lettered_chunk": ("Bloque de plan abandonado", "Falló todos sus reintentos."),
    "dead_lettered_chunks_recent": ("Bloques de plan abandonados", "Fallaron todos sus reintentos."),
    "plan_persist_failed": ("No se pudo guardar un plan", "El plan se generó pero no se guardó."),
    "plan_data_corrupted": ("Plan con datos dañados", "Hay que revisarlo a mano."),
    "pipeline_crash_fallback": ("Plan de emergencia entregado", "La generación falló y se entregó el plan de respaldo."),
    "pipeline_emergency_fallback_p1_5": ("Plan de emergencia entregado",
                                         "La generación falló y se entregó el plan de respaldo."),
    "plan_fallback_rate_high": ("Muchos planes de respaldo", "Demasiados planes salen del camino de respaldo."),
    "degraded_rate_high": ("Muchos planes degradados", "Demasiados planes salen con calidad reducida."),
    "pipeline_metrics_silence": ("La telemetría dejó de llegar", "Hace rato que no se registra actividad."),
    "auth_failure_rate_high": ("Muchos fallos al iniciar sesión", "Puede ser un problema del login o un ataque."),
    "deploy_lag_drift_vs_expected": ("El servidor no corre la última versión",
                                     "El despliegue no se aplicó o se revirtió."),
    "deploy_lag_marker_stale": ("El servidor no corre la última versión", "El despliegue no se aplicó."),
    "billing_orphan_subscription_unrecoverable": ("Pago sin cuenta asociada",
                                                  "PayPal cobró y no se sabe a qué cuenta dar el plan."),
    "billing_cancel_failed": ("No se pudo cancelar una suscripción", "Riesgo de doble cobro."),
    "scheduler_error": ("Una tarea programada falló", "Un proceso automático dio error."),
    "scheduler_missed": ("Una tarea programada no corrió", "Un proceso automático se saltó su turno."),
    "otro": ("Otro aviso", "Un aviso con un tipo que no se pudo leer."),
}
# Familias cuyo código lleva el nombre de la tarea pegado detrás (`scheduler_error_<tarea>`): una fila por familia.
_FAMILIAS_ALERTA = ("scheduler_error", "scheduler_missed", "post_swap_critical_divergence")
_NIVEL_ALERTA = {"critical": "critico", "error": "critico", "high": "aviso", "warning": "aviso", "medium": "aviso"}
_ORDEN_NIVEL = {"critico": 0, "aviso": 1, "info": 2}
# Un bloque de la cola cuyo turno pasó hace más de esto no está programado: está atrasado.
_ATRASO = "2 hours"
# Por debajo de esto, un porcentaje del escáner es ruido: 1 de 2 fotos «fallidas» no dice nada.
_MUESTRA_ESCANER = 20
_GASTO_TOP = 8
_NOTA_LIBRE_PROHIBIDA = re.compile(r"\S*@\S*|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


@dataclass
class _Ctx:
    """El periodo pedido y las cuentas que NO cuentan como usuarios (las de administración)."""
    dias: int
    fuera: list = field(default_factory=list)


def _texto_de_nota(v) -> str:
    """Las notas del banco las escribe quien corre el banco, no un usuario; aun así, ni correos ni ids."""
    t = " ".join(_NOTA_LIBRE_PROHIBIDA.sub(" ", str(v or "")).split())
    if len(t) <= 60:
        return t
    corte = t[:60].rsplit(" ", 1)[0] if " " in t[:60] else t[:60]      # en una palabra entera, no a media
    return corte.rstrip(" ,;:") + "…"


def _estado_plan(v) -> str:
    v = str(v or "")
    return v if v in _ESTADOS_PLAN else "otro"


def _etiqueta_de_codigo(v) -> str:
    v = str(v or "")
    return v if _ETIQUETA_DE_CODIGO.match(v) else "otro"


def _tipo_alerta(clave) -> str:
    tipo = _etiqueta_de_codigo(str(clave or "").split(":", 1)[0])
    for familia in _FAMILIAS_ALERTA:
        if tipo.startswith(familia):
            return familia
    return tipo


def _agrupar(filas: list, clave: str, etiqueta, *sumas: str) -> list:
    """Suma las filas cuya etiqueta segura coincide (dos «otro» son UNA fila), de mayor a menor por la 1.ª suma."""
    grupos: dict = {}
    for r in filas:
        k = etiqueta(r.get(clave))
        g = grupos.setdefault(k, dict.fromkeys(sumas, 0))
        for campo in sumas:
            g[campo] += int(r.get(campo) or 0)
    return sorted(grupos.items(), key=lambda kv: -kv[1][sumas[0]])


def _entero(n) -> str:
    return f"{int(n or 0):,}"


def _pct(x) -> str:
    return "—" if x is None else f"{float(x) * 100:.0f} %"


def _usd(micros) -> str:
    return f"US${(micros or 0) / 1_000_000:,.2f}"


def _seg(ms) -> str:
    return "—" if ms is None else f"{float(ms) / 1000:.1f} s"


def _plural(n: int, uno: str, varios: str) -> str:
    return f"{n} {uno if n == 1 else varios}"


def _uno(sql: str, params: tuple = ()) -> dict:
    return execute_sql_query(sql, params, fetch_one=True) or {}


def _todos(sql: str, params: tuple = ()) -> list:
    return execute_sql_query(sql, params, fetch_all=True) or []


def _rango(col: str, dias: int, anterior: bool = False) -> tuple:
    """El periodo actual, o el de la misma duración justo antes (para el «frente a»)."""
    if anterior:
        return (f"{col} >= now() - make_interval(days => %s) AND {col} < now() - make_interval(days => %s)",
                (dias * 2, dias))
    return f"{col} >= now() - make_interval(days => %s)", (dias,)


def _kpis(bid: str, titulo: str, filas: list, nota: str | None = None) -> dict:
    """Cada fila: (etiqueta, valor) o (etiqueta, valor, opciones) con `destacado` (cifra grande), `nivel` (subfila) o
    `ayuda` (qué significa, en una línea)."""
    salida = []
    for f in filas:
        fila = {"etiqueta": f[0], "valor": f[1]}
        if len(f) > 2:
            fila.update(f[2])
        salida.append(fila)
    bloque = {"id": bid, "titulo": titulo, "tipo": "kpis", "filas": salida}
    if nota:
        bloque["nota"] = nota
    return bloque


_DESTACADO = {"destacado": True}
_SUBFILA = {"nivel": 1}


def _ids_fuera() -> list:
    """Las cuentas de administración: las del .env del panel y las de tier `admin`. Si la consulta falla, al menos las
    del .env (un panel que cuenta al dueño es peor que uno que no carga un bloque, pero no tanto como para no cargar)."""
    ids = set(admin_ids())
    try:
        ids |= {str(r.get("id")).lower() for r in _todos(
            "SELECT id::text AS id FROM public.user_profiles WHERE plan_tier = 'admin'") if r.get("id")}
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-637] no se pudieron leer las cuentas admin: {e!r}")
    return sorted(ids)


# ---------------------------------------------------------------------------------------------------------- consultas

def _actividad_sql(ctx: _Ctx, col_consumo: str = "consumed_at", anterior: bool = False) -> tuple:
    """(momento, usuario) de cada comida registrada y cada mensaje al coach, sin las cuentas admin. Un mensaje viejo
    sin `user_id` (antes del 15-sep) toma el de su sesión."""
    w1, p1 = _rango(f"c.{col_consumo}", ctx.dias, anterior)
    w2, p2 = _rango("m.created_at", ctx.dias, anterior)
    sql = (f"SELECT c.{col_consumo} AS t, c.user_id::text AS u FROM public.consumed_meals c "
           f"WHERE {w1} AND c.user_id::text <> ALL(%s::text[]) "
           "UNION ALL SELECT m.created_at, COALESCE(m.user_id, s.user_id)::text FROM public.agent_messages m "
           "LEFT JOIN public.agent_sessions s ON s.id = m.session_id "
           f"WHERE m.role = 'user' AND COALESCE(m.user_id, s.user_id) IS NOT NULL AND {w2} "
           "AND COALESCE(m.user_id, s.user_id)::text <> ALL(%s::text[])")
    return sql, (*p1, ctx.fuera, *p2, ctx.fuera)


def _activos(ctx: _Ctx, anterior: bool = False) -> int:
    sql, params = _actividad_sql(ctx, anterior=anterior)
    return int(_uno(f"SELECT COUNT(DISTINCT u) AS n FROM ({sql}) a", params).get("n") or 0)


def _cuentas_nuevas(ctx: _Ctx, anterior: bool = False) -> int:
    w, p = _rango("created_at", ctx.dias, anterior)
    return int(_uno(f"SELECT COUNT(*) AS n FROM public.user_profiles WHERE {w} AND id::text <> ALL(%s::text[])",
                    (*p, ctx.fuera)).get("n") or 0)


def _gasto_micros(ctx: _Ctx, anterior: bool = False) -> int:
    w, p = _rango("created_at", ctx.dias, anterior)
    return int(_uno(f"SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros FROM public.llm_usage_events WHERE {w}",
                    p).get("micros") or 0)


def _alertas_abiertas() -> list:
    """Alertas sin resolver agrupadas por tipo: [(tipo, nivel, n, primera)] de más grave a menos."""
    filas = _todos("SELECT alert_key, severity, triggered_at FROM public.system_alerts WHERE resolved_at IS NULL "
                   "ORDER BY triggered_at LIMIT 1000")
    grupos: dict = {}
    for r in filas:
        tipo = _tipo_alerta(r.get("alert_key"))
        nivel = _NIVEL_ALERTA.get(str(r.get("severity") or "").lower(), "info")
        g = grupos.setdefault(tipo, {"nivel": nivel, "n": 0, "primera": r.get("triggered_at")})
        g["n"] += 1
        if _ORDEN_NIVEL[nivel] < _ORDEN_NIVEL[g["nivel"]]:
            g["nivel"] = nivel
    return sorted(((t, g["nivel"], g["n"], g["primera"]) for t, g in grupos.items()),
                  key=lambda x: (_ORDEN_NIVEL[x[1]], -x[2]))


def _cola_ahora() -> dict:
    """La cola de generación en este momento (todas las cuentas: es la salud del sistema, no el uso)."""
    return _uno("SELECT COUNT(*) FILTER (WHERE status = 'pending' AND execute_after > now()) AS programados, "
                f"COUNT(*) FILTER (WHERE status = 'pending' AND execute_after <= now() "
                f"AND execute_after >= now() - interval '{_ATRASO}') AS listos, "
                "COUNT(*) FILTER (WHERE status = 'processing') AS en_curso, "
                f"COUNT(*) FILTER (WHERE status = 'pending' AND execute_after < now() - interval '{_ATRASO}') "
                "AS atrasados, COUNT(*) FILTER (WHERE status = 'pending_user_action') AS esperan_usuario "
                "FROM public.plan_chunk_queue WHERE status IN ('pending', 'processing', 'pending_user_action')")


# ------------------------------------------------------------------------------------------------------------ bloques

def _cambio(actual: int, antes: int, dias: int, fmt=_entero, sube_es_bueno: bool | None = True) -> tuple:
    """(texto, tono) del cambio frente al periodo anterior. `sube_es_bueno=None`: el cambio se dice, no se juzga."""
    diff = actual - antes
    if diff == 0:
        return f"Igual que los {dias} días anteriores", "neutro"
    texto = f"{'↑' if diff > 0 else '↓'} {fmt(abs(diff))} frente a los {dias} días anteriores ({fmt(antes)})"
    if sube_es_bueno is None:
        return texto, "neutro"
    return texto, "bueno" if (diff > 0) == sube_es_bueno else "malo"


def bloque_resumen(ctx: _Ctx) -> dict:
    d = ctx.dias
    activos, activos_antes = _activos(ctx), _activos(ctx, anterior=True)
    nuevas, nuevas_antes = _cuentas_nuevas(ctx), _cuentas_nuevas(ctx, anterior=True)
    gasto, gasto_antes = _gasto_micros(ctx), _gasto_micros(ctx, anterior=True)
    alertas = _alertas_abiertas()
    criticas = sum(n for _, nivel, n, _ in alertas if nivel == "critico")
    avisos = sum(n for _, nivel, n, _ in alertas if nivel == "aviso")
    notas = sum(n for _, nivel, n, _ in alertas if nivel == "info")
    atrasados = int(_cola_ahora().get("atrasados") or 0)
    avisos += atrasados
    if criticas:
        estado, tono = _plural(criticas, "crítico", "críticos"), "malo"
    elif avisos:
        estado, tono = _plural(avisos, "aviso", "avisos"), "aviso"
    else:
        estado, tono = "Todo bien", "bueno"
    detalle_estado = ("Nada que requiera acción." if not (criticas or avisos) else "Detalle en «Requiere atención».")
    if notas:
        detalle_estado += f" {_plural(notas, 'nota informativa', 'notas informativas')}."

    t_act, tono_act = _cambio(activos, activos_antes, d)
    t_nue, tono_nue = _cambio(nuevas, nuevas_antes, d)
    t_gas, _ = _cambio(gasto, gasto_antes, d, fmt=_usd, sube_es_bueno=None)
    por_usuario = f"{_usd(gasto // activos)} por usuario activo. " if activos else ""
    return {"id": "resumen", "seccion": "Resumen", "titulo": f"Últimos {d} días", "tipo": "resumen", "tarjetas": [
        {"etiqueta": "Usuarios activos", "valor": _entero(activos), "cambio": t_act, "tono": tono_act,
         "ayuda": "Personas que registraron una comida o escribieron al coach. Sin tus cuentas."},
        {"etiqueta": "Cuentas nuevas", "valor": _entero(nuevas), "cambio": t_nue, "tono": tono_nue,
         "ayuda": "Registros en el periodo. Sin tus cuentas."},
        {"etiqueta": "Gasto de IA", "valor": _usd(gasto), "cambio": t_gas, "tono": "neutro",
         "ayuda": f"{por_usuario}Incluye tus pruebas: el dinero sale igual."},
        {"etiqueta": "Estado del sistema", "valor": estado, "tono": tono, "ayuda": detalle_estado},
    ]}


def bloque_atencion(ctx: _Ctx) -> dict:
    items = []
    atrasados = int(_cola_ahora().get("atrasados") or 0)
    if atrasados:
        items.append({"nivel": "aviso", "titulo": "Bloques de plan atrasados", "valor": _entero(atrasados),
                      "detalle": "Debían empezar hace más de 2 horas y siguen esperando."})
    fallidos = int(_uno("SELECT COUNT(*) AS n FROM public.meal_plans WHERE plan_data->>'generation_status' = 'failed' "
                        f"AND created_at >= {_VENTANA} AND user_id::text <> ALL(%s::text[])",
                        (ctx.dias, ctx.fuera)).get("n") or 0)
    if fallidos:
        items.append({"nivel": "aviso", "titulo": "Planes fallidos", "valor": _entero(fallidos),
                      "detalle": f"Planes creados en {ctx.dias} días que terminaron en error."})
    for tipo, nivel, n, primera in _alertas_abiertas():
        titulo, que_es = _ALERTA.get(tipo) or (f"Aviso técnico: {tipo.replace('_', ' ')}", "Sin descripción todavía.")
        desde = f" Abierta desde el {primera:%d-%m}." if isinstance(primera, datetime) else ""
        items.append({"nivel": nivel, "titulo": titulo, "valor": _entero(n), "detalle": que_es + desde})
    items.sort(key=lambda i: _ORDEN_NIVEL[i["nivel"]])
    return {"id": "atencion", "seccion": "Requiere atención", "titulo": "Avisos abiertos", "tipo": "avisos",
            "items": items, "vacio": "Nada pendiente.",
            "nota": "Rojo: actuar ya. Ámbar: revisar. Gris: informativo, no requiere acción."}


def _etiqueta_dia(d, semanal: bool) -> str:
    return f"{'sem. ' if semanal else ''}{d.day} {_MES[d.month]}"


def _casillas(ctx: _Ctx) -> tuple:
    """Las casillas del eje: un día cada una hasta 31 días; con más, una semana (lunes)."""
    semanal = ctx.dias > 31
    hoy = datetime.now(ZoneInfo(_ZONA)).date()
    inicio = hoy - timedelta(days=ctx.dias - 1)
    if semanal:
        inicio -= timedelta(days=inicio.weekday())
        paso = 7
    else:
        paso = 1
    casillas = []
    d = inicio
    while d <= hoy:
        casillas.append(d)
        d += timedelta(days=paso)
    return semanal, casillas


def _serie(ctx: _Ctx, filas: list, fmt) -> list:
    semanal, casillas = _casillas(ctx)
    por_fecha = {}
    for r in filas:
        f = r.get("d")
        f = f.date() if isinstance(f, datetime) else f
        por_fecha[f] = por_fecha.get(f, 0) + (r.get("n") or 0)
    return [{"etiqueta": _etiqueta_dia(c, semanal), "valor": float(por_fecha.get(c, 0)), "texto": fmt(por_fecha.get(c, 0))}
            for c in casillas]


def bloque_activos_dia(ctx: _Ctx) -> dict:
    unidad = "week" if ctx.dias > 31 else "day"
    sql, params = _actividad_sql(ctx)
    filas = _todos(f"SELECT date_trunc(%s, t AT TIME ZONE %s)::date AS d, COUNT(DISTINCT u) AS n FROM ({sql}) a "
                   "GROUP BY 1", (unidad, _ZONA, *params))
    return {"id": "activos_dia", "seccion": "Usuarios",
            "titulo": "Usuarios activos por semana" if unidad == "week" else "Usuarios activos por día",
            "tipo": "serie", "puntos": _serie(ctx, filas, _entero),
            "nota": "Personas distintas que registraron una comida o escribieron al coach ese día. Sin tus cuentas."}


def bloque_embudo(ctx: _Ctx) -> dict:
    zona = _ZONA
    r = _uno(
        "SELECT COUNT(*) AS cuentas, "
        "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM public.meal_plans mp WHERE mp.user_id = p.id)) AS con_plan, "
        "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM public.consumed_meals cm WHERE cm.user_id = p.id)) AS con_comida, "
        "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM public.agent_messages am WHERE am.user_id = p.id "
        "AND am.role = 'user')) AS con_coach, "
        "COUNT(*) FILTER (WHERE (SELECT COUNT(DISTINCT x.dia) FROM ("
        " SELECT (cm.consumed_at AT TIME ZONE %s)::date AS dia FROM public.consumed_meals cm WHERE cm.user_id = p.id"
        " UNION SELECT (am.created_at AT TIME ZONE %s)::date FROM public.agent_messages am"
        " WHERE am.user_id = p.id AND am.role = 'user') x) >= 2) AS volvieron "
        f"FROM public.user_profiles p WHERE p.created_at >= {_VENTANA} AND p.id::text <> ALL(%s::text[])",
        (zona, zona, ctx.dias, ctx.fuera))
    total = int(r.get("cuentas") or 0)
    pasos = [("Se registraron", total), ("Generaron un plan", r.get("con_plan")),
             ("Registraron una comida", r.get("con_comida")), ("Escribieron al coach", r.get("con_coach")),
             ("Volvieron otro día", r.get("volvieron"))]
    return {"id": "embudo", "seccion": "Usuarios", "titulo": f"Qué hicieron las cuentas nuevas ({ctx.dias} días)",
            "tipo": "embudo",
            "pasos": [{"etiqueta": e, "valor": _entero(n), "pct": (int(n or 0) / total) if total else 0.0,
                       "texto": _pct(int(n or 0) / total) if total else "—"} for e, n in pasos],
            "nota": (f"Nadie se registró en estos {ctx.dias} días." if not total else
                     "Cada paso cuenta sobre las cuentas nuevas del periodo, hasta hoy. «Volvieron»: actividad en dos "
                     "o más días distintos.")}


def bloque_cuentas(ctx: _Ctx) -> dict:
    tiers = _todos("SELECT plan_tier AS tier, COUNT(*) AS n FROM public.user_profiles "
                   "WHERE id::text <> ALL(%s::text[]) GROUP BY 1", (ctx.fuera,))
    subs = _uno("SELECT COUNT(*) AS n FROM public.user_profiles WHERE subscription_status = 'active' "
                "AND id::text <> ALL(%s::text[])", (ctx.fuera,))
    comidas = _uno(f"SELECT COUNT(*) AS n FROM public.consumed_meals WHERE consumed_at >= {_VENTANA} "
                   "AND user_id::text <> ALL(%s::text[])", (ctx.dias, ctx.fuera))

    def tier(v):
        v = str(v or "")
        return v if v in _ETIQUETA_TIER else "otro"
    grupos = _agrupar(tiers, "tier", tier, "n")
    filas = [("Cuentas", _entero(sum(g["n"] for _, g in grupos)), {"ayuda": f"Sin contar {len(ctx.fuera)} de admin."})]
    filas += [(_ETIQUETA_TIER[k], _entero(g["n"]), _SUBFILA) for k, g in grupos]
    filas += [("Suscripciones de pago activas", _entero(subs.get("n"))),
              (f"Comidas registradas en {ctx.dias} días", _entero(comidas.get("n")))]
    return _kpis("cuentas", "Cuentas", filas) | {"seccion": "Usuarios"}


def bloque_coach(ctx: _Ctx) -> dict:
    c = _uno("SELECT COUNT(*) FILTER (WHERE m.role = 'user') AS preguntas, "
             "COUNT(*) FILTER (WHERE m.role = 'model') AS respuestas, "
             "COUNT(DISTINCT COALESCE(m.user_id, s.user_id)) FILTER (WHERE m.role = 'user') AS personas, "
             "COUNT(*) FILTER (WHERE m.feedback = 'up') AS up, COUNT(*) FILTER (WHERE m.feedback = 'down') AS down "
             "FROM public.agent_messages m LEFT JOIN public.agent_sessions s ON s.id = m.session_id "
             f"WHERE m.created_at >= {_VENTANA} "
             "AND COALESCE(COALESCE(m.user_id, s.user_id)::text, '') <> ALL(%s::text[])", (ctx.dias, ctx.fuera))
    up, down = int(c.get("up") or 0), int(c.get("down") or 0)
    filas = [("Mensajes de usuarios", _entero(c.get("preguntas")), _DESTACADO),
             ("Personas que lo usaron", _entero(c.get("personas")), _DESTACADO),
             ("Respuestas del coach", _entero(c.get("respuestas")))]
    if up + down:
        filas += [("Valoraciones", _entero(up + down), {"ayuda": "El usuario puede marcar 👍 o 👎 en cada respuesta."}),
                  ("👍", _entero(up), _SUBFILA), ("👎", _entero(down), _SUBFILA),
                  ("Respuestas con 👎", _pct(down / (up + down)))]
    else:
        filas.append(("Valoraciones", "Ninguna todavía",
                      {"ayuda": "El usuario puede marcar 👍 o 👎 en cada respuesta; nadie lo ha hecho en el periodo."}))
    return _kpis("coach", "Coach", filas, "Sin tus cuentas.") | {"seccion": "Producto"}


def bloque_planes(ctx: _Ctx) -> dict:
    planes = _todos("SELECT COALESCE(plan_data->>'generation_status', 'sin estado') AS estado, COUNT(*) AS n "
                    f"FROM public.meal_plans WHERE created_at >= {_VENTANA} AND user_id::text <> ALL(%s::text[]) "
                    "GROUP BY 1 ORDER BY 2 DESC", (ctx.dias, ctx.fuera))
    cola = _cola_ahora()
    total = sum(int(r.get("n") or 0) for r in planes)
    filas = [("Planes creados", _entero(total), _DESTACADO)]
    if total:
        filas.append(("Estado de esos planes", ""))
        filas += [(_ETIQUETA_ESTADO_PLAN.get(k, k), _entero(g["n"]), _SUBFILA)
                  for k, g in _agrupar(planes, "estado", _estado_plan, "n")]
    en_cola = sum(int(cola.get(k) or 0) for k in ("programados", "listos", "en_curso", "atrasados", "esperan_usuario"))
    filas.append(("Bloques en la cola ahora", _entero(en_cola),
                  {"ayuda": "Cada plan largo se genera por bloques de días; esto es lo que espera turno ahora mismo."}))
    filas += [("Programados para más adelante", _entero(cola.get("programados")), _SUBFILA),
              ("Listos para generarse", _entero(cola.get("listos")), _SUBFILA),
              ("Generándose", _entero(cola.get("en_curso")), _SUBFILA),
              ("Esperan al usuario", _entero(cola.get("esperan_usuario")), _SUBFILA),
              ("Atrasados (más de 2 h)", _entero(cola.get("atrasados")), _SUBFILA)]
    return _kpis("planes", "Planes", filas, "Planes: sin tus cuentas. Cola: todas las cuentas.") | {"seccion": "Producto"}


def bloque_escaner(ctx: _Ctx) -> dict:
    dias = ctx.dias
    # Revisión final: «fallido» es solo `error` (no_comida y una compra sin totales son respuestas legítimas), y solo
    # las fotos del ESCÁNER (`purpose='diary'`): las del chat no son este bloque.
    v = _uno("SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE metadata->>'resultado' = 'error') AS fallidos, "
             "COUNT(*) FILTER (WHERE metadata->>'resultado' = 'no_comida') AS no_comida, "
             "COUNT(*) FILTER (WHERE metadata->>'resultado' = 'sin_totales') AS sin_totales, "
             "percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50, "
             "percentile_cont(0.9) WITHIN GROUP (ORDER BY duration_ms) AS p90 "
             "FROM public.pipeline_metrics WHERE node = 'vision_scan_resultado' AND metadata->>'purpose' = 'diary' "
             f"AND created_at >= {_VENTANA} AND COALESCE(user_id, '') <> ALL(%s::text[])", (dias, ctx.fuera))
    s = _uno("SELECT COUNT(*) AS n, "
             "COUNT(*) FILTER (WHERE (metadata->>'corregido')::boolean) AS corregidos, "
             "COUNT(*) FILTER (WHERE (metadata->>'cambiados')::int > 0) AS cambiar, "
             "COUNT(*) FILTER (WHERE (metadata->>'redescrito')::boolean) AS describelo, "
             "COUNT(*) FILTER (WHERE (metadata->>'cantidades_editadas')::int > 0) AS cantidades, "
             "COUNT(*) FILTER (WHERE (metadata->>'dudas_cambiadas')::int > 0) AS dudas, "
             "COUNT(*) FILTER (WHERE (metadata->>'macros_tecleadas')::boolean) AS macros, "
             "percentile_cont(0.5) WITHIN GROUP (ORDER BY (metadata->>'desvio_kcal')::float) AS desvio "
             f"FROM public.pipeline_metrics WHERE node = 'scan_outcome' AND created_at >= {_VENTANA} "
             "AND COALESCE(user_id, '') <> ALL(%s::text[])", (dias, ctx.fuera))
    n_v, n_s = int(v.get("n") or 0), int(s.get("n") or 0)
    # [P1-PLAN-LOTE-620] la señal nació el 27-sep: sin esta nota, «1 foto en 30 días» se lee como que nadie escanea
    primero = _uno("SELECT MIN(created_at) AS desde FROM public.pipeline_metrics WHERE node = 'vision_scan_resultado'")
    desde = primero.get("desde")
    notas = ["Sin tus cuentas."]
    if desde is not None and desde > datetime.now(timezone.utc) - timedelta(days=dias):
        notas.append(f"Se registra desde el {desde:%d-%m-%Y}.")

    # [P1-PLAN-LOTE-637] con pocas fotos, los porcentajes son ruido: se dicen los conteos y el tiempo, y se avisa
    if n_v < _MUESTRA_ESCANER:
        notas.append(f"Muestra pequeña: los porcentajes aparecen desde {_MUESTRA_ESCANER} fotos.")
        return _kpis("escaner", "Escáner de comidas", [
            ("Fotos analizadas", _entero(n_v), _DESTACADO),
            ("Platos registrados con el escáner", _entero(n_s), _DESTACADO),
            ("Tiempo de análisis (mediana)", _seg(v.get("p50"))),
        ], " ".join(notas)) | {"seccion": "Producto"}

    def frac(k):
        return int(s.get(k) or 0) / n_s if n_s else None

    # [P1-PLAN-LOTE-620] Una fila destacada se pinta aparte, arriba: si «Corregidos» lo fuera, sus cinco razones
    # quedarían colgando de la fila de encima. Las subfilas siguen siempre a su total, que es una fila normal.
    return _kpis("escaner", "Escáner de comidas", [
        ("Fotos analizadas", _entero(n_v), _DESTACADO),
        ("Platos registrados con el escáner", _entero(n_s), _DESTACADO),
        ("Análisis fallidos", _pct(int(v.get("fallidos") or 0) / n_v if n_v else None),
         {"ayuda": "La IA no pudo leer la foto."}),
        ("No era comida", _pct(int(v.get("no_comida") or 0) / n_v if n_v else None)),
        ("Sin totales (compra o etiqueta)", _pct(int(v.get("sin_totales") or 0) / n_v if n_v else None)),
        ("Tiempo de análisis (mediana / p90)", f"{_seg(v.get('p50'))} / {_seg(v.get('p90'))}",
         {"ayuda": "p90: 9 de cada 10 análisis tardan menos que esto."}),
        ("Corregidos por el usuario", _pct(frac("corregidos")),
         {"ayuda": "Platos que el usuario tocó antes de guardarlos; menos es mejor."}),
        ("Cambió un ingrediente", _pct(frac("cambiar")), _SUBFILA),
        ("«Descríbelo»", _pct(frac("describelo")), _SUBFILA),
        ("Editó cantidades", _pct(frac("cantidades")), _SUBFILA),
        ("Cambió la respuesta a una duda", _pct(frac("dudas")), _SUBFILA),
        ("Tecleó las macros", _pct(frac("macros")), _SUBFILA),
        ("Desvío mediano de calorías (IA → registrado)", _pct(s.get("desvio")),
         {"ayuda": "Cuánto cambió el usuario las calorías que propuso la IA."}),
    ], " ".join(notas)) | {"ancho": True, "seccion": "Producto"}


def bloque_gasto_dia(ctx: _Ctx) -> dict:
    unidad = "week" if ctx.dias > 31 else "day"
    filas = _todos("SELECT date_trunc(%s, created_at AT TIME ZONE %s)::date AS d, "
                   "COALESCE(SUM(cost_usd_micros), 0) AS n FROM public.llm_usage_events "
                   f"WHERE created_at >= {_VENTANA} GROUP BY 1", (unidad, _ZONA, ctx.dias))
    return {"id": "gasto_dia", "seccion": "Costes",
            "titulo": "Gasto de IA por semana" if unidad == "week" else "Gasto de IA por día", "tipo": "serie",
            "puntos": _serie(ctx, filas, _usd), "nota": "Incluye tus pruebas."}


def bloque_gasto(ctx: _Ctx) -> dict:
    dias = ctx.dias
    por_funcion = _todos("SELECT COALESCE(node, 'sin atribuir') AS funcion, COUNT(*) AS llamadas, "
                         "COALESCE(SUM(cost_usd_micros), 0) AS micros FROM public.llm_usage_events "
                         f"WHERE created_at >= {_VENTANA} GROUP BY 1 ORDER BY 3 DESC LIMIT 200", (dias,))
    total = _uno("SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros, COUNT(*) AS n FROM public.llm_usage_events "
                 f"WHERE created_at >= {_VENTANA}", (dias,))
    grupos = _agrupar(por_funcion, "funcion", _etiqueta_de_codigo, "micros", "llamadas")
    # [P1-PLAN-LOTE-620] las 8 que más gastan con nombre legible; el resto en UNA fila, y la proporción para las barras
    filas = [(_ETIQUETA_FUNCION.get(k, k), g["llamadas"], g["micros"]) for k, g in grupos[:_GASTO_TOP]]
    resto = grupos[_GASTO_TOP:]
    if resto:
        filas.append((f"Otras ({len(resto)} funciones)", sum(g["llamadas"] for _, g in resto),
                      sum(g["micros"] for _, g in resto)))
    total_micros = int(total.get("micros") or 0)
    return {"id": "gasto", "seccion": "Costes", "titulo": f"Gasto por función ({dias} días): {_usd(total_micros)}",
            "tipo": "tabla", "columnas": ["Función", "Llamadas", "Coste"],
            "filas": [[nombre, _entero(llamadas), _usd(micros)] for nombre, llamadas, micros in filas],
            "barras": [round(micros / total_micros, 4) if total_micros else 0.0 for _, _, micros in filas]}


def bloque_analizador(_ctx: _Ctx) -> dict:
    corridas = _todos("SELECT ran_at, model, n, ok, notes, metrics FROM public.analyzer_benchmark_runs "
                      "ORDER BY ran_at DESC LIMIT 10")
    filas = []
    for r in corridas:
        m = r.get("metrics") or {}
        # [P1-PLAN-LOTE-620] qué se midió (la nota de la corrida o, sin ella, el modelo) y si la corrida no vale
        corrida = _texto_de_nota(r.get("notes")) or _texto_de_nota(r.get("model")) or "—"
        if not m.get("valida"):
            corrida += " (no válida)"
        filas.append([r["ran_at"].strftime("%Y-%m-%d %H:%M"), corrida,
                      _pct((m.get("kcal") or {}).get("mediana")), _pct((m.get("proteina_g") or {}).get("mediana")),
                      _pct(m.get("recall_componentes"))])
    return {"id": "analizador", "seccion": "Calidad del escáner", "titulo": "Banco del analizador", "tipo": "tabla",
            "columnas": ["Fecha (UTC)", "Corrida", "kcal", "Proteína", "Componentes"], "filas": filas,
            "nota": "kcal y Proteína: error mediano frente a lo pesado (Nutrition5k); menos es mejor. Componentes: "
                    "los que reconoce. Una corrida sola varía ±2-4 puntos: se decide con la regla pareada."}


_BLOQUES = (
    ("resumen", "Resumen", "Resumen", bloque_resumen),
    ("atencion", "Requiere atención", "Avisos abiertos", bloque_atencion),
    ("activos_dia", "Usuarios", "Usuarios activos por día", bloque_activos_dia),
    ("embudo", "Usuarios", "Qué hicieron las cuentas nuevas", bloque_embudo),
    ("cuentas", "Usuarios", "Cuentas", bloque_cuentas),
    ("coach", "Producto", "Coach", bloque_coach),
    ("planes", "Producto", "Planes", bloque_planes),
    ("escaner", "Producto", "Escáner de comidas", bloque_escaner),
    ("gasto_dia", "Costes", "Gasto de IA por día", bloque_gasto_dia),
    ("gasto", "Costes", "Gasto por función", bloque_gasto),
    ("analizador", "Calidad del escáner", "Banco del analizador", bloque_analizador),
)


def metricas(dias: int) -> dict:
    ctx = _Ctx(dias=max(1, min(90, int(dias))), fuera=_ids_fuera())
    bloques = []
    for bid, seccion, titulo, fn in _BLOQUES:
        try:
            bloques.append(fn(ctx))
        except Exception as e:
            logger.warning(f"⚠️ [P1-PLAN-LOTE-576] bloque {bid} no disponible: {e!r}")
            bloques.append({"id": bid, "seccion": seccion, "titulo": titulo, "tipo": "error", "error": "No disponible"})
    return {"dias": ctx.dias, "generado": datetime.now(timezone.utc).isoformat(), "bloques": bloques}
