# backend/admin_avisos.py
"""[P1-PLAN-LOTE-993 · 2026-11-12] «Requiere atención» del panel: qué es un problema DE VERDAD, desde cuándo, y qué ya
se resolvió.

El dueño, ante 9 filas ámbar y rojas: «no creo que haya tantos errores, ¿o sí?». No los había. Medido en producción el
12-nov: los 2 «críticos» (bloques abandonados) y el «plan stranded partial» eran del MISMO plan, de una cuenta que se
registró el 1-oct y no volvió; los 4 bloques «atrasados» eran de 3 cuentas sin actividad desde septiembre u octubre
(una, de admin); 4 de los 7 «planes sin aprobar la revisión» eran de esas cuentas; y los «abandonados» no fallaron
nada — se descartaron por caer después del final del plan (`execute_after_beyond_plan_window`, 0 intentos), aunque el
panel decía «Falló todos sus reintentos». Nadie resuelve esas alertas (modelo Manual), así que se acumulaban para
siempre con la misma alarma.

Ahora cada aviso se lee contra el estado ACTUAL antes de pintarse:
  - `activo`    — afecta a alguien que usa la app (señal en los últimos `_INACTIVO_DIAS` días) o al sistema entero.
  - `revision`  — hay que mirarlo, pero no rompe nada ahora (un plan entregado sin aprobar, una contradicción).
  - `espera`    — es de una cuenta inactiva: no hace falta actuar; se retoma si vuelve.
  - `historial` — eventos informativos, lo que ya no aplica (el plan se borró o fue reemplazado; el bloque se
                  descartó por diseño) y lo RESUELTO en los últimos 7 días, con su fecha.
Cada fila dice sus fechas en llano («Desde el 27 oct · hace 16 días») y qué hacer. «Marcar como revisado» cierra las
alertas de una fila (`resolved_at`): si la condición sigue, su productor la reabre en su próxima pasada — el botón no
puede esconder un problema vivo.

Nunca sale un id: las alertas viajan como un resumen de 16 hex de su clave (`_huella`), que solo el endpoint de
«revisado» sabe volver a casar con su fila.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_INACTIVO_DIAS = 14
_RESUELTOS_DIAS = 7
_ORDEN_GRUPO = {"activo": 0, "revision": 1, "espera": 2, "historial": 3}
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_HUELLA = re.compile(r"^[0-9a-f]{16}$")

# Condiciones del SISTEMA: no dependen de un usuario, así que no «esperan» a nadie.
_TIPOS_SISTEMA = frozenset({
    "llm_circuit_breaker_open", "llm_provider_balance_exhausted", "gemini_spend_cap_exceeded",
    "pipeline_metrics_silence", "auth_failure_rate_high", "deploy_lag_drift_vs_expected", "deploy_lag_marker_stale",
    "scheduler_error", "scheduler_missed", "plan_fallback_rate_high", "degraded_rate_high",
    "billing_orphan_subscription_unrecoverable", "billing_cancel_failed", "dead_lettered_chunks_recent",
    "chunks_stuck_processing",
})
# Avisos sobre UN plan: si ese plan ya no es el actual de su dueño (lo borró o generó otro), ya no aplican.
_TIPOS_DE_PLAN = frozenset({
    "plan_quality_degraded", "dead_lettered_chunk", "plan_stranded_partial", "chunk_overdue", "plan_chunk_zombie",
    "plan_data_corrupted", "plan_persist_failed", "pipeline_crash_fallback", "pipeline_emergency_fallback_p1_5",
})
# Razones de descarte que NO son un fallo: el bloque caía fuera del plan y se tiró a propósito.
_DESCARTE_POR_DISENO = frozenset({"execute_after_beyond_plan_window"})

_ACCION = {
    "plan_quality_degraded": "Ábrelo en «Cuentas» y comprueba que los platos encajan con su perfil.",
    "dream_contradiction": "Mira en «Cuentas» cuál de los dos datos es el bueno; si toca una alergia, corrígelo.",
    "review_login_used": "Si no fue Apple revisando la app, cambia el código de App Review (MEALFIT_REVIEW_LOGIN_CODE).",
    "review_failed_delivered_rate_high": "Mira qué rechazó la revisión clínica en los planes recientes.",
    "registry_dishes_unused": "Revisa por qué el generador casi no elige platos del catálogo.",
    "dead_lettered_chunk": "Regenera el plan de esa cuenta o pídele que lo renueve.",
    "plan_stranded_partial": "Regenera el plan de esa cuenta o pídele que lo renueve.",
    "country_beta_first_plan": "Revisa que ese primer plan salió bien.",
    "llm_provider_balance_exhausted": "Recarga la cuenta del proveedor.",
}
# Títulos y textos que faltaban en `admin_metricas._ALERTA` (salían como «Aviso técnico: …» con su código).
_TEXTOS = {
    "review_login_used": ("Se usó el acceso de App Review",
                          "Alguien entró en la cuenta de demostración con el código fijo para Apple."),
    "plan_stranded_partial": ("Plan a medias sin días", "Un plan se quedó sin ningún día generado."),
    "chunk_pantry_snapshots_stale": ("Nevera desactualizada en la cola",
                                     "Bloques en cola con una foto vieja de la Nevera."),
    "chunk_overdue": ("Bloques de plan atrasados", "Debían haber empezado y no lo hicieron."),
}
# El descarte por diseño tiene su propio título: «abandonado / falló todos sus reintentos» era falso (0 intentos).
_TEXTO_DESCARTE = ("Bloques descartados al final del plan",
                   "Se programaron para después del final del plan y se descartaron a propósito: no fallaron.")
# Avisos de seguridad o de producto que NO esperan a nadie aunque lleven el id de una cuenta.
_TIPOS_SIN_ESPERA = frozenset({"review_login_used"})


def _huella(clave: str) -> str:
    return hashlib.sha256(str(clave).encode("utf-8")).hexdigest()[:16]


def _local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    import admin_metricas as am
    return dt.astimezone(ZoneInfo(am._ZONA))


def fecha(dt, ahora: datetime | None = None) -> str:
    """«12 nov» (con el año si no es este); hoy y ayer con su hora, que es lo que importa de algo reciente."""
    import admin_metricas as am
    ahora = _local(ahora or datetime.now(timezone.utc))
    d = _local(dt)
    hora = f"{d.hour % 12 or 12}:{d.minute:02d} {'a. m.' if d.hour < 12 else 'p. m.'}"
    if d.date() == ahora.date():
        return f"hoy, {hora}"
    if d.date() == ahora.date() - timedelta(days=1):
        return f"ayer, {hora}"
    texto = f"{d.day} {am._MES[d.month]}"
    return texto if d.year == ahora.year else f"{texto} {d.year}"


def hace(dt, ahora: datetime | None = None) -> str:
    ahora = _local(ahora or datetime.now(timezone.utc))
    dias = (ahora.date() - _local(dt).date()).days
    if dias <= 0:
        return "hoy"
    if dias == 1:
        return "ayer"
    return f"hace {dias} días"


def _momento(primera, ultima, ahora=None) -> str | None:
    """«Detectado …»: `triggered_at` es la ÚLTIMA vez que el productor lo vio (los que se reemiten la mueven), así que
    «desde» mentía en esos."""
    if not isinstance(primera, datetime):
        return None
    ultima = ultima if isinstance(ultima, datetime) else primera
    if _local(primera).date() == _local(ultima).date():
        f = fecha(ultima, ahora)
        return f"Detectado {f}" if f.startswith(("hoy", "ayer")) else f"Detectado el {f} · {hace(ultima, ahora)}"
    return f"Detectados del {fecha(primera, ahora)} al {fecha(ultima, ahora)} · el último {hace(ultima, ahora)}"


def _rango_fechas(desde, hasta) -> str:
    if _local(desde).date() == _local(hasta).date():
        return f"el {fecha(desde)}"
    return f"entre el {fecha(desde)} y el {fecha(hasta)}"


def _uuid(v) -> str | None:
    v = str(v or "").strip().lower()
    return v if _UUID.match(v) else None


def _ids_de(r: dict) -> tuple:
    """(usuario, plan) de una alerta: de su metadata o, si falta, de los segmentos de su clave."""
    partes = str(r.get("alert_key") or "").split(":")
    uid = _uuid(r.get("uid")) or next((_uuid(p) for p in partes[1:2] if _uuid(p)), None)
    pid = _uuid(r.get("pid"))
    return uid, pid


def _ultima_senal(uids: list) -> dict:
    """Último momento en que cada cuenta dio señal: registró comida, escribió al coach o se creó la cuenta."""
    import admin_metricas as am
    if not uids:
        return {}
    filas = am._todos(
        "SELECT x.u, MAX(x.t) AS ultima FROM ("
        "SELECT user_id::text AS u, consumed_at AS t FROM public.consumed_meals WHERE user_id::text = ANY(%s::text[]) "
        "UNION ALL SELECT COALESCE(m.user_id, s.user_id)::text, m.created_at FROM public.agent_messages m "
        "LEFT JOIN public.agent_sessions s ON s.id = m.session_id "
        "WHERE m.role = 'user' AND COALESCE(m.user_id, s.user_id)::text = ANY(%s::text[]) "
        "UNION ALL SELECT id::text, created_at FROM public.user_profiles WHERE id::text = ANY(%s::text[])"
        ") x GROUP BY 1", (uids, uids, uids))
    if not isinstance(filas, list):
        return {}
    return {str(r.get("u")): r.get("ultima") for r in filas if isinstance(r, dict) and r.get("u")}


def _planes(pids: list) -> dict:
    """De cada plan: si existe y si es el más reciente de su dueño."""
    import admin_metricas as am
    if not pids:
        return {}
    filas = am._todos(
        "SELECT mp.id::text AS id, (mp.created_at = (SELECT MAX(m2.created_at) FROM public.meal_plans m2 "
        "WHERE m2.user_id = mp.user_id)) AS actual FROM public.meal_plans mp WHERE mp.id::text = ANY(%s::text[])",
        (pids,))
    if not isinstance(filas, list):
        return {}
    return {str(r.get("id")): bool(r.get("actual")) for r in filas if isinstance(r, dict) and r.get("id")}


def _inactiva(uid, senales: dict, ahora: datetime) -> datetime | None:
    """La fecha de la última señal si la cuenta lleva más de `_INACTIVO_DIAS` sin darla; si no, None."""
    if not uid or uid not in senales:
        return None
    ultima = senales.get(uid)
    if not isinstance(ultima, datetime):
        return None
    if ultima.tzinfo is None:
        ultima = ultima.replace(tzinfo=timezone.utc)
    return ultima if ultima < ahora - timedelta(days=_INACTIVO_DIAS) else None


def _clasificar(tipo: str, nivel: str, razon, plan_actual, inactiva_desde, cuando=None, ahora=None) -> tuple:
    """(grupo, nivel, estado, variante) de UNA alerta abierta. `variante` cambia el texto de la fila."""
    if tipo in _TIPOS_DE_PLAN and plan_actual is False:
        return "historial", "info", "obsoleto", None
    if str(razon or "") in _DESCARTE_POR_DISENO:
        return "historial", "info", "abierto", "descarte"
    if tipo in _TIPOS_SIN_ESPERA:
        viejo = isinstance(cuando, datetime) and cuando < (ahora or datetime.now(timezone.utc)) - timedelta(
            days=_INACTIVO_DIAS)
        return ("historial", "info", "abierto", None) if viejo else ("revision", "aviso", "abierto", None)
    if tipo not in _TIPOS_SISTEMA and inactiva_desde is not None:
        return "espera", "info", "abierto", None
    if nivel == "info":
        return "historial", "info", "abierto", None
    if nivel == "critico" or tipo in _TIPOS_SISTEMA:
        return "activo", nivel, "abierto", None
    return "revision", nivel, "abierto", None


def _textos(tipo: str, variante=None) -> tuple:
    import admin_metricas as am
    if variante == "descarte":
        return _TEXTO_DESCARTE
    if tipo in _TEXTOS:
        return _TEXTOS[tipo]
    return am._ALERTA.get(tipo) or (f"Aviso técnico: {tipo.replace('_', ' ')}", "Sin descripción todavía.")


def items(ctx, ahora: datetime | None = None) -> list:
    """Las filas de «Requiere atención», con su conteo interno `n`, ya agrupadas y en orden (grupo, gravedad)."""
    import admin_metricas as am
    ahora = ahora or datetime.now(timezone.utc)
    fuera = {str(x).lower() for x in (ctx.fuera or [])}
    abiertas = am._todos(
        "SELECT alert_key, severity, triggered_at, metadata->>'user_id' AS uid, metadata->>'plan_id' AS pid, "
        "metadata->>'escalation_reason' AS razon FROM public.system_alerts WHERE resolved_at IS NULL "
        "ORDER BY triggered_at LIMIT 1000")
    resueltas = am._todos(
        "SELECT alert_key, severity, triggered_at, resolved_at FROM public.system_alerts "
        "WHERE resolved_at >= now() - make_interval(days => %s) ORDER BY resolved_at DESC LIMIT 300",
        (_RESUELTOS_DIAS,))
    abiertas = [r for r in (abiertas if isinstance(abiertas, list) else []) if isinstance(r, dict)]
    resueltas = [r for r in (resueltas if isinstance(resueltas, list) else [])
                 if isinstance(r, dict) and isinstance(r.get("resolved_at"), datetime)]

    # La cola: los bloques atrasados, partidos por si su cuenta sigue viva.
    cola = am._cola_ahora()
    atrasados = int(cola.get("atrasados") or 0) if isinstance(cola, dict) else 0
    por_cuenta = []
    if atrasados:
        filas = am._todos(f"SELECT user_id::text AS uid, COUNT(*) AS n FROM public.plan_chunk_queue "
                          f"WHERE status = 'pending' AND execute_after < now() - interval '{am._ATRASO}' GROUP BY 1")
        por_cuenta = [r for r in (filas if isinstance(filas, list) else []) if isinstance(r, dict)]

    ids = [_ids_de(r) for r in abiertas]
    uids = sorted({u for u, _ in ids if u} | {u for u in (_uuid(r.get("uid")) for r in por_cuenta) if u})
    senales = _ultima_senal(uids)
    planes = _planes(sorted({p for _, p in ids if p}))

    filas: dict = {}

    def sumar(clave, **f):
        g = filas.setdefault(clave, {**f, "n": 0, "primera": None, "ultima": None, "huellas": [], "tuyas": 0,
                                     "nivel": f["nivel"]})
        return g

    for r, (uid, pid) in zip(abiertas, ids):
        tipo = am._tipo_alerta(r.get("alert_key"))
        nivel = am._NIVEL_ALERTA.get(str(r.get("severity") or "").lower(), "info")
        actual = planes.get(pid) if pid else None
        if pid and pid not in planes:
            actual = False                                       # el plan no existe: lo borraron
        inactiva = None if (uid and uid in fuera) else _inactiva(uid, senales, ahora)
        grupo, nivel, estado, variante = _clasificar(tipo, nivel, r.get("razon"), actual, inactiva,
                                                     r.get("triggered_at"), ahora)
        g = sumar((grupo, tipo, estado, variante), grupo=grupo, tipo=tipo, estado=estado, variante=variante,
                  nivel=nivel)
        g["n"] += 1
        if inactiva is not None:
            g.setdefault("inactivas", []).append(inactiva)
        if am._ORDEN_NIVEL[nivel] < am._ORDEN_NIVEL[g["nivel"]]:
            g["nivel"] = nivel
        t = r.get("triggered_at")
        if isinstance(t, datetime):
            g["primera"] = t if g["primera"] is None or t < g["primera"] else g["primera"]
            g["ultima"] = t if g["ultima"] is None or t > g["ultima"] else g["ultima"]
        g["huellas"].append(_huella(r.get("alert_key")))
        if uid and uid in fuera:
            g["tuyas"] += 1

    salida = []
    for g in filas.values():
        titulo, detalle = _textos(g["tipo"], g["variante"])
        if g["estado"] == "obsoleto":
            detalle += " Ya no aplica: el plan se borró o fue reemplazado por otro."
        if g["grupo"] == "espera" and g.get("inactivas"):
            uno = g["n"] == 1
            detalle += (f" No hace falta actuar: {'su' if uno else 'la'} última actividad "
                        f"{'fue' if uno else 'de esas cuentas fue'} "
                        f"{_rango_fechas(min(g['inactivas']), max(g['inactivas']))}; se retoma si "
                        f"{'vuelve' if uno else 'vuelven'}.")
        if g["tuyas"] == g["n"]:
            detalle += " Es de una de tus cuentas." if g["n"] == 1 else " Son de tus cuentas."
        elif g["tuyas"]:
            detalle += f" {g['tuyas']} de {g['n']} {'es' if g['tuyas'] == 1 else 'son'} de tus cuentas."
        item = {"grupo": g["grupo"], "nivel": g["nivel"], "titulo": titulo, "n": g["n"], "detalle": detalle,
                "momento": _momento(g["primera"], g["ultima"], ahora), "resolver": g["huellas"]}
        if g["estado"] == "obsoleto":
            item["estado"] = "obsoleto"
        accion = _ACCION.get(g["tipo"])
        if accion and g["grupo"] in ("activo", "revision"):
            item["accion"] = accion
        salida.append(item)

    # Bloques atrasados: los de cuentas vivas son un problema; los de cuentas inactivas, esperan.
    if atrasados:
        en_espera, desde = 0, None
        for r in por_cuenta:
            uid = _uuid(r.get("uid"))
            ina = None if (uid and uid in fuera) else _inactiva(uid, senales, ahora)
            if ina is not None:
                en_espera += int(r.get("n") or 0)
                desde = ina if desde is None or ina > desde else desde
        en_espera = min(en_espera, atrasados)
        vivos = atrasados - en_espera
        if vivos:
            salida.append({"grupo": "activo", "nivel": "aviso", "titulo": "Bloques de plan atrasados", "n": vivos,
                           "detalle": "Debían empezar hace más de 2 horas y siguen esperando, en cuentas que se usan.",
                           "accion": "Revisa el worker de la cola: debería haberlos generado."})
        if en_espera:
            salida.append({"grupo": "espera", "nivel": "info", "titulo": "Bloques de plan en pausa", "n": en_espera,
                           "detalle": ("Su turno ya pasó, pero son de cuentas que no se usan desde hace más de "
                                       f"{_INACTIVO_DIAS} días: se generan si la persona vuelve.")})

    # Planes fallidos del periodo (no son alertas: no se pueden «marcar como revisados»).
    fallidos = int(am._uno("SELECT COUNT(*) AS n FROM public.meal_plans "
                           "WHERE plan_data->>'generation_status' = 'failed' "
                           f"AND created_at >= {am._VENTANA} AND user_id::text <> ALL(%s::text[])",
                           (ctx.dias, ctx.fuera)).get("n") or 0)
    if fallidos:
        salida.append({"grupo": "revision", "nivel": "aviso", "titulo": "Planes fallidos", "n": fallidos,
                       "detalle": f"Planes creados en {ctx.dias} días que terminaron en error."})

    # Lo resuelto en los últimos días, una fila por tipo, con la fecha de la última resolución.
    hechas: dict = {}
    for r in resueltas:
        tipo = am._tipo_alerta(r.get("alert_key"))
        h = hechas.setdefault(tipo, {"n": 0, "ultima": None})
        h["n"] += 1
        t = r["resolved_at"]
        h["ultima"] = t if h["ultima"] is None or t > h["ultima"] else h["ultima"]
    for tipo, h in hechas.items():
        titulo, _ = _textos(tipo)
        salida.append({"grupo": "historial", "nivel": "info", "estado": "resuelto", "titulo": titulo, "n": h["n"],
                       "detalle": "Ya no está pasando.",
                       "momento": f"Resuelto {_resuelto(h['ultima'], ahora)}"})

    salida.sort(key=lambda i: (_ORDEN_GRUPO[i["grupo"]], i.get("estado") == "resuelto",
                               am._ORDEN_NIVEL[i["nivel"]], -i["n"]))
    return salida


def _resuelto(dt, ahora) -> str:
    f = fecha(dt, ahora)
    return f if f.startswith(("hoy", "ayer")) else f"el {f} · {hace(dt, ahora)}"


def marcar_revisados(huellas: list) -> int:
    """Cierra las alertas ABIERTAS cuya huella está en la lista. Devuelve cuántas cerró."""
    import admin_metricas as am
    limpias = sorted({h for h in (str(x).lower() for x in huellas or []) if _HUELLA.match(h)})
    if not limpias:
        return 0
    filas = am.execute_sql_write(
        "UPDATE public.system_alerts SET resolved_at = now(), "
        "metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('revisado_en_panel', now()) "
        "WHERE resolved_at IS NULL "
        "AND left(encode(sha256(convert_to(alert_key, 'UTF8')), 'hex'), 16) = ANY(%s::text[]) RETURNING 1 AS ok",
        (limpias,), returning=True)
    return len(filas or [])
