# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1059 · plan 1029/1035 · 2026-11-17] El juego de las semanas, en el servidor.

Las reglas son las de la tarjeta «Semana N» de Progreso (`frontend/src/utils/progresoDeLosDias.js`), portadas tal cual:

- Día en meta = kcal del día dentro de la banda de la meta vigente ESE día (`informe_medico.estado_del_dia`, con test de
  paridad contra `calorieGoalStatus.js`). Sin registros: vacío.
- Semana = domingo a sábado. Semana 1 = la semana del primer día biológico con registro (`user_journey`).
- Estrellas: 3 días en meta = 1, 5 = 2, 7 = 3. Racha = días seguidos en meta; hoy en curso no la rompe.

El servidor es la fuente de la verdad (D-F): `resumen()` calcula todo en vivo desde `consumed_meals` (UNA agregación por
día desde la Semana 1) y deja las semanas cerradas en `user_week_summaries`. Los eventos del juego (cierre de semana e
hitos, D-G) son códigos con datos: el texto lo pone el cliente con su i18n. Nada aquí llama a un modelo.

Logs: solo recuentos y códigos; nunca kcal ni comidas (`log_sin_datos_de_salud`).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from informe_medico import estado_del_dia

logger = logging.getLogger(__name__)

DIAS_DE_LA_SEMANA = 7
ESTRELLAS_MAX = 3
SEMANAS_MAX_POR_PETICION = 104
SEMANAS_MAX_COACH = 26

#: Rangos por estrellas totales (plan §7.5: ajustables aquí). (clave, desde)
RANGOS = (("bronce", 0), ("plata", 10), ("oro", 25), ("platino", 50), ("diamante", 100))

#: Hitos (lote 1035). Cada uno se emite UNA vez por persona.
HITOS_RACHA = (3, 7, 14, 30)
HITOS_SEMANA = (4, 12, 26, 52)
SEMANAS_SEGUIDAS_CON_ESTRELLA = 4
EVENTOS_MAX_GUARDADOS = 200


# ═══════════════════════════════════════════════════════════════════ puras
def domingo_de(d: date) -> date:
    return d - timedelta(days=(d.weekday() + 1) % 7)


def numero_de_semana(week1_start: date, d: date) -> int:
    return max(1, (d - week1_start).days // DIAS_DE_LA_SEMANA + 1)


def fechas_de_la_semana(week1_start: date, n: int) -> list[date]:
    base = week1_start + timedelta(days=(n - 1) * DIAS_DE_LA_SEMANA)
    return [base + timedelta(days=i) for i in range(DIAS_DE_LA_SEMANA)]


def estrellas(dias_en_meta: int) -> int:
    if dias_en_meta >= DIAS_DE_LA_SEMANA:
        return 3
    if dias_en_meta >= 5:
        return 2
    return 1 if dias_en_meta >= 3 else 0


def faltan_para_otra_estrella(dias_en_meta: int) -> int:
    for escalon in (3, 5, DIAS_DE_LA_SEMANA):
        if dias_en_meta < escalon:
            return escalon - dias_en_meta
    return 0


def estado(calories: Any, meals_count: Any, meta: Any, objetivo: Any) -> str:
    """'vacio' | 'meta' | 'bajo' | 'pasado' — los nombres del cliente (`estadoDelDia`)."""
    kcal = float(calories or 0)
    if not (int(meals_count or 0) > 0) and not (kcal > 0):
        return "vacio"
    e = estado_del_dia(kcal, float(meta or 0), objetivo)
    return {"en_meta": "meta", "sobre": "pasado"}.get(e, "bajo")


def racha(estados_ordenados: list[str], hoy_en_curso: bool = True) -> int:
    """Días seguidos en meta al final de la lista (el último es hoy). Hoy sin meta aún no rompe: se cuenta hasta ayer."""
    i = len(estados_ordenados) - 1
    if hoy_en_curso and i >= 0 and estados_ordenados[i] != "meta":
        i -= 1
    n = 0
    while i >= 0 and estados_ordenados[i] == "meta":
        n += 1
        i -= 1
    return n


def mejor_racha(estados_ordenados: Iterable[str]) -> int:
    mejor = actual = 0
    for e in estados_ordenados:
        actual = actual + 1 if e == "meta" else 0
        mejor = max(mejor, actual)
    return mejor


def rango(estrellas_totales: int) -> dict:
    clave, siguiente, faltan = RANGOS[0][0], None, None
    for i, (k, desde) in enumerate(RANGOS):
        if estrellas_totales >= desde:
            clave = k
            if i + 1 < len(RANGOS):
                siguiente, faltan = RANGOS[i + 1][0], RANGOS[i + 1][1] - estrellas_totales
            else:
                siguiente, faltan = None, None
    return {"clave": clave, "siguiente": siguiente, "estrellas_para_el_siguiente": faltan}


def _media(xs: list[float]) -> Optional[int]:
    return int(round(sum(xs) / len(xs))) if xs else None


def armar_resumen(*, hoy: date, dia_1: Optional[date], dias: dict[str, dict], metas_por_dia: dict[str, dict],
                  meta_de_hoy: Optional[dict], objetivo_del_perfil: Any = None) -> dict:
    """Todo el juego a partir de datos YA leídos. `dias`: {iso: {calories, protein, meals_count}} (solo con registro).

    Devuelve el `resumen` del plan (§1029.6) + `semanas` (todas, con su detalle) para quien lo necesite.
    """
    if not dia_1 or dia_1 > hoy:
        return {}
    w1 = domingo_de(dia_1)
    actual = numero_de_semana(w1, hoy)
    estados: list[str] = []          # desde w1 hasta hoy, en orden
    por_semana: dict[int, dict] = {}
    d = w1
    while d <= hoy:
        iso = d.isoformat()
        r = dias.get(iso) or {}
        es_hoy = d == hoy
        m = (meta_de_hoy if es_hoy else (metas_por_dia.get(iso) or meta_de_hoy)) or {}
        objetivo = (objetivo_del_perfil if es_hoy else (m.get("main_goal") or objetivo_del_perfil)) or ""
        e = estado(int(r.get("calories") or 0), r.get("meals_count"), m.get("calories"), objetivo)
        estados.append(e)
        n = numero_de_semana(w1, d)
        s = por_semana.setdefault(n, {"numero": n, "empieza": (w1 + timedelta(days=(n - 1) * 7)).isoformat(),
                                      "dias_registrados": 0, "dias_en_meta": 0, "dias_bajo": 0, "dias_sobre": 0,
                                      "_kcal": [], "_prot": []})
        if e != "vacio":
            s["dias_registrados"] += 1
            s["_kcal"].append(float(r.get("calories") or 0))
            s["_prot"].append(float(r.get("protein") or 0))
            s[{"meta": "dias_en_meta", "bajo": "dias_bajo", "pasado": "dias_sobre"}[e]] += 1
        d += timedelta(days=1)

    semanas = []
    for n in sorted(por_semana):
        s = por_semana[n]
        semanas.append({
            "numero": n, "empieza": s["empieza"],
            "termina": (date.fromisoformat(s["empieza"]) + timedelta(days=6)).isoformat(),
            "dias_registrados": s["dias_registrados"], "dias_en_meta": s["dias_en_meta"],
            "dias_bajo": s["dias_bajo"], "dias_sobre": s["dias_sobre"],
            "estrellas": estrellas(s["dias_en_meta"]),
            "kcal_media": _media(s["_kcal"]), "proteina_media": _media(s["_prot"]),
            "cerrada": n < actual,
        })
    cur = semanas[-1]
    cerradas = [s for s in semanas if s["cerrada"]]
    total_estrellas = sum(s["estrellas"] for s in semanas)
    mejor = max(cerradas + [cur], key=lambda s: (s["dias_en_meta"], s["numero"]))
    out = {
        "dia_1": dia_1.isoformat(), "semana_1_empieza": w1.isoformat(), "hoy": hoy.isoformat(),
        "semana": {"numero": actual, "empieza": cur["empieza"], "termina": cur["termina"],
                   "dia": (hoy - date.fromisoformat(cur["empieza"])).days + 1,
                   "dias_en_meta": cur["dias_en_meta"], "dias_registrados": cur["dias_registrados"],
                   "estrellas": cur["estrellas"], "faltan_para_otra_estrella": faltan_para_otra_estrella(cur["dias_en_meta"])},
        "racha": racha(estados), "mejor_racha": mejor_racha(estados),
        "total": {"semanas": actual, "estrellas": total_estrellas,
                  "semanas_perfectas": sum(1 for s in semanas if s["estrellas"] == 3),
                  "dias_en_meta": sum(s["dias_en_meta"] for s in semanas)},
        "rango": rango(total_estrellas),
        "semana_pasada": ({k: cerradas[-1][k] for k in ("numero", "dias_en_meta", "estrellas")} if cerradas else None),
        "mejor_semana": ({k: mejor[k] for k in ("numero", "dias_en_meta", "estrellas")} if mejor["dias_en_meta"] else None),
        "semanas": semanas,
    }
    return out


# ─────────────────────────────────────────────────────────────── eventos (lote 1035, puro)
def hitos_cumplidos(res: dict) -> list[dict]:
    """Todos los eventos que el estado actual ya cumple, con su clave única. El llamador filtra los ya emitidos."""
    if not res:
        return []
    out: list[dict] = []
    semanas = res.get("semanas") or []
    cerradas = [s for s in semanas if s.get("cerrada")]
    if cerradas:
        u = cerradas[-1]
        out.append({"clave": f"semana_cerrada:{u['numero']}", "codigo": "semana_cerrada",
                    "datos": {"numero": u["numero"], "estrellas": u["estrellas"], "dias_en_meta": u["dias_en_meta"],
                              "siguiente": u["numero"] + 1}})
    if res["total"]["estrellas"] >= 1:
        out.append({"clave": "primera_estrella", "codigo": "primera_estrella", "datos": {}})
    perfectas = [s for s in semanas if s["estrellas"] == 3]
    if perfectas:
        out.append({"clave": "primera_semana_perfecta", "codigo": "primera_semana_perfecta",
                    "datos": {"numero": perfectas[0]["numero"]}})
    for x in HITOS_RACHA:
        if res["racha"] >= x:
            out.append({"clave": f"racha:{x}", "codigo": "racha", "datos": {"dias": x}})
    seguidas = 0
    for s in cerradas:
        seguidas = seguidas + 1 if s["estrellas"] >= 1 else 0
        if seguidas >= SEMANAS_SEGUIDAS_CON_ESTRELLA:
            out.append({"clave": "semanas_seguidas_con_estrella:4", "codigo": "semanas_seguidas_con_estrella",
                        "datos": {"semanas": SEMANAS_SEGUIDAS_CON_ESTRELLA}})
            break
    rg = res["rango"]["clave"]
    if rg != RANGOS[0][0]:
        out.append({"clave": f"rango:{rg}", "codigo": "rango", "datos": {"clave": rg}})
    for x in HITOS_SEMANA:
        if res["semana"]["numero"] >= x:
            out.append({"clave": f"semana_numero:{x}", "codigo": "semana_numero", "datos": {"numero": x}})
    return out


def eventos_nuevos(cumplidos: list[dict], estado_kv: dict, hoy_iso: str) -> tuple[list[dict], dict]:
    """(los que hay que mostrar, estado_kv nuevo). La primera vez (sin estado) todo lo cumplido se da por visto: una
    cuenta vieja no recibe de golpe el historial entero. Con varias semanas cerradas de golpe solo cuenta la última
    (`hitos_cumplidos` solo propone la última)."""
    emitidos = dict((estado_kv or {}).get("emitidos") or {})
    vistos = set((estado_kv or {}).get("vistos") or [])
    primera = not estado_kv
    for ev in cumplidos:
        if ev["clave"] not in emitidos:
            emitidos[ev["clave"]] = hoy_iso
            if primera:
                vistos.add(ev["clave"])
    pendientes = [dict(ev, emitido=emitidos[ev["clave"]]) for ev in cumplidos
                  if ev["clave"] in emitidos and ev["clave"] not in vistos]
    if len(emitidos) > EVENTOS_MAX_GUARDADOS:
        viejas = sorted(emitidos, key=lambda k: emitidos[k])[: len(emitidos) - EVENTOS_MAX_GUARDADOS]
        for k in viejas:
            emitidos.pop(k, None)
            vistos.discard(k)
    return pendientes, {"emitidos": emitidos, "vistos": sorted(vistos)}


def eventos_de_hoy(estado_kv: dict, hoy_iso: str) -> list[str]:
    """Claves emitidas HOY (para que el coach lo celebre una vez, el día que ocurre)."""
    return sorted(k for k, v in ((estado_kv or {}).get("emitidos") or {}).items() if v == hoy_iso)


# ═══════════════════════════════════════════════════════════════════ datos
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 30.0


def invalidar(user_id: str) -> None:
    with _CACHE_LOCK:
        _CACHE.pop(str(user_id), None)


def _kv_clave(user_id: str) -> str:
    return f"progreso_eventos:{user_id}"


def asegurar_dia_1(user_id: str, fecha_local: Any, tz_offset_min: Optional[int] = None) -> None:
    """El día 1 solo se mueve hacia ATRÁS (§3 del plan): borrar la primera comida no reinicia el viaje."""
    from db_core import execute_sql_write
    if not user_id or user_id == "guest" or not fecha_local:
        return
    d = fecha_local if isinstance(fecha_local, date) else date.fromisoformat(str(fecha_local)[:10])
    execute_sql_write(
        "INSERT INTO user_journey (user_id, first_log_date, tz_offset_min) VALUES (%s, %s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET first_log_date = LEAST(user_journey.first_log_date, EXCLUDED.first_log_date), "
        "updated_at = now() WHERE EXCLUDED.first_log_date < user_journey.first_log_date",
        (user_id, d, tz_offset_min),
    )


def tras_escribir_el_diario(user_id: str, consumed_at: Any = None, *, es_registro: bool = False) -> None:
    """Enganche de `db_facts` tras un registro/corrección/borrado guardado. Nunca lanza (la comida ya está guardada).

    Asegura el día 1 (solo en registros) y deja sin cerrar la semana tocada: la siguiente lectura la recalcula.
    """
    try:
        invalidar(user_id)
        if not user_id or user_id == "guest":
            return
        from db_core import execute_sql_write
        from diary_targets import resolve_user_effective_off_min
        eff = resolve_user_effective_off_min(user_id, None)
        if isinstance(consumed_at, str):
            try:
                consumed_at = datetime.fromisoformat(consumed_at.replace("Z", "+00:00"))
            except ValueError:
                consumed_at = None
        if isinstance(consumed_at, datetime):
            ca = consumed_at if consumed_at.tzinfo else consumed_at.replace(tzinfo=timezone.utc)
            dia = (ca.astimezone(timezone.utc) - timedelta(minutes=eff)).date()
        else:
            dia = (datetime.now(timezone.utc) - timedelta(minutes=eff)).date()
        if es_registro:
            try:
                from db_facts import _diary_rollover_hour
                tz = eff - int(_diary_rollover_hour()) * 60
            except Exception:  # noqa: BLE001
                tz = None
            asegurar_dia_1(user_id, dia, tz)
        execute_sql_write(
            "UPDATE user_week_summaries SET closed = false WHERE user_id = %s AND week_start = %s",
            (user_id, domingo_de(dia)),
        )
    except Exception as exc:  # noqa: BLE001
        try:
            from fail_open import fail_open
            fail_open("progreso_dia_1", exc)
        except Exception:  # noqa: BLE001
            logger.warning(f"[LOTE-1059] enganche de progreso falló: {type(exc).__name__}")


def _dia_1(user_id: str, eff_off: int) -> Optional[date]:
    from db_core import execute_sql_query
    f = execute_sql_query("SELECT first_log_date FROM user_journey WHERE user_id = %s", (user_id,), fetch_one=True)
    if f and f.get("first_log_date"):
        v = f["first_log_date"]
        return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])
    # Cuenta sin fila (registró antes del enganche o el enganche falló): se deduce y se guarda.
    p = execute_sql_query(
        "SELECT min((consumed_at - make_interval(mins => %s))::date) AS d FROM consumed_meals WHERE user_id = %s",
        (eff_off, user_id), fetch_one=True,
    ) or {}
    d = p.get("d")
    if not d:
        return None
    d = d if isinstance(d, date) else date.fromisoformat(str(d)[:10])
    try:
        from db_facts import _diary_rollover_hour
        asegurar_dia_1(user_id, d, eff_off - int(_diary_rollover_hour()) * 60)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1059] día 1 no guardado: {type(exc).__name__}")
    return d


def _leer(user_id: str, tz_offset: Optional[int]) -> dict:
    from db_core import execute_sql_query
    from diary_targets import resolve_targets_by_day, resolve_user_effective_off_min

    eff = resolve_user_effective_off_min(user_id, tz_offset)
    hoy = (datetime.now(timezone.utc) - timedelta(minutes=eff)).date()
    dia_1 = _dia_1(user_id, eff)
    if not dia_1:
        return {}
    w1 = domingo_de(min(dia_1, hoy))
    filas = execute_sql_query(
        """
        SELECT (consumed_at - make_interval(mins => %s))::date AS dia,
               COALESCE(SUM(calories), 0) AS calories, COALESCE(SUM(protein), 0) AS protein, COUNT(*) AS meals_count
        FROM consumed_meals
        WHERE user_id = %s AND consumed_at >= %s::date - interval '1 day'
        GROUP BY 1
        """,
        (eff, user_id, w1), fetch_all=True,
    ) or []
    dias = {str(r.get("dia")): {"calories": int(r.get("calories") or 0), "protein": int(r.get("protein") or 0),
                                "meals_count": int(r.get("meals_count") or 0)} for r in filas}
    metas: dict = {}
    try:
        metas = resolve_targets_by_day(user_id, list(dias) + [hoy.isoformat()], eff) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1059] metas no resueltas: {type(exc).__name__}")
    perfil = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True) or {}
    hp = perfil.get("health_profile") or {}
    if isinstance(hp, str):
        try:
            hp = json.loads(hp)
        except Exception:  # noqa: BLE001
            hp = {}
    return armar_resumen(hoy=hoy, dia_1=min(dia_1, hoy), dias=dias, metas_por_dia=metas,
                         meta_de_hoy=metas.get(hoy.isoformat()), objetivo_del_perfil=(hp or {}).get("mainGoal"))


def _sincronizar_semanas(user_id: str, semanas: list[dict]) -> int:
    """Deja en `user_week_summaries` las semanas cerradas que falten o hayan cambiado. Devuelve cuántas escribió."""
    from db_core import execute_sql_query, execute_sql_write
    cerradas = [s for s in semanas if s.get("cerrada")]
    if not cerradas:
        return 0
    guardadas = {int(f["week_number"]): f for f in (execute_sql_query(
        "SELECT week_number, days_logged, days_on_goal, days_under, days_over, stars, closed "
        "FROM user_week_summaries WHERE user_id = %s", (user_id,), fetch_all=True) or [])}
    n = 0
    for s in cerradas:
        g = guardadas.get(s["numero"])
        if g and g.get("closed") and (int(g["days_logged"]), int(g["days_on_goal"]), int(g["days_under"]),
                                      int(g["days_over"]), int(g["stars"])) == (
                s["dias_registrados"], s["dias_en_meta"], s["dias_bajo"], s["dias_sobre"], s["estrellas"]):
            continue
        execute_sql_write(
            """
            INSERT INTO user_week_summaries (user_id, week_number, week_start, days_logged, days_on_goal, days_under,
                                             days_over, stars, kcal_avg, protein_avg, closed, computed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, now())
            ON CONFLICT (user_id, week_number) DO UPDATE SET
                week_start = EXCLUDED.week_start, days_logged = EXCLUDED.days_logged,
                days_on_goal = EXCLUDED.days_on_goal, days_under = EXCLUDED.days_under,
                days_over = EXCLUDED.days_over, stars = EXCLUDED.stars, kcal_avg = EXCLUDED.kcal_avg,
                protein_avg = EXCLUDED.protein_avg, closed = true, computed_at = now()
            """,
            (user_id, s["numero"], s["empieza"], s["dias_registrados"], s["dias_en_meta"], s["dias_bajo"],
             s["dias_sobre"], s["estrellas"], s["kcal_media"], s["proteina_media"]),
        )
        n += 1
    return n


def _estado_kv(user_id: str) -> dict:
    from db_core import execute_sql_query
    f = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (_kv_clave(user_id),), fetch_one=True)
    v = (f or {}).get("value")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            v = None
    return v if isinstance(v, dict) else {}


def _guardar_kv(user_id: str, valor: dict) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
        (_kv_clave(user_id), json.dumps(valor)),
    )


def resumen(user_id: str, tz_offset: Optional[int] = None, *, con_semanas: bool = False,
            con_eventos: bool = True) -> dict:
    """El resumen del juego (plan §1029.6) + `eventos` pendientes (lote 1035). Invitado o sin registros ⇒ {}."""
    if not user_id or user_id == "guest":
        return {}
    clave = f"{user_id}:{tz_offset}"
    with _CACHE_LOCK:
        hit = _CACHE.get(str(user_id))
    if hit and hit[1].get("_clave") == clave and time.monotonic() - hit[0] < _CACHE_TTL_S:
        res = hit[1]
    else:
        res = _leer(user_id, tz_offset)
        if res:
            try:
                escritas = _sincronizar_semanas(user_id, res["semanas"])
                if escritas:
                    logger.info(f"[LOTE-1059] semanas cerradas guardadas: {escritas}")
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[LOTE-1059] semanas no sincronizadas: {type(exc).__name__}")
            res["_clave"] = clave
            try:
                kv = _estado_kv(user_id)
                pendientes, nuevo = eventos_nuevos(hitos_cumplidos(res), kv, res["hoy"])
                if nuevo != kv:
                    _guardar_kv(user_id, nuevo)
                res["_eventos"] = pendientes
                res["_eventos_de_hoy"] = eventos_de_hoy(nuevo, res["hoy"])
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[LOTE-1059] eventos no evaluados: {type(exc).__name__}")
                res["_eventos"], res["_eventos_de_hoy"] = [], []
        with _CACHE_LOCK:
            _CACHE[str(user_id)] = (time.monotonic(), res)
    if not res:
        return {}
    out = {k: v for k, v in res.items() if not k.startswith("_") and (con_semanas or k != "semanas")}
    if con_eventos:
        out["eventos"] = [{"clave": e["clave"], "codigo": e["codigo"], "datos": e["datos"]} for e in res.get("_eventos") or []]
    out["eventos_de_hoy"] = list(res.get("_eventos_de_hoy") or [])
    return out


def marcar_vistos(user_id: str, claves: Iterable[str]) -> int:
    claves = {str(c)[:80] for c in (claves or []) if c}
    if not claves:
        return 0
    kv = _estado_kv(user_id)
    emitidos = (kv or {}).get("emitidos") or {}
    vistos = set((kv or {}).get("vistos") or [])
    nuevos = {c for c in claves if c in emitidos} - vistos
    if nuevos:
        _guardar_kv(user_id, {"emitidos": emitidos, "vistos": sorted(vistos | nuevos)})
        invalidar(user_id)
    return len(nuevos)


def semanas(user_id: str, tz_offset: Optional[int] = None, desde: Optional[int] = None,
            hasta: Optional[int] = None) -> list[dict]:
    """Lista de semanas (numero, empieza, dias_en_meta, dias_registrados, estrellas, kcal_media), tope 104."""
    res = resumen(user_id, tz_offset, con_semanas=True, con_eventos=False)
    lista = res.get("semanas") or []
    if desde is not None:
        lista = [s for s in lista if s["numero"] >= int(desde)]
    if hasta is not None:
        lista = [s for s in lista if s["numero"] <= int(hasta)]
    lista = lista[-SEMANAS_MAX_POR_PETICION:]
    return [{k: s[k] for k in ("numero", "empieza", "termina", "dias_en_meta", "dias_registrados", "estrellas",
                                "kcal_media", "cerrada")} for s in lista]
