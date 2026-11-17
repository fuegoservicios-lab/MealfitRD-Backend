# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1032/1033 · 2026-11-17] Crear, listar, leer y borrar informes para el médico (`user_health_reports`).

Lo usan el router (`routers/reports.py`, origen 'app') y la herramienta del coach (`generar_informe_para_medico`, origen
'coach'). Toda sentencia filtra `AND user_id = %s` (invariante I2). Tope: 3 informes por persona y día biológico.
La redacción con IA va por canario `informe_medico` (coach_canario) y exige el permiso de IA de terceros; sin ella el
informe sale `data_only`, que ya es útil. Ni los datos ni la redacción van a los logs: solo recuentos.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

TOPE_DIARIO = 3
LISTA_MAX = 20


class TopeDiario(Exception):
    """Ya hizo los informes permitidos hoy."""


def _inicio_del_dia_utc(user_id: str, tz_offset: Optional[int]) -> datetime:
    from diary_targets import resolve_user_effective_off_min
    off = resolve_user_effective_off_min(user_id, tz_offset)
    ahora_local = datetime.now(timezone.utc) - timedelta(minutes=off)
    inicio_local = ahora_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return inicio_local + timedelta(minutes=off)


def informes_de_hoy(user_id: str, tz_offset: Optional[int] = None) -> int:
    from db_core import execute_sql_query
    fila = execute_sql_query(
        "SELECT count(*)::int AS n FROM user_health_reports WHERE user_id = %s AND created_at >= %s",
        (user_id, _inicio_del_dia_utc(user_id, tz_offset)), fetch_one=True,
    ) or {}
    return int(fila.get("n") or 0)


def _locale_de(user_id: str, locale: Optional[str]) -> str:
    if locale:
        return str(locale)[:10]
    try:
        from db_core import execute_sql_query
        f = execute_sql_query("SELECT locale FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True) or {}
        return str(f.get("locale") or "es")[:10]
    except Exception:  # noqa: BLE001
        return "es"


def redaccion_activa(user_id: str) -> bool:
    try:
        from coach_canario import brazo
        return brazo(user_id, "informe_medico") == "on"
    except Exception:  # noqa: BLE001
        return False


def crear_informe(user_id: str, *, dias: Any = 30, tz_offset: Optional[int] = None, locale: Optional[str] = None,
                  con_redaccion: bool = True, con_permiso: bool = True, origin: str = "app") -> dict:
    """Crea y guarda un informe. Lanza `TopeDiario` si ya hizo 3 hoy. Devuelve el informe como lo lee el cliente."""
    from db_core import execute_sql_write
    from informe_medico import construir_datos

    if informes_de_hoy(user_id, tz_offset) >= TOPE_DIARIO:
        raise TopeDiario()
    loc = _locale_de(user_id, locale)
    data = construir_datos(user_id, dias, tz_offset, loc)

    red = {"status": "data_only", "narrative": None, "model": None, "cost_usd_micros": 0, "motivo": "no_pedida"}
    if con_redaccion:
        if not redaccion_activa(user_id):
            red["motivo"] = "no_disponible"
        else:
            from informe_redaccion import redactar
            red = redactar(data, loc, con_permiso=con_permiso)

    filas = execute_sql_write(
        """
        INSERT INTO user_health_reports
            (user_id, period_start, period_end, locale, status, data, narrative, narrative_model, cost_usd_micros, origin)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
        RETURNING id, created_at
        """,
        (user_id, data["periodo"]["desde"], data["periodo"]["hasta"], loc, red["status"],
         json.dumps(data, ensure_ascii=False), json.dumps(red["narrative"], ensure_ascii=False) if red["narrative"] else None,
         red.get("model"), int(red.get("cost_usd_micros") or 0), "coach" if origin == "coach" else "app"),
        returning=True,
    ) or []
    fila = filas[0] if filas else {}
    logger.info(f"[LOTE-1032] informe creado: status={red['status']} motivo={red.get('motivo')} dias={data['periodo']['dias']} "
                f"secciones={len(data)} origen={origin}")
    return {
        "id": str(fila.get("id")) if fila.get("id") else None,
        "created_at": fila.get("created_at").isoformat() if hasattr(fila.get("created_at"), "isoformat") else None,
        "status": red["status"], "motivo": red.get("motivo"), "data": data, "narrative": red["narrative"],
        "locale": loc,
    }


def _fila_a_informe(f: dict, con_contenido: bool = True) -> dict:
    out = {
        "id": str(f.get("id")),
        "periodo": {"desde": str(f.get("period_start")), "hasta": str(f.get("period_end"))},
        "status": f.get("status"),
        "created_at": f.get("created_at").isoformat() if hasattr(f.get("created_at"), "isoformat") else str(f.get("created_at")),
        "origin": f.get("origin"),
        "locale": f.get("locale"),
    }
    if con_contenido:
        data = f.get("data")
        out["data"] = json.loads(data) if isinstance(data, str) else data
        narr = f.get("narrative")
        try:
            out["narrative"] = json.loads(narr) if narr else None
        except Exception:  # noqa: BLE001
            out["narrative"] = None
    return out


def listar(user_id: str) -> list[dict]:
    from db_core import execute_sql_query
    filas = execute_sql_query(
        "SELECT id, period_start, period_end, status, created_at, origin, locale FROM user_health_reports "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
        (user_id, LISTA_MAX), fetch_all=True,
    ) or []
    return [_fila_a_informe(f, con_contenido=False) for f in filas]


def leer(user_id: str, informe_id: str) -> Optional[dict]:
    from db_core import execute_sql_query
    f = execute_sql_query(
        "SELECT id, period_start, period_end, status, created_at, origin, locale, data, narrative "
        "FROM user_health_reports WHERE id = %s AND user_id = %s",
        (informe_id, user_id), fetch_one=True,
    )
    return _fila_a_informe(f) if f else None


def borrar(user_id: str, informe_id: str) -> bool:
    from db_core import execute_sql_write
    filas = execute_sql_write(
        "DELETE FROM user_health_reports WHERE id = %s AND user_id = %s RETURNING id",
        (informe_id, user_id), returning=True,
    ) or []
    return bool(filas)


# [P1-PLAN-LOTE-1058 · plan 1036] Retención: los informes pasan de `MEALFIT_HEALTH_REPORTS_RETENTION_DAYS` (365) ⇒ fuera.
def dias_de_retencion() -> int:
    try:
        from knobs import _env_int
        return max(30, min(int(_env_int("MEALFIT_HEALTH_REPORTS_RETENTION_DAYS", 365)), 3650))
    except Exception:  # noqa: BLE001
        return 365


def purgar_informes_antiguos() -> int:
    """Cron diario; nunca lanza (un fallo se reintenta mañana). Solo cuenta lo borrado: nada del contenido."""
    from db_core import execute_sql_write
    dias = dias_de_retencion()
    try:
        r = execute_sql_write(
            "DELETE FROM public.user_health_reports WHERE created_at < now() - make_interval(days => %s) RETURNING id",
            (dias,), returning=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [LOTE-1058] no se pudo purgar informes antiguos: {type(e).__name__}")
        return 0
    n = len(r) if isinstance(r, list) else 0
    if n:
        logger.info(f"[LOTE-1058] informes para el médico: {n} con más de {dias} días purgados.")
    return n


def resumen_para_admin(horas: int = 168) -> dict:
    """[plan 1036] Recuentos para /admin/chat-insights: cuántos, de qué estado, origen y coste medio. NUNCA contenido."""
    from db_core import execute_sql_query
    h = max(1, min(int(horas or 168), 720))
    filas = execute_sql_query(
        "SELECT status, origin, count(*)::int AS n, COALESCE(avg(cost_usd_micros), 0)::int AS coste_medio "
        "FROM user_health_reports WHERE created_at > now() - make_interval(hours => %s) GROUP BY 1, 2",
        (h,), fetch_all=True,
    ) or []
    out = {"total": 0, "por_estado": {}, "por_origen": {}, "coste_medio_usd": 0.0}
    coste, con_coste = 0, 0
    for f in filas:
        n = int(f.get("n") or 0)
        out["total"] += n
        out["por_estado"][f.get("status")] = out["por_estado"].get(f.get("status"), 0) + n
        out["por_origen"][f.get("origin")] = out["por_origen"].get(f.get("origin"), 0) + n
        if f.get("status") == "ready":
            coste += int(f.get("coste_medio") or 0) * n
            con_coste += n
    out["coste_medio_usd"] = round(coste / con_coste / 1_000_000, 4) if con_coste else 0.0
    return out
