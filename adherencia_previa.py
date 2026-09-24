# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-206 · 2026-09-24] Lo que el siguiente bloque aprende del anterior: sólo lo que el usuario DIJO.

Producción: el bloque «🔁 ADHERENCIA REAL DEL BLOQUE N (APRENDIZAJE OBLIGATORIO)» del prompt salió en los 15 rellenos
desde el 4-sep con `consumed=0 skipped=N`. Ningún usuario con plan por bloques registra comidas, y el desglose convertía
cada plato NO registrado en «⛔ El usuario NO consumió … NO repitas estos platos ni recetas con la misma combinación
principal de ingredientes/técnica. Si una comida fue saltada repetidamente, simplifica esa franja al máximo». Sin
registros no hay evidencia de nada —el propio gate lo llama `zero_log_proxy`—: el modelo recibía en cada relleno la
orden de huir de la familia de platos del bloque anterior y de simplificar TODAS las franjas.

Y la ventana: el desglose y el gate eligen los días del bloque previo por NÚMERO (`day` entre offset+1 y offset+count),
pero cada shift renumera los días vivos desde 1 y archiva los pasados (en `_archived_days` todos quedan con `day=1`), y
el offset del bloque previo ya completado no se re-ancla. «window=days 2-6 skipped=4» (17-sep) son las 4 comidas de UN
día de un bloque de cinco. Los registros, además, se leían desde un inicio calculado con el ancla que el shift ya
movió a hoy: a quien registre, se le perdían los primeros días del bloque.

Aquí:
  · el bloque previo son sus FECHAS: los `n` últimos días planificados (vivos ∪ archivados), `n` = días del bloque
    anterior en la cola, y de ellos sólo los ya VIVIDOS (fecha < hoy local); sin fechas, la selección de siempre;
  · los registros del desglose se leen de ESA ventana (medianoche local de su primer día → fin del último);
  · «no consumió» sólo con registro representativo (≥ max(2, 25 % de lo planificado) — el mismo umbral con el que el
    gate separa `sparse_logging_proxy`); por debajo se dice sólo lo que SÍ comió, y sin registros, nada.
En el gate sólo se corrige la VENTANA cuando ya iba a evaluar (ver `ventana_gate`): si un usuario sin registros debe
esperar en pausa al cambiar de bloque es una decisión de producto que este lote no toma.
Knob `MEALFIT_PREV_ADHERENCE_HONEST` (True).
"""
from __future__ import annotations

import logging
import unicodedata
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

UMBRAL_REPRESENTATIVO = 0.25   # el de `_calculate_chunk_consumption_ratio._SPARSE_LOGGING_THRESHOLD`
_FUERA = ("swapped_out", "skipped", "rejected")


def activo() -> bool:
    """tooltip-anchor: MEALFIT_PREV_ADHERENCE_HONEST"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PREV_ADHERENCE_HONEST", True)
    except Exception:
        return True


def _norm(texto) -> str:
    """Igual que `cron_tasks._normalize_meal_name`: minúsculas, sin acentos, sin bordes."""
    if not texto:
        return ""
    try:
        from constants import strip_accents
        return strip_accents(str(texto).lower()).strip()
    except Exception:
        s = unicodedata.normalize("NFKD", str(texto).lower())
        return "".join(c for c in s if not unicodedata.combining(c)).strip()


def _fecha(d) -> Optional[date]:
    if not isinstance(d, dict):
        return None
    try:
        return date.fromisoformat(str(d.get("date") or "")[:10])
    except ValueError:
        return None


def representativo(registros: int, planificadas: int) -> bool:
    return planificadas > 0 and registros >= max(2, planificadas * UMBRAL_REPRESENTATIVO)


def dias_del_bloque(plan_data, n, hoy: Optional[date] = None) -> Optional[list]:
    """Los días (vivos ∪ archivados) cuyas fechas están entre las `n` últimas planificadas; con `hoy`, sólo los de
    fecha anterior a hoy (los ya vividos). None = el plan no trae fechas (el caller conserva su selección)."""
    if not isinstance(plan_data, dict):
        return None
    dias = []
    for clave in ("_archived_days", "days"):
        lista = plan_data.get(clave)
        dias.extend(d for d in (lista if isinstance(lista, list) else []) if _fecha(d) is not None)
    if not dias:
        return None
    try:
        n = max(1, int(n))
    except (TypeError, ValueError):
        return None
    ultimas = sorted({_fecha(d) for d in dias})[-n:]
    elegidas = set(ultimas)
    out = [d for d in dias if _fecha(d) in elegidas and (hoy is None or _fecha(d) < hoy)]
    return sorted(out, key=_fecha)


def dias_del_bloque_en_cola(meal_plan_id, week_number, respaldo, consultar: Optional[Callable] = None) -> int:
    """Días del bloque N-1 según la cola (también para el bloque 2, que `_resolve_previous_chunk_window` responde con
    el offset YA re-anclado del bloque actual). Sin fila, `respaldo`. `consultar` = el `execute_sql_query` del caller
    (el de cron_tasks, parcheable en sus tests)."""
    try:
        if consultar is None:
            from db_core import execute_sql_query as consultar
        fila = consultar(
            "SELECT days_count FROM plan_chunk_queue WHERE meal_plan_id = %s AND week_number = %s "
            "AND status IN ('completed', 'processing') ORDER BY status = 'completed' DESC, updated_at DESC NULLS LAST "
            "LIMIT 1",
            (str(meal_plan_id), int(week_number) - 1),
            fetch_one=True,
        )
        n = int((fila or {}).get("days_count") or 0)
        if n > 0:
            return n
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-206] days_count del bloque previo no disponible: {e}")
    try:
        return max(1, int(respaldo))
    except (TypeError, ValueError):
        return 1


def medianoche_utc(dia: date, tz_min: int) -> datetime:
    """Medianoche LOCAL de `dia` como instante UTC (convención getTimezoneOffset: RD = +240 → 04:00Z)."""
    return datetime.combine(dia, time(0), tzinfo=timezone.utc) + timedelta(minutes=int(tz_min or 0))


def _cuando(registro) -> Optional[datetime]:
    v = registro.get("consumed_at") if isinstance(registro, dict) else None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def construir(dias, registros, numero_bloque) -> Optional[dict]:
    """El desglose con la forma de `_compute_prev_chunk_meal_breakdown`, honesto: sin registro representativo no hay
    «no consumió»."""
    planificados, vistos, n_planificadas = [], set(), 0
    for dia in dias or []:
        for comida in (dia.get("meals") or []) if isinstance(dia, dict) else []:
            if not isinstance(comida, dict) or comida.get("status") in _FUERA or not comida.get("name"):
                continue
            k = _norm(comida.get("name"))
            if not k:
                continue
            n_planificadas += 1
            if k not in vistos:
                vistos.add(k)
                planificados.append((comida.get("name"), k))
    if not planificados:
        return None
    comidos = {_norm(r.get("meal_name") or r.get("name")) for r in registros or [] if isinstance(r, dict)}
    comidos.discard("")
    n_registros = sum(1 for r in registros or [] if isinstance(r, dict) and _norm(r.get("meal_name") or r.get("name")))
    consumidos = [nombre for nombre, k in planificados if k in comidos]
    saltados = [nombre for nombre, k in planificados if k not in comidos]
    evidencia = "representative" if representativo(n_registros, n_planificadas) else ("sparse" if n_registros else "none")
    if evidencia != "representative":
        saltados = []
    if not consumidos and not saltados:
        return None
    fechas = [f for f in (_fecha(d) for d in dias or []) if f]
    return {
        "chunk_number": int(numero_bloque),
        "prev_start_day": min(fechas).isoformat() if fechas else None,
        "prev_end_day": max(fechas).isoformat() if fechas else None,
        "consumed_meals": consumidos[:8],
        "skipped_meals": saltados[:8],
        "consumed_count": len(consumidos),
        "skipped_count": len(saltados),
        "logged_meals": n_registros,
        "planned_meals": n_planificadas,
        "evidence": evidencia,
    }


def desglose(*, legado: Callable[[], Optional[dict]], plan_data, meal_plan_id, week_number, prev_offset, prev_count,
             registros, leer_registros: Optional[Callable] = None, user_id=None, tz_min: int = 240,
             consultar: Optional[Callable] = None, ahora: Optional[datetime] = None) -> Optional[dict]:
    """Punto de entrada del worker. Knob apagado ⇒ `legado()` tal cual."""
    if not activo():
        return legado()
    ahora = ahora or datetime.now(timezone.utc)
    hoy = (ahora - timedelta(minutes=int(tz_min or 0))).date()
    n = dias_del_bloque_en_cola(meal_plan_id, week_number, prev_count, consultar)
    dias = dias_del_bloque(plan_data, n, hoy=hoy)
    if dias is None:
        # Sin fechas: la selección por número de siempre, pero con la misma honestidad.
        ini, fin = int(prev_offset) + 1, int(prev_offset) + int(prev_count)
        dias = [d for d in (plan_data or {}).get("days") or [] if isinstance(d, dict) and ini <= int(d.get("day") or 0) <= fin]
        return construir(dias, registros, int(week_number) - 1)
    if not dias:
        return None
    inicio = medianoche_utc(_fecha(dias[0]), tz_min)
    fin = medianoche_utc(_fecha(dias[-1]) + timedelta(days=1), tz_min)
    propios = registros
    if leer_registros is not None and user_id:
        try:
            propios = leer_registros(user_id, inicio.isoformat()) or []
        except Exception as e:
            logger.debug(f"[P1-PLAN-LOTE-206] registros de la ventana no disponibles: {e}")
    propios = [r for r in propios or [] if (_cuando(r) is None) or (inicio <= _cuando(r) < fin)]
    return construir(dias, propios, int(week_number) - 1)


def ventana_gate(plan_data, meal_plan_id, week_number, prev_count, legado_dias, prev_start_iso, tz_min, hoy,
                 consultar: Optional[Callable] = None):
    """Para `_check_chunk_learning_ready`: (días, inicio_iso) del bloque previo por FECHAS.

    Sólo corrige cuando la selección por número ya encontró días en ESTE `plan_data` (el gate iba a evaluar): si
    encontró cero, se conserva —el gate sigue su rama de siempre (cola / relectura / fail-open)— y si los días vienen
    de la relectura fresca de P0-3 (otro objeto), también: este `plan_data` es el viejo. Así un usuario sin registros
    no pasa de «genera» a «pausa de 6 h» por este lote. El inicio nunca se mueve hacia adelante."""
    if not legado_dias or not activo():
        return legado_dias, prev_start_iso
    vivos = (plan_data or {}).get("days") if isinstance(plan_data, dict) else None
    if not all(any(d is v for v in (vivos or [])) for d in legado_dias):
        return legado_dias, prev_start_iso
    try:
        dias = dias_del_bloque(plan_data, dias_del_bloque_en_cola(meal_plan_id, week_number, prev_count, consultar),
                               hoy=hoy)
        if not dias:
            return legado_dias, prev_start_iso
        inicio = medianoche_utc(_fecha(dias[0]), tz_min)
        try:
            previo = datetime.fromisoformat(str(prev_start_iso).replace("Z", "+00:00"))
            previo = previo if previo.tzinfo else previo.replace(tzinfo=timezone.utc)
            inicio = min(inicio, previo)
        except (TypeError, ValueError):
            pass
        return dias, inicio.isoformat()
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-206] ventana del gate por fechas no disponible: {e}")
        return legado_dias, prev_start_iso
