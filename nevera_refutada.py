# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-747 · 2026-09-28] Una Nevera que la guarda ya refutó no se vuelve a probar con 3 corridas del LLM.

Caso REAL (plan 3957a669, la única usuaria externa con relleno ligado a su Nevera, 04→27-sep). Su Nevera nació de UNA
compra (04-sep, 41 alimentos) y no volvió a cambiar: sin compras ni consumos; sólo se movían las reservas. Los bloques
3, 4, 7 y 9 del relleno recorrieron el mismo camino: 3 intentos del pipeline (17-28 min) rechazados por la guarda de
existencia («INEXISTENTES: aguacate, edamame, guineo…») → pausa `pantry_violation_after_retries` → 12 h de TTL →
modo flexible → entrega con 🚨 Compra Urgente. Esos intentos descartados fueron el 49 % de todo su gasto de LLM. Desde
el segundo bloque el desenlace era predecible ANTES de llamar al modelo: la misma guarda ya había refutado esa Nevera
y la Nevera no había cambiado.

Aquí vive el pre-chequeo determinista que el worker consulta justo antes del bucle de reintentos, y la evidencia que
lo alimenta:

  `registrar` — la guarda de existencia agota sus reintentos: se guarda en `plan_data._pantry_refutation` la unión de
      las líneas INEXISTENTES de sus últimos intentos y la huella BRUTA de la Nevera (`user_inventory`, lo que el
      usuario tiene, sin reservas). `jsonb_set` quirúrgico + `AND user_id` + sello `_plan_modified_at` (patrón de
      `_mark_first_purchase_pause`).
  `prechequeo` — entra directo por la pausa de siempre SÓLO si se cumplen las cuatro:
      1. La SSOT de las guardas no exime al bloque (`_pantry_gate_waiver_reason`, mismos argumentos que la guarda de
         existencia): flexible, advisory, invitado, Nevera virtual y la autonomía del `initial_plan` —y con ella
         P1-FIRST-PURCHASE-PAUSE, que vive antes, en el gate pre-pipeline— siguen mandando. Ninguna guarda decide sola.
      2. Hay Nevera exigida (la misma condición `if _pantry_snapshot:` de la guarda).
      3. La evidencia existe, tiene líneas y no caducó (`MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H`, 336 h: cada dos
         semanas se vuelve a probar de verdad aunque nada cambie).
      4. La Nevera no creció: ni alimento nuevo ni más cantidad en la huella bruta (las reservas no son compras: suben
         y bajan solas) Y la MISMA medida de la guarda (`validate_ingredients_against_pantry` + `compras_pequenas.
         tolerar`) sigue rechazando lo refutado contra la Nevera de hoy.
      ⇒ `_pause_chunk_for_pantry_refresh(reason="pantry_violation_after_retries")` + el mismo push: el estado que el
      worker dejaba tras los 3 intentos, sin los 3 intentos. El recovery decide lo demás igual que siempre (TTL →
      flexible, tope de ciclos P2-PANTRY-PAUSE-MAX-CYCLES).
  La comprobación 4 usa el modo sonda del validador (sin paso vectorial: una llamada de embeddings por línea no cabe
  en un pre-chequeo). La sonda sólo rechaza DE MÁS; como lo refutado ya lo rechazó la guarda completa contra una
  Nevera que no ha crecido, el único hueco es un alimento que estaba reservado entero al refutar y hoy el vector
  casaría con una línea refutada. Acotado por la caducidad y por el knob.

Cualquier fallo al DECIDIR deja correr al LLM (conducta previa). Knob `MEALFIT_PANTRY_REFUTED_PRECHECK` (True) —
apagado, ni lee ni escribe. tooltip-anchor: P1-PLAN-LOTE-747
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MARCA = "_pantry_refutation"
#: El motivo de la pausa que el worker dejaba tras agotar los reintentos: el pre-chequeo entra por el MISMO.
MOTIVO = "pantry_violation_after_retries"
_MAX_LINEAS = 40
_MAX_FILAS = 300


def activo() -> bool:
    """Knob. tooltip-anchor: MEALFIT_PANTRY_REFUTED_PRECHECK"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PANTRY_REFUTED_PRECHECK", True)
    except Exception:
        return True


def max_edad_h() -> int:
    """Horas que vale una refutación. tooltip-anchor: MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H"""
    try:
        from knobs import _env_int
        return _env_int("MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H", 336, validator=lambda v: 1 <= v <= 1440)
    except Exception:
        return 336


def _sa(s) -> str:
    t = "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t.lower()).strip()


# ─────────────── acceso a datos (los tests los sustituyen) ───────────────

def _leer_bruto(user_id):
    """Filas de la Nevera REAL del usuario (lo que tiene, sin descontar reservas). `kind='food'` (P1-PLAN-LOTE-290)."""
    from db import execute_sql_query
    return execute_sql_query(
        "SELECT ingredient_name, quantity::float8 AS quantity, unit FROM user_inventory "
        "WHERE user_id = %s AND kind = 'food' AND quantity > 0",
        (user_id,), fetch_all=True,
    )


def _escribir(sql, params) -> None:
    from db import execute_sql_write
    execute_sql_write(sql, params)


def _metrica(**meta) -> None:
    """Una fila en `pipeline_metrics` por bloque que se ahorró sus reintentos (best-effort)."""
    try:
        from db import execute_sql_write
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
            (meta.get("user_id"), None, "pantry_refuted_precheck", 0, 0,
             json.dumps(meta, ensure_ascii=False, default=str)),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-747] pipeline_metrics no-op: {type(e).__name__}: {e}")


# ─────────────── la medida ───────────────

def huella_bruta(filas) -> dict:
    """`{"nombre|unidad": cantidad}` de la Nevera bruta. Por NOMBRE crudo (no la base normalizada, que junta tilapia y
    salmón en «pescado»): un alimento nuevo siempre aparece como clave nueva."""
    out: dict = {}
    for f in (filas or [])[:_MAX_FILAS]:
        if not isinstance(f, dict):
            continue
        nombre = _sa(f.get("ingredient_name"))
        if not nombre:
            continue
        try:
            q = float(f.get("quantity") or 0)
        except (TypeError, ValueError):
            q = 0.0
        if q <= 0:
            continue
        k = f"{nombre}|{_sa(f.get('unit') or 'unidad')}"
        out[k] = round(out.get(k, 0.0) + q, 4)
    return out


def mas_rica(ahora: dict, antes: dict) -> bool:
    """¿La Nevera de `ahora` tiene algo que la de `antes` no tenía (alimento, unidad o cantidad mayor)?"""
    if not isinstance(ahora, dict) or not isinstance(antes, dict):
        return True
    for k, q in ahora.items():
        previo = antes.get(k)
        if previo is None:
            return True
        try:
            if float(q) > float(previo) * 1.001 + 1e-6:
                return True
        except (TypeError, ValueError):
            return True
    return False


def lineas_refutadas(*resultados) -> list:
    """Unión ordenada y sin repetir de las líneas INEXISTENTES de los resultados de la guarda (texto de
    `validate_ingredients_against_pantry`); lo que no es un rechazo de existencia no aporta nada."""
    from compras_pequenas import faltantes
    out, vistos = [], set()
    for r in resultados:
        for linea in faltantes(r):
            k = _sa(linea)
            if k and k not in vistos:
                vistos.add(k)
                out.append(linea)
    return out[:_MAX_LINEAS]


def _edad_h(at) -> "float | None":
    try:
        t = datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0


def evidencia_vigente(plan_data, nevera_neta, user_id, country="DO") -> "dict | None":
    """La refutación que sigue valiendo para la Nevera de hoy, o None. Lanza sólo si falla el acceso a datos."""
    ev = plan_data.get(MARCA) if isinstance(plan_data, dict) else None
    if not isinstance(ev, dict):
        return None
    lineas = [x for x in (ev.get("lines") or []) if isinstance(x, str) and x.strip()]
    bruto_antes = ev.get("bruto")
    edad = _edad_h(ev.get("at"))
    if not lineas or not isinstance(bruto_antes, dict) or not bruto_antes or edad is None or edad > max_edad_h():
        return None
    filas = _leer_bruto(user_id)
    if filas is None:
        return None
    if mas_rica(huella_bruta(filas), bruto_antes):
        return None
    # La MISMA medida de la guarda de existencia contra la Nevera que verá el bloque (modo sonda, sin vector).
    from constants import validate_ingredients_against_pantry
    from compras_pequenas import tolerar
    veredicto = tolerar(validate_ingredients_against_pantry(
        lineas, list(nevera_neta or []), strict_quantities=False, country=country or "DO", probe_only=True))
    if veredicto is True:
        return None
    return ev


# ─────────────── los dos ganchos del worker ───────────────

def registrar(meal_plan_id, user_id, week_number, chunk_kind, resultados) -> bool:
    """La guarda de existencia agotó sus reintentos: deja la evidencia en el plan. Best-effort, nunca lanza."""
    try:
        if not activo() or not meal_plan_id or not user_id:
            return False
        lineas = lineas_refutadas(*(resultados or ()))
        if not lineas:
            return False
        filas = _leer_bruto(user_id)
        bruto = huella_bruta(filas)
        if not bruto:
            return False
        ev = {
            "v": 1,
            "at": datetime.now(timezone.utc).isoformat(),
            "week": int(week_number) if str(week_number).lstrip("-").isdigit() else week_number,
            "chunk_kind": chunk_kind,
            "lines": lineas,
            "bruto": bruto,
        }
        _escribir(
            "UPDATE meal_plans SET plan_data = jsonb_set(jsonb_set(COALESCE(plan_data, '{}'::jsonb), "
            "'{_pantry_refutation}', %s::jsonb), '{_plan_modified_at}', to_jsonb(NOW()::text)) "
            "WHERE id = %s AND user_id = %s",
            (json.dumps(ev, ensure_ascii=False), str(meal_plan_id), str(user_id)),
        )
        logger.info(f"🧊 [P1-PLAN-LOTE-747] refutación de la Nevera registrada plan={meal_plan_id} "
                    f"bloque={week_number}: {len(lineas)} línea(s) fuera ({', '.join(lineas[:6])}).")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-747] no se pudo registrar la refutación de plan={meal_plan_id}: "
                       f"{type(e).__name__}: {e} (sin evidencia, el próximo bloque corre como siempre).")
        return False


def prechequeo(*, task_id, user_id, meal_plan_id, week_number, chunk_kind, snap, form_data, plan_data,
               country=None) -> bool:
    """True ⇒ el bloque quedó en la pausa de siempre sin gastar LLM y el worker debe volver. False ⇒ sigue igual."""
    if not activo():
        return False
    fd = form_data if isinstance(form_data, dict) else {}
    nevera = fd.get("current_pantry_ingredients") or []
    if not nevera or not isinstance(plan_data, dict) or MARCA not in plan_data:
        return False
    try:
        import cron_tasks as _ct
        motivo = _ct._pantry_gate_waiver_reason(
            chunk_kind=chunk_kind,
            snapshot=snap,
            form_data=fd,
            fresh_inventory_source=fd.get("_fresh_pantry_source"),
        )
        if motivo:
            return False
        ev = evidencia_vigente(plan_data, nevera, user_id, country)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-747] pre-chequeo no concluyente plan={meal_plan_id} bloque={week_number}: "
                       f"{type(e).__name__}: {e} → el LLM corre como siempre.")
        return False
    if not ev:
        return False

    lineas = list(ev.get("lines") or [])
    edad = _edad_h(ev.get("at")) or 0.0
    logger.warning(
        f"🧊 [P1-PLAN-LOTE-747] plan={meal_plan_id} bloque={week_number}: la Nevera ya fue refutada por la guarda "
        f"(bloque {ev.get('week')}, hace {edad:.0f} h) y no ha cambiado; lo refutado sigue fuera "
        f"({', '.join(lineas[:6])}). Pausa directa `{MOTIVO}` SIN los {1 + int(getattr(_ct, 'CHUNK_PANTRY_MAX_RETRIES', 2))} "
        f"intentos del pipeline."
    )
    fd["_pantry_correction"] = (
        "ERRORES DE DESPENSA HALLADOS OBLIGANDO A CORREGIR:\n"
        f"- Ingredientes COMPLETAMENTE INEXISTENTES en inventario: {', '.join(lineas)}.\n"
    )[:1000]
    # El MISMO final que el worker tras agotar los reintentos (cron_tasks, rama `[P1-1] Violación persistente`).
    _ct._pause_chunk_for_pantry_refresh(task_id, user_id, week_number, fresh_inventory=nevera, reason=MOTIVO)
    try:
        _ct._dispatch_push_notification(
            user_id=user_id,
            title="Tu plan necesita revisión de ingredientes",
            body=(
                "No pudimos generar los próximos días con los "
                "ingredientes que tienes. Actualiza tu nevera para continuar."
            ),
            url="/dashboard",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-747] push no enviado plan={meal_plan_id}: {type(e).__name__}: {e}")
    _metrica(user_id=user_id, plan_id=str(meal_plan_id), week=week_number, refuted_week=ev.get("week"),
             age_h=round(edad, 1), lineas=len(lineas), chunk_kind=chunk_kind)
    return True


__all__ = ["MARCA", "MOTIVO", "activo", "max_edad_h", "huella_bruta", "mas_rica", "lineas_refutadas",
           "evidencia_vigente", "registrar", "prechequeo"]
