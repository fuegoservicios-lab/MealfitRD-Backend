# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-746 · 2026-09-28] La alerta de revisión fallida cuenta ENTREGAS, no corridas del pipeline.

`review_failed_delivered_rate_high` (P2-REVIEW-FAILED-RATE) contaba filas `pipeline_metrics.node='clinical_band'`, y esa
fila se emite al final de CADA corrida del pipeline (`_compute_pipeline_holistic_score_and_emit`), no en la entrega. El
worker de bloques re-corre el pipeline entero cuando la nevera no cuadra o el pickup falla, y el camino inicial igual
(`_run_pantry_validation_for_initial_chunk`); solo la última corrida llega al plan. Producción, 27-sep: el bloque 9 del
plan 3957a669 corrió a las 04:37 (aprobada), 04:43 (aprobada), 04:51 (rechazada) y 16:59 (rechazada) y se entregó UNA
vez, a las 17:02. La alerta del 28-sep 14:53 leyó 2 fallidas de 6 «entregas» (33 %); en entregas eran 1 de 3.

Ahora:
  · la fila `clinical_band` lleva `entrega` = {clave, plan_id, contexto, semana} (`clave_de_entrega`, desde el
    `_caller_target_plan_id` / `_caller_context` que ya estampan el worker, el chunk inicial y el JIT; sin plan, la
    correlación de la petición; sin nada, {} y la fila cuenta sola, como antes);
  · por clave se toma la ÚLTIMA corrida; si la clave es de un bloque de la cola (`semana`), solo cuenta si ese bloque
    está `completed` y la corrida es la última ANTES de completarse (un bloque que no terminó no entregó nada);
  · el fallback se mira en la corrida ENTREGADA (filtrarlo antes elegiría una corrida anterior que no se entregó).

Las filas sin `entrega` (anteriores al despliegue) cuentan una por fila hasta que salgan de la ventana. Solo lectura,
salvo el cierre de la alerta heredada (abajo). Knob `MEALFIT_REVFAIL_COUNT_DELIVERIES` (True): apagado, una por corrida
como antes (y ventana de 72 h).

[P1-PLAN-LOTE-746 · 2026-09-28]
  · VENTANA: prod completó 15 bloques en 14 días (≈3,2 por 72 h) contra un mínimo de 5 muestras: con 72 h la alerta
    casi nunca evaluaba. En modo entregas la ventana por defecto es 168 h (`lookback_por_defecto`); el knob
    `MEALFIT_REVFAIL_RATE_LOOKBACK_H` explícito sigue mandando.
  · ALERTA HEREDADA: la fila abierta el 28-sep la escribió el conteo por CORRIDAS (su metadata no trae `n_corridas`) y,
    con muestra insuficiente, el cron nunca la tocaba. `cerrar_alerta_heredada` la cierra si las entregas de la ventana
    (≥1) están bajo el umbral; una alerta abierta ya por entregas espera la muestra mínima, como siempre.
  · HORA DE ENTREGA: `learning_persisted_at` (T2 del worker y el chunk inicial la estampan en el MISMO UPDATE que
    `status='completed'`); `updated_at` lo mueven también la GC de snapshots y `reservation_status`. Y un margen de
    2 min: la fila `clinical_band` va por `_METRICS_EXECUTOR` (en cola) y su `created_at` puede caer tras la compleción.
  · SOLO BLOQUES LLM: un bloque completado por shuffle/edge/emergencia (`quality_tier`) no entregó la corrida LLM; no se
    le atribuye su revisión. El chunk inicial no estampa `quality_tier` (NULL = llm).
  · LÍMITE CONOCIDO: `plan_chunk_queue.meal_plan_id` es ON DELETE CASCADE — si se borra el plan o la cuenta, sus
    entregas salen de la cuenta (por eso un replay desde el journal da más entregas que las filas `completed` de hoy).
  · TRANSICIÓN TRAS DESPLEGAR (revisión 2): las filas `clinical_band` sin `entrega` (anteriores al despliegue) siguen
    contando UNA POR CORRIDA hasta salir de la ventana, y con la ventana de 168 h eso son hasta 7 DÍAS, no 3: durante
    esa semana la tasa puede salir inflada por los reintentos (el caso del 27-sep: 4 corridas, 1 entrega). SOP: una
    alerta `review_failed_delivered_rate_high` en los 7 días siguientes al despliegue se lee con el `n_corridas` de su
    metadata — si `n_corridas` supera con mucho a `n_delivered`, es el conteo heredado; no se toca el umbral.
tooltip-anchor: P1-PLAN-LOTE-746-ENTREGAS
"""
from __future__ import annotations

import logging
import re
from datetime import timedelta

logger = logging.getLogger(__name__)

_SEMANA_RX = re.compile(r"^chunk_worker:week_(\d+)$")
_MARGEN_COLA_METRICAS = timedelta(minutes=2)

_SQL_CORRIDAS = """
    SELECT created_at,
           metadata->>'review_passed' AS review_passed,
           COALESCE(metadata->>'delivered_was_fallback', 'false') AS fallback,
           metadata->'entrega' AS entrega
      FROM pipeline_metrics
     WHERE node = 'clinical_band'
       AND created_at > NOW() - (%s || ' hours')::interval
"""
_SQL_COMPLETADOS = """
    SELECT meal_plan_id::text AS plan_id, week_number AS semana, updated_at, quality_tier,
           COALESCE(learning_persisted_at, updated_at) AS completado_en
      FROM plan_chunk_queue
     WHERE status = 'completed'
       AND COALESCE(quality_tier, 'llm') = 'llm'
       AND COALESCE(learning_persisted_at, updated_at) > NOW() - (%s || ' hours')::interval
"""
# La fila abierta por el conteo por CORRIDAS (anterior a este lote) no trae `n_corridas` en su metadata.
_SQL_CERRAR_HEREDADA = """
    UPDATE system_alerts SET resolved_at = NOW()
     WHERE alert_key = %s AND resolved_at IS NULL
       AND NOT (COALESCE(metadata, '{}'::jsonb) ? 'n_corridas')
"""


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_REVFAIL_COUNT_DELIVERIES", True)
    except Exception:                                                          # noqa: BLE001
        return True


def lookback_por_defecto() -> int:
    """Ventana por defecto del cron: 168 h contando entregas (hay pocas), 72 h contando corridas (como antes)."""
    return 168 if activo() else 72


def cerrar_alerta_heredada(alert_key, entregas, fallidas, umbral, escribir) -> bool:
    """Con muestra insuficiente: cierra la alerta abierta por el conteo por CORRIDAS si las entregas de la ventana (≥1)
    están bajo el umbral. `escribir` = el `execute_sql_write` del cron. Devuelve si lanzó el UPDATE; nunca lanza."""
    try:
        if not activo() or int(entregas) < 1 or int(fallidas) > float(umbral) * int(entregas):
            return False
        escribir(_SQL_CERRAR_HEREDADA, (alert_key,))
        logger.info(f"✅ [P1-PLAN-LOTE-746] `{alert_key}`: {fallidas}/{entregas} entregas bajo el umbral → se cierra "
                    f"la fila heredada del conteo por corridas si sigue abierta (una abierta por entregas espera la "
                    f"muestra mínima).")
        return True
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-746] cierre de la alerta heredada no-op: {type(e).__name__}: {e}")
        return False


def clave_de_entrega(form_data) -> dict:
    """La identidad de lo que esta corrida entregaría: {clave, plan_id, contexto, semana}, o {} si no hay con qué."""
    try:
        fd = form_data if isinstance(form_data, dict) else {}
        ctx = str(fd.get("_caller_context") or "initial_generate")
        m = _SEMANA_RX.match(ctx)
        semana = 1 if ctx == "chunk_worker:initial" else (int(m.group(1)) if m else None)
        pid = fd.get("_caller_target_plan_id")
        if pid:
            return {"clave": f"{pid}:{ctx}", "plan_id": str(pid), "contexto": ctx, "semana": semana}
        import correlation
        corr = correlation.get_correlation_id()
        if corr and str(corr).strip() not in ("", "-"):
            return {"clave": f"corr:{corr}:{ctx}", "plan_id": None, "contexto": ctx, "semana": None}
        return {}
    except Exception:                                                          # noqa: BLE001
        return {}


def _es(valor, esperado: str) -> bool:
    return str(valor if valor is not None else "").strip().lower() == esperado


def contar_entregas(corridas, completados, por_entrega=None) -> dict:
    """Puro. `corridas`: filas {created_at, review_passed, fallback, entrega}; `completados`: {plan_id, semana,
    updated_at}. Devuelve {entregas, fallidas, corridas, modo} — entregas y fallidas SIN fallback."""
    if por_entrega is None:
        por_entrega = activo()
    filas = [c for c in (corridas or []) if isinstance(c, dict)]
    if not por_entrega:
        elegidas = filas
    else:
        grupos, elegidas = {}, []
        for c in filas:
            e = c.get("entrega") if isinstance(c.get("entrega"), dict) else {}
            if e.get("clave"):
                grupos.setdefault(str(e["clave"]), []).append(c)
            else:
                elegidas.append(c)                       # sin clave (legado): una por fila, como antes
        cierres = {}
        for q in completados or []:
            try:
                if str(q.get("quality_tier") or "llm").strip().lower() != "llm":
                    continue                             # shuffle/edge/emergencia: no se entregó la corrida LLM
                t = q.get("completado_en") or q.get("updated_at")
                cierres.setdefault((str(q.get("plan_id")), int(q.get("semana"))), []).append(t)
            except (TypeError, ValueError):
                continue
        for cs in grupos.values():
            try:
                cs = sorted(cs, key=lambda c: c.get("created_at"))
                e = cs[-1]["entrega"]
                if e.get("semana") is None:              # sin cola que consultar (JIT, SSE): la última corrida
                    elegidas.append(cs[-1])
                    continue
                vistas = set()
                for t in sorted(x for x in cierres.get((str(e.get("plan_id")), int(e["semana"])), []) if x is not None):
                    previas = [c for c in cs
                               if c.get("created_at") is not None and c["created_at"] <= t + _MARGEN_COLA_METRICAS]
                    if previas and id(previas[-1]) not in vistas:
                        vistas.add(id(previas[-1]))
                        elegidas.append(previas[-1])
            except Exception as ex:                                            # noqa: BLE001
                logger.debug(f"[P1-PLAN-LOTE-746] grupo no contado: {type(ex).__name__}: {ex}")
    vivas = [c for c in elegidas if not _es(c.get("fallback"), "true")]
    return {"entregas": len(vivas), "fallidas": sum(1 for c in vivas if _es(c.get("review_passed"), "false")),
            "corridas": len(filas), "modo": "entregas" if por_entrega else "corridas"}


def contar_entregas_revisadas(lookback_h) -> tuple:
    """(entregas, entregas_con_revision_fallida, corridas) de las últimas `lookback_h` horas. Solo SELECT; (0, 0, 0)
    si la base falla (el cron lo registra como muestra insuficiente, nunca como alerta)."""
    try:
        import db_core
        h = str(int(lookback_h))
        corridas = db_core.execute_sql_query(_SQL_CORRIDAS, (h,), fetch_all=True) or []
        completados = (db_core.execute_sql_query(_SQL_COMPLETADOS, (h,), fetch_all=True) or []) if activo() else []
        r = contar_entregas(corridas, completados)
        return r["entregas"], r["fallidas"], r["corridas"]
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-746] conteo de entregas revisadas no disponible: {type(e).__name__}: {e}")
        return 0, 0, 0


__all__ = ["clave_de_entrega", "contar_entregas", "contar_entregas_revisadas", "activo", "lookback_por_defecto",
           "cerrar_alerta_heredada"]
