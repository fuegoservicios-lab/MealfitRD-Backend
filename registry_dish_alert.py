# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-814 · 2026-09-29] La alerta `registry_dishes_unused`, contada por ENTREGAS y sin mentir por omisión.

El cron `cron_tasks._registry_dish_rate_alert_job` (P1-FIDELIDAD-PLATO-DEL-REGISTRY) sumaba filas de
`pipeline_metrics` node=`plan_policy_fidelity`. Medido el 28-sep con SELECT sobre producción:

  · **Una fila por intento de revisión, no por entrega.** La alerta abierta desde el 26-sep 11:28 UTC («1/76 platos
    en 72 h, n_runs 5») eran 3 entregas: 2 de una cuenta admin y UNA de dbrito contada tres veces por sus
    reintentos (rebanada `150515fd`). Deduplicada, con n=3 < 5, no habría saltado.
  · **Juzgaba «aplicables»**, el rendimiento de una costura (`apply_library_recipe`) que no tiene llamadores en
    producción. La pregunta que sí tiene respuesta es la PROCEDENCIA: ¿el plato salió del catálogo?
  · **La población era sobre todo el dueño.** Las cuentas de `MEALFIT_ADMIN_USER_IDS` no son la flota.
  · **Sin muestra, la alerta quedaba abierta con un dato viejo** y nada lo decía. Resolverla por antigüedad la
    dejaría muda (deduplicada y a 72 h sólo habría sido evaluable 66 horas en 20 días) y reabriría el punto ciego
    que P1-FIDELIDAD-PLATO-DEL-REGISTRY cerró. «No concluyente» no colapsa a ningún lado: ni abre ni cierra; la
    alerta abierta se marca `stale` y guarda `last_evaluable_at` de la última vez que SÍ se pudo evaluar.

Este módulo cuenta y decide; el `alert_key` y el tick siguen en `cron_tasks` (su SSOT y el escáner de
`test_p2_audit_4_alert_keys_documented`). Las funciones SQL llegan como parámetro: el cron le pasa las suyas.
Knob `MEALFIT_REGISTRY_PROVENANCE_V2` (en `recipe_library.provenance_v2_enabled`); apagado, el cron de antes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

COUNTING = "v2_entrega_plan_rebanada"

# `as_of` NULL ⇒ NOW(): el cron pasa None; el replay pasa cada hora del pasado y corre ESTE mismo SQL.
SQL_FILAS_V2 = """
SELECT id, created_at, user_id,
       metadata->>'plan_id'    AS plan_id,
       metadata->>'slice_hash' AS slice_hash,
       COALESCE((metadata->>'registry_dishes_total')::int, 0)      AS total,
       COALESCE((metadata->>'registry_dishes_matched')::int, 0)    AS matched,
       COALESCE((metadata->>'registry_dishes_applicable')::int, 0) AS applicable
  FROM pipeline_metrics
 WHERE node = 'plan_policy_fidelity'
   AND created_at >  COALESCE(%s::timestamptz, NOW()) - (%s || ' hours')::interval
   AND created_at <= COALESCE(%s::timestamptz, NOW())
   AND COALESCE(metadata->>'registry_in_prompt', 'false') = 'true'
   AND COALESCE((metadata->>'registry_dishes_total')::int, 0) > 0
 ORDER BY created_at, id
 LIMIT 50000
"""


def _momento(v) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def contar_entregas(filas: list, admin_ids) -> dict:
    """Una entrega = (plan_id, slice_hash); de sus reintentos cuenta el ÚLTIMO, que es el entregado. Una fila
    sin `plan_id` no se puede deduplicar y cuenta sola (queda dicho en `sin_plan`)."""
    admin = {str(x).strip().lower() for x in (admin_ids or ()) if str(x).strip()}
    propias, n_admin, sin_plan = [], 0, 0
    for f in (filas or []):
        if not isinstance(f, dict):
            continue
        if str(f.get("user_id") or "").strip().lower() in admin:
            n_admin += 1
            continue
        propias.append(f)
    entregas: dict = {}
    for f in sorted(propias, key=lambda r: (_momento(r.get("created_at")), _int(r.get("id")))):
        pid = f.get("plan_id")
        if not pid:
            sin_plan += 1
        entregas[(str(pid) if pid else f"fila:{f.get('id')}", str(f.get("slice_hash") or ""))] = f
    vals = list(entregas.values())
    return {"filas": len(propias), "filas_admin": n_admin, "sin_plan": sin_plan, "entregas": len(vals),
            "platos": sum(_int(f.get("total")) for f in vals),
            "del_catalogo": sum(_int(f.get("matched")) for f in vals),
            "aplicables": sum(_int(f.get("applicable")) for f in vals)}


def decidir(cuenta: dict, *, min_samples: int, floor: float) -> tuple:
    """(`insuficiente`|`alerta`|`resuelve`, tasa de procedencia o None). Pura: la usa también el replay."""
    if _int(cuenta.get("entregas")) < min_samples or _int(cuenta.get("platos")) <= 0:
        return "insuficiente", None
    tasa = round(_int(cuenta.get("del_catalogo")) / _int(cuenta.get("platos")), 3)
    return ("alerta" if tasa < floor else "resuelve"), tasa


def run_v2(alert_key: str, query: Callable, write: Callable, *, lookback_h: int, min_samples: int,
           floor: float, as_of: Optional[str] = None) -> dict:
    """Cuenta, decide y actúa sobre `system_alerts`. Devuelve lo que el tick del cron publica."""
    try:
        from admin_acceso import admin_ids
        ids = sorted(admin_ids())
    except Exception as e:
        # Sin la lista no se sabe qué filas son del dueño: evaluar sería evaluar la cuenta equivocada.
        logger.warning(f"⚠️ [P1-PLAN-LOTE-814] sin MEALFIT_ADMIN_USER_IDS legible, no se evalúa: {e!r}")
        return {"n": 0, "rate": None, "alert_emitted": False, "skip": "admin_ids_unavailable",
                "tick": {"provenance_v2": True, "counting": COUNTING}}
    filas = query(SQL_FILAS_V2, (as_of, str(lookback_h), as_of), fetch_all=True) or []
    c = contar_entregas(filas, ids)
    decision, tasa = decidir(c, min_samples=min_samples, floor=floor)
    ahora = datetime.now(timezone.utc).isoformat()
    tick = {"provenance_v2": True, "counting": COUNTING, "decision": decision, "n_deliveries": c["entregas"],
            "n_rows": c["filas"], "n_rows_admin_excluded": c["filas_admin"], "n_rows_sin_plan": c["sin_plan"],
            "n_dishes": c["platos"], "n_from_registry": c["del_catalogo"], "n_applicable": c["aplicables"],
            "applicable_rate": (round(c["aplicables"] / c["platos"], 3) if c["platos"] else None),
            "min_samples": min_samples}
    out = {"n": c["entregas"], "rate": tasa, "alert_emitted": False, "skip": None, "tick": tick}
    if decision == "insuficiente":
        out["skip"] = f"insufficient_samples ({c['entregas']}<{min_samples} entregas)"
        # `verdict_counting` dice qué conteo emitió el veredicto que sigue abierto: el de antes de este lote no
        # llevaba `counting` (filas, no entregas) y un v2 sin muestra no puede confirmarlo NI desmentirlo.
        write(
            "UPDATE system_alerts SET metadata = COALESCE(metadata, '{}'::jsonb) "
            "|| jsonb_build_object('verdict_counting', COALESCE(metadata->>'counting', 'v1_filas')) || %s::jsonb "
            "WHERE alert_key = %s AND resolved_at IS NULL",
            (json.dumps({"stale": True, "stale_reason": "insufficient_samples", "stale_checked_at": ahora,
                         "stale_n_deliveries": c["entregas"], "stale_min_samples": min_samples,
                         "stale_lookback_h": lookback_h}, ensure_ascii=False), alert_key),
        )
        logger.info(f"[P1-PLAN-LOTE-814] {c['entregas']}/{min_samples} entregas en {lookback_h}h: no concluyente, "
                    f"la alerta abierta (si la hay) queda stale, ni se abre ni se cierra")
        return out
    if decision == "resuelve":
        write("UPDATE system_alerts SET resolved_at = NOW() WHERE alert_key = %s AND resolved_at IS NULL",
              (alert_key,))
        logger.info(f"✅ [P1-PLAN-LOTE-814] {int(tasa * 100)}% de platos del catálogo "
                    f"({c['del_catalogo']}/{c['platos']}, {c['entregas']} entregas) ≥ piso {int(floor * 100)}%")
        return out
    meta = {"counting": COUNTING, "registry_provenance_rate": tasa, "n_deliveries": c["entregas"],
            "n_rows": c["filas"], "n_rows_admin_excluded": c["filas_admin"], "n_dishes": c["platos"],
            "n_from_registry": c["del_catalogo"], "n_applicable": c["aplicables"], "floor": floor,
            "lookback_h": lookback_h, "min_samples": min_samples, "last_evaluable_at": ahora, "stale": False}
    write(
        """
        INSERT INTO system_alerts
            (alert_key, alert_type, severity, title, message, metadata)
        VALUES (%s, 'reliability_degradation', 'warning', %s, %s, %s::jsonb)
        ON CONFLICT (alert_key) DO UPDATE
        SET triggered_at = NOW(), message = EXCLUDED.message,
            metadata = EXCLUDED.metadata, resolved_at = NULL
        """,
        (alert_key, "El catálogo de platos viaja en el prompt y no se usa",
         f"Con la biblioteca ENCENDIDA, sólo el {int(tasa * 100)}% de los platos servidos "
         f"({c['del_catalogo']}/{c['platos']} en {c['entregas']} entregas, {lookback_h}h, sin cuentas admin) "
         f"sale del catálogo — bajo el piso {int(floor * 100)}%. De ellos, {c['aplicables']} traen además los "
         f"alimentos de su plantilla (dato informativo: la receta curada aún no tiene llamador).",
         json.dumps(meta, ensure_ascii=False)),
    )
    out["alert_emitted"] = True
    logger.warning(f"🚨 [P1-PLAN-LOTE-814] {int(tasa * 100)}% de platos del catálogo ({c['del_catalogo']}/"
                   f"{c['platos']}, {c['entregas']} entregas, {lookback_h}h) < piso {int(floor * 100)}% "
                   f"→ alert `{alert_key}`")
    return out
