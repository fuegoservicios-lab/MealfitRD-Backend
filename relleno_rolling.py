# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-811 · 2026-09-29] El relleno de un plan rolling: UNA decisión y UN snapshot para el cron y `/shift-plan`.

Qué estaba roto (medido el 28-sep, solo SELECT y journal):
  1. `/shift-plan` detectaba el «gap huérfano» de un plan de 7 días (0 bloques vivos, ventana incompleta) y SOLO lo
     escribía en el log («Habilitando rolling refill de recuperación»): esa rama era un `elif` de la misma cadena que la
     que encola, así que se quedaba con el caso y el relleno no salía nunca (commit 29889c97, P0-4). Como el shift
     reescribe el ancla a hoy, en un plan de 7 días `days_remaining` no llega a 0 y la renovación semanal no corre: el
     gap es la ÚNICA vía de relleno. Hoy no tiene víctimas (la única usuaria de 7 días no usa el chat, así que la
     rellena el cron de inactivos), pero un usuario semanal que abre el chat quedaba fuera del cron y su plan se vaciaba.
  2. El snapshot de TODA renovación (`rolling_refill`, cron y HTTP) era `{**health_profile}`: la política efectiva del
     plan (`plan_data._plan_policy.effective`) sólo se inyecta al CREAR el plan (`generation_inputs.py`,
     `routers/plans.py::_horizon_inject` / `_enqueue_remaining_chunks`). Las renovaciones de whitney (7 bloques desde el
     08-sep) salieron sin anclas, sin banda de recurrencia, sin compra única y sin el bloque 📐 — en silencio, y sin
     métrica de fidelidad (la revisión sale antes de medir cuando no hay política).
  3. El HTTP no ponía `_is_continuation` / `_continuation_anchor_iso` (el cron sí, P1-7): sin ellas el gate temporal
     cae en la fórmula antigua de `prev_end`, la de la tormenta de aplazamientos, sólo tapada por un knob.

`decidir(...)` es la condición pura y compartida; `snapshot_relleno(...)` arma el snapshot del bloque de renovación con
las marcas de continuación y la política. Los dos call sites (cron `_background_shift_plan_for_user`, HTTP
`api_shift_plan`) la usan tanto para el relleno de la ventana como para la renovación semanal (P0-1).

Por qué NO viaja la rebanada del blueprint (`_blueprint_slice`): la rebanada se indexa por día del CICLO (0..N-1 del run)
y la cola del relleno cuenta días desde el ancla MÓVIL (el shift la reescribe a hoy; `_shift_days_accumulated` sólo lo
lleva el HTTP). No hay un mapeo fiable día-del-ciclo ↔ offset del relleno, y uno inventado (p. ej. módulo 7) repartiría
anclas y candidatos del registry en días equivocados. El relleno lleva la política como la llevan el swap y la
regeneración de día (`horizon.attach_policy_to_swap_form`): política del plan vivo, sin rebanada. Los validadores ya
soportan esa forma (`fidelity_issues` escala la banda a la ventana; `policy_prompt_block` omite el reparto por día). El
worker recalcula `_policy_enforced` al ejecutar también en ese caso (`cron_tasks`, tooltip-anchor
P1-PLAN-LOTE-811-ENFORCE).

Knobs:
  · `MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL` (True): `/shift-plan` rellena el gap huérfano de 7 días con la misma condición
    del cron (bloques vivos ⇒ no), y sus snapshots llevan las marcas de continuación. False ⇒ la conducta anterior del
    HTTP entera (el gap sólo se registra, los planes de 15/30 días no miran los bloques vivos, sin marcas).
  · `MEALFIT_REFILL_CARRIES_POLICY` (True): el snapshot de la renovación lleva la política efectiva del plan. Default
    True porque (a) es la política con la que NACIÓ el plan y la que el usuario vio en el panel «solicitaste /
    aplicamos»: perderla en la renovación es exactamente la «modificación silenciosa» que F2 prohíbe, y F3 dice que la
    renovación HEREDA la política; (b) no la recompila: si `MEALFIT_PLAN_POLICY_MODE=off`, `effective_policy_for_plan`
    devuelve None y no viaja nada, igual que en la creación; (c) queda gemela de la CREACIÓN también en `shadow`:
    ahí el bloque 📐 queda vacío (`policy_prompt_block` exige enforce), pero NO es «sólo medición», porque los
    consumidores de compra única leen `_plan_policy_effective` SIN mirar `_policy_enforced` —
    `compra_unica.nevera_virtual` (Nevera virtual en los bloques 2+), `compra_unica.candidatos_del_dia` y
    `ai_helpers._age_pantry_for_block` / `ai_helpers._single_trip_durable_filter` (sembrador filtrado por
    durabilidad)—, así que llevarla cambia lo que se genera en un plan de compra única. Es lo mismo que ya pasa al
    crear el plan en `shadow` (`horizon.inject_policy_into_pipeline_data` inyecta la política en todo modo ≠ off).
    Producción está en `enforce` con gate `warn` (las 8 últimas filas de `plan_policy_fidelity`, 29-sep).
    False ⇒ el snapshot de antes, byte a byte.

Aproximación conocida — el día del ciclo [P1-PLAN-LOTE-817, corregida por el revisor del 811]: el bloque lleva
`_days_offset` contado desde el ancla MÓVIL, no desde el día del ciclo de compra. El defecto no nace aquí: ya existe en
todo el sistema, porque el shift re-ancla también la cola pendiente (el rebase de offsets,
`constants.rebase_pending_chunk_offsets`), así que todo bloque —creación, renovación semanal (P0-1), relleno por gap—
cuenta desde la ventana viva, desplazado por los días archivados. Llevar la política lo EXTIENDE a las renovaciones y
rellenos de 15/30 días con compra única, que antes no activaban nada: los filtros de durabilidad
(`single_trip_requirements` en `_single_trip_durable_filter`, `_age_pantry_for_block`, `candidatos_del_dia`) actúan
SIEMPRE y evalúan un día temprano (el 3 cuando es el 20: exigencia nula o laxa); `compra_unica.nevera_virtual` sólo con
la Nevera real vacía o apagada, y entonces ofrece frescos del día 1 que ya no están (replay 29-sep, 6594aae1:
`_days_offset` 5 frente al día 10, 13 frescos que no llegan). El default True se sostiene: la renovación queda igual
que la creación (mismos consumidores, mismo índice desplazado); anclas, banda, presupuesto y bloque 📐 sí son
correctos. Arreglo en el lote 816: el día del ciclo a esos consumidores (`_blueprint_slice.days_offset`, o inicio del
ciclo + días archivados). tooltip-anchor: P1-PLAN-LOTE-811-DIA-DEL-CICLO

Presupuesto duro y `waiting_user` en una renovación: `budget_below_floor` (`action=waiting_user`) es una relajación que
el COMPILADOR emite al crear el plan y que sólo consume el formulario (`frontend/src/config/planPolicy.js`,
`relaxationIsBlocking`: CTA antes de gastar el crédito). En el backend nada convierte esa acción en una pausa: el
estado `pending_user_action` de un bloque lo ponen únicamente las guardas de Nevera. Aquí además no se recompila: el
relleno lleva la MISMA política (mismo `policy_hash`) con la que ya se generaron los bloques anteriores, con su
`budget.status` tal cual (salvo las anclas que choquen con el perfil VIGENTE, que se retiran y cambian el hash:
`anclas_vigentes`, P1-PLAN-LOTE-817). Una renovación con la política no puede, por tanto, quedar en `waiting_user` ni pausarse por
el presupuesto; lo que cambia es que el precio del registry y el tier (`budget.tier`) vuelven a guiar los candidatos
como en la creación. Si algún día el backend hiciera cumplir `waiting_user`, este sería el punto a revisar: la
renovación no pasa por el formulario y el usuario no tendría dónde resolverlo.

Doc: backend/docs/plan_policy_f3.md (sección «Renovación»). Tests: tests/test_p1_plan_lote_811.py y
tests/test_p1_plan_lote_817.py (anclas frente al perfil vigente, `anclas_vigentes`).
tooltip-anchor: P1-PLAN-LOTE-811
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Estados en los que hay una generación en curso: ni el cron ni el HTTP rellenan encima (gemelo de `is_partial`).
EN_GENERACION = frozenset({"partial", "generating_next"})


def gap_7d_http_activo() -> bool:
    """tooltip-anchor: MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL", True)
    except Exception:
        return True


def lleva_politica() -> bool:
    """tooltip-anchor: MEALFIT_REFILL_CARRIES_POLICY"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_REFILL_CARRIES_POLICY", True)
    except Exception:
        return True


def decidir(total_dias: int, dias_visibles: int, dias_restantes: int, ventana_necesaria: int, bloques_vivos: int,
            status: Optional[str], *, gap_7d: bool = True) -> tuple[bool, str]:
    """¿Hay que encolar el relleno de la ventana de un plan rolling? (encolar, motivo). Pura, la misma para cron y HTTP.

    `dias_visibles` son los que quedan TRAS el shift; `ventana_necesaria` = min(tamaño del bloque siguiente, días
    restantes). La renovación de un plan expirado (`dias_restantes == 0`) es otra rama y aquí da `sin_dias_restantes`.
    Un plan en `partial`/`generating_next` no se rellena (hay generación en curso): un plan de 7 días en `partial` con 0
    bloques vivos queda sin relleno por las dos vías — hueco conocido, fuera de este lote.
    `gap_7d=False` reproduce el HTTP de antes: el gap de 7 días no se rellenaba."""
    if str(status or "") in EN_GENERACION:
        return False, "plan_en_generacion"
    if int(dias_restantes or 0) <= 0:
        return False, "sin_dias_restantes"
    if int(dias_visibles or 0) >= int(ventana_necesaria or 0):
        return False, "ventana_completa"
    if int(bloques_vivos or 0) > 0:
        return False, "bloques_vivos"
    if int(total_dias or 0) == 7:
        return (True, "gap_huerfano_7d") if gap_7d else (False, "gap_7d_apagado")
    return True, "ventana_incompleta"


def politica_del_plan(plan_data: Optional[dict]) -> Optional[dict]:
    """La política efectiva persistida del plan (`_plan_policy.effective`), o None (motor apagado / plan sin política).
    NO se recompila desde el perfil: el formulario de la renovación no es el del wizard."""
    try:
        from horizon import effective_policy_for_plan
        eff = effective_policy_for_plan(plan_data if isinstance(plan_data, dict) else {}, None)
        return eff if isinstance(eff, dict) and eff else None
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-811] política del plan no disponible: {e}")
        return None


def snapshot_relleno(*, hp: dict, user_id: str, chunk_count: int, ancla_iso: Optional[str], plan_data: Optional[dict],
                     previous_meals: list, triggered_by: Optional[str] = None, semanal: bool = False,
                     form_extra: Optional[dict] = None, continuacion: bool = True) -> dict:
    """El `pipeline_snapshot` de un bloque `rolling_refill` (relleno de la ventana o renovación semanal).

    `ancla_iso` es el ancla vigente tras el shift (o el inicio de la renovación): va en `_plan_start_date` y, con
    `continuacion`, en `_continuation_anchor_iso` (P1-7: el gate usa `ancla − 1 día` como fin del bloque previo cuando
    éste es del plan original). `form_extra` añade la Nevera de la renovación semanal."""
    form = {**(hp or {}), "user_id": user_id, "totalDays": chunk_count, "_plan_start_date": ancla_iso}
    if form_extra:
        form.update(form_extra)
    if continuacion:
        form["_is_continuation"] = True
        form["_continuation_anchor_iso"] = ancla_iso
    if lleva_politica():
        # Siempre server-side: lo que traiga el perfil no decide la política que obedece el motor.
        for k in ("_blueprint_slice", "_plan_policy_effective", "_policy_enforced", "_policy_day_index"):
            form.pop(k, None)
        eff = politica_del_plan(plan_data)
        if eff:
            # [P1-PLAN-LOTE-817 · 2026-09-29] las anclas congeladas frente a las alergias, la dieta y los rechazos
            # VIGENTES, con la puerta del guard (sin conflicto: el mismo objeto, snapshot idéntico al del 811).
            # Knob MEALFIT_REFILL_POLICY_ANCHORS_RECHECK. tooltip-anchor: P1-PLAN-LOTE-817
            from anclas_vigentes import anclas_vigentes, recheck_activo
            if recheck_activo():
                eff, _ = anclas_vigentes(eff, hp, user_id=user_id)
        if eff:
            form["_plan_policy_effective"] = eff
            try:
                from horizon import policy_enforced
                form["_policy_enforced"] = bool(policy_enforced(user_id))
            except Exception:
                form["_policy_enforced"] = False
    snapshot = {
        "form_data": form,
        "taste_profile": "",
        "memory_context": "",
        "previous_meals": list(previous_meals or []),
        "totalDays": chunk_count,
        "_is_rolling_refill": True,
    }
    if semanal:
        snapshot["_is_weekly_renewal"] = True
    if triggered_by:
        snapshot["_triggered_by"] = triggered_by
    return snapshot
