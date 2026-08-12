# plan_mode.py
"""[P1-PLAN-MODE · 2026-08-11] El interruptor de generación de planes — SSOT.

Hay usuarios que quieren la app como contador de macros y recomendador, sin que la IA
les genere planes. `plan_mode` en `user_profiles` es su postura: `'plan'` (default) o
`'tracking'`. Este módulo es el ÚNICO sitio que la cambia, porque cambiarla no es
editar un escalar: es una transacción que cancela filas de `plan_chunk_queue`, libera
locks y estampa el estado del plan. Una regla que vive en dos puertas se cumple en una
(por eso `plan_mode` NO entra en `_PROFILE_SCALAR_WHITELIST`).

── POR QUÉ DOS CAPAS ────────────────────────────────────────────────────────────────
El pickup de `process_plan_chunk_queue` no hace JOIN con `meal_plans`: lo único que
decide si un chunk gasta LLM es su propia fila. El precedente `_frozen_at` demuestra
que un flag que el pickup no lee no apaga nada. Por eso:

  1. El GATE del pickup (`P1-PLAN-MODE-PICKUP-GATE`, cron_tasks.py) — la capa que
     sobrevive a CUALQUIER cron que encole detrás del apagado.
  2. La cancelación de la cola, aquí — para que la cola no quede «pendiente para
     siempre» acumulando alertas.

── POR QUÉ `cancelled` Y NO `pending_user_action` ──────────────────────────────────
No es hipótesis: la bandera del congelado YA existe en producción y no apaga nada.
`_recover_pantry_paused_chunks` corre cada minuto, selecciona TODOS los
`pending_user_action` sin filtrar por plan, y a las 12 horas los resucita
(`_activate_flexible_mode` + `status='pending'`). El propio repo lo escribe
(cron_tasks.py: «un pausado NO está muerto — el recovery cron lo resucita»).
`failed` tampoco sirve: `recover_failed_chunks_long_plans` lo re-encola sin mirar
`generation_status`. `cancelled` es el único TERMINAL DURO que ningún recovery toca.

Estados que se cancelan: `pending`, `processing`, `stale`, `pending_user_action` y
`failed`. Dejar fuera `pending_user_action` sería P1-CHUNK-REBASE-PAUSED con otra
ropa. NO se tocan `completed` (días ya entregados) ni filas con `dead_lettered_at`
(el forense las necesita).

── EL ORDEN ES LA MITAD DEL DISEÑO ─────────────────────────────────────────────────
Al pausar: LA BANDERA PRIMERO. Si el proceso muere a mitad, la cola queda viva pero
INERTE (el gate ya la bloquea). Al revés, el bg-refill reencola en ≤4 h y el plan
resucita solo. Al reanudar: bandera primero también (encolar con el gate puesto deja
chunks que el pickup ignora), y el snapshot del estado se restaura ANTES de cualquier
shift — `api_shift_plan` decide su rama leyendo `generation_status`.

`generation_status='paused_by_user'` silencia ~7 crons SIN editarlos: todos filtran
por lista blanca de estados, y un valor fuera de la lista los deja inertes. El
snapshot `_paused_prev_generation_status` existe porque reanudar a un `'complete'`
literal con `days=[]` violaría el CHECK I8.
"""
from __future__ import annotations

import logging
import os

from db import execute_sql_query, execute_sql_write

logger = logging.getLogger("mealfit")

# Kill switch del feature completo. False ⇒ el gate del pickup no se inyecta y estos
# helpers se vuelven no-op: todo se comporta como antes de P1-PLAN-MODE.
PLAN_MODE_SWITCH_ENABLED = str(os.environ.get("MEALFIT_PLAN_MODE_SWITCH", "true")).strip().lower() in ("1", "true", "yes", "on")


def _resume_max_days() -> int:
    """Días de pausa tras los cuales reanudar NO auto-shiftea (el plan venció y su
    ventana temporal ya no significa nada). Clamp [1, 365]."""
    try:
        v = int(os.environ.get("MEALFIT_PLAN_PAUSE_MAX_RESUME_DAYS", "30"))
    except (TypeError, ValueError):
        v = 30
    return max(1, min(365, v))


# El fragmento del gate, como CONSTANTE — cero entrada de usuario en el SQL. Se
# inyecta en las DOS ramas del pickup y en el bg-refill cuando el knob está on.
PICKUP_GATE_SQL = """
                -- [P1-PLAN-MODE-PICKUP-GATE · 2026-08-11] El freno de verdad. Este
                -- UPDATE es el ÚNICO sitio por el que pasa TODO el gasto LLM de
                -- chunks: run_plan_pipeline solo corre sobre lo que esta CTE
                -- devuelve. Un usuario en modo seguimiento no genera, encole quien
                -- encole detrás. Va aquí y no en un flag de plan_data porque el
                -- precedente _frozen_at demuestra que un flag que el pickup no lee
                -- no apaga nada.
                AND NOT EXISTS (
                    SELECT 1 FROM user_profiles up
                    WHERE up.id = q1.user_id AND up.plan_mode = 'tracking'
                )
"""

# Los cinco estados que el apagado cancela. Módulo-nivel para que el test los ancle.
CANCELLABLE_STATES = ("pending", "processing", "stale", "pending_user_action", "failed")

# [P1-RESUME-REVIVES-QUEUE · 2026-08-12] La firma que la pausa deja en
# `dead_letter_reason` de sus cancelados. Es EL contrato de la reanudación: solo
# las filas con esta firma exacta (y dead_lettered_at NULL) se reviven — un
# cancelled de cualquier otro origen sigue siendo terminal, como siempre.
PAUSE_CANCEL_REASON = "[P1-PLAN-MODE] paused_by_user"


def get_plan_mode(user_id: str) -> dict:
    row = execute_sql_query(
        "SELECT plan_mode, plan_mode_changed_at FROM user_profiles WHERE id = %s",
        (user_id,), fetch_one=True,
    ) or {}
    return {
        "plan_mode": row.get("plan_mode") or "plan",
        "plan_mode_changed_at": str(row.get("plan_mode_changed_at") or "") or None,
    }


def pause_plan_generation(user_id: str) -> dict:
    """Apagar. Idempotente: pausar dos veces no cancela nada nuevo ni re-estampa."""
    if not PLAN_MODE_SWITCH_ENABLED:
        return {"plan_mode": "plan", "skipped": "switch_off"}

    # 1. LA BANDERA PRIMERO (ver cabecera: el orden es la mitad del diseño).
    execute_sql_write(
        """
        UPDATE user_profiles
        SET plan_mode = 'tracking',
            plan_mode_changed_at = CASE WHEN plan_mode <> 'tracking' THEN NOW()
                                        ELSE plan_mode_changed_at END
        WHERE id = %s
        """,
        (user_id,),
    )

    # 2. Cancelar la cola. RETURNING para contar y para el log forense.
    #    [P1-RESUME-REVIVES-QUEUE · 2026-08-12] La pausa FIRMA sus cancelados en
    #    `dead_letter_reason` (dead_lettered_at queda NULL — no es dead-letter).
    #    Sin la firma, reanudar no sabía cuáles filas eran suyas: el primer ciclo
    #    real de pausa→reanuda dejó el plan en 3/30 PARA SIEMPRE, porque ningún
    #    otro mecanismo rellena un plan partial de un usuario activo (el catch-up
    #    del shift es solo no-partial, el bg-refill es solo inactivos ≥3 días, y
    #    el recovery ignora cancelled a propósito).
    cancelados = execute_sql_write(
        """
        UPDATE plan_chunk_queue
        SET status = 'cancelled', dead_letter_reason = %s, updated_at = NOW()
        WHERE user_id = %s
          AND status IN ('pending', 'processing', 'stale', 'pending_user_action', 'failed')
          AND dead_lettered_at IS NULL
        RETURNING id
        """,
        (PAUSE_CANCEL_REASON, user_id), returning=True,
    ) or []

    # 3. Locks. Sin FK a meal_plans: nadie los limpia por CASCADE (el DELETE del plan
    #    los borra a mano, plans.py). Un apagado que no lo haga deja locks zombi.
    execute_sql_write("DELETE FROM chunk_user_locks WHERE user_id = %s", (user_id,))

    # 4. Estampar el plan vivo. jsonb merge `||` (atómico, exento de I7), AND user_id
    #    (I2), y el SNAPSHOT del estado anterior para poder reanudar sin adivinar.
    execute_sql_write(
        """
        UPDATE meal_plans
        SET plan_data = plan_data || jsonb_build_object(
                'generation_status', 'paused_by_user',
                '_paused_at', NOW()::text,
                '_paused_prev_generation_status', plan_data->>'generation_status'
            ),
            updated_at = NOW()
        WHERE user_id = %s
          AND plan_data->>'generation_status' NOT IN ('abandoned', 'paused_by_user')
        """,
        (user_id,),
    )

    n = len(cancelados)
    logger.info(f"⏸ [P1-PLAN-MODE] user {user_id}: pausa — {n} chunk(s) cancelados, locks liberados")
    return {"plan_mode": "tracking", "chunks_cancelled": n}


def _revive_paused_chunks(user_id: str) -> dict:
    """[P1-RESUME-REVIVES-QUEUE · 2026-08-12] Revive los chunks que ESTA pausa
    canceló (firma exacta en dead_letter_reason) y rebasea sus offsets contra la
    ventana viva de cada plan con el SSOT del ancla.

    Por qué existe: el primer ciclo real pausa→reanuda dejó un plan partial en
    3/30 sin NADIE que lo rellenara — catch-up del shift (solo no-partial),
    bg-refill (solo inactivos ≥3 días) y recovery (ignora cancelled a propósito)
    lo cubren todos MENOS este caso. La reanudación reencola lo que la pausa
    apagó; `cancelled` sigue siendo terminal para todo lo demás.

    El rebase NO es opcional: los offsets de esas filas quedaron anclados al
    ancla de ANTES de la pausa, y el shift ya la movió a hoy — dejarlos quietos
    es la clase f380821a (dos generaciones sobre los mismos días / relleno
    tarde). Best-effort: si el pool no está o algo revienta, se loggea GRITADO y
    resume sigue — el banner de plan-incompleto queda como red manual."""
    try:
        from db_core import connection_pool
        from psycopg.rows import dict_row
        if not connection_pool:
            logger.warning("⚠️ [P1-RESUME-REVIVES-QUEUE] sin pool: cola NO revivida (red manual: banner).")
            return {"revived": 0, "plans": 0}
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending', dead_letter_reason = NULL,
                            attempts = 0, updated_at = NOW()
                        WHERE user_id = %s
                          AND status = 'cancelled'
                          AND dead_letter_reason = %s
                          AND dead_lettered_at IS NULL
                        RETURNING id, meal_plan_id
                        """,
                        (user_id, PAUSE_CANCEL_REASON),
                    )
                    filas = cursor.fetchall() or []
                    planes = sorted({str(f["meal_plan_id"]) for f in filas if f.get("meal_plan_id")})
                    for pid in planes:
                        cursor.execute(
                            "SELECT jsonb_array_length(COALESCE(plan_data->'days', '[]'::jsonb)) AS d "
                            "FROM meal_plans WHERE id = %s",
                            (pid,),
                        )
                        dias_vivos = int((cursor.fetchone() or {}).get("d") or 0)
                        # Lazy: cron_tasks importa plan_mode (el gate del pickup) —
                        # a nivel de módulo sería circular.
                        from cron_tasks import _rebase_pending_chunk_offsets_sql
                        movidas = _rebase_pending_chunk_offsets_sql(cursor, pid, dias_vivos)
                        logger.info(
                            f"▶ [P1-RESUME-REVIVES-QUEUE] plan {pid[:8]}: rebase post-revive "
                            f"({dias_vivos} días vivos, {movidas} fila(s) movidas)"
                        )
        if filas:
            logger.info(f"▶ [P1-RESUME-REVIVES-QUEUE] user {user_id}: {len(filas)} chunk(s) revividos en {len(planes)} plan(es)")
        return {"revived": len(filas), "plans": len(planes)}
    except Exception as e:
        logger.error(f"❌ [P1-RESUME-REVIVES-QUEUE] revive falló (resume sigue; red manual: banner): {e}")
        return {"revived": 0, "plans": 0}


def resume_plan_generation(user_id: str) -> dict:
    """Encender. Restaura el snapshot (nunca un 'complete' literal) y declara si el
    plan venció la ventana de reanudación — el caller decide si shiftear."""
    if not PLAN_MODE_SWITCH_ENABLED:
        return {"plan_mode": "plan", "skipped": "switch_off"}

    # 1. Bandera primero: encolar con el gate puesto deja chunks que el pickup ignora.
    execute_sql_write(
        """
        UPDATE user_profiles
        SET plan_mode = 'plan',
            plan_mode_changed_at = CASE WHEN plan_mode <> 'plan' THEN NOW()
                                        ELSE plan_mode_changed_at END
        WHERE id = %s
        """,
        (user_id,),
    )

    # 2. Restaurar el estado del plan desde el SNAPSHOT — con guard del CHECK I8:
    #    jamás devolver a 'complete' un plan cuyos days quedaron vacíos.
    filas = execute_sql_write(
        """
        UPDATE meal_plans
        SET plan_data = (plan_data - '_paused_at' - '_paused_prev_generation_status')
                || jsonb_build_object(
                    'generation_status',
                    CASE
                        WHEN COALESCE(plan_data->>'_paused_prev_generation_status', 'partial') = 'complete'
                             AND jsonb_array_length(COALESCE(plan_data->'days', '[]'::jsonb)) = 0
                        THEN 'partial'
                        ELSE COALESCE(plan_data->>'_paused_prev_generation_status', 'partial')
                    END
                ),
            updated_at = NOW()
        WHERE user_id = %s
          AND plan_data->>'generation_status' = 'paused_by_user'
        RETURNING id,
                  plan_data->>'generation_status' AS restored_status,
                  (plan_data->>'_paused_at') AS paused_at
        """,
        (user_id,), returning=True,
    ) or []

    # 3. Revivir la cola que ESTA pausa canceló (firma exacta) + rebase de offsets
    #    contra la ventana viva. DESPUÉS de la bandera (paso 1): revivir con el gate
    #    puesto dejaría chunks pending que el pickup ignora — invisibles.
    _revive = _revive_paused_chunks(user_id)

    # 4. ¿Venció la ventana? Se calcula sobre plan_mode_changed_at (la pausa pudo no
    #    tener plan vivo). El caller (endpoint) decide shiftear o pedir plan nuevo.
    row = execute_sql_query(
        """
        SELECT GREATEST(0, EXTRACT(EPOCH FROM (NOW() - plan_mode_changed_at)) / 86400.0)::int AS paused_days
        FROM user_profiles WHERE id = %s
        """,
        (user_id,), fetch_one=True,
    ) or {}
    paused_days = int(row.get("paused_days") or 0)
    expired = paused_days > _resume_max_days()

    restored = filas[0].get("restored_status") if filas else None
    logger.info(
        f"▶ [P1-PLAN-MODE] user {user_id}: reanuda — plan restaurado a "
        f"{restored or 'sin plan pausado'}, {paused_days} día(s) de pausa, vencido={expired}"
    )
    return {
        "plan_mode": "plan",
        "plan_status": restored,
        "paused_days": paused_days,
        "plan_expired": expired,
        "chunks_revived": _revive["revived"],
    }


def ensure_plan_generation_enabled(user_id: str) -> bool:
    """[P1-PLAN-MODE] El re-encendido automático de `_postprocess_pipeline_result`.

    Generar un plan ES el consentimiento de generar. Sin esto, un usuario en pausa que
    pulsa «Generar plan» paga su crédito, recibe la semana 1 por SSE, y las semanas
    2..N quedan bloqueadas por nuestro propio gate — en silencio, con `chunk_overdue`
    alertando cada hora sobre un plan pagado que no puede completarse.

    No-op (False) si ya estaba en 'plan'. True si encendió."""
    if not PLAN_MODE_SWITCH_ENABLED:
        return False
    filas = execute_sql_write(
        """
        UPDATE user_profiles
        SET plan_mode = 'plan', plan_mode_changed_at = NOW()
        WHERE id = %s AND plan_mode = 'tracking'
        RETURNING id
        """,
        (user_id,), returning=True,
    ) or []
    if filas:
        logger.info(f"▶ [P1-PLAN-MODE] user {user_id}: re-encendido automático al generar plan")
    return bool(filas)
