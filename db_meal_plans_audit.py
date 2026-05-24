"""[P2-DOC-1 · 2026-05-12] Helper SSOT para `meal_plans_audit`.

La tabla `meal_plans_audit` (creada en `p2_new_5_meal_plans_audit_table.sql`)
es backup defensivo append-only del `plan_data` ANTES de cualquier mutación
correctiva (rollback de corrupción, hotfix, pre-delete backup). NO es
telemetría automática — su SOP P3-AUDIT-6 dice "el SRE inserta una fila
manualmente antes de mutar". Audit 2026-05-11 detectó que la tabla seguía
con n_live_tup=1 (single row legacy) porque el SOP requiere SQL copy-paste
y los SRE evitaban hacerlo bajo presión.

Este helper Python cubre el SOP con una única call site reutilizable
desde scripts de hotfix, endpoints `/admin/*` o cron-tasks correctivos.

NO se invoca automáticamente desde `update_meal_plan_data` ni similares
— eso convertiría la tabla en log de write-amplification (cada mutación
nominal generaría una fila, llenando GB rápido). La convención sigue
siendo: SRE/operador decide explícitamente cuándo registrar el backup.

Tooltip-anchor: P2-DOC-1-MEAL-PLANS-AUDIT-HELPER.

Uso típico desde un script SRE:

    from db_meal_plans_audit import record_meal_plan_audit_backup

    audit_id = record_meal_plan_audit_backup(
        meal_plan_id="<plan-uuid>",
        action="corruption_repair",
        actor="sre_manual",
        note="Reparando _chunk_lessons drift post-incidente 2026-05-12",
    )
    # → INSERT INTO meal_plans_audit (..., plan_data_before=<snapshot>, ...)
    # → retorna el id BIGINT de la fila insertada (o None si falló).

Uso típico desde un endpoint `/admin/<thing>` que va a mutar plan_data:

    record_meal_plan_audit_backup(
        meal_plan_id=plan_id,
        action="manual_rollback",
        actor=f"endpoint_{request.url.path}",
        note=f"User {verified_user_id} solicitó rollback",
    )

Test: `test_p2_doc_1_meal_plans_audit_helper.py`.
"""
from __future__ import annotations

import logging
import uuid as _uuid
from typing import Optional

from db_core import execute_sql_query, execute_sql_write

logger = logging.getLogger(__name__)


# Enum cerrado del CHECK constraint en `meal_plans_audit.action`. Si
# necesitas un valor nuevo, extender el CHECK con migración explícita
# ANTES de añadir entrada acá.
_VALID_ACTIONS = (
    "corruption_repair",
    "manual_rollback",
    "pre_delete_backup",
    "schema_migration",
)


def record_meal_plan_audit_backup(
    meal_plan_id: str,
    action: str,
    actor: str,
    note: Optional[str] = None,
) -> Optional[int]:
    """Captura un snapshot del `plan_data` de un meal_plan ANTES de mutarlo.

    Args:
        meal_plan_id: UUID del plan a backup-ear. Si no existe en
            `meal_plans`, el INSERT falla con FK violation (mantenemos
            la integridad referencial — no permitimos backups huérfanos).
            NOTA: la tabla NO tiene FK declarada hacia meal_plans, pero
            la columna sí es UUID — un valor inválido (non-UUID) levantará
            error al cast.
        action: uno de {`corruption_repair`, `manual_rollback`,
            `pre_delete_backup`, `schema_migration`}. CHECK constraint
            en DB rechaza otros valores.
        actor: convención `sre_manual` | `cron_<name>` | `endpoint_<name>`.
            String libre pero recomendado mantener el formato para
            forensics.
        note: contexto del incidente (opcional, recomendado).

    Returns:
        El `id` BIGINT de la fila insertada en `meal_plans_audit`, o
        `None` si la inserción falló (DB no disponible, plan_id inválido,
        action no permitido, etc.). En caso de fallo, el caller debe
        decidir si abortar la mutación correctiva o proceder bajo riesgo.

    Side-effects:
        - 1 INSERT en `meal_plans_audit`.
        - 1 SELECT contra `meal_plans` para snapshotear `plan_data`.
        - Logger WARN si el plan no existe (FK silenciosa sin foreign key).
    """
    if not isinstance(meal_plan_id, str) or not meal_plan_id.strip():
        logger.warning(
            "[P2-DOC-1] record_meal_plan_audit_backup: meal_plan_id vacío."
        )
        return None
    try:
        _uuid.UUID(meal_plan_id)
    except (ValueError, AttributeError):
        logger.warning(
            f"[P2-DOC-1] record_meal_plan_audit_backup: meal_plan_id "
            f"{meal_plan_id!r} no es UUID válido."
        )
        return None

    if action not in _VALID_ACTIONS:
        logger.warning(
            f"[P2-DOC-1] record_meal_plan_audit_backup: action {action!r} "
            f"no está en {_VALID_ACTIONS}. Rechazado por el CHECK constraint."
        )
        return None

    if not isinstance(actor, str) or not actor.strip():
        logger.warning(
            "[P2-DOC-1] record_meal_plan_audit_backup: actor vacío."
        )
        return None

    # Snapshot del plan_data + user_id desde meal_plans. user_id puede
    # ser NULL si el caller llama tras DELETE de la fila (caso
    # pre_delete_backup invocado DESPUÉS — antipatrón pero soportado:
    # la column user_id de la tabla audit es nullable a propósito).
    try:
        row = execute_sql_query(
            """
            SELECT user_id, plan_data
              FROM meal_plans
             WHERE id = %s::uuid
            """,
            (meal_plan_id,),
            fetch_one=True,
        )
    except Exception as e:
        logger.error(
            f"[P2-DOC-1] SELECT plan_data falló para plan {meal_plan_id!r}: {e}"
        )
        return None

    if not row:
        logger.warning(
            f"[P2-DOC-1] meal_plans WHERE id={meal_plan_id!r} no devolvió "
            f"fila. ¿Plan ya borrado? Continuando con plan_data_before=NULL "
            f"causaría NOT NULL violation — abortando."
        )
        return None

    plan_data = row.get("plan_data")
    user_id = row.get("user_id")

    if plan_data is None:
        logger.warning(
            f"[P2-DOC-1] plan_data IS NULL para plan {meal_plan_id!r}. "
            f"meal_plans_audit.plan_data_before es NOT NULL — abortando."
        )
        return None

    try:
        result = execute_sql_write(
            """
            INSERT INTO meal_plans_audit
                (meal_plan_id, user_id, plan_data_before, action, actor, note)
            VALUES (%s::uuid, %s, %s::jsonb, %s, %s, %s)
            RETURNING id
            """,
            (
                meal_plan_id,
                user_id,
                # plan_data viene como dict de psycopg con jsonb. Reserializar
                # vía json.dumps para que psycopg lo encapsule correctamente.
                # Si ya es string, dejarlo tal cual.
                _serialize_jsonb(plan_data),
                action,
                actor,
                note,
            ),
            returning=True,
        )
    except Exception as e:
        logger.error(
            f"[P2-DOC-1] INSERT meal_plans_audit falló (plan={meal_plan_id!r}, "
            f"action={action!r}): {e}"
        )
        return None

    if isinstance(result, list) and result:
        new_id = result[0].get("id") if isinstance(result[0], dict) else None
        logger.info(
            f"[P2-DOC-1] Backup registrado: meal_plans_audit.id={new_id} "
            f"plan={meal_plan_id!r} action={action!r} actor={actor!r}"
        )
        return new_id
    return None


def _serialize_jsonb(value) -> str:
    """Convierte un dict/list a string JSON para psycopg. Si ya es string,
    lo devuelve tal cual (asumiendo que es JSON válido)."""
    if isinstance(value, str):
        return value
    import json as _json
    return _json.dumps(value, ensure_ascii=False)


def list_recent_audit_backups(
    meal_plan_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """Lista los últimos backups, filtrando opcionalmente por plan o user.

    Útil desde un script SRE post-incidente: ver qué se backupeó las
    últimas 24h, qué actors, qué actions. NO retorna `plan_data_before`
    (puede ser grande) — el caller hace un SELECT específico por id.

    Returns:
        Lista de dicts con keys `{id, meal_plan_id, user_id, action,
        actor, note, created_at}`. Vacía si no hay matches o si falla.
    """
    if not isinstance(limit, int) or limit <= 0:
        limit = 20
    if limit > 500:
        limit = 500

    conditions = []
    params: list = []
    if meal_plan_id:
        try:
            _uuid.UUID(meal_plan_id)
            conditions.append("meal_plan_id = %s::uuid")
            params.append(meal_plan_id)
        except (ValueError, AttributeError):
            return []
    if user_id:
        try:
            _uuid.UUID(user_id)
            conditions.append("user_id = %s::uuid")
            params.append(user_id)
        except (ValueError, AttributeError):
            return []
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    try:
        # [P2-PROD-AUDIT-1-SQL-FSTRING-NOQA] `where` se construye desde
        # lista `conditions` con fragments parametrizados (`"col = %s"`).
        # UUIDs validados con `_uuid.UUID(...)` antes de append. params
        # tuple separada → safe.
        rows = execute_sql_query(
            f"""
            SELECT id, meal_plan_id, user_id, action, actor, note, created_at
              FROM meal_plans_audit
              {where}
             ORDER BY created_at DESC
             LIMIT %s
            """,  # noqa: S608
            tuple(params),
            fetch_all=True,
        )
    except Exception as e:
        logger.error(f"[P2-DOC-1] list_recent_audit_backups falló: {e}")
        return []

    return list(rows or [])
