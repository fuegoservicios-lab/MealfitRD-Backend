"""[P0-3] Regression guard for chunk_user_locks schema.

Historia:
    chunk_user_locks.locked_by_chunk_id se creó originalmente como bigint, pero
    el worker (cron_tasks.py:11325) pasa plan_chunk_queue.id (uuid) al INSERT:
        INSERT INTO chunk_user_locks (user_id, locked_at, locked_by_chunk_id, heartbeat_at)
        VALUES (%s, NOW(), %s, NOW()) ON CONFLICT (user_id) DO NOTHING ...
    Cada llamada lanzaba:
        column "locked_by_chunk_id" is of type bigint but expression is of type uuid
    El except silencioso del _chunk_worker continuaba sin lock, desactivando de
    facto la protección user-level de concurrencia. Múltiples chunks del mismo
    usuario podían ejecutar LLM en paralelo y corromper aprendizaje cross-chunk.

Fix:
    `supabase/p1_chunk_user_locks_uuid_fix.sql` aplica
        ALTER TABLE chunk_user_locks
            ALTER COLUMN locked_by_chunk_id DROP NOT NULL,
            ALTER COLUMN locked_by_chunk_id TYPE uuid USING NULL::uuid,
            ALTER COLUMN locked_by_chunk_id SET NOT NULL;
    Verificado en producción (mpoodlmnzaeuuazsazbj): la columna es uuid.

Este test sella el invariante a nivel schema. Si una migración futura revertiera
el tipo (poco probable pero posible vía rollback equivocado o esquema regenerado
desde un dump viejo), este test caería ANTES de que el bug llegue a producción.
"""
import pytest

from db_core import execute_sql_query


pytestmark = pytest.mark.e2e


def test_chunk_user_locks_locked_by_chunk_id_is_uuid():
    """La columna `locked_by_chunk_id` DEBE ser uuid para que el INSERT del
    worker no falle con type mismatch contra plan_chunk_queue.id (uuid).
    """
    row = execute_sql_query(
        """
        SELECT data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunk_user_locks'
          AND column_name = 'locked_by_chunk_id'
        """,
        fetch_one=True,
    )
    assert row is not None, (
        "chunk_user_locks.locked_by_chunk_id no existe. ¿Tabla renombrada o "
        "schema regenerado sin la columna?"
    )
    assert row["data_type"] == "uuid", (
        f"chunk_user_locks.locked_by_chunk_id es {row['data_type']!r}, "
        f"se esperaba 'uuid'. Si recientemente se aplicó una migración que "
        f"revierte al tipo bigint, restablecer ejecutando "
        f"`supabase/p1_chunk_user_locks_uuid_fix.sql` o equivalente."
    )
    assert row["is_nullable"] == "NO", (
        "chunk_user_locks.locked_by_chunk_id debería ser NOT NULL. La "
        "migración del fix re-aplica NOT NULL después del cambio de tipo."
    )


def test_chunk_user_locks_user_id_is_uuid_and_pk():
    """`user_id` debe ser uuid + PK. Si el PK cambia, los ON CONFLICT
    (user_id) DO NOTHING del worker dejan de comportarse como upsert
    user-único y la serialización por usuario se rompe silenciosamente.
    """
    col = execute_sql_query(
        """
        SELECT data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunk_user_locks'
          AND column_name = 'user_id'
        """,
        fetch_one=True,
    )
    assert col is not None and col["data_type"] == "uuid", (
        f"chunk_user_locks.user_id debe ser uuid, encontrado: {col!r}"
    )

    pk = execute_sql_query(
        """
        SELECT pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE conrelid = 'public.chunk_user_locks'::regclass
          AND contype = 'p'
        """,
        fetch_one=True,
    )
    assert pk is not None, "chunk_user_locks no tiene primary key definida."
    assert "user_id" in pk["definition"], (
        f"PK de chunk_user_locks no es por user_id: {pk['definition']!r}. "
        "Sin esto, ON CONFLICT (user_id) en el worker no funciona como upsert."
    )


def test_chunk_user_locks_insert_accepts_uuid_task_id():
    """Smoke integration: el INSERT del worker (con uuid task_id) NO debe
    lanzar type mismatch. Si alguna migración futura cambia el tipo de
    locked_by_chunk_id a algo no-uuid, este test rompe inmediatamente con
    el error real ('column ... is of type X but expression is of type uuid').

    Usa user_id de prueba determinístico + ON CONFLICT DO NOTHING para no
    interferir con datos reales y para ser idempotente.
    """
    import uuid as _uuid

    from db_core import execute_sql_write

    test_user_id = "00000000-0000-0000-0000-0000000000a3"  # P0-3 sentinel
    test_chunk_id = str(_uuid.uuid4())

    # Limpiar cualquier residuo previo (idempotencia ante runs anteriores).
    execute_sql_write(
        "DELETE FROM chunk_user_locks WHERE user_id = %s", (test_user_id,)
    )

    try:
        # Replica EXACTAMENTE el INSERT del worker (cron_tasks.py:11325).
        # Si el tipo de locked_by_chunk_id no es uuid-compatible, esto falla
        # con "column ... is of type bigint but expression is of type uuid"
        # ANTES de que el bug llegue a un usuario real.
        execute_sql_write(
            """
            INSERT INTO chunk_user_locks (user_id, locked_at, locked_by_chunk_id, heartbeat_at)
            VALUES (%s, NOW(), %s, NOW())
            ON CONFLICT (user_id) DO NOTHING
            """,
            (test_user_id, test_chunk_id),
        )

        # Verificar que se insertó con el uuid intacto.
        row = execute_sql_query(
            "SELECT locked_by_chunk_id::text AS lid FROM chunk_user_locks WHERE user_id = %s",
            (test_user_id,),
            fetch_one=True,
        )
        assert row is not None, "INSERT no persistió la fila"
        assert row["lid"] == test_chunk_id, (
            f"locked_by_chunk_id se persistió como {row['lid']!r}, "
            f"esperado {test_chunk_id!r}"
        )
    finally:
        execute_sql_write(
            "DELETE FROM chunk_user_locks WHERE user_id = %s", (test_user_id,)
        )
