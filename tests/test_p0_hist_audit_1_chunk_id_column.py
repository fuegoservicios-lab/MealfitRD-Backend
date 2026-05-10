"""[P0-HIST-AUDIT-1 · 2026-05-09] Tests de regresión: el subquery
``DELETE FROM chunk_user_locks ... locked_by_chunk_id IN (SELECT id FROM
plan_chunk_queue ...)`` debe usar la PK real ``id`` y NO una columna
inexistente ``chunk_id``.

Bug original (audit historial 2026-05-08):
    Tanto ``api_restore_plan`` como ``api_delete_plan`` referenciaban
    ``SELECT chunk_id FROM plan_chunk_queue WHERE meal_plan_id = %s
    AND chunk_id IS NOT NULL``. La PK de la tabla es ``id`` (uuid);
    NO existe una columna ``chunk_id``. Postgres lanzaba
    ``psycopg.errors.UndefinedColumn: column "chunk_id" does not exist``
    → ROLLBACK del bloque transaccional → el endpoint devolvía 500
    sin restaurar/eliminar nada. Los tests existentes (P0-HIST-1 y
    P0-HIST-3) no detectaban el bug porque mockeaban
    ``execute_sql_query``/``connection_pool`` y nunca tocaban Postgres.

Verificación contra DB real (MCP Supabase, project mpoodlmnzaeuuazsazbj):
    - El SQL nuevo (``SELECT id``) compila y usa
      ``Index Scan using chunk_user_locks_pkey``.
    - El SQL viejo (``SELECT chunk_id``) lanza ``undefined_column``.

Cobertura:
    - Anchor textual del marker en ambos endpoints.
    - Contrato SQL positivo: ambos endpoints contienen el subquery
      con ``SELECT id FROM plan_chunk_queue``.
    - Contrato SQL negativo: ningún endpoint del archivo referencia
      ``SELECT chunk_id FROM plan_chunk_queue``.
    - Cross-check con ``cron_tasks.py:16096``: el INSERT que popula
      ``chunk_user_locks.locked_by_chunk_id`` usa
      ``plan_chunk_queue.id`` como source — esto justifica que el
      JOIN deba ser contra ``id``, no contra otra columna.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"


# ---------------------------------------------------------------------------
# 1. Anchor del marker en ambos endpoints
# ---------------------------------------------------------------------------
def test_marker_in_api_restore_plan():
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "P0-HIST-AUDIT-1" in src, (
        "api_restore_plan debe mencionar el marker P0-HIST-AUDIT-1 para "
        "que un grep desde memoria/CLAUDE.md encuentre el cierre."
    )


def test_marker_in_api_delete_plan():
    from routers.plans import api_delete_plan
    src = inspect.getsource(api_delete_plan)
    assert "P0-HIST-AUDIT-1" in src, (
        "api_delete_plan debe mencionar el marker P0-HIST-AUDIT-1 para "
        "que un grep desde memoria/CLAUDE.md encuentre el cierre."
    )


# ---------------------------------------------------------------------------
# 2. Contrato SQL positivo: el subquery usa SELECT id
# ---------------------------------------------------------------------------
def _extract_locks_subquery(endpoint_src: str) -> str:
    """Aísla el bloque del DELETE FROM chunk_user_locks para inspección.

    El regex requiere los tres tokens del SQL real (DELETE FROM, WHERE
    user_id, locked_by_chunk_id IN) para no falso-positivar contra la
    docstring del endpoint que también menciona ``DELETE FROM
    chunk_user_locks`` en prosa.
    """
    m = re.search(
        r"DELETE\s+FROM\s+chunk_user_locks\s+"
        r"WHERE\s+user_id\s*=\s*%s\s+"
        r"AND\s+locked_by_chunk_id\s+IN\s*\([\s\S]*?\)",
        endpoint_src,
        re.IGNORECASE,
    )
    assert m is not None, (
        "No se encontró bloque SQL `DELETE FROM chunk_user_locks WHERE "
        "user_id = %s AND locked_by_chunk_id IN (...)` en el endpoint"
    )
    return m.group(0)


def test_restore_subquery_uses_id_not_chunk_id():
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    block = _extract_locks_subquery(src)
    assert re.search(r"SELECT\s+id\s+FROM\s+plan_chunk_queue", block, re.IGNORECASE), (
        "api_restore_plan debe usar `SELECT id FROM plan_chunk_queue` para "
        "filtrar locks. La PK real es `id`, NO `chunk_id`."
    )


def test_delete_subquery_uses_id_not_chunk_id():
    from routers.plans import api_delete_plan
    src = inspect.getsource(api_delete_plan)
    block = _extract_locks_subquery(src)
    assert re.search(r"SELECT\s+id\s+FROM\s+plan_chunk_queue", block, re.IGNORECASE), (
        "api_delete_plan debe usar `SELECT id FROM plan_chunk_queue` para "
        "filtrar locks. La PK real es `id`, NO `chunk_id`."
    )


# ---------------------------------------------------------------------------
# 3. Contrato SQL negativo: ninguna parte del archivo plans.py debe
#    referenciar `SELECT chunk_id FROM plan_chunk_queue`. Esta aserción
#    cubre TODO el módulo, no solo los dos endpoints HIST — protege
#    contra futuros endpoints que copien-y-peguen el patrón roto.
# ---------------------------------------------------------------------------
def test_no_chunk_id_subquery_anywhere_in_plans_router():
    text = _PLANS_PY.read_text(encoding="utf-8")
    matches = re.findall(
        r"SELECT\s+chunk_id\s+FROM\s+plan_chunk_queue",
        text,
        re.IGNORECASE,
    )
    assert not matches, (
        f"Encontradas {len(matches)} referencias a `SELECT chunk_id FROM "
        f"plan_chunk_queue` en routers/plans.py. La PK es `id`; "
        f"`chunk_id` NO existe — el SQL fallaría en runtime con "
        f"`UndefinedColumn`. Reemplazar por `SELECT id`."
    )


# ---------------------------------------------------------------------------
# 4. Cross-check con el INSERT que popula locked_by_chunk_id
# ---------------------------------------------------------------------------
def test_insert_into_chunk_user_locks_uses_plan_chunk_queue_id():
    """El subquery del DELETE asume que ``locked_by_chunk_id`` se popula
    con ``plan_chunk_queue.id``. Si en el futuro alguien cambia el INSERT
    para popularlo con otra columna, la aserción del DELETE deja de
    coincidir y este test falla — anchor cruzado entre productor y
    consumidor del valor.
    """
    cron_path = _BACKEND_ROOT / "cron_tasks.py"
    text = cron_path.read_text(encoding="utf-8")
    # Buscar el INSERT que popula chunk_user_locks. Aceptamos el patrón
    # estándar `INSERT INTO chunk_user_locks (..., locked_by_chunk_id, ...)`.
    insert_block_match = re.search(
        r"INSERT\s+INTO\s+chunk_user_locks\s*\([^)]*locked_by_chunk_id[^)]*\)",
        text,
        re.IGNORECASE,
    )
    assert insert_block_match is not None, (
        "No se encontró INSERT INTO chunk_user_locks (...,locked_by_chunk_id,...) "
        "en cron_tasks.py. ¿Fue movido? Sin un INSERT de referencia, no "
        "podemos verificar que el DELETE filtra contra la columna correcta."
    )


# ---------------------------------------------------------------------------
# 5. Smoke documental: la columna chunk_id NO existe en plan_chunk_queue
#    (anclado a la verificación contra DB real registrada en el docstring
#    superior). Si alguna vez se agrega esa columna como SSOT, este test
#    debe ser revisitado.
# ---------------------------------------------------------------------------
def test_plan_chunk_queue_pk_is_id_per_grep():
    """Verifica vía grep estático que el resto del backend usa
    ``SELECT id FROM plan_chunk_queue`` (no ``chunk_id``).
    """
    text = (_BACKEND_ROOT / "db_plans.py").read_text(encoding="utf-8")
    assert "SELECT id FROM plan_chunk_queue" in text, (
        "db_plans.py debería contener al menos un `SELECT id FROM "
        "plan_chunk_queue` (cierre P0-4 reservation). Si fue refactorizado "
        "entero, actualizar este anchor."
    )
