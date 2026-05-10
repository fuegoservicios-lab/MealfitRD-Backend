"""[P1-HIST-AUDIT-7 · 2026-05-09] Tests del advisory lock per-user
en los tres mutators del Historial.

Bug original (audit historial 2026-05-08):
    Doble-click "Reactivar" desde dos tabs sobre planes distintos:
    ambos endpoints resolvían el MISMO `target` (latest), ambos
    sobreescribían — el último gana, el primero perdía silenciosamente.
    Mismo riesgo en restore vs delete vs rename simultáneo del mismo
    user (race en el SELECT inicial vs el UPDATE/DELETE final).

Fix:
    `pg_advisory_xact_lock` per-user (key
    `user:history_mutator:<user_id>` hashed con `hashtextextended`)
    PRIMERO en la transacción de cada mutator. El lock se libera al
    COMMIT/ROLLBACK; users distintos NO se bloquean entre sí.

Cobertura:
    - Anchor del marker en cada uno de los 3 endpoints.
    - Helper `acquire_user_history_advisory_lock` definido en db_plans.
    - Cada endpoint llama al helper como PRIMER statement de la
      transacción (orden importa: si el SELECT target precede al
      lock, el bug original sigue presente).
    - El helper usa la misma hash function (`hashtextextended`) que
      `acquire_meal_plan_advisory_lock` para consistencia.
    - El lock vive dentro de `conn.transaction()` (transaccional, se
      libera automáticamente).
    - Ningún mutator usa `pg_advisory_lock` (non-xact) — eso fugaría
      el lock si el endpoint crash sin DELETE explícito.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_PLANS_PATH = _BACKEND_ROOT / "db_plans.py"
_PLANS_PY_PATH = _BACKEND_ROOT / "routers" / "plans.py"


# ---------------------------------------------------------------------------
# 1. Helper module-level definido en db_plans.py
# ---------------------------------------------------------------------------
def test_helper_acquire_user_history_advisory_lock_exists():
    """`acquire_user_history_advisory_lock(cursor, user_id)` está
    definido en `db_plans.py` y es importable."""
    from db_plans import acquire_user_history_advisory_lock
    assert callable(acquire_user_history_advisory_lock)


def test_helper_uses_hashtextextended():
    """El helper usa `hashtextextended(text, 0)` — misma hash function
    que `acquire_meal_plan_advisory_lock` (cierre P1-5 unificación).
    Un cambio futuro a otra hash dejaría dos call sites incoherentes
    sobre el mismo lock."""
    from db_plans import acquire_user_history_advisory_lock
    src = inspect.getsource(acquire_user_history_advisory_lock)
    assert "hashtextextended" in src, (
        "Helper debe usar `hashtextextended` para consistencia con "
        "`acquire_meal_plan_advisory_lock` y `_p05_acquire_lock` en "
        "cron_tasks.py."
    )


def test_helper_uses_xact_lock_not_session_lock():
    """`pg_advisory_xact_lock` (transactional) — se libera al COMMIT/
    ROLLBACK automáticamente. NO `pg_advisory_lock` (session) que
    quedaría colgado si el endpoint crashea sin DELETE explícito.
    """
    from db_plans import acquire_user_history_advisory_lock
    src = inspect.getsource(acquire_user_history_advisory_lock)
    assert "pg_advisory_xact_lock" in src
    # Negativo: no usar session-level lock.
    assert "pg_advisory_lock(" not in src.replace("pg_advisory_xact_lock", ""), (
        "El helper NO debe usar `pg_advisory_lock` (session). Si el "
        "endpoint crash sin DELETE explícito, el lock queda colgado "
        "hasta que la conexión muera."
    )


def test_helper_namespace_uses_user_prefix():
    """Key del lock: `user:history_mutator:<user_id>`. Prefix `user:`
    distingue de `meal_plan:<purpose>:<plan_id>` para que dos endpoints
    del mismo user, uno per-plan-id y otro per-user, NO colisionen
    accidentalmente entre sí.
    """
    from db_plans import acquire_user_history_advisory_lock
    fn_src = inspect.getsource(acquire_user_history_advisory_lock)
    # El namespace `user:` debe estar en el f-string que construye la key.
    assert "user:" in fn_src, (
        "La key del lock debe empezar con `user:` para namespace separation."
    )
    # `history_mutator` puede estar referenciado por la constante
    # `_USER_HISTORY_LOCK_PURPOSE` (verificado en
    # `test_purpose_constant_present`); aquí verificamos que el valor
    # final de la key sí lo incluye via la constante.
    from db_plans import _USER_HISTORY_LOCK_PURPOSE
    assert _USER_HISTORY_LOCK_PURPOSE == "history_mutator", (
        f"`_USER_HISTORY_LOCK_PURPOSE` debe ser exactamente "
        f"`history_mutator`, got {_USER_HISTORY_LOCK_PURPOSE!r}."
    )


# ---------------------------------------------------------------------------
# 2. Anchor del marker en los 3 endpoints
# ---------------------------------------------------------------------------
def test_marker_in_api_restore_plan():
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "P1-HIST-AUDIT-7" in src, (
        "api_restore_plan debe mencionar P1-HIST-AUDIT-7."
    )


def test_marker_in_api_delete_plan():
    from routers.plans import api_delete_plan
    src = inspect.getsource(api_delete_plan)
    assert "P1-HIST-AUDIT-7" in src


def test_marker_in_api_rename_plan():
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    assert "P1-HIST-AUDIT-7" in src


# ---------------------------------------------------------------------------
# 3. Cada endpoint invoca el helper
# ---------------------------------------------------------------------------
def test_restore_invokes_acquire_user_history_lock():
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    assert "acquire_user_history_advisory_lock" in src, (
        "api_restore_plan debe llamar `acquire_user_history_advisory_lock` "
        "al inicio de la transacción."
    )


def test_delete_invokes_acquire_user_history_lock():
    from routers.plans import api_delete_plan
    src = inspect.getsource(api_delete_plan)
    assert "acquire_user_history_advisory_lock" in src


def test_rename_invokes_acquire_user_history_lock():
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    assert "acquire_user_history_advisory_lock" in src


# ---------------------------------------------------------------------------
# 4. ORDEN: el lock va PRIMERO en la transacción
# ---------------------------------------------------------------------------
def _find_transaction_block(endpoint_src: str) -> str:
    """Aísla el bloque dentro de `with conn.transaction():` para
    inspeccionar el orden de los statements."""
    m = re.search(
        r"with\s+conn\.transaction\(\)[\s\S]*?(?=\n        logger|\n    except)",
        endpoint_src,
    )
    assert m is not None, (
        f"No se encontró bloque `with conn.transaction()` en el endpoint."
    )
    return m.group(0)


def test_restore_lock_precedes_target_select():
    """El advisory_lock DEBE invocarse ANTES del SELECT target.
    Si el SELECT precede al lock, dos restores concurrentes leen el
    mismo target → bug original.
    """
    from routers.plans import api_restore_plan
    src = inspect.getsource(api_restore_plan)
    block = _find_transaction_block(src)
    lock_pos = block.find("acquire_user_history_advisory_lock")
    target_pos = block.find("SELECT id\n                        FROM meal_plans")
    if target_pos < 0:
        # Fallback: buscar cualquier SELECT id FROM meal_plans dentro
        # del bloque transaction.
        target_pos = block.find("FROM meal_plans")
    assert lock_pos > -1, "Lock no encontrado dentro del transaction block"
    assert target_pos > -1, "Target SELECT no encontrado dentro del transaction block"
    assert lock_pos < target_pos, (
        f"El advisory_lock (pos {lock_pos}) debe ir ANTES del SELECT target "
        f"(pos {target_pos}) en api_restore_plan. Si va después, el bug "
        f"original (race en target SELECT) sigue presente."
    )


def test_delete_lock_precedes_release_locks():
    """El advisory_lock DEBE preceder al DELETE FROM chunk_user_locks
    para serializar restore concurrente.
    """
    from routers.plans import api_delete_plan
    src = inspect.getsource(api_delete_plan)
    block = _find_transaction_block(src)
    lock_pos = block.find("acquire_user_history_advisory_lock")
    delete_pos = block.find("DELETE FROM chunk_user_locks")
    assert lock_pos > -1 and delete_pos > -1
    assert lock_pos < delete_pos, (
        "El advisory_lock debe ir ANTES del DELETE FROM chunk_user_locks "
        "en api_delete_plan."
    )


def test_rename_lock_precedes_update():
    from routers.plans import api_rename_plan
    src = inspect.getsource(api_rename_plan)
    block = _find_transaction_block(src)
    lock_pos = block.find("acquire_user_history_advisory_lock")
    update_pos = block.find("UPDATE meal_plans")
    assert lock_pos > -1 and update_pos > -1
    assert lock_pos < update_pos, (
        "El advisory_lock debe ir ANTES del UPDATE meal_plans en "
        "api_rename_plan."
    )


# ---------------------------------------------------------------------------
# 5. El lock vive dentro de transaction() (xact lock requiere TX)
# ---------------------------------------------------------------------------
def test_all_three_endpoints_use_explicit_transaction():
    """Los tres endpoints deben usar `with conn.transaction():` para
    que `pg_advisory_xact_lock` se libere automáticamente al COMMIT/
    ROLLBACK. El endpoint rename antes usaba `execute_sql_write` (TX
    implícita single-statement) — no soporta el lock + UPDATE en la
    misma TX. Reescrito en P1-HIST-AUDIT-7.
    """
    for endpoint_name in ["api_restore_plan", "api_delete_plan", "api_rename_plan"]:
        from routers import plans as plans_module
        endpoint = getattr(plans_module, endpoint_name)
        src = inspect.getsource(endpoint)
        assert "conn.transaction()" in src, (
            f"{endpoint_name} debe usar `conn.transaction()` para que el "
            f"advisory_lock se libere automáticamente."
        )


# ---------------------------------------------------------------------------
# 6. Cross-check: namespace en _USER_HISTORY_LOCK_PURPOSE
# ---------------------------------------------------------------------------
def test_purpose_constant_present():
    """La constante `_USER_HISTORY_LOCK_PURPOSE` está definida y
    contiene `history_mutator`. Si alguien la cambia, el bug del
    audit puede regresar (locks con namespace diferente no colisionan).
    """
    text = _DB_PLANS_PATH.read_text(encoding="utf-8")
    assert re.search(
        r'_USER_HISTORY_LOCK_PURPOSE\s*=\s*["\']history_mutator["\']',
        text,
    ), (
        "Constante `_USER_HISTORY_LOCK_PURPOSE` ausente o con valor "
        "distinto de `history_mutator` en db_plans.py."
    )
