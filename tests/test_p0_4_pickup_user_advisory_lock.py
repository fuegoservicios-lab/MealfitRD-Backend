"""[P0-4 FIX] Tests para el advisory lock por user_id en la pickup query.

Race que el fix cierra (a nivel de pickup, no de _chunk_worker):
    El filtro `q1.user_id NOT IN (SELECT user_id FROM plan_chunk_queue WHERE
    status='processing')` evalúa la subquery contra el snapshot pre-UPDATE de
    cada TX. Dos pickups paralelos (cron threads distintos en el mismo tick)
    NO se ven entre sí: ambos pueden seleccionar chunks distintos del mismo
    user (con planes diferentes) y commitear simultáneamente.

    Pre-fix:
      1. Pickup A: SELECT ve user_U sin chunks 'processing' → toma chunk_a.
      2. Pickup B (paralelo): SELECT también ve user_U sin chunks 'processing'
         → toma chunk_b (otro plan del mismo user).
      3. Ambos commitean → ambos chunks 'processing' para user_U.
      4. _chunk_worker(A) hace INSERT en chunk_user_locks → OK.
      5. _chunk_worker(B) hace INSERT → CONFLICT (user_id) → demote a 'pending'.

    El bounce processing→pending→processing en B desperdicia ciclos, ensucia
    métricas y agrega lag visible al usuario.

    Post-fix:
      pg_try_advisory_xact_lock per user_id en el WHERE del pickup serializa
      a nivel SQL: la TX que adquiere primero el lock (TX A) excluye al
      mismo user_id de TX B (pg_try retorna FALSE → fila excluida). Una vez
      TX A commitea, el lock se libera, pero la fila ya está en 'processing',
      así que el filtro `user_id NOT IN` excluye al user en pickups subsecuentes
      sin necesidad del lock.

    chunk_user_locks (tabla de DB) sigue siendo defensa-en-profundidad contra
    paths que bypassean el pickup (recovery cron, sync API directos, etc.).

Ejecutar:
    cd backend && python -m pytest tests/test_p0_4_pickup_user_advisory_lock.py -v
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_source():
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. Ambas pickup queries (target_plan_id y global) tienen el advisory lock
# ---------------------------------------------------------------------------
def test_both_pickup_queries_have_per_user_advisory_lock():
    """Las dos ramas del pickup (target_plan_id y global) deben incluir el
    `pg_try_advisory_xact_lock` per user_id. Sin él, el bounce
    processing→pending→processing reaparece para esa rama y degrada SLA.
    """
    source = _load_source()

    lock_pattern = (
        r"pg_try_advisory_xact_lock\(\s*"
        r"hashtextextended\(\s*'chunk_pickup_user:'\s*\|\|\s*"
        r"q1\.user_id::text"
    )
    matches = re.findall(lock_pattern, source)
    # Debe aparecer al menos 2 veces: una en cada rama del pickup.
    assert len(matches) >= 2, (
        f"pg_try_advisory_xact_lock per user_id encontrado {len(matches)} veces; "
        f"esperaba >=2 (una por cada rama del pickup: target_plan_id y global). "
        f"Sin esto, el race SELECT→UPDATE en una de las ramas reabre el bounce."
    )


# ---------------------------------------------------------------------------
# 2. El lock está DESPUÉS de filtros baratos (status, user_id NOT IN)
# ---------------------------------------------------------------------------
def test_advisory_lock_sits_after_cheap_filters():
    """Para minimizar adquisiciones desperdiciadas, el `pg_try_advisory_xact_lock`
    debe evaluarse DESPUÉS de filtros baratos (`status IN`, `user_id NOT IN`).
    Si Postgres lo evalúa primero, adquirimos locks para filas que serían
    descartadas por filtros más selectivos — desperdicio inocuo (auto-release
    al COMMIT) pero ruidoso bajo carga.
    """
    source = _load_source()

    # Buscar el advisory lock y verificar que las cláusulas anteriores en el
    # mismo bloque WHERE (de la misma rama) incluyen `user_id NOT IN`.
    # Tomamos los primeros ~600 caracteres ANTES de cada lock como contexto.
    for m in re.finditer(
        r"pg_try_advisory_xact_lock\(\s*\n?\s*hashtextextended\(\s*'chunk_pickup_user:",
        source,
    ):
        prelude = source[max(0, m.start() - 1500): m.start()]
        assert "user_id NOT IN" in prelude, (
            f"Advisory lock en posición {m.start()} no tiene `user_id NOT IN` "
            f"en su prelude — el lock podría adquirirse ANTES del filtro barato, "
            f"causando adquisiciones masivas en cargas altas. Reordenar el WHERE."
        )


# ---------------------------------------------------------------------------
# 3. Namespace del lock no colisiona con acquire_meal_plan_advisory_lock
# ---------------------------------------------------------------------------
def test_pickup_lock_namespace_does_not_collide_with_meal_plan_lock():
    """`acquire_meal_plan_advisory_lock` usa keys 'meal_plan:<purpose>:<id>'
    (ver db_plans.py). El lock del pickup usa 'chunk_pickup_user:<id>'. Si
    accidentalmente alguien copy-pastea 'meal_plan:' en el pickup o
    'chunk_pickup_user:' en el meal_plan helper, dos paths ortogonales se
    bloquearían entre sí (e.g., catchup en routers/plans.py competiría con
    pickup en cron_tasks.py).
    """
    source = _load_source()

    # El namespace pickup es exclusivo de cron_tasks.py.
    pickup_ns_count = source.count("'chunk_pickup_user:'")
    assert pickup_ns_count >= 2, (
        f"Namespace 'chunk_pickup_user:' no encontrado en ambas pickups "
        f"(found {pickup_ns_count}, expected >=2)"
    )

    # Asegurar que NO se usa el namespace 'meal_plan:' del helper unificado
    # — eso colisionaría con catchup/tz_resync/general locks.
    meal_plan_ns_in_pickup = re.search(
        r"pg_try_advisory_xact_lock\([^)]*meal_plan:",
        source,
    )
    assert meal_plan_ns_in_pickup is None, (
        f"El pickup usa el namespace 'meal_plan:' que ya tiene `acquire_meal_plan"
        f"_advisory_lock`. Esto causaría colisiones cross-purpose: un catchup "
        f"esperando el lock de un meal_plan bloquearía pickups de otros chunks."
    )


# ---------------------------------------------------------------------------
# 4. El lock es transaccional (xact_lock), NO session-level
# ---------------------------------------------------------------------------
def test_pickup_uses_xact_lock_not_session_lock():
    """`pg_try_advisory_xact_lock` se libera al COMMIT del UPDATE; una versión
    session-level (`pg_try_advisory_lock`) requeriría `pg_advisory_unlock`
    explícito y, en pool de conexiones, podría dejar locks zombie si la
    conexión vuelve al pool sin liberarlos.

    Por la misma razón, el pickup NO debe usar `pg_advisory_lock` (sin _xact)
    — auto-release vs leak es un trade crítico en pools.
    """
    source = _load_source()

    # En la región del pickup: solo xact_lock, no session-level.
    # Buscamos las líneas alrededor del pickup query y verificamos.
    pickup_match = re.search(
        r"UPDATE plan_chunk_queue\s+SET status = 'processing'.*?RETURNING id,",
        source,
        re.DOTALL,
    )
    assert pickup_match, "No se encontró el bloque del pickup query"

    # El bloque debe incluir _xact_lock al menos una vez.
    # (en realidad incluye DOS, una por cada rama; el regex toma la primera).
    # Buscamos en el área del archivo donde están AMBAS pickups.
    pickups_region_start = source.find(
        "UPDATE plan_chunk_queue\n            SET status = 'processing'"
    )
    if pickups_region_start < 0:
        pickups_region_start = source.find(
            "UPDATE plan_chunk_queue\r\n            SET status = 'processing'"
        )
    pickups_region_end = source.find("RETURNING id,", pickups_region_start + 1)
    pickups_region_end = source.find(
        "RETURNING id,", pickups_region_end + 1
    ) + 200  # incluir la segunda RETURNING también
    pickups_region = source[pickups_region_start:pickups_region_end]

    assert "pg_try_advisory_xact_lock" in pickups_region, (
        "El pickup NO usa pg_try_advisory_xact_lock. Sin transactional auto-release, "
        "los locks pueden quedar zombie en el pool de conexiones."
    )

    # Buscar accidentes: pg_advisory_lock (session-level) o pg_try_advisory_lock
    # SIN _xact en la región del pickup.
    accidental_session = re.search(
        r"pg_(try_)?advisory_lock\(\s*hashtextextended\([^)]*chunk_pickup_user",
        pickups_region,
    )
    assert accidental_session is None, (
        f"Detectado uso de pg_advisory_lock (session-level) en el pickup. "
        f"DEBE ser pg_try_advisory_xact_lock para auto-release transaccional. "
        f"Match: {accidental_session.group()!r}"
    )


# ---------------------------------------------------------------------------
# 5. Defensa-en-profundidad chunk_user_locks INSERT sigue presente
# ---------------------------------------------------------------------------
def test_chunk_user_locks_table_defense_still_present():
    """Aun con el advisory lock en el pickup, `chunk_user_locks` sigue siendo
    defensa-en-profundidad para paths que NO pasan por el pickup (recovery,
    sync API). Quitarlo abriría una nueva ventana donde un chunk recién
    encolado vía `_enqueue_plan_chunk` podría tomar inventario en paralelo
    a un chunk recién recogido por el pickup.
    """
    source = _load_source()

    insert_pattern = re.search(
        r"INSERT INTO chunk_user_locks[^;]*ON CONFLICT \(user_id\) DO NOTHING",
        source,
        re.DOTALL,
    )
    assert insert_pattern, (
        "El INSERT de chunk_user_locks con ON CONFLICT (user_id) DO NOTHING ha "
        "sido removido. Era defensa-en-profundidad contra paths no-pickup; sin "
        "él, un chunk encolado vía _enqueue_plan_chunk podría racear con uno "
        "recogido por pickup en el mismo user."
    )


# ---------------------------------------------------------------------------
# 6. db_plans.acquire_meal_plan_advisory_lock sigue intacto (no afectado)
# ---------------------------------------------------------------------------
def test_meal_plan_lock_helper_remains_independent():
    """El fix del pickup NO debe haber tocado el helper unificado de
    `acquire_meal_plan_advisory_lock`. Cualquier modificación accidental a su
    namespace o hash function rompería catchup, tz_resync y los advisory
    locks de T1/T2.
    """
    from db_plans import (
        acquire_meal_plan_advisory_lock,
        _MEAL_PLAN_LOCK_PURPOSES,
    )

    assert callable(acquire_meal_plan_advisory_lock)
    # Los purposes canónicos no deben haberse perdido.
    for required in ("general", "catchup", "tz_resync"):
        assert required in _MEAL_PLAN_LOCK_PURPOSES, (
            f"Purpose canónico {required!r} desaparecido de "
            f"_MEAL_PLAN_LOCK_PURPOSES — el fix de P0-4 no debe haber tocado "
            f"este helper."
        )
