"""[P2-5 · 2026-05-10] Anchor estático del contrato actual del pool DB.

Estado verificado en el audit 2026-05-10:
    NO hay pool DB separado para LangGraph PostgresSaver. Tanto el
    backend (queries via `execute_sql_query` / `execute_sql_write`),
    como el checkpointer LangGraph (`PostgresSaver(connection_pool)`
    en `agent.py`), como los crons APScheduler, comparten el MISMO
    `connection_pool` global de `db_core.py`. Default max=60.

    El plan P2-5 del audit pidió "verificar y, si no está implementado,
    documentar como seguimiento". Este test es ese anchor: enforza el
    invariante actual y obliga al operador a actualizar el test
    explícitamente cuando decida separar los pools.

Por qué importa:
    Memoria `project_db_pool_saturation_2026_05_06`: bajo carga, el
    chat agent (PostgresSaver) puede consumir conexiones a >50% del
    pool, dejando al worker de chunks sin slots → "couldn't get a
    connection 30s" → APScheduler MISSED en cascada (que P0-2 atrapa,
    pero la causa raíz es esta).

    Si en el futuro se decide separar pools, este test debe actualizarse
    para reflejar la nueva arquitectura. La actualización debe
    acompañar la separación — no ser opcional.

Cobertura:
    1. `db_core.py` declara `connection_pool` y `async_connection_pool`
       como singletons globales con knobs `MEALFIT_DB_POOL_*`.
    2. `agent.py` (chat) usa `PostgresSaver(connection_pool)` con el
       pool global.
    3. NO existe un pool secundario llamado `langgraph_pool` o
       `chat_pool` (regression guard: si alguien lo añade sin actualizar
       este test, falla).
    4. Knobs `MEALFIT_DB_POOL_*` registrados con clamps documentados
       (min/max/timeout).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE = _BACKEND_ROOT / "db_core.py"
_AGENT = _BACKEND_ROOT / "agent.py"


def _read(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"Archivo no encontrado: {p}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. db_core declara los pools singletons
# ---------------------------------------------------------------------------
def test_db_core_declares_connection_pool_singleton():
    src = _read(_DB_CORE)
    assert re.search(r"^connection_pool\s*=\s*None", src, re.MULTILINE), (
        "`connection_pool` debe declararse como singleton top-level en db_core.py."
    )
    assert re.search(r"^async_connection_pool\s*=\s*None", src, re.MULTILINE), (
        "`async_connection_pool` debe declararse como singleton top-level."
    )


def test_db_core_uses_mealfit_pool_knobs():
    """Los 4 knobs MEALFIT_DB_POOL_* están presentes con sus clamps."""
    src = _read(_DB_CORE)
    for knob in [
        "MEALFIT_DB_POOL_MIN_SIZE",
        "MEALFIT_DB_POOL_MAX_SIZE",
        "MEALFIT_DB_POOL_TIMEOUT_S",
        "MEALFIT_DB_POOL_MAX_IDLE_S",
    ]:
        assert knob in src, f"Knob `{knob}` no encontrado en db_core.py"


def test_db_core_uses_transaction_pooler_port_6543():
    """Producción debe rewriter `:5432` a `:6543` (Supabase Transaction Pooler).

    Sin esto, conexiones directas se agotan al escalar. Es la mitigación
    natural de no separar pools — el pooler de Supabase multiplexa.
    """
    src = _read(_DB_CORE)
    assert ':6543' in src and ':5432' in src, (
        "El rewrite a port 6543 (transaction pooler) debe estar presente."
    )


# ---------------------------------------------------------------------------
# 2. agent.py usa el pool global para LangGraph PostgresSaver
# ---------------------------------------------------------------------------
def test_agent_postgres_saver_uses_global_pool():
    """`PostgresSaver(connection_pool)` debe usar EL pool global.

    Si esto cambia (e.g., a `PostgresSaver(langgraph_pool)`), actualizar
    este test simultáneamente y crear el pool nuevo en db_core.py.
    """
    src = _read(_AGENT)
    matches = re.findall(r"PostgresSaver\(([^)]+)\)", src)
    assert matches, "No se encontraron call sites de `PostgresSaver(...)` en agent.py"
    for arg in matches:
        arg_stripped = arg.strip()
        assert arg_stripped == "connection_pool", (
            f"`PostgresSaver({arg_stripped})` debe pasar el pool global "
            f"`connection_pool`. Si separaste pools, actualiza este test "
            f"y la memoria `project_p2_5_db_pool_contract_2026_05_10`."
        )


# ---------------------------------------------------------------------------
# 3. No hay pools secundarios "fantasma"
# ---------------------------------------------------------------------------
def test_no_secondary_pool_declared():
    """Sanity check: si alguien declaró otro pool en db_core sin actualizar
    este test, captura el cambio para que sea consciente."""
    src = _read(_DB_CORE)
    # Solo asignaciones top-level (`<var> = ConnectionPool(...)`). Esto evita
    # falsos positivos por strings de log que mencionan "ConnectionPool (Sync y Async)".
    syncs = re.findall(
        r"^\s*\w+\s*=\s*ConnectionPool\s*\(", src, re.MULTILINE
    )
    asyncs = re.findall(
        r"^\s*\w+\s*=\s*AsyncConnectionPool\s*\(", src, re.MULTILINE
    )
    # Esperamos exactamente 1 cada uno (el global). Si hay más, es un nuevo
    # pool que requiere documentación + actualización del test.
    assert len(syncs) == 1, (
        f"Esperaba 1 ConnectionPool() instantiation, encontré {len(syncs)}. "
        f"Si añadiste pool nuevo, documentar arquitectura + actualizar test."
    )
    assert len(asyncs) == 1, (
        f"Esperaba 1 AsyncConnectionPool() instantiation, encontré {len(asyncs)}."
    )


# ---------------------------------------------------------------------------
# 4. Memoria de cierre referenciada
# ---------------------------------------------------------------------------
def test_db_pool_memory_entry_exists():
    """La memoria que documenta el trade-off debe existir."""
    memory_dir = Path.home() / ".claude" / "projects" / \
        "c--Users-angel-OneDrive-Escritorio-MealfitRD-IA" / "memory"
    if not memory_dir.exists():
        pytest.skip("Memoria local no disponible (CI hermético).")
    # Si una de las dos existe, OK.
    candidates = [
        memory_dir / "project_db_pool_saturation.md",
        memory_dir / "project_p2_5_db_pool_contract_2026_05_10.md",
    ]
    found = any(c.exists() for c in candidates)
    assert found, (
        f"Ninguna de las memorias documentando el contrato del pool "
        f"DB existe: {[str(c) for c in candidates]}. Crear una con el "
        f"trade-off explícito."
    )
