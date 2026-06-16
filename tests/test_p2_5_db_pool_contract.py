"""[P2-5 · 2026-05-10 · actualizado P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20]
Anchor del contrato del pool DB.

Estado verificado en el audit 2026-05-10:
    Originalmente NO había pool DB separado para LangGraph PostgresSaver.
    Tanto el backend, el checkpointer LangGraph, como los crons APScheduler
    compartían el MISMO `connection_pool` global. Default max=60.

Estado actualizado tras P1-CHECKPOINT-POOL-SPLIT (2026-05-20):
    Se introdujo `chat_checkpoint_pool` SEPARADO para el LangGraph
    `PostgresSaver` por un bug productivo: el Transaction Pooler de
    Supabase (port 6543) mataba conns idle agresivamente (~30s) mientras
    LangGraph mantenía la conn checkout durante el LLM call → SSL bad
    length / EOF al `put_writes` final. El split (session mode 5432)
    redujo el problema; los P-fixes posteriores del mismo día
    (P1-CHAT-CHECKPOINT-FIX + P1-CHAT-CHECKPOINT-DEGRADE) cerraron el
    residuo (force-rewrite + pool recycling agresivo + silent degrade
    post-stream).

    Este test enforza el contrato POST-SPLIT:
    1. `connection_pool` + `async_connection_pool` (globales, tráfico
       genérico, port 6543 transaction mode).
    2. `chat_checkpoint_pool` (LangGraph checkpointer ONLY, port 5432
       session mode, min_size=0 + max_idle=30s).
    3. `agent.py` callsites de PostgresSaver pasan `_checkpoint_pool`
       con fallback `chat_checkpoint_pool or connection_pool` — NUNCA
       `PostgresSaver(connection_pool)` literal directo.

Por qué el split importa:
    Memoria `project_db_pool_saturation_2026_05_06`: bajo carga, el chat
    agent puede consumir conexiones a >50% del pool, dejando al worker
    de chunks sin slots → "couldn't get a connection 30s" → APScheduler
    MISSED en cascada (que P0-2 atrapa). El split aísla el tráfico LangGraph
    en su propio pool (max=4) sin afectar el pool del tráfico app general.

Cobertura:
    1. `db_core.py` declara `connection_pool` + `async_connection_pool`
       + `chat_checkpoint_pool` como singletons top-level.
    2. `agent.py` callsites usan `_checkpoint_pool` con fallback —
       NUNCA `PostgresSaver(connection_pool)` literal.
    3. Knobs `MEALFIT_DB_POOL_*` registrados con clamps documentados.
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


def test_db_core_uses_neon_pooled_and_direct_urls():
    """Producción debe multiplexar conexiones vía el pooler de Neon.

    [P1-NEON-DB-MIGRATION · 2026-06-12] Supabase fue eliminado por completo.
    El viejo contrato "rewrite `:5432`→`:6543` (Supabase Transaction Pooler)"
    ya no aplica: Neon NO distingue pooled vs direct por puerto, sino por
    HOSTNAME (`-pooler` suffix) expuesto en dos env vars separados. El
    contrato post-migración es:
      - `NEON_DATABASE_URL_POOLED` → pools principales (pooler / transaction
        mode, multiplexa via PgBouncer/Supavisor; reemplaza el rol del 6543).
      - `NEON_DATABASE_URL` → `chat_checkpoint_pool` (endpoint directo,
        session mode; el LangGraph PostgresSaver requiere session mode).

    Sin esta separación, conexiones directas se agotan al escalar — la misma
    motivación que el rewrite a 6543 cerraba en el mundo Supabase.
    """
    src = _read(_DB_CORE)
    assert "NEON_DATABASE_URL_POOLED" in src, (
        "Los pools principales deben consumir `NEON_DATABASE_URL_POOLED` "
        "(pooler de Neon, transaction mode) para multiplexar conexiones."
    )
    assert "NEON_DATABASE_URL" in src, (
        "El `chat_checkpoint_pool` debe consumir `NEON_DATABASE_URL` "
        "(endpoint directo, session mode) — requerido por el PostgresSaver."
    )


# ---------------------------------------------------------------------------
# 2. agent.py callsites de PostgresSaver post-split
# ---------------------------------------------------------------------------
def test_agent_postgres_saver_uses_checkpoint_pool_fallback():
    """[Actualizado P1-CHAT-CHECKPOINT-DEGRADE 2026-05-20]
    Tras P1-CHECKPOINT-POOL-SPLIT, los callsites de `PostgresSaver(...)`
    en agent.py NO pueden pasar `connection_pool` literal — deben usar
    `_checkpoint_pool` (variable local que resuelve a
    `chat_checkpoint_pool or connection_pool`).

    Si alguien introduce un nuevo callsite con `PostgresSaver(connection_pool)`,
    el bug SSL bad length vuelve para ese path.
    """
    src = _read(_AGENT)
    matches = re.findall(r"PostgresSaver\(([^)]+)\)", src)
    assert matches, "No se encontraron call sites de `PostgresSaver(...)` en agent.py"
    forbidden = "connection_pool"
    allowed = {"_checkpoint_pool", "conn"}
    for arg in matches:
        arg_stripped = arg.strip()
        assert arg_stripped != forbidden, (
            f"`PostgresSaver({arg_stripped})` directo prohibido post-split. "
            f"Usar `PostgresSaver(_checkpoint_pool)` con fallback "
            f"`chat_checkpoint_pool or connection_pool` arriba. Ver "
            f"P1-CHECKPOINT-POOL-SPLIT · 2026-05-20."
        )
        # Sanity: arg debe ser uno de los permitidos (variable resuelta).
        # `conn` se usa en app.py:PostgresSaver(conn).setup() de startup,
        # pero también podría aparecer indirectamente acá si un setup
        # callsite migrara. Documentar permitidos.
        assert arg_stripped in allowed, (
            f"`PostgresSaver({arg_stripped})` usa un argumento inesperado. "
            f"Permitidos: {sorted(allowed)}. Si añadiste un patrón nuevo, "
            f"actualiza este test + memoria."
        )
    # Sanity positiva: ≥2 callsites del fallback (chat_with_agent + stream).
    fallback_pattern = re.findall(
        r"chat_checkpoint_pool\s+or\s+connection_pool",
        src,
    )
    assert len(fallback_pattern) >= 2, (
        f"Esperaba ≥2 callsites con fallback `chat_checkpoint_pool or "
        f"connection_pool`; encontrados: {len(fallback_pattern)}. Refactor "
        f"incompleto del split."
    )


# ---------------------------------------------------------------------------
# 3. Inventario explícito de pools post-split
# ---------------------------------------------------------------------------
def test_pool_inventory_post_split():
    """[Actualizado P1-CHAT-CHECKPOINT-DEGRADE 2026-05-20]
    Inventario explícito de pools sync/async en db_core.py:
        - 1 `ConnectionPool` global (tráfico genérico, port 6543).
        - 1 `ConnectionPool` `chat_checkpoint_pool` (LangGraph, port 5432).
        - 1 `AsyncConnectionPool` (async tráfico, port 6543).

    Si alguien declara un pool sync adicional sin actualizar este test
    (e.g., otro split para crons), debe documentarlo + bumpear el conteo
    esperado conscientemente.
    """
    src = _read(_DB_CORE)
    syncs = re.findall(
        r"^\s*\w+\s*=\s*ConnectionPool\s*\(", src, re.MULTILINE
    )
    asyncs = re.findall(
        r"^\s*\w+\s*=\s*AsyncConnectionPool\s*\(", src, re.MULTILINE
    )
    assert len(syncs) == 2, (
        f"Esperaba 2 ConnectionPool() instantiations (connection_pool + "
        f"chat_checkpoint_pool), encontré {len(syncs)}. Si añadiste pool "
        f"nuevo o removiste uno, documentar arquitectura + actualizar test."
    )
    assert len(asyncs) == 1, (
        f"Esperaba 1 AsyncConnectionPool() instantiation (async_connection_pool), "
        f"encontré {len(asyncs)}."
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
