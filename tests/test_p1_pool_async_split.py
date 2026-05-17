"""[P1-POOL-ASYNC-SPLIT · 2026-05-16] Regression guard: pool ASYNC tiene
knobs separados del SYNC para no saturar pgBouncer Transaction Mode.

Bug observado en test E2E del 2026-05-15 21:40-21:53 (plan_id=ae29c7a9):
  - 10-15 warnings "DB async cache/CB reset error: couldn't get a connection
    after 20.00 sec" durante cada plan, persistiendo incluso después de bajar
    `MEALFIT_DB_POOL_MAX_SIZE=60→25` (fix P0-DB-POOL-PGBOUNCER-SATURATION).
  - El plan SÍ se persistió en `meal_plans` (sync OK), pero el async path
    saturaba constantemente.

Root cause: `async_connection_pool` reutilizaba los mismos knobs `DB_POOL_*` que
el `connection_pool` sync. Cada uno reservaba hasta `max_size=25` conexiones
INDEPENDIENTES. Total real = 50 conexiones contra el cap de pgBouncer Transaction
Mode (~15-30 client conns free tier) → el async se quedaba sin slots porque el
sync se calentaba primero y consumía los 15-30 disponibles.

Fix: knobs separados `MEALFIT_DB_ASYNC_POOL_{MIN,MAX,TIMEOUT_S}_SIZE` con
defaults conservadores (min=2, max=8, timeout=20s). Total sync+async = 25+8 = 33,
mucho más cercano al cap pgBouncer pero dejando headroom razonable.

Si migras a Supabase Pro / dedicated, restaurar ambos a valores generosos
(sync=60, async=20).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE_PATH = _BACKEND_ROOT / "db_core.py"


def _read_db_core() -> str:
    return _DB_CORE_PATH.read_text(encoding="utf-8")


def test_async_pool_min_size_knob_exists():
    """`MEALFIT_DB_ASYNC_POOL_MIN_SIZE` debe estar definido como knob propio
    via `_int_env`, separado de `MEALFIT_DB_POOL_MIN_SIZE` (sync)."""
    text = _read_db_core()
    assert "MEALFIT_DB_ASYNC_POOL_MIN_SIZE" in text, (
        "Falta knob `MEALFIT_DB_ASYNC_POOL_MIN_SIZE` en db_core.py. "
        "P1-POOL-ASYNC-SPLIT requiere que el pool async tenga min separado."
    )
    m = re.search(
        r'DB_ASYNC_POOL_MIN_SIZE\s*=\s*_int_env\(\s*["\']MEALFIT_DB_ASYNC_POOL_MIN_SIZE["\']\s*,\s*(\d+)',
        text,
    )
    assert m, "DB_ASYNC_POOL_MIN_SIZE no usa `_int_env` con default explícito."
    default = int(m.group(1))
    assert default <= 5, (
        f"Default MIN_SIZE async={default} > 5: demasiadas conexiones idle "
        f"reservadas; el sync debe poder usar la mayoría del cap pgBouncer."
    )


def test_async_pool_max_size_knob_exists_and_conservative():
    """`MEALFIT_DB_ASYNC_POOL_MAX_SIZE` debe existir y default <=15 para
    que sync (max=25) + async (max=8) ≈ 33 conexiones contra cap pgBouncer
    free tier (~15-30). Si subes a 20+, suma >45 y reintroduce la saturación."""
    text = _read_db_core()
    m = re.search(
        r'DB_ASYNC_POOL_MAX_SIZE\s*=\s*_int_env\(\s*["\']MEALFIT_DB_ASYNC_POOL_MAX_SIZE["\']\s*,\s*(\d+)',
        text,
    )
    assert m, "Falta `MEALFIT_DB_ASYNC_POOL_MAX_SIZE` definido con `_int_env`."
    default = int(m.group(1))
    assert default <= 15, (
        f"Default MAX_SIZE async={default} > 15: con sync max=25 default, "
        f"el total contra pgBouncer free tier (cap ~15-30) puede saturar. "
        f"Si migras a Pro / dedicated, actualiza este threshold."
    )


def test_async_pool_timeout_knob_exists():
    """`MEALFIT_DB_ASYNC_POOL_TIMEOUT_S` debe existir con default >=15s para
    absorber bursts de cache/CB-reset bajo concurrencia."""
    text = _read_db_core()
    m = re.search(
        r"DB_ASYNC_POOL_TIMEOUT_S\s*=\s*_float_env\(\s*[\"']MEALFIT_DB_ASYNC_POOL_TIMEOUT_S[\"']\s*,\s*([\d.]+)",
        text,
    )
    assert m, "Falta `MEALFIT_DB_ASYNC_POOL_TIMEOUT_S` definido con `_float_env`."
    default = float(m.group(1))
    assert default >= 15.0, (
        f"Default TIMEOUT={default}s < 15s: queries async (cache + CB resets) "
        f"bajo carga pueden necesitar más colchón."
    )


def test_async_pool_uses_async_knobs_not_sync():
    """`AsyncConnectionPool` debe instanciarse con DB_ASYNC_POOL_* knobs,
    NO con DB_POOL_* (sync). Si vuelven a compartirse, el split se rompe."""
    text = _read_db_core()
    # Find AsyncConnectionPool(...) callsite block.
    async_pool_match = re.search(
        r"async_connection_pool\s*=\s*AsyncConnectionPool\((.*?)\)",
        text,
        re.DOTALL,
    )
    assert async_pool_match, (
        "No se encontró asignación `async_connection_pool = AsyncConnectionPool(...)`. "
        "¿Movido o renombrado? Actualizar este test."
    )
    block = async_pool_match.group(1)
    # El bloque debe usar DB_ASYNC_POOL_* knobs (al menos min y max).
    assert "DB_ASYNC_POOL_MIN_SIZE" in block, (
        "`AsyncConnectionPool` no usa `DB_ASYNC_POOL_MIN_SIZE`. "
        "P1-POOL-ASYNC-SPLIT requiere que el pool async use knobs separados."
    )
    assert "DB_ASYNC_POOL_MAX_SIZE" in block, (
        "`AsyncConnectionPool` no usa `DB_ASYNC_POOL_MAX_SIZE`."
    )
    assert "DB_ASYNC_POOL_TIMEOUT_S" in block, (
        "`AsyncConnectionPool` no usa `DB_ASYNC_POOL_TIMEOUT_S`."
    )
    # Y NO debe usar DB_POOL_* (sync). Excepción: DB_POOL_MAX_IDLE_S es compartido
    # (no hay razón para tener idle distinto entre sync y async).
    assert "min_size=DB_POOL_MIN_SIZE" not in block, (
        "`AsyncConnectionPool` aún usa `DB_POOL_MIN_SIZE` (sync). Refactor incompleto."
    )
    assert "max_size=DB_POOL_MAX_SIZE" not in block, (
        "`AsyncConnectionPool` aún usa `DB_POOL_MAX_SIZE` (sync). Refactor incompleto."
    )


def test_sync_pool_still_uses_sync_knobs():
    """Sanity: el pool SYNC debe seguir usando DB_POOL_* (NO los async).
    Defensiva contra refactors que muevan accidentalmente el sync al async."""
    text = _read_db_core()
    sync_pool_match = re.search(
        r"connection_pool\s*=\s*ConnectionPool\((.*?)\)",
        text,
        re.DOTALL,
    )
    assert sync_pool_match, (
        "No se encontró asignación `connection_pool = ConnectionPool(...)`."
    )
    block = sync_pool_match.group(1)
    assert "DB_POOL_MIN_SIZE" in block and "DB_ASYNC_POOL_MIN_SIZE" not in block, (
        "Sync pool debe usar `DB_POOL_MIN_SIZE`, no el async knob."
    )
    assert "DB_POOL_MAX_SIZE" in block and "DB_ASYNC_POOL_MAX_SIZE" not in block, (
        "Sync pool debe usar `DB_POOL_MAX_SIZE`, no el async knob."
    )
