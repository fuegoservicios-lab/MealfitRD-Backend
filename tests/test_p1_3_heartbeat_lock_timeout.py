"""[P1-3 · 2026-05-10] Regression guard: el UPDATE de
`chunk_user_locks SET heartbeat_at = NOW()` aplica `SET LOCAL lock_timeout`
para no colgarse indefinidamente.

Bug original (audit 2026-05-10):
    Dos sites del heartbeat (`_touch_chunk_heartbeat` inline + `_do_update`
    en el thread daemon `_heartbeat_loop`) ejecutaban
    `UPDATE chunk_user_locks SET heartbeat_at = NOW() WHERE locked_by_chunk_id = %s`
    sin `SET LOCAL lock_timeout`. Si otro proceso (zombie rescue, otro
    worker) tenía la fila bloqueada, el heartbeat esperaba indefinidamente
    hasta que el `statement_timeout` de Supavisor (~60s+) lo abortaba.
    Resultado: el heartbeat_at del chunk NO se refrescaba durante minutos,
    el zombie rescue (P0-4 housekeeping) eventualmente lo mataba como
    "stale" tras `CHUNK_LOCK_STALE_MINUTES`, y el chunk reiniciaba con
    métricas corruptas. Cierre P1-LOCK-1 cubría `meal_plans` FOR UPDATE
    pero NO este site.

Fix:
    1. `db_core.execute_sql_write` extendido con param opcional
       `lock_timeout_ms` (backward-compat: `None` = comportamiento
       anterior). Cuando se provee, envuelve el query en transacción
       explícita con `SET LOCAL lock_timeout = '<N>ms'` aplicado
       ANTES del query principal.
    2. Los 2 callsites del heartbeat (`_touch_chunk_heartbeat` y
       `_do_update` en `_heartbeat_loop`) leen el knob
       `MEALFIT_CHUNK_HEARTBEAT_LOCK_TIMEOUT_MS` (default 3000ms)
       y lo pasan al `execute_sql_write`. Si el lock no se adquiere
       en 3s, psycopg propaga `LockNotAvailable`; el except genérico
       atrapa, loguea, y el siguiente cycle del thread daemon reintenta.

Cobertura de este test (parser-based, no DB):
    1. `db_core.execute_sql_write` declara el parámetro `lock_timeout_ms`.
    2. La implementación incluye `SET LOCAL lock_timeout`.
    3. Los 2 callsites del heartbeat pasan `lock_timeout_ms=...`.
    4. El knob `MEALFIT_CHUNK_HEARTBEAT_LOCK_TIMEOUT_MS` se lee vía
       `_env_int` (auto-registro en `_KNOBS_REGISTRY`).

Out of scope:
    - Test runtime contra DB real (necesita contención simulada;
      queda como P3 con staging DB).
    - Auditar TODOS los otros UPDATEs sobre `chunk_user_locks` (DELETE
      en cleanup, INSERT inicial). Esos no son hot-path del heartbeat
      y se tratan en P1-LOCK-2 si surgen problemas.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CORE_PATH = _BACKEND_ROOT / "db_core.py"
_CRON_TASKS_PATH = _BACKEND_ROOT / "cron_tasks.py"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. `execute_sql_write` acepta `lock_timeout_ms`
# ---------------------------------------------------------------------------
def test_execute_sql_write_accepts_lock_timeout_ms():
    src = _read(_DB_CORE_PATH)
    # Buscamos la signature de la función.
    sig_match = re.search(
        r"def\s+execute_sql_write\s*\(([^)]*)\)",
        src,
    )
    assert sig_match is not None, "execute_sql_write no encontrada."
    sig = sig_match.group(1)
    assert "lock_timeout_ms" in sig, (
        "execute_sql_write debe aceptar `lock_timeout_ms` como parámetro. "
        "Sin esto, los callsites del heartbeat no pueden pasar el timeout."
    )


def test_execute_sql_write_emits_set_local_lock_timeout():
    """Cuando `lock_timeout_ms` está presente, el body debe ejecutar
    `SET LOCAL lock_timeout = '<N>ms'` antes del query principal."""
    src = _read(_DB_CORE_PATH)
    func_match = re.search(
        r"def\s+execute_sql_write\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None
    body = func_match.group(0)
    assert "SET LOCAL lock_timeout" in body, (
        "execute_sql_write debe ejecutar `SET LOCAL lock_timeout = '<N>ms'` "
        "cuando `lock_timeout_ms` está presente. Sin esto el parámetro es "
        "cosmético — el lock seguiría sin timeout local."
    )


def test_execute_sql_write_wraps_in_transaction():
    """`SET LOCAL` requiere transacción explícita para tener scope (sin
    `conn.transaction()` el autocommit aplica el SET LOCAL inmediatamente
    y lo pierde en el siguiente comando)."""
    src = _read(_DB_CORE_PATH)
    func_match = re.search(
        r"def\s+execute_sql_write\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    assert "conn.transaction()" in body, (
        "execute_sql_write debe usar `conn.transaction()` cuando aplica "
        "lock_timeout_ms — sin tx explícita, el `SET LOCAL` se pierde "
        "por autocommit."
    )


# ---------------------------------------------------------------------------
# 2. Los callsites del heartbeat usan el knob
# ---------------------------------------------------------------------------
def test_touch_chunk_heartbeat_uses_lock_timeout_knob():
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        r"def\s+_touch_chunk_heartbeat\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None
    body = func_match.group(0)
    assert "MEALFIT_CHUNK_HEARTBEAT_LOCK_TIMEOUT_MS" in body, (
        "`_touch_chunk_heartbeat` debe leer el knob "
        "`MEALFIT_CHUNK_HEARTBEAT_LOCK_TIMEOUT_MS`. Sin esto, vuelve al "
        "comportamiento legacy que se cuelga indefinidamente bajo "
        "contención."
    )
    assert "lock_timeout_ms=" in body, (
        "`_touch_chunk_heartbeat` debe pasar `lock_timeout_ms=...` a "
        "`execute_sql_write`. Sin esto el knob es inútil."
    )


def test_heartbeat_thread_loop_uses_lock_timeout_knob():
    """El thread daemon `_heartbeat_loop._do_update` (anidado dentro de
    `_chunk_worker`) también debe pasar lock_timeout_ms."""
    src = _read(_CRON_TASKS_PATH)
    # Buscar el bloque del thread (entre `def _do_update` y el siguiente
    # `def`, dentro del scope de _chunk_worker).
    update_block_pattern = (
        r'UPDATE chunk_user_locks SET heartbeat_at = NOW\(\) WHERE locked_by_chunk_id = %s",\s*\(lock_chunk_id,\),\s*lock_timeout_ms='
    )
    assert re.search(update_block_pattern, src, re.DOTALL), (
        "El UPDATE del thread daemon (`_do_update`) debe incluir "
        "`lock_timeout_ms=...` como kwarg al execute_sql_write. Sin esto, "
        "el thread se cuelga bajo contención (chunk muere zombie pese al "
        "fix P1-3)."
    )


def test_heartbeat_lock_timeout_knob_uses_env_int():
    """El knob debe pasar por `_env_int` para auto-registro en
    `_KNOBS_REGISTRY` y visibilidad en `/health/version`."""
    src = _read(_CRON_TASKS_PATH)
    pattern = r"_env_int\(\s*[\"']MEALFIT_CHUNK_HEARTBEAT_LOCK_TIMEOUT_MS[\"']\s*,\s*3000\s*\)"
    assert re.search(pattern, src), (
        "El knob `MEALFIT_CHUNK_HEARTBEAT_LOCK_TIMEOUT_MS` debe leerse vía "
        "`_env_int(\"MEALFIT_CHUNK_HEARTBEAT_LOCK_TIMEOUT_MS\", 3000)`. "
        "Default 3000ms = 3s, balance entre fail-fast (vs. 60s+ del "
        "statement_timeout legacy) y tolerancia a contención real."
    )
