"""[P1-2 · 2026-05-10] Regression guard: los 4 helpers de push
notifications (`_maybe_notify_user_*`) usan CAS atómico vía
`_claim_push_cooldown_slot` en vez del patrón SELECT-then-UPDATE.

Bug original (audit 2026-05-10):
    Los 4 helpers (`_maybe_notify_user_live_degraded`,
    `_maybe_notify_user_pantry_degraded`,
    `_maybe_notify_user_stale_snapshot_paused`,
    `_maybe_notify_user_tz_unresolved`) hacían:
      1. SELECT health_profile WHERE id=%s
      2. parse last_iso, comparar contra now - cooldown
      3. if cooldown vivo: return False
      4. send push
      5. UPDATE health_profile SET last_iso = NOW()

    Bajo concurrencia (2 chunks del mismo usuario emitiendo flexible_mode
    simultáneamente), dos workers podían:
      T1: SELECT → last_iso=None
      T2: SELECT → last_iso=None (antes que T1 commitee)
      T1, T2: ambos pasan cooldown check
      T1, T2: ambos emiten push
      T1, T2: ambos UPDATE (último gana, pero el daño ya está hecho)
    El usuario recibía 2 pushes potencialmente contradictorios (un
    chunk con `reason=live_fetch_degraded`, otro con
    `reason=stale_snapshot_auto_flex`, etc.).

Fix:
    Nuevo helper `_claim_push_cooldown_slot(user_id, slot_key, cooldown_hours)`
    ejecuta un solo `UPDATE … RETURNING id` con la condición de cooldown
    EMBEBIDA en el WHERE:
      WHERE id = %s
        AND (health_profile->>slot_key IS NULL or '' OR
             (health_profile->>slot_key)::timestamptz < NOW() - make_interval(hours => %s::int))
    Si la condición se cumple → UPDATE muta + RETURNING devuelve la fila →
    True. Si no → 0 rows → False. PostgreSQL serializa el UPDATE-de-misma-
    fila → solo un caller gana.

    Los 4 helpers ahora delegan el check+update a `_claim_push_cooldown_slot`
    y solo emiten el push si el CAS retorna True.

Cobertura de este test (parser-based, no DB):
    1. El helper `_claim_push_cooldown_slot` existe en cron_tasks.py.
    2. Usa un único `UPDATE … RETURNING` con la condición CAS.
    3. Pasa el slot_key como param Python (no interpolado → safe).
    4. Cada uno de los 4 helpers invoca `_claim_push_cooldown_slot`.
    5. Ninguno de los 4 helpers conserva el patrón legacy
       (SELECT health_profile WHERE id=%s + parse + comparar
       fechas en Python).

Smoke runtime hecho durante el cierre (no en CI):
    Round 1 (sin slot) → CAS gana (1 row).
    Round 2 (cooldown vivo) → CAS pierde (0 rows).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS_PATH = _BACKEND_ROOT / "cron_tasks.py"

_HELPERS_REFACTORED = (
    "_maybe_notify_user_live_degraded",
    "_maybe_notify_user_pantry_degraded",
    "_maybe_notify_user_stale_snapshot_paused",
    "_maybe_notify_user_tz_unresolved",
)


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Helper `_claim_push_cooldown_slot` existe + estructura correcta
# ---------------------------------------------------------------------------
def test_claim_helper_exists():
    src = _read(_CRON_TASKS_PATH)
    assert re.search(
        r"def\s+_claim_push_cooldown_slot\s*\(",
        src,
    ), (
        "`_claim_push_cooldown_slot` debe existir en cron_tasks.py. "
        "Sin este helper, los push notifications vuelven al patrón "
        "SELECT-then-UPDATE con race."
    )


def test_claim_helper_uses_single_atomic_update():
    """El helper debe usar UN solo UPDATE con condición CAS embebida —
    si vuelven a aparecer SELECT separado + UPDATE separado, la race
    se reintroduce."""
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        r"def\s+_claim_push_cooldown_slot\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None
    body = func_match.group(0)

    # Debe haber un UPDATE con RETURNING.
    assert re.search(r"UPDATE\s+user_profiles", body, re.IGNORECASE), (
        "Helper debe ejecutar `UPDATE user_profiles`."
    )
    assert re.search(r"RETURNING\s+id", body, re.IGNORECASE), (
        "Helper debe usar `RETURNING id` para saber si el CAS ganó."
    )

    # NO debe haber SELECT explícito sobre user_profiles (rompería el CAS).
    assert "execute_sql_query" not in body, (
        "Helper NO debe llamar a `execute_sql_query` (= SELECT). El check "
        "del cooldown debe estar EMBEBIDO en el WHERE del UPDATE — sin "
        "esto, la race se reintroduce."
    )


def test_claim_helper_uses_make_interval_with_int():
    """Lección P0-HIST-FIX-1: `make_interval(hours => N)` requiere int
    explícito. Sin `::int`, Postgres rechaza floats/strings."""
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        r"def\s+_claim_push_cooldown_slot\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    assert re.search(
        r"make_interval\s*\(\s*hours\s*=>\s*%s\s*::\s*int\s*\)",
        body, re.IGNORECASE,
    ), (
        "Helper debe usar `make_interval(hours => %s::int)`. "
        "Sin `::int`, viene como string vía parámetro de psycopg y "
        "Postgres rechaza."
    )


def test_claim_helper_passes_slot_key_as_param_not_interpolated():
    """El `slot_key` debe pasarse como parámetro Python, no interpolado
    en el SQL string — defensa contra inyección."""
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        r"def\s+_claim_push_cooldown_slot\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    # El query debe usar `ARRAY[%s]::text[]` para jsonb_set path (param-binding).
    assert "ARRAY[%s]::text[]" in body, (
        "Helper debe construir el path de jsonb_set como "
        "`ARRAY[%s]::text[]` (param-binding), no como f-string del slot_key."
    )
    # Y `health_profile ->> %s` para lecturas de la key (también param).
    assert "health_profile ->> %s" in body, (
        "Helper debe leer `health_profile ->> %s` con slot_key como param. "
        "Si interpola con f-string, hay inyección posible."
    )
    # Aislar el bloque del SQL (entre triple-quoted strings) y verificar
    # que NO contiene `{slot_key}` ni `{cooldown_hours}` (f-string substitution).
    sql_block_match = re.search(r'"""(.*?)"""', body, re.DOTALL)
    assert sql_block_match is not None, "No se encontró el SQL triple-quoted."
    sql_block = sql_block_match.group(1)
    assert "{slot_key}" not in sql_block, (
        "El SQL contiene `{slot_key}` — interpolación f-string del param. "
        "Cambiar a `%s` con param binding."
    )
    assert "{cooldown_hours}" not in sql_block, (
        "El SQL contiene `{cooldown_hours}` — interpolación f-string. "
        "Cambiar a `%s` con param binding."
    )


# ---------------------------------------------------------------------------
# 2. Los 4 helpers de push invocan el nuevo CAS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("helper_name", _HELPERS_REFACTORED)
def test_helper_invokes_claim_helper(helper_name: str):
    """Cada uno de los 4 helpers debe llamar a `_claim_push_cooldown_slot`."""
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        rf"def\s+{re.escape(helper_name)}\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None, f"{helper_name} no encontrado."
    body = func_match.group(0)
    assert "_claim_push_cooldown_slot" in body, (
        f"`{helper_name}` no llama a `_claim_push_cooldown_slot`. "
        f"Si no usa el helper, la race SELECT-then-UPDATE vuelve para "
        f"este flow."
    )


@pytest.mark.parametrize("helper_name", _HELPERS_REFACTORED)
def test_helper_no_legacy_select_then_update(helper_name: str):
    """El patrón legacy era: `execute_sql_query("SELECT health_profile ...")`
    + parse `last_iso` + comparar + `execute_sql_write("UPDATE ...")`.
    Si alguno de los helpers vuelve a tener AMBAS llamadas, la race
    se reintrodujo."""
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        rf"def\s+{re.escape(helper_name)}\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    has_select = "execute_sql_query" in body and "SELECT health_profile" in body
    has_write_helper_call = (
        "execute_sql_write" in body
        and "_claim_push_cooldown_slot" not in body
    )
    assert not (has_select and has_write_helper_call), (
        f"`{helper_name}` parece haber regresado al patrón "
        f"SELECT-then-UPDATE (sin pasar por `_claim_push_cooldown_slot`). "
        f"La race vuelve."
    )


def test_legacy_safe_fromisoformat_import_removed():
    """Si quedó algún `from constants import safe_fromisoformat as _sfi` en
    los helpers refactorizados, indica que aún parsean fechas en Python
    (señal del patrón legacy SELECT-then-UPDATE)."""
    src = _read(_CRON_TASKS_PATH)
    for helper_name in _HELPERS_REFACTORED:
        func_match = re.search(
            rf"def\s+{re.escape(helper_name)}\(.*?(?=\ndef\s+|\Z)",
            src, re.DOTALL,
        )
        body = func_match.group(0) if func_match else ""
        assert "safe_fromisoformat" not in body, (
            f"`{helper_name}` aún importa `safe_fromisoformat` — señal "
            f"de parseo de fechas Python (patrón legacy SELECT-then-"
            f"UPDATE). El CAS atómico no necesita parsear fechas en "
            f"el caller."
        )
