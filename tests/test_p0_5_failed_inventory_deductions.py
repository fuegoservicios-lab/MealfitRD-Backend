"""[P0-5 · 2026-05-10] Regression guard: `failed_inventory_deductions` ya NO
es tabla huérfana — recibe INSERTs cuando una deducción falla y un cron
lee + alerta cuando el backlog supera umbral.

Bug original (audit 2026-05-10):
    La tabla `failed_inventory_deductions` existía con RLS forzado
    (`p1_hist_audit_6_rls_force.sql`) e índice KEEP en `user_id`
    (`p2_perf_1_consolidate_unused_index_comments.sql`), pero:
      - NADIE escribía a ella. `deduct_consumed_meal_from_inventory`
        capturaba excepciones con `logger.error` y seguía sin persistir.
      - NADIE leía la tabla. No había cron ni endpoint admin que la
        surfacie. Tabla write-only orphan.
    Resultado: ingredientes que no se podían deducir (parse errors,
    excepción en RPC, master_ingredients missing) desaparecían en logs
    locales sin observabilidad post-mortem.

Fix dual (P0-5):
    A) WRITE — `db_inventory._persist_failed_inventory_deductions` +
       captura per-item en `deduct_consumed_meal_from_inventory`. Items
       con `parse_failed_or_invalid_qty`, `deduction_returned_false`, o
       `exception` se persisten como jsonb array.
    B) READ — cron `_alert_failed_inventory_deductions_backlog`:
       lee `failed_inventory_deductions` en ventana móvil (knob
       `MEALFIT_FAILED_DEDUCTIONS_ALERT_LOOKBACK_H` default 24h),
       cuenta filas; si supera umbral (`MEALFIT_FAILED_DEDUCTIONS_ALERT_THRESHOLD`
       default 25) emite `system_alerts` (severity=warning, idempotente
       vía UPSERT por `alert_key`).

Cobertura de este test (parser-based, no DB):
    1. `db_inventory.py` declara `_persist_failed_inventory_deductions`
       y `deduct_consumed_meal_from_inventory` la invoca.
    2. La función de cron `_alert_failed_inventory_deductions_backlog`
       existe en cron_tasks.py.
    3. El cron está registrado en `register_plan_chunk_scheduler` con
       job_id canónico.
    4. Los 3 knobs nuevos (`THRESHOLD`, `LOOKBACK_H`, `INTERVAL_MIN`)
       están en el código y pasan por `_env_int` (auto-registro en
       `_KNOBS_REGISTRY`).

Out of scope (gaps para P-fixes posteriores):
    - Retry automático per-row con `attempts` counter. La columna
      `attempts` existe (default 0) pero el cron actual NO incrementa
      ni reintenta. Decisión P0-5: el retry es invasivo (necesita
      preservar contexto de unidad/master) y deserve diseño explícito.
    - Endpoint admin `/admin/failed-deductions` para inspección manual.
    - Schema-aware test contra DB real (smoke INSERT + SELECT + cron run).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_INVENTORY_PATH = _BACKEND_ROOT / "db_inventory.py"
_CRON_TASKS_PATH = _BACKEND_ROOT / "cron_tasks.py"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. WRITE path: `_persist_failed_inventory_deductions` existe y se invoca.
# ---------------------------------------------------------------------------
def test_persist_failed_function_exists():
    src = _read(_DB_INVENTORY_PATH)
    assert re.search(
        r"def\s+_persist_failed_inventory_deductions\s*\(",
        src,
    ), (
        "`_persist_failed_inventory_deductions` debe existir en db_inventory.py. "
        "Si lo movieron a otro módulo, actualizar este test."
    )


def test_persist_failed_inserts_to_correct_table():
    src = _read(_DB_INVENTORY_PATH)
    # Buscamos el body de la función específica.
    func_match = re.search(
        r"def\s+_persist_failed_inventory_deductions\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None
    body = func_match.group(0)
    assert "failed_inventory_deductions" in body, (
        "`_persist_failed_inventory_deductions` debe escribir a la tabla "
        "`failed_inventory_deductions` (no a otra)."
    )
    # [P1-NEON-DB-MIGRATION · 2026-06-12] El INSERT viaja por SQL directo
    # (antes `supabase.table(...).insert(...)`). Las columnas canónicas
    # (user_id, ingredients, attempts) deben preservarse.
    assert re.search(
        r"INSERT INTO failed_inventory_deductions\s*\(user_id,\s*ingredients,\s*attempts\)",
        body,
    ), (
        "Debe usar `execute_sql_write(\"INSERT INTO failed_inventory_deductions "
        "(user_id, ingredients, attempts) VALUES ...\")`."
    )


def test_deduct_consumed_meal_calls_persist_failed():
    """`deduct_consumed_meal_from_inventory` debe invocar al helper
    de persistencia. Si alguien refactoriza y olvida el call, los
    fallos vuelven a desaparecer en logs."""
    src = _read(_DB_INVENTORY_PATH)
    func_match = re.search(
        r"def\s+deduct_consumed_meal_from_inventory\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None
    body = func_match.group(0)
    assert "_persist_failed_inventory_deductions" in body, (
        "`deduct_consumed_meal_from_inventory` debe invocar a "
        "`_persist_failed_inventory_deductions` con los items que fallaron. "
        "Sin esto, los fallos vuelven a desaparecer (P0-5 regresión)."
    )


def test_deduct_captures_three_failure_reasons():
    """Las 3 razones canónicas de fallo deben quedar registradas:
    parse_failed_or_invalid_qty, deduction_returned_false, exception."""
    src = _read(_DB_INVENTORY_PATH)
    func_match = re.search(
        r"def\s+deduct_consumed_meal_from_inventory\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    expected_reasons = [
        "parse_failed_or_invalid_qty",
        "deduction_returned_false",
        "exception",
    ]
    for reason in expected_reasons:
        assert reason in body, (
            f"`deduct_consumed_meal_from_inventory` no captura el reason "
            f"`{reason}`. Las 3 razones canónicas deben coexistir para "
            f"que el cron de alerta tenga señal suficiente."
        )


# ---------------------------------------------------------------------------
# 2. READ path: cron de alerta existe + registrado.
# ---------------------------------------------------------------------------
def test_alert_cron_function_exists():
    src = _read(_CRON_TASKS_PATH)
    assert re.search(
        r"def\s+_alert_failed_inventory_deductions_backlog\s*\(",
        src,
    ), "`_alert_failed_inventory_deductions_backlog` debe existir en cron_tasks.py."


def test_alert_cron_queries_correct_table():
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        r"def\s+_alert_failed_inventory_deductions_backlog\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None
    body = func_match.group(0)
    assert "failed_inventory_deductions" in body, (
        "Cron debe consultar la tabla `failed_inventory_deductions`."
    )
    # Debe usar `make_interval(hours => …::int)` (no `interval N hours` literal —
    # el knob viene como int variable).
    assert re.search(r"make_interval\s*\(\s*hours\s*=>", body, re.IGNORECASE), (
        "Cron debe usar `make_interval(hours => ...::int)` (lección P0-HIST-FIX-1: "
        "Postgres requiere int explícito, no float ni string)."
    )


def test_alert_cron_emits_system_alerts():
    src = _read(_CRON_TASKS_PATH)
    func_match = re.search(
        r"def\s+_alert_failed_inventory_deductions_backlog\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    assert "system_alerts" in body, "Cron debe emitir a `system_alerts`."
    assert "failed_inventory_deductions_backlog" in body, (
        "Cron debe usar `alert_type='failed_inventory_deductions_backlog'` "
        "para que el operador pueda filtrar."
    )
    assert "ON CONFLICT (alert_key)" in body, (
        "INSERT debe ser idempotente vía UPSERT por `alert_key` "
        "(no flooding de alertas duplicadas)."
    )


def test_cron_registered_in_scheduler():
    """El cron debe estar registrado en `register_plan_chunk_scheduler` —
    si solo se define pero nadie lo registra, jamás corre."""
    src = _read(_CRON_TASKS_PATH)
    reg_match = re.search(
        r"def\s+register_plan_chunk_scheduler\(.*?(?=\Z)",
        src, re.DOTALL,
    )
    assert reg_match is not None, "register_plan_chunk_scheduler no encontrado."
    reg_body = reg_match.group(0)
    assert "alert_failed_inventory_deductions_backlog" in reg_body, (
        "El cron NO está registrado en `register_plan_chunk_scheduler`. "
        "Sin registro, APScheduler no lo dispara. Job_id esperado: "
        "`alert_failed_inventory_deductions_backlog`."
    )
    assert "_alert_failed_inventory_deductions_backlog" in reg_body, (
        "El registro debe pasar la función `_alert_failed_inventory_deductions_backlog` "
        "como callable. Sin esto, APScheduler ejecuta el job vacío."
    )


# ---------------------------------------------------------------------------
# 3. Knobs nuevos pasan por `_env_int` (auto-registro)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("knob_name,default", [
    ("MEALFIT_FAILED_DEDUCTIONS_ALERT_THRESHOLD", 25),
    ("MEALFIT_FAILED_DEDUCTIONS_ALERT_LOOKBACK_H", 24),
    ("MEALFIT_FAILED_DEDUCTIONS_ALERT_INTERVAL_MIN", 60),
])
def test_knob_uses_env_int_helper(knob_name, default):
    """Cada knob nuevo debe pasar por `_env_int(...)` con el default
    correcto. Esto los auto-registra en `_KNOBS_REGISTRY` y los expone
    en `/health/version` y `/admin/knobs`. Knobs raw `os.environ.get`
    pasan invisibles al operador."""
    src = _read(_CRON_TASKS_PATH)
    pattern = rf"_env_int\(\s*[\"']{re.escape(knob_name)}[\"']\s*,\s*{default}\s*\)"
    assert re.search(pattern, src), (
        f"Knob `{knob_name}` debe usar `_env_int(\"{knob_name}\", {default})` "
        f"con el default correcto. Sin esto, no aparece en /health/version."
    )
