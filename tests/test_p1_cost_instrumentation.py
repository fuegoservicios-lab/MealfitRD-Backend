"""[P1-COST-INSTRUMENTATION · 2026-05-15] Regression guards para la
instrumentación de costo por llamada LLM.

Pre-fix: `api_usage` solo contaba invocaciones para el paywall mensual
(gratis=15/basic=50/plus=200), sin tokens ni costo monetario. Audit
2026-05-15 estimó ~$0.06-$0.15/plan pero el sistema estaba ciego al
costo real → imposible decidir optimizaciones (context caching, retry
budgets, model swaps).

Fix:
  1. Tabla `llm_usage_events` (migración SSOT
     `migrations/p1_cost_instrumentation_2026_05_15.sql`).
  2. `db_profiles.compute_llm_cost_micros(model, in, out, cached)` con
     pricing dict (override via knob `MEALFIT_LLM_PRICING_JSON`).
  3. `db_profiles.log_llm_usage_event(...)` — INSERT best-effort.
  4. `graph_orchestrator._emit_llm_usage_event_best_effort(...)` invocado
     desde el path exitoso de `_safe_ainvoke` (tras `wait_for` ok, ANTES
     del `return result`).

Estos tests son parser-based (no requieren DB ni red) + un grupo
funcional puro sobre `compute_llm_cost_micros`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_PROFILES_PATH = _BACKEND_ROOT / "db_profiles.py"
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"
_MIGRATION_PATH = (
    _BACKEND_ROOT / "migrations"
    / "p1_cost_instrumentation_2026_05_15.sql"
)


# ----- Migración SSOT ------------------------------------------------------

def test_migration_file_exists():
    assert _MIGRATION_PATH.exists(), (
        "P1-COST-INSTRUMENTATION: falta la migración "
        "migrations/p1_cost_instrumentation_2026_05_15.sql"
    )


def test_migration_creates_table_idempotent():
    text = _MIGRATION_PATH.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.llm_usage_events" in text, (
        "Migración debe usar CREATE TABLE IF NOT EXISTS para ser idempotente "
        "(convención P3-MIGRATION-IDEMPOTENCE-DOC)."
    )
    for col in (
        "user_id uuid",
        "plan_id uuid",
        "model text NOT NULL",
        "node text",
        "input_tokens int",
        "output_tokens int",
        "cached_tokens int",
        "cost_usd_micros bigint",
        "metadata jsonb",
    ):
        assert col in text, f"Migración debe definir columna `{col}`."


def test_migration_enables_rls_service_role_only():
    text = _MIGRATION_PATH.read_text(encoding="utf-8")
    assert "ENABLE ROW LEVEL SECURITY" in text
    assert "FORCE ROW LEVEL SECURITY" in text
    assert "TO service_role" in text, (
        "RLS debe restringir a service_role (tabla operacional, no consumida "
        "por frontend — mismo patrón que pipeline_metrics / meal_plans_audit)."
    )


def test_migration_indexes_present():
    text = _MIGRATION_PATH.read_text(encoding="utf-8")
    # Queries esperadas: time-window + group by model/user.
    assert "idx_llm_usage_events_created_at" in text
    assert "idx_llm_usage_events_model_created" in text
    assert "idx_llm_usage_events_user_id" in text


# ----- db_profiles.py helpers ---------------------------------------------

def test_db_profiles_exports_compute_and_logger():
    text = _DB_PROFILES_PATH.read_text(encoding="utf-8")
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] renombrada desde compute_gemini_*.
    assert re.search(r"^def compute_llm_cost_micros\(", text, re.MULTILINE), (
        "P1-COST-INSTRUMENTATION: `compute_llm_cost_micros` debe existir "
        "como función top-level en db_profiles.py."
    )
    assert re.search(r"^def log_llm_usage_event\(", text, re.MULTILINE), (
        "P1-COST-INSTRUMENTATION: `log_llm_usage_event` debe existir "
        "como función top-level en db_profiles.py."
    )


def test_log_llm_usage_event_kill_switch_present():
    """Knob `MEALFIT_LLM_COST_TRACKING_ENABLED` debe servir como kill switch
    sin redeploy si la instrumentación causa contención inesperada."""
    text = _DB_PROFILES_PATH.read_text(encoding="utf-8")
    assert "MEALFIT_LLM_COST_TRACKING_ENABLED" in text, (
        "P1-COST-INSTRUMENTATION: falta knob kill-switch "
        "`MEALFIT_LLM_COST_TRACKING_ENABLED` en db_profiles.py."
    )


def test_compute_llm_cost_micros_pro_pricing():
    """[P0-DEEPSEEK-MIGRATION] deepseek-v4-pro a 1M in + 1M out debe coincidir
    con la tabla de pricing default (input $0.435 + output $0.87 = $1.305 →
    1_305_000 micros)."""
    from db_profiles import compute_llm_cost_micros
    cost = compute_llm_cost_micros(
        "deepseek-v4-pro",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cached_tokens=0,
    )
    assert cost == 1_305_000, (
        f"Pricing Pro inesperado: {cost} (esperado 1_305_000 = $1.305). "
        "Si DeepSeek cambió precios, actualizar `_DEFAULT_LLM_PRICING_MICROS_PER_M` "
        "Y este assert juntos."
    )


def test_compute_llm_cost_micros_flash_pricing():
    """deepseek-v4-flash a 1M in + 1M out = $0.14 + $0.28 = $0.42 → 420_000."""
    from db_profiles import compute_llm_cost_micros
    cost = compute_llm_cost_micros(
        "deepseek-v4-flash",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cached_tokens=0,
    )
    assert cost == 420_000, (
        f"Pricing Flash inesperado: {cost} (esperado 420_000 = $0.42)."
    )


def test_compute_llm_cost_micros_cached_discount():
    """Cached tokens facturan al rate de cache-hit. Pro: 2M input TODO
    cacheado, 0 output = 2M × $0.003625/M = $0.00725 → 7_250 micros."""
    from db_profiles import compute_llm_cost_micros
    cost = compute_llm_cost_micros(
        "deepseek-v4-pro",
        input_tokens=2_000_000,
        output_tokens=0,
        cached_tokens=2_000_000,
    )
    # billable_input = max(0, 2M - 2M) = 0; cached = 2M × 3_625 / 1M = 7_250.
    assert cost == 7_250, (
        f"Cache discount mal aplicado: {cost} (esperado 7_250)."
    )


def test_compute_llm_cost_micros_unknown_model_returns_none():
    from db_profiles import compute_llm_cost_micros
    assert compute_llm_cost_micros("claude-opus-9000", 1000, 500) is None
    # Los modelos Gemini retirados también son "desconocidos" post-migración.
    assert compute_llm_cost_micros("gemini-3.5-flash", 1000, 500) is None


def test_compute_llm_cost_micros_handles_missing_tokens():
    from db_profiles import compute_llm_cost_micros
    assert compute_llm_cost_micros("deepseek-v4-pro", None, 500) is None
    assert compute_llm_cost_micros("deepseek-v4-pro", 1000, None) is None
    assert compute_llm_cost_micros(None, 1000, 500) is None


# ----- _safe_ainvoke instrumentación --------------------------------------

def _extract_function_body(text: str, func_name: str) -> str:
    """Extrae el cuerpo de una `async def` o `def` top-level por nombre."""
    pattern = re.compile(
        rf"^(async\s+)?def\s+{re.escape(func_name)}\b.*?(?=^\S)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    assert m, f"No encontré la función `{func_name}` en graph_orchestrator.py"
    return m.group(0)


def test_safe_ainvoke_emits_usage_event_on_success():
    """En el path exitoso (tras `wait_for` ok, ANTES del `return result`)
    `_safe_ainvoke` debe llamar a `_emit_llm_usage_event_best_effort`.
    Sin esto, la tabla `llm_usage_events` queda vacía y la instrumentación
    no captura nada en producción."""
    text = _GRAPH_PATH.read_text(encoding="utf-8")
    body = _extract_function_body(text, "_safe_ainvoke")
    assert "_emit_llm_usage_event_best_effort" in body, (
        "P1-COST-INSTRUMENTATION: `_safe_ainvoke` debe invocar "
        "`_emit_llm_usage_event_best_effort(...)` en el happy-path antes "
        "del `return`. Sin esto, la instrumentación es no-op."
    )
    # Debe estar ANTES del return (no en una rama de error).
    emit_idx = body.find("_emit_llm_usage_event_best_effort")
    # El primer `return result` del happy-path debe venir después del emit.
    try_idx = body.find("try:")
    return_result_idx = body.find("return result", try_idx)
    assert emit_idx > 0 and return_result_idx > emit_idx, (
        "El emit debe estar entre `wait_for` exitoso y `return result`."
    )


def test_helper_emit_usage_event_defined_top_level():
    text = _GRAPH_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"^def _emit_llm_usage_event_best_effort\(", text, re.MULTILINE
    ), (
        "Helper `_emit_llm_usage_event_best_effort` debe estar definido "
        "top-level en graph_orchestrator.py."
    )


def test_helper_uses_log_llm_usage_event():
    """El helper debe delegar la persistencia a `db_profiles.log_llm_usage_event`
    (SSOT) — no debe duplicar la lógica de INSERT inline."""
    text = _GRAPH_PATH.read_text(encoding="utf-8")
    body = _extract_function_body(text, "_emit_llm_usage_event_best_effort")
    assert "log_llm_usage_event" in body, (
        "Helper debe delegar persistencia a db_profiles.log_llm_usage_event."
    )


def test_helper_is_defensive_against_missing_usage_metadata():
    """Responses sin `usage_metadata` (modelos legacy, fallbacks) no deben
    persistir filas placeholder vacías ni crashear."""
    text = _GRAPH_PATH.read_text(encoding="utf-8")
    body = _extract_function_body(text, "_emit_llm_usage_event_best_effort")
    # Debe checkear que usage existe Y es dict ANTES de leer sub-claves.
    assert "if not usage" in body or "if usage is None" in body, (
        "Helper debe early-return si `usage_metadata` está ausente."
    )
    assert "best-effort" in body.lower() or "try:" in body, (
        "Helper debe ser best-effort (try/except global)."
    )
