"""[P2-REGEN-DAY-IDEMPOTENCY · 2026-06-26] (audit 3-flujos P2) `/regenerate-day` debe ser idempotente
ante un retry tras corte de red: una segunda llamada del MISMO día dentro de una ventana corta NO debe
re-correr el loop LLM ni RE-COBRAR — debe devolver los platos ya persistidos (already_applied).

Pre-fix: `api_regenerate_day` es PAGADO (1 crédito) + LENTO (~1 min, N swaps LLM) y SÍNCRONO (persiste +
cobra aunque el cliente se desconecte). Sin guard de idempotencia, un retry tras perder la respuesta HTTP
re-corría el loop, RE-COBRABA y pisaba el día (mismo modo que cerró P1-DEDUP-RECENT-PLAN, pero a nivel-día).

Fix: marker en `app_kv_store` keyed por (user, plan, day), seteado post-persist + post-cobro
(`mark_regen_day_done`); el inicio del endpoint lo lee (`check_recent_regen_day`) y corta el retry.
Knob MEALFIT_REGEN_DAY_DEDUP_SECONDS (default 45, 0=off, clamp [0,600]).

Parser-based para el wiring del router + unit del key del KV (sin DB).
"""
import ast
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


PLANS = _read("routers/plans.py")
DBP = _read("db_plans.py")
REGEN = _func_src(PLANS, "api_regenerate_day")


# ── Helpers del KV en db_plans ─────────────────────────────────────────────────
def test_kv_helpers_defined():
    for fn in ("def mark_regen_day_done(", "def check_recent_regen_day(", "def _regen_day_dedup_kv_key("):
        assert fn in DBP, f"{fn} ausente en db_plans.py"


def test_check_is_fail_open_and_windowed():
    chk = _func_src(DBP, "check_recent_regen_day")
    assert "max_seconds <= 0" in chk, "max_seconds<=0 debe desactivar (fail-open)"
    assert "return False" in chk, "ante error/ausencia → False (el update procede)"
    assert "total_seconds()" in chk and "max_seconds" in chk, "debe comparar la edad contra la ventana"


# ── Wiring en el router ─────────────────────────────────────────────────────────
def test_dedup_check_precedes_charge():
    assert "check_recent_regen_day(" in REGEN, "falta el guard de idempotencia al inicio"
    assert '"already_applied": True' in REGEN, "debe devolver already_applied en el retry duplicado"
    assert "MEALFIT_REGEN_DAY_DEDUP_SECONDS" in REGEN, "knob de la ventana ausente"
    check_idx = REGEN.index("check_recent_regen_day(")
    charge_idx = REGEN.index('log_api_usage(user_id, "llm_regenerate_day")')
    assert check_idx < charge_idx, "el check de idempotencia debe PRECEDER al cobro (no re-cobrar el retry)"


def test_already_applied_returns_persisted_meals_and_no_recharge():
    # el early-return del retry duplicado devuelve los meals ya persistidos y NO llega al cobro.
    early = REGEN[:REGEN.index('log_api_usage(user_id, "llm_regenerate_day")')]
    assert '"already_applied": True' in early, "el return already_applied va ANTES del cobro"
    assert '"meals": meals' in early, "devuelve los platos del día ya persistido"


def test_mark_done_after_charge_and_gated_by_ai_ok():
    assert "mark_regen_day_done(user_id, plan_id, day_index)" in REGEN, "falta el set del marker post-éxito"
    charge_idx = REGEN.index('log_api_usage(user_id, "llm_regenerate_day")')
    mark_idx = REGEN.index("mark_regen_day_done(user_id, plan_id, day_index)")
    assert charge_idx < mark_idx, "el marker se setea DESPUÉS del cobro (post-éxito)"
    # gateado por el mismo `if not _ai_unavailable` que el cobro (una interrupción NO marca el día)
    block = REGEN[REGEN.index("if not _ai_unavailable:"): mark_idx + 60]
    assert "if not _ai_unavailable:" in block and "mark_regen_day_done" in block, \
        "mark_regen_day_done debe vivir dentro del bloque `if not _ai_unavailable`"


# ── Unit del key del KV (sin DB) ────────────────────────────────────────────────
try:
    from db_plans import _regen_day_dedup_kv_key
    _DB_ERR = None
except Exception as _e:  # pragma: no cover
    _regen_day_dedup_kv_key = None
    _DB_ERR = _e


@pytest.mark.skipif(_regen_day_dedup_kv_key is None, reason=f"db_plans no importable: {_DB_ERR}")
def test_kv_key_is_namespaced_and_distinct():
    assert _regen_day_dedup_kv_key("u1", "p1", 3) == "regen_day_done:u1:p1:3"
    # day_index coercionado a int (acepta str del body)
    assert _regen_day_dedup_kv_key("u1", "p1", "3") == "regen_day_done:u1:p1:3"
    # claves distintas por día / plan / user → sin colisión cross-entidad
    k = _regen_day_dedup_kv_key
    assert k("u1", "p1", 0) != k("u1", "p1", 1)
    assert k("u1", "p1", 0) != k("u2", "p1", 0)
    assert k("u1", "p1", 0) != k("u1", "p2", 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
