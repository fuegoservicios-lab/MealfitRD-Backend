"""[P0-POOL-FREETIER-RETUNE · 2026-05-16] Tune del pool DB + spending-cap
fast-fail en embeddings.

Bug observado en logs prod 2026-05-16 17:46-17:59 (775s pipeline):
  - 10-15× warning "[CB-RESET] DB async no disponible: couldn't get a
    connection after 20.00 sec" durante un único plan.
  - Async pool max=12 + sync max=25 = 37 conns vs cap pgBouncer free tier
    ~15-30 → saturación permanente.
  - Cada embed query con spending cap activo gastaba 12s en 3 reintentos
    garantizados a fallar.

Fixes anclados por este test:
  1. `.env` tunea pool sync 25→20 + async 12→6 + async timeout 20→8s.
  2. `shopping_calculator._gemini_call_with_retry` detecta spending cap →
     set backoff global de 30 min → fast-fail en lugar de 3 retries.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"
_SHOPPING_PY = _BACKEND_ROOT / "shopping_calculator.py"


def test_env_pool_sync_lowered_for_freetier():
    """Pool sync max debe estar en ≤ 20 (no 25) para alinear con cap pgBouncer
    free tier ~15-30 combinado con async."""
    env = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(r"^MEALFIT_DB_POOL_MAX_SIZE=(\d+)\s*$", env, re.MULTILINE)
    assert m, "MEALFIT_DB_POOL_MAX_SIZE no seteado en .env"
    val = int(m.group(1))
    assert val <= 20, (
        f"MEALFIT_DB_POOL_MAX_SIZE={val} sigue alto para free tier. "
        f"Con async max=6, total {val}+6 debe ≤ 30 (cap pgBouncer)."
    )


def test_env_async_pool_knobs_present():
    """Los 3 knobs del pool async (MIN/MAX/TIMEOUT) deben estar en .env con
    valores conservadores para free tier.

    [stale-parser fix 2026-06-16] El umbral original de P0-POOL-FREETIER-RETUNE
    (async max ≤ 6, timeout ≤ 8) fue SUPERSEDED por dos retunes documentados
    posteriores en el mismo `.env`:
      - P1-BESTEFFORT-DB-CB (2026-05-21): async max 6→10 + timeout 8→12. El
        `BestEffortDBCB` local ahora fail-fasts tras 3 timeouts, así que el
        timeout bajo dejó de ser la palanca de fail-fast.
      - P1-ORCH-1-DBPOOL (2026-05-28): sync max 20→12 para dar headroom real
        al async dentro del cap pgBouncer.
    El invariante operacional real — `sync_max + async_max ≤ 30` — sigue
    enforced por `test_total_pool_within_pgbouncer_cap` (actual 12+10=22).
    Estas dos aserciones per-knob solo guardan contra la regresión al estado
    saturado (async max=12, timeout=20)."""
    env = _ENV_PATH.read_text(encoding="utf-8")
    # MAX async ≤ 10 (P1-BESTEFFORT-DB-CB; default code 12 satura combinado con sync)
    max_m = re.search(r"^MEALFIT_DB_ASYNC_POOL_MAX_SIZE=(\d+)\s*$", env, re.MULTILINE)
    assert max_m, "MEALFIT_DB_ASYNC_POOL_MAX_SIZE faltante en .env"
    assert int(max_m.group(1)) <= 10, (
        "Async pool max no debe exceder 10 (P1-BESTEFFORT-DB-CB); el default "
        "code es 12 que satura pgBouncer cuando se combina con sync."
    )
    # TIMEOUT async ≤ 12s (P1-BESTEFFORT-DB-CB; el BestEffortDBCB hace fail-fast,
    # no el timeout bajo — vs default 20s que desperdicia wall-clock en saturación)
    to_m = re.search(r"^MEALFIT_DB_ASYNC_POOL_TIMEOUT_S=(\d+)\s*$", env, re.MULTILINE)
    assert to_m, "MEALFIT_DB_ASYNC_POOL_TIMEOUT_S faltante en .env"
    assert int(to_m.group(1)) <= 12, (
        "Async pool timeout debe ser ≤12s (P1-BESTEFFORT-DB-CB) para que "
        "best-effort writes (CB-RESET, LLM-CACHE, etc.) no bloqueen el plan; "
        "el BestEffortDBCB fail-fasts tras 3 timeouts."
    )


def test_total_pool_within_pgbouncer_cap():
    """Sum sync_max + async_max ≤ 30 (pgBouncer free tier upper bound).
    Cualquier configuración por encima vuelve a saturar."""
    env = _ENV_PATH.read_text(encoding="utf-8")
    sync = int(re.search(r"^MEALFIT_DB_POOL_MAX_SIZE=(\d+)", env, re.MULTILINE).group(1))
    asyc = int(re.search(r"^MEALFIT_DB_ASYNC_POOL_MAX_SIZE=(\d+)", env, re.MULTILINE).group(1))
    total = sync + asyc
    assert total <= 30, (
        f"Pool total {sync}+{asyc}={total} excede cap pgBouncer free tier (~30). "
        f"Esto causa saturación + 'couldn't get a connection after Ns' warnings. "
        f"Si migraste a Supabase Pro, ignorar este test y subir el bound."
    )


def test_spending_cap_detector_helper_exists():
    """`_is_gemini_spending_cap_error` debe existir y detectar las frases
    canónicas del mensaje de Google ('spending cap', 'ai.studio/spend')."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    assert "def _is_gemini_spending_cap_error(exc: Exception)" in src, (
        "Helper de detección de spending cap ausente."
    )
    # Tres substrings canónicos
    for needle in ('"spending cap"', '"monthly spending"', '"ai.studio/spend"'):
        assert needle in src, (
            f"Detector no chequea substring {needle!r} — el mensaje de Google "
            f"puede mutar; cubrir múltiples variantes."
        )


def test_global_backoff_window_set_on_spending_cap():
    """Al detectar spending cap, el helper debe setear el backoff global +
    raise. Cualquier subsecuente call entra en fast-fail sin retry."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    assert "_GEMINI_SPENDING_CAP_BACKOFF_S" in src, (
        "Constante de duración del backoff global ausente."
    )
    assert "_gemini_spending_cap_backoff_until" in src, (
        "Variable de estado del backoff window ausente."
    )
    # En la rama spending cap, se setea el backoff antes de raise.
    cap_branch = re.search(
        r"if _is_gemini_spending_cap_error\(exc\):.*?raise",
        src,
        re.DOTALL,
    )
    assert cap_branch, "Rama de detección de spending cap no encontrada en retry."
    body = cap_branch.group(0)
    assert "_gemini_spending_cap_backoff_until" in body, (
        "La rama de spending cap no setea el backoff global — el siguiente "
        "embed call volverá a hacer los 3 retries inútiles."
    )


def test_fast_fail_check_at_entry_of_retry():
    """Al entrar `_gemini_call_with_retry`, si el backoff está activo,
    raise inmediato SIN iniciar los 3 reintentos."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def _gemini_call_with_retry\([^)]+\):.*?(?=\n_master_cache|\Z)",
        src,
        re.DOTALL,
    )
    assert fn, "Función `_gemini_call_with_retry` no encontrada."
    body = fn.group(0)
    # El check del backoff debe estar ANTES del for loop de retries
    backoff_idx = body.find("_gemini_spending_cap_backoff_until > _time.time()")
    for_idx = body.find("for attempt in range(3):")
    assert backoff_idx >= 0, "Check de backoff window ausente al entrar."
    assert for_idx >= 0, "Loop de retries no encontrado."
    assert backoff_idx < for_idx, (
        "Check del backoff window debe ejecutar ANTES del loop. Si está "
        "después, los 3 retries corren igual y desperdician 10-12s/call."
    )
