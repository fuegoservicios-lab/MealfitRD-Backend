"""[P3-NEW-9 · 2026-05-11] Coalesce inter-thread del emit
`plan_quality_degraded:*` vía advisory lock en `app_kv_store`.

Bug original (audit 2026-05-11):
    Dos requests concurrentes del mismo user terminando en `should_retry
    "end"` simultáneamente disparaban 2 INSERTs paralelos. La DB dedupea
    el row por `ON CONFLICT (alert_key)`, pero webhooks downstream
    (Sentry/Slack) podían N-firing.

Fix:
    Pre-INSERT, intentar UPSERT condicional sobre
    `app_kv_store.key = 'plan_quality_emit_lock:<user_id>:<plan_id>'`
    con `WHERE updated_at < NOW() - 60s`:
      - Fila nueva → INSERT → RETURNING → procede.
      - Fila stale (>60s) → UPDATE → RETURNING → procede.
      - Fila fresh (<60s) → WHERE falla → no RETURNING → skip.

    Best-effort: si el lock falla por KV down, caemos al comportamiento
    pre-P3-NEW-9 (emit sin coalesce, mejor que perder señal).

Tests parser-based:
    1. La query UPSERT condicional existe en el helper.
    2. La key format es `plan_quality_emit_lock:<user_id>:<plan_id>`.
    3. La ventana de coalesce es `60 seconds` (alineada con TTL razonable
       de should_retry concurrency).
    4. Si `lock_row` es falsy → return temprano (NO emit).
    5. Si el lock falla (excepción), el código DEBE proceder con emit
       (fallback best-effort).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_FP = _REPO_ROOT / "backend" / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _GRAPH_FP.read_text(encoding="utf-8")


def _extract_helper_body(src: str) -> str:
    start = src.find("def _emit_plan_quality_degraded_alert(")
    assert start > 0, "_emit_plan_quality_degraded_alert no encontrado"
    end = src.find("\ndef should_retry(", start)
    assert end > start, "fin de helper no encontrado"
    return src[start:end]


def test_lock_key_format(src: str):
    body = _extract_helper_body(src)
    assert re.search(
        r'f["\']plan_quality_emit_lock:\{user_id\}:\{plan_id\}["\']',
        body,
    ), (
        "P3-NEW-9 regresión: la key del lock ya no tiene formato canónico "
        "`plan_quality_emit_lock:{user_id}:{plan_id}`. Sin este formato, "
        "el coalesce no agrupa concurrent emits del mismo plan."
    )


def test_upsert_conditional_present(src: str):
    body = _extract_helper_body(src)
    # Patrón canónico: INSERT INTO app_kv_store ... ON CONFLICT ... DO UPDATE
    # WHERE app_kv_store.updated_at < NOW() - INTERVAL '60 seconds'.
    required_tokens = [
        "INSERT INTO app_kv_store",
        "ON CONFLICT",
        "DO UPDATE",
        "WHERE app_kv_store.updated_at < NOW() - INTERVAL '60 seconds'",
        "RETURNING key",
    ]
    for tok in required_tokens:
        assert tok in body, (
            f"P3-NEW-9 regresión: el query del lock ya no contiene `{tok}`. "
            f"Sin este token, el coalesce condicional no funciona como "
            f"diseñado."
        )


def test_skip_when_lock_fresh(src: str):
    """Si lock_row es falsy (UPSERT no actualizó nada) → return temprano."""
    body = _extract_helper_body(src)
    # Patrón: `if not lock_row:` seguido de `return` dentro de N líneas.
    pattern = re.compile(
        r"if\s+not\s+lock_row\s*:\s*\n"
        r"(?:[^\n]*\n){0,8}?"  # docs/log lines
        r"\s+return\s*\n",
    )
    assert pattern.search(body), (
        "P3-NEW-9 regresión: el path de skip cuando `lock_row` es falsy "
        "ya no retorna temprano. Sin return, el emit se ejecuta igualmente "
        "y el coalesce no tiene efecto."
    )


def test_fallback_on_lock_exception(src: str):
    """Si el lock falla (KV down), DEBE proceder con emit (best-effort)."""
    body = _extract_helper_body(src)
    # Patrón: try del UPSERT → except → asignar `lock_row` a un valor truthy
    # para que el if not lock_row NO dispare.
    pattern = re.compile(
        r"except\s+Exception\s+as\s+_lock_err:\s*\n"
        r"(?:[^\n]*\n){0,10}?"
        r"\s+lock_row\s*=\s*\{",
    )
    assert pattern.search(body), (
        "P3-NEW-9 regresión: el except del lock ya no asigna `lock_row` a "
        "un dict truthy (fallback best-effort). Sin esto, si KV está "
        "down, el helper salta TODOS los emits → perdemos señal SRE."
    )


def test_emit_call_after_lock_check(src: str):
    """El INSERT INTO system_alerts DEBE estar DESPUÉS del check `if not lock_row`."""
    body = _extract_helper_body(src)
    lock_check_idx = body.find("if not lock_row:")
    insert_idx = body.find("INSERT INTO system_alerts")
    assert lock_check_idx > 0 and insert_idx > 0
    assert lock_check_idx < insert_idx, (
        "P3-NEW-9 regresión: el check `if not lock_row` está DESPUÉS del "
        "INSERT INTO system_alerts. Eso es lógicamente inválido — el "
        "lock debe consultarse ANTES del emit."
    )
