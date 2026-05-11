"""[P1-NEW-3 · 2026-05-11] Helper `_emit_plan_quality_degraded_alert`
en `graph_orchestrator.py` emite `system_alerts.plan_quality_degraded:*`
cuando `should_retry` decide entregar plan SIN `review_passed=True`.

Bug original (audit 2026-05-11):
    Antes, las 5 ramas "end" de `should_retry` (critical, high_contextual,
    max_attempts, invalid_pipeline_start, budget_exhausted) entregaban
    el plan degradado sin señal a SRE. Solo el
    `_shopping_coherence_block_history.action_taken=reject_minor` quedaba
    en `plan_data` — no escalable.

Fix: helper que upsert idempotente en `system_alerts` con
`alert_key=plan_quality_degraded:<user_id>:<plan_id>`. Invocado en
las 5 ramas con `if not state.get("review_passed", False)` defensivo.

Estrategia del test (parser-based):
    1. Helper definido top-level.
    2. Hace INSERT INTO system_alerts ON CONFLICT (alert_key) DO UPDATE.
    3. alert_key sigue patrón `plan_quality_degraded:{user_id}:{plan_id}`.
    4. Las 5 ramas "end" de should_retry invocan el helper.
    5. Cada call site usa un `exit_reason` distinto.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_ORCHESTRATOR = _REPO_ROOT / "backend" / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _GRAPH_ORCHESTRATOR.read_text(encoding="utf-8")


def _extract_function_body(text: str, name: str) -> str:
    m = re.search(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"función `{name}` no encontrada"
    return m.group(1)


def test_helper_defined(src: str):
    """`_emit_plan_quality_degraded_alert` debe existir como top-level."""
    assert re.search(
        r"^def\s+_emit_plan_quality_degraded_alert\s*\(",
        src,
        re.MULTILINE,
    ), (
        "P1-NEW-3 regresión: helper `_emit_plan_quality_degraded_alert` "
        "no existe. Sin él, no hay observabilidad de planes degradados "
        "entregados al usuario."
    )


def test_helper_inserts_into_system_alerts(src: str):
    """El helper debe persistir a `system_alerts` (no solo log)."""
    body = _extract_function_body(src, "_emit_plan_quality_degraded_alert")
    assert re.search(r"INSERT\s+INTO\s+system_alerts", body, re.IGNORECASE), (
        "P1-NEW-3 regresión: helper no escribe a system_alerts. "
        "Solo logs sería insuficiente — SRE necesita la tabla para "
        "dashboards."
    )


def test_helper_uses_alert_key_namespace(src: str):
    """alert_key debe seguir patrón `plan_quality_degraded:<user_id>:<plan_id>`
    para ser greppable + único por incidente."""
    body = _extract_function_body(src, "_emit_plan_quality_degraded_alert")
    assert "plan_quality_degraded" in body, (
        "P1-NEW-3 regresión: alert_key no usa namespace "
        "`plan_quality_degraded`. Sin namespace consistente, no se "
        "puede greppear ni filtrar en dashboards."
    )


def test_helper_uses_upsert_pattern(src: str):
    """ON CONFLICT (alert_key) DO UPDATE — idempotencia + dedupe."""
    body = _extract_function_body(src, "_emit_plan_quality_degraded_alert")
    assert re.search(
        r"ON\s+CONFLICT\s*\(\s*alert_key\s*\)\s+DO\s+UPDATE",
        body,
        re.IGNORECASE,
    ), (
        "P1-NEW-3 regresión: helper no usa UPSERT. Sin idempotencia, "
        "repetir la misma exit_reason crea filas duplicadas."
    )


def test_helper_is_best_effort(src: str):
    """Try/except envuelve el body — un fallo del emit NO debe abortar
    el routing del pipeline."""
    body = _extract_function_body(src, "_emit_plan_quality_degraded_alert")
    assert "try:" in body and "except" in body, (
        "P1-NEW-3 regresión: helper sin try/except. Si DB blip, el "
        "fallo propaga al routing del pipeline → user sufre un error "
        "técnico por una alerta opcional."
    )


def test_should_retry_invokes_helper_in_all_end_branches(src: str):
    """Las 5 ramas "end" de should_retry deben invocar el helper con
    un `exit_reason` distinto."""
    body = _extract_function_body(src, "should_retry")
    expected_reasons = {
        "critical",
        "high_contextual",
        "max_attempts",
        "invalid_pipeline_start",
        "budget_exhausted",
    }
    found = set()
    for reason in expected_reasons:
        pattern = re.compile(
            rf'_emit_plan_quality_degraded_alert\([^)]*exit_reason\s*=\s*["\']{re.escape(reason)}["\']',
        )
        if pattern.search(body):
            found.add(reason)
    missing = expected_reasons - found
    assert not missing, (
        f"P1-NEW-3 regresión: ramas de should_retry sin invocar el "
        f"helper con exit_reason esperado: {sorted(missing)}. "
        f"Cada rama 'end' donde review_passed=False debe emitir alert."
    )


def test_alert_key_documented_in_claude_md():
    """CLAUDE.md tabla 'Política system_alerts' debe documentar
    `plan_quality_degraded:*` con productor + resolver + modelo."""
    claude_md = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "plan_quality_degraded:<user_id>:<plan_id>" in claude_md, (
        "P1-NEW-3 regresión: CLAUDE.md no documenta el alert_key "
        "`plan_quality_degraded:<user_id>:<plan_id>`. Sin documentación, "
        "el test `test_p2_audit_4_alert_keys_documented` fallará."
    )
