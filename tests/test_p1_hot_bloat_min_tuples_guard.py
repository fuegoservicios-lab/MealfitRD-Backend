"""[P1-HOT-BLOAT-MIN-TUPLES · 2026-05-13] Guard contra false-positives del
cron `_emit_hot_table_bloat_tick` sobre tablas tiny.

Contexto:
    Audit production-readiness 2026-05-12 detectó 2 alerts `hot_table_bloat`
    vivas sobre `meal_plans` (3 live + 17 dead → 85%) y `agent_sessions`
    (3 live + 14 dead → 82%) que jamás auto-resolvían. Postgres autovacuum
    no corre sobre tablas con tan pocas filas: threshold = 50 + (0.20 ×
    n_live) → para 3 filas vivas necesita ≥51 dead tuples antes de
    disparar. El check `dead_pct > 70%` del cron (P2-AUDIT-2) es entonces
    false positive permanente sobre tablas tiny → alert fatigue.

    Fix añade knob `MEALFIT_AUTOVACUUM_MIN_TOTAL_TUPLES` (default 100,
    clamp [10, 10000]) que skipea el check si `n_live + n_dead <
    min_total_tuples`. Skip emite `_hot_table_bloat_skipped_low_volume` a
    pipeline_metrics (observabilidad) y AUTO-RESUELVE cualquier alerta
    previa (para que el deploy cierre las 2 vivas sin MCP manual).

Lo que este test enforza:
  A) Knob `MEALFIT_AUTOVACUUM_MIN_TOTAL_TUPLES` leído via `_env_int` con
     default literal 100.
  B) Clamp `max(10, min(..., 10000))` aplicado al valor del knob.
  C) Guard `if total_tuples < min_total_tuples` PRECEDE al cómputo
     `is_bloated = ... and ...` (orden importa — si va después, el alert
     ya se upserted antes del skip).
  D) Branch del skip llama `INSERT INTO pipeline_metrics` con node
     `_hot_table_bloat_skipped_low_volume` (observabilidad continua).
  E) Branch del skip llama `UPDATE system_alerts SET resolved_at = NOW()`
     (auto-resuelve alertas previas — cierra las 2 vivas en deploy).
  F) Branch del skip termina con `continue` (no cae al alert decision).
  G) Anchor textual `P1-HOT-BLOAT-MIN-TUPLES` presente en el cuerpo.

Tooltip-anchor: P1-HOT-BLOAT-MIN-TUPLES.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_TASKS = _REPO_ROOT / "backend" / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


def _extract_function(src: str, name: str) -> str:
    pattern = re.compile(
        rf"def {re.escape(name)}\b.*?(?=^def\s)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"P1-HOT-BLOAT-MIN-TUPLES: no localicé `def {name}` en cron_tasks.py."
    return m.group(0)


# A) Knob leído via _env_int con default 100.
def test_a_knob_read_with_default_100(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    assert re.search(
        r'_env_int\(\s*"MEALFIT_AUTOVACUUM_MIN_TOTAL_TUPLES"\s*,\s*100\s*\)',
        body,
    ), (
        "P1-HOT-BLOAT-MIN-TUPLES: el knob "
        "`MEALFIT_AUTOVACUUM_MIN_TOTAL_TUPLES` debe leerse via "
        "`_env_int(\"MEALFIT_AUTOVACUUM_MIN_TOTAL_TUPLES\", 100)`. Default "
        "100 cubre las 7 tablas P1-B sin disparar sobre estado normal "
        "(meal_plans/agent_sessions viven con <20 tuplas totales)."
    )


# B) Clamp [10, 10000] aplicado.
def test_b_clamp_applied(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    # Regex tolera `int(...)` o variable plana dentro del `min(...)`: el
    # patrón `.+?` no-greedy entre `min(` y `,\s*10000` matchea ambos
    # `min(_min_total_raw, 10000)` y `min(int(_min_total_raw), 10000)`.
    assert re.search(
        r"max\(\s*10\s*,\s*min\(.+?,\s*10000\s*\)\s*\)",
        body,
    ), (
        "P1-HOT-BLOAT-MIN-TUPLES: clamp `max(10, min(..., 10000))` no "
        "presente. Sin clamp un operador puede setear `=0` (degrada al "
        "comportamiento pre-fix) o `=1000000` (bloquea siempre)."
    )


# C) Guard PRECEDE al cómputo de `is_bloated`.
def test_c_guard_precedes_is_bloated_decision(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    guard_match = re.search(
        r"if\s+total_tuples\s*<\s*min_total_tuples\s*:",
        body,
    )
    is_bloated_match = re.search(r"is_bloated\s*=\s*\(", body)
    assert guard_match, (
        "P1-HOT-BLOAT-MIN-TUPLES: guard `if total_tuples < min_total_tuples:` "
        "no presente."
    )
    assert is_bloated_match, (
        "P1-HOT-BLOAT-MIN-TUPLES: cómputo `is_bloated = (...)` no presente "
        "(se removió por accidente)."
    )
    assert guard_match.start() < is_bloated_match.start(), (
        "P1-HOT-BLOAT-MIN-TUPLES: el guard del min_total_tuples debe correr "
        "ANTES de calcular `is_bloated`. Si va después, la UPSERT del alert "
        "ya se ejecutó y el continue no la previene."
    )


# D) Skip emite tick observable a pipeline_metrics.
def test_d_skip_emits_pipeline_metric(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    # Localizar el bloque del guard y verificar que dentro hay el INSERT.
    # Tomamos desde `if total_tuples < min_total_tuples:` hasta el `continue`.
    block_match = re.search(
        r"if\s+total_tuples\s*<\s*min_total_tuples\s*:.*?\bcontinue\b",
        body,
        re.DOTALL,
    )
    assert block_match, (
        "P1-HOT-BLOAT-MIN-TUPLES: bloque del guard (desde `if total_tuples "
        "< min_total_tuples:` hasta `continue`) no localizable — verificar "
        "indentación / orden."
    )
    block = block_match.group(0)
    assert "INSERT INTO pipeline_metrics" in block, (
        "P1-HOT-BLOAT-MIN-TUPLES: el skip debe emitir tick a "
        "`pipeline_metrics` (observabilidad continua: SRE necesita saber "
        "cuántas tablas se saltan y por qué)."
    )
    assert "_hot_table_bloat_skipped_low_volume" in block, (
        "P1-HOT-BLOAT-MIN-TUPLES: el `node` del tick del skip debe ser "
        "`_hot_table_bloat_skipped_low_volume` para distinguir del tick "
        "normal `_hot_table_bloat_tick` en dashboards."
    )


# E) Skip auto-resuelve alerta previa.
def test_e_skip_auto_resolves_existing_alert(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    block_match = re.search(
        r"if\s+total_tuples\s*<\s*min_total_tuples\s*:.*?\bcontinue\b",
        body,
        re.DOTALL,
    )
    assert block_match, (
        "P1-HOT-BLOAT-MIN-TUPLES: bloque del guard no localizable."
    )
    block = block_match.group(0)
    assert re.search(
        r"UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*NOW\(\)",
        block,
        re.IGNORECASE,
    ), (
        "P1-HOT-BLOAT-MIN-TUPLES: el skip debe `UPDATE system_alerts SET "
        "resolved_at = NOW()` para cerrar cualquier alerta previa de la "
        "misma tabla. Sin esto, las 2 alertas vivas (meal_plans + "
        "agent_sessions) seguirían abiertas tras el deploy hasta MCP manual."
    )


# F) Skip termina en `continue` (no cae al bloque de decisión).
def test_f_skip_ends_with_continue(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    # Verificar que después del bloque del guard viene `# 3) Decisión` o
    # cualquier código posterior (i.e. `continue` no es el último statement
    # de la función). La regex de D/E ya valida `continue` está dentro.
    block_match = re.search(
        r"if\s+total_tuples\s*<\s*min_total_tuples\s*:.*?\bcontinue\b",
        body,
        re.DOTALL,
    )
    assert block_match, "P1-HOT-BLOAT-MIN-TUPLES: bloque del guard no localizable."
    # Verificar que `is_bloated = ` aparece DESPUÉS del `continue` del guard.
    tail = body[block_match.end():]
    assert "is_bloated" in tail, (
        "P1-HOT-BLOAT-MIN-TUPLES: el cómputo `is_bloated` debe quedar "
        "DESPUÉS del `continue` del guard (orden esperado: guard → "
        "is_bloated → UPSERT/auto-resolve). Sin `continue`, el código del "
        "skip y del alert se ejecutarían en serie sobre la misma tabla."
    )


# G) Anchor textual presente.
def test_g_anchor_present(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    assert "P1-HOT-BLOAT-MIN-TUPLES" in body, (
        "P1-HOT-BLOAT-MIN-TUPLES: cuerpo del cron perdió el anchor textual "
        "`P1-HOT-BLOAT-MIN-TUPLES`. Sin anchor, un futuro grep no encuentra "
        "el contexto del fix."
    )
