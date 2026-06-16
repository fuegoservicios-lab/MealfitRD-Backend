"""[P2-AUDIT-2 · 2026-05-12] Cron observable de bloat para las 7 tablas
tuneadas en P1-B.

Contexto:
    P1-B (2026-05-12) aplicó `scale_factor=0.05 + threshold=25` a 7 tablas
    chronic-UPDATE. Sin observabilidad continua del bloat post-tuning, un
    autovacuum que deje de disparar (bug de postgres, configuración perdida,
    lock contention) pasaría desapercibido hasta dead_pct >90%.

    P2-AUDIT-2 añade el cron `_emit_hot_table_bloat_tick` (default cada 6h)
    que emite por tabla a `pipeline_metrics` + UPSERT alert
    `hot_table_bloat:<table>` si dead_pct y horas-sin-autovacuum superan
    los umbrales. Auto-resolve cuando tabla sana de nuevo.

Lo que este test enforza:
  A) Función `_emit_hot_table_bloat_tick` definida en cron_tasks.py.
  B) Cubre las 7 tablas P1-B exactas (sin drift).
  C) Registrada en `register_plan_chunk_scheduler` con knob
     `MEALFIT_AUTOVACUUM_TICK_INTERVAL_MIN` + clamp [60, 1440].
  D) Emite tick a `pipeline_metrics` (telemetría continua).
  E) UPSERT alert con `alert_key = f"hot_table_bloat:{relname}"`.
  F) Auto-resolve via `UPDATE … SET resolved_at = NOW()` cuando tabla sana.
  G) `hot_table_bloat:<table>` documentado en tabla de CLAUDE.md.

Tooltip-anchor: P2-AUDIT-2-AUTOVACUUM-TICK.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "backend"
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"
# [P3-CLAUDEMD-CAP / P2-NEW-3 drift] La tabla canónica de alert_keys de
# `system_alerts` se movió de CLAUDE.md al doc SSOT
# `backend/docs/system_alerts_resolution_table.md` (doc-first, para respetar
# el cap de tamaño de CLAUDE.md). El drift bidireccional lo enforza
# `test_p2_audit_4_alert_keys_documented` contra ese mismo doc.
_ALERT_KEYS_DOC = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"

_P1B_TABLES = (
    "app_kv_store",
    "plan_chunk_queue",
    "meal_plans",
    "agent_sessions",
    "plan_chunk_metrics",
    "user_profiles",
    "system_alerts",
)


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


def _extract_function(src: str, name: str) -> str:
    pattern = re.compile(
        rf"def {re.escape(name)}\b.*?(?=^def\s)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(src)
    assert m, f"P2-AUDIT-2: no localicé `def {name}` en cron_tasks.py."
    return m.group(0)


# A) Función existe.
def test_a_function_defined(cron_src: str):
    assert re.search(r"^def _emit_hot_table_bloat_tick\b", cron_src, re.MULTILINE), (
        "P2-AUDIT-2: cron_tasks.py no define `_emit_hot_table_bloat_tick`."
    )


# B) Cubre las 7 tablas P1-B exactas.
def test_b_covers_seven_p1b_tables(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    missing = [t for t in _P1B_TABLES if f'"{t}"' not in body]
    assert not missing, (
        f"P2-AUDIT-2: `_emit_hot_table_bloat_tick` no cubre las tablas: "
        f"{missing}. Si una tabla se eliminó del tuning P1-B, también "
        f"debe removerse de esta tupla. Si se añadió otra, añadirla aquí."
    )


# C) Registrada en register_plan_chunk_scheduler con knob + clamp.
def test_c_registered_in_scheduler(cron_src: str):
    body = _extract_function(cron_src, "register_plan_chunk_scheduler")
    assert "_emit_hot_table_bloat_tick" in body, (
        "P2-AUDIT-2: `register_plan_chunk_scheduler` no referencia "
        "`_emit_hot_table_bloat_tick`. Sin el registro, el cron no corre."
    )
    assert "hot_table_bloat_tick" in body, (
        "P2-AUDIT-2: job id `hot_table_bloat_tick` no está en el scheduler "
        "(necesario para idempotencia `scheduler.get_job(...)`)."
    )
    assert "MEALFIT_AUTOVACUUM_TICK_INTERVAL_MIN" in body, (
        "P2-AUDIT-2: knob `MEALFIT_AUTOVACUUM_TICK_INTERVAL_MIN` no se lee "
        "en `register_plan_chunk_scheduler`. Sin esto, el operador no "
        "puede ajustar el intervalo sin redeploy."
    )
    # Clamp [60, 1440] visible.
    assert re.search(
        r"max\(\s*60\s*,\s*min\([^,)]*tick_interval[^,)]*,\s*1440\s*\)\s*\)",
        body,
    ), (
        "P2-AUDIT-2: clamp `max(60, min(_autovacuum_tick_interval, 1440))` "
        "no presente. Sin clamp, un operador puede setear 1 → autovacuum "
        "tick cada minuto (overhead innecesario)."
    )


# D) Emite tick a pipeline_metrics.
def test_d_emits_pipeline_metric_tick(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    assert "INSERT INTO pipeline_metrics" in body, (
        "P2-AUDIT-2: el cron no emite a `pipeline_metrics`. Sin tick "
        "observable, no podemos distinguir 'cron no corrió' vs 'cron "
        "corrió y todas las tablas sanas'."
    )
    assert "_hot_table_bloat_tick" in body, (
        "P2-AUDIT-2: el `node` del tick debe ser `_hot_table_bloat_tick` "
        "para correlacionar en dashboards."
    )


# E) UPSERT alert con clave correcta.
def test_e_upserts_alert_with_correct_key(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    assert 'alert_key = f"hot_table_bloat:{relname}"' in body or \
           'f"hot_table_bloat:{relname}"' in body, (
        "P2-AUDIT-2: alert_key debe ser `f\"hot_table_bloat:{relname}\"` "
        "(matchea documentación en CLAUDE.md tabla system_alerts)."
    )
    assert "INSERT INTO system_alerts" in body, (
        "P2-AUDIT-2: cron no hace UPSERT a `system_alerts`. Sin UPSERT, "
        "cada tick crea una row nueva → tabla infla."
    )
    assert "ON CONFLICT (alert_key) DO UPDATE" in body, (
        "P2-AUDIT-2: el UPSERT debe usar `ON CONFLICT (alert_key) DO UPDATE` "
        "para preservar la idempotencia documentada en CLAUDE.md sección "
        "'Política de system_alerts resolution'."
    )


# F) Auto-resolve cuando tabla sana de nuevo.
def test_f_auto_resolves_when_healthy(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    assert re.search(
        r"UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*NOW\(\)",
        body,
        re.IGNORECASE,
    ), (
        "P2-AUDIT-2: cron no auto-resuelve la alert cuando la tabla vuelve "
        "a estar sana. El modelo Auto (explicit) de CLAUDE.md requiere "
        "el `UPDATE … SET resolved_at = NOW()` en el path sano."
    )


# G) Documentado en el doc SSOT de alert_keys.
def test_g_alert_key_documented_in_claude_md():
    # [P3-CLAUDEMD-CAP drift] La tabla de alert_keys vive ahora en el doc
    # canónico `backend/docs/system_alerts_resolution_table.md` (no inline en
    # CLAUDE.md). El test global de drift
    # (test_p2_audit_4_alert_keys_documented) parsea ESE doc; aquí lo
    # verificamos específicamente para claridad del fix.
    src = _ALERT_KEYS_DOC.read_text(encoding="utf-8")
    assert "hot_table_bloat:<table>" in src, (
        "P2-AUDIT-2: pattern `hot_table_bloat:<table>` no documentado en "
        "`backend/docs/system_alerts_resolution_table.md` (tabla canónica de "
        "system_alerts). Añadir fila con productor + resolver antes de "
        "mergear — el test global test_p2_audit_4_alert_keys_documented "
        "también fallará."
    )


def test_h_anchor_present(cron_src: str):
    body = _extract_function(cron_src, "_emit_hot_table_bloat_tick")
    assert "P2-AUDIT-2" in body, (
        "P2-AUDIT-2: cuerpo del cron perdió el anchor textual `P2-AUDIT-2`."
    )
