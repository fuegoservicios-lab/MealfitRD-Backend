"""[P2-OBSERVABILITY-1 · 2026-05-12] Cron `_alert_pipeline_metrics_silence`
detecta cuando el tick observable `_hardfloor_autoheal_tick` no aparece
en `pipeline_metrics` durante una ventana — síntoma de deploy lag u otra
causa de silencio del loop hardfloor.

Vector observado (audit 2026-05-11):
    MCP snapshot mostró pipeline_metrics con solo 3 nodos distintos
    emitiendo en últimas 2h. Las telemetrías de `_hardfloor_autoheal_tick`,
    `_chunk_heartbeat_baseline`, `_chunk_lesson_synth_*` estaban
    silenciadas — síntoma del deploy lag P0-PROD-1 (binario en prod
    crasheaba inserts con `is_guest`). Ningún alert avisó.

Fix:
    Cron `_alert_pipeline_metrics_silence` cada `MEALFIT_PIPELINE_METRICS_SILENCE_INTERVAL_MIN`
    (default 10min). Si `SELECT 1 FROM pipeline_metrics WHERE node =
    <observed_node> AND created_at > NOW() - threshold` devuelve 0 rows,
    UPSERT alert `pipeline_metrics_silence` (warning). Auto-resolve cuando
    el tick vuelve.

Lo que este test enforza:
    A) Función `_alert_pipeline_metrics_silence` declarada en cron_tasks.py.
    B) Cron registrado en `register_plan_chunk_scheduler` con id
       `alert_pipeline_metrics_silence`.
    C) Default observado node = `_hardfloor_autoheal_tick`.
    D) Default threshold = 30min, clamp [5, 720].
    E) Default interval = 10min, clamp [5, 120].
    F) Auto-resolve: cuando hay row encontrado, UPDATE `resolved_at = NOW()`
       sobre `alert_key = 'pipeline_metrics_silence'`.
    G) Alert key `pipeline_metrics_silence` documentado en CLAUDE.md tabla
       de system_alerts policy.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"
_CLAUDE_MD = _BACKEND_ROOT.parent / "CLAUDE.md"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def claude_md_src() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


def _isolate_func(src: str, name: str) -> str:
    m = re.search(
        rf"def\s+{re.escape(name)}\b(.*?)(?=^def\s+\w|\Z)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"Función `{name}` no encontrada en cron_tasks.py."
    return m.group(1)


def test_a_function_defined(cron_src: str):
    assert "def _alert_pipeline_metrics_silence(" in cron_src, (
        "P2-OBSERVABILITY-1: función `_alert_pipeline_metrics_silence` "
        "ausente. Sin esta función, deploy lag P0-PROD-1 vuelve a ser "
        "invisible."
    )


def test_b_registered_in_scheduler(cron_src: str):
    """El cron debe registrarse en `register_plan_chunk_scheduler` con
    `id="alert_pipeline_metrics_silence"`."""
    assert 'id="alert_pipeline_metrics_silence"' in cron_src, (
        "P2-OBSERVABILITY-1: cron no registrado en "
        "`register_plan_chunk_scheduler`. La función existe pero NO se "
        "ejecuta automáticamente."
    )
    assert "_alert_pipeline_metrics_silence," in cron_src, (
        "P2-OBSERVABILITY-1: referencia al callable del cron ausente "
        "en el bloque de registro."
    )


def test_c_default_observed_node_is_hardfloor_tick(cron_src: str):
    body = _isolate_func(cron_src, "_alert_pipeline_metrics_silence")
    assert "_hardfloor_autoheal_tick" in body, (
        "P2-OBSERVABILITY-1: el observed_node default no es "
        "`_hardfloor_autoheal_tick`. El plan original asume ese tick "
        "como canario."
    )


def test_d_threshold_default_and_clamp(cron_src: str):
    body = _isolate_func(cron_src, "_alert_pipeline_metrics_silence")
    # Default 30
    m = re.search(
        r'_env_int\(\s*["\']MEALFIT_PIPELINE_METRICS_SILENCE_ALERT_MIN["\']\s*,\s*(\d+)\s*\)',
        body,
    )
    assert m, "Knob `MEALFIT_PIPELINE_METRICS_SILENCE_ALERT_MIN` no leído."
    assert int(m.group(1)) == 30, (
        "P2-OBSERVABILITY-1: default del threshold debe ser 30min. "
        "Más bajo dispara falsos positivos; más alto pierde feedback rápido."
    )
    # Clamp explícito.
    assert "threshold_min < 5" in body, (
        "P2-OBSERVABILITY-1: clamp inferior ausente. Sin clamp un operador "
        "setea 0 y el alert dispara cada tick."
    )
    assert "threshold_min > 720" in body, (
        "P2-OBSERVABILITY-1: clamp superior ausente. Sin clamp un operador "
        "setea 99999 y el alert nunca dispara."
    )


def test_e_interval_default_and_clamp_in_registration(cron_src: str):
    """En el bloque de registro: default 10, clamp [5, 120]."""
    # Aislar el bloque de registro.
    block_match = re.search(
        r'_PIPE_SILENCE_INT\s*=\s*_env_int.*?_add_job_jittered',
        cron_src,
        re.DOTALL,
    )
    assert block_match, "Bloque de registro del cron no aislable."
    block = block_match.group(0)
    m = re.search(
        r'_env_int\(\s*["\']MEALFIT_PIPELINE_METRICS_SILENCE_INTERVAL_MIN["\']\s*,\s*(\d+)\s*\)',
        block,
    )
    assert m and int(m.group(1)) == 10, (
        "P2-OBSERVABILITY-1: default del interval debe ser 10min."
    )
    assert "_PIPE_SILENCE_INT < 5" in block and "_PIPE_SILENCE_INT > 120" in block, (
        "P2-OBSERVABILITY-1: clamp del interval ausente. Esperado [5, 120]."
    )


def test_f_auto_resolves_on_tick_recovery(cron_src: str):
    """Cuando hay un row encontrado en la ventana, UPDATE resolved_at."""
    body = _isolate_func(cron_src, "_alert_pipeline_metrics_silence")
    # El UPDATE debe estar en la rama del `if row:`.
    assert re.search(
        r"if\s+row\s*:[\s\S]+?UPDATE\s+system_alerts[\s\S]+?resolved_at\s*=\s*NOW\(\)",
        body,
    ), (
        "P2-OBSERVABILITY-1: auto-resolve ausente. Sin ese UPDATE, una "
        "alert previa permanece abierta indefinidamente aunque el tick "
        "vuelva."
    )
    # El UPDATE debe targetear EL alert_key específico.
    assert '"pipeline_metrics_silence"' in body, (
        "P2-OBSERVABILITY-1: alert_key `pipeline_metrics_silence` ausente "
        "en el body del cron."
    )


def test_g_alert_key_documented_in_claude_md(claude_md_src: str):
    """La alert_key debe estar en la tabla de system_alerts policy."""
    assert "pipeline_metrics_silence" in claude_md_src, (
        "P2-OBSERVABILITY-1: alert_key `pipeline_metrics_silence` no "
        "documentado en CLAUDE.md. El test "
        "`test_p2_audit_4_alert_keys_documented` falla si se omite."
    )


def test_h_anchor_present(cron_src: str):
    assert "P2-OBSERVABILITY-1" in cron_src, (
        "P2-OBSERVABILITY-1: anchor ausente del bloque doc del cron."
    )
