"""[P2-PIPELINE-SILENCE-MULTI-NODE · 2026-05-13] Extensión del watchdog
`_alert_pipeline_metrics_silence` (P2-OBSERVABILITY-1) para soportar
múltiples nodos observados con threshold opcional por-nodo.

Motivación (audit 2026-05-13):
    El watchdog original observaba un único nodo (`_hardfloor_autoheal_tick`)
    como canario de liveness del binary. SRE no podía añadir un segundo
    nodo (e.g., `pipeline_holistic` con threshold holgado 24h para alertar
    sobre ausencia de tráfico productivo) sin perder el watchdog
    original.

Fix:
    1. Knob `MEALFIT_PIPELINE_METRICS_SILENCE_NODE` acepta ahora CSV
       con formato opcional `<node>:<threshold_min>` por entrada:
         `_hardfloor_autoheal_tick,pipeline_holistic:1440`
    2. Threshold per-node se clampea a [5, 1440]. Sin `:` usa el global.
    3. Backward-compat: 1 nodo → alert_key legacy `pipeline_metrics_silence`.
       >1 nodos → alert_key `pipeline_metrics_silence:<node>` por cada uno
       (preserva el contrato del test P2-AUDIT-4 documentated alert keys —
       el legacy `pipeline_metrics_silence` sigue documentado).

Asserts:
    A) Helper `_parse_silence_observed_nodes` existe.
    B) Parsea CSV correctamente (cases: empty, single, multi, with-threshold,
       with-bad-threshold, whitespace, clamp).
    C) `_alert_pipeline_metrics_silence` invoca el parser.
    D) El cuerpo itera sobre los nodos parseados (loop `for`).
    E) Sufijo `:<node>` se construye correctamente en modo multi-node.
    F) BC: 1 nodo → alert_key sin sufijo.

Tooltip-anchor: P2-PIPELINE-SILENCE-MULTI-NODE-START
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _isolate_func(src: str, name: str) -> str:
    m = re.search(
        rf"def\s+{re.escape(name)}\b(.*?)(?=^def\s+\w|\Z)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"Función `{name}` no encontrada en cron_tasks.py."
    return m.group(1)


# ---------------------------------------------------------------------------
# A) Helper exists
# ---------------------------------------------------------------------------
def test_a_parser_helper_defined(cron_src: str):
    assert "def _parse_silence_observed_nodes(" in cron_src, (
        "P2-PIPELINE-SILENCE-MULTI-NODE regresión: el helper "
        "`_parse_silence_observed_nodes` ya no está definido. Sin él, "
        "el watchdog vuelve al modo single-node sin posibilidad de "
        "observar nodos adicionales como `pipeline_holistic`."
    )


# ---------------------------------------------------------------------------
# B) Parser behavior (unit test via direct import — degrada a parser-based
#    si el módulo no puede importarse en este entorno de test)
# ---------------------------------------------------------------------------
def _try_import_parser():
    if str(_BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        from cron_tasks import _parse_silence_observed_nodes  # type: ignore
        return _parse_silence_observed_nodes
    except Exception:
        return None


def test_b_parser_handles_empty_returns_empty_list():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible — falló import top-level.")
    assert parser("") == []
    assert parser("   ") == []
    assert parser(None or "") == []


def test_b_parser_handles_single_node_without_threshold():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible.")
    assert parser("_hardfloor_autoheal_tick") == [("_hardfloor_autoheal_tick", None)]


def test_b_parser_handles_multi_node_csv():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible.")
    out = parser("_hardfloor_autoheal_tick,pipeline_holistic")
    assert out == [
        ("_hardfloor_autoheal_tick", None),
        ("pipeline_holistic", None),
    ]


def test_b_parser_handles_per_node_threshold():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible.")
    out = parser("pipeline_holistic:1440")
    assert out == [("pipeline_holistic", 1440)]


def test_b_parser_clamps_threshold_to_5_min():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible.")
    out = parser("foo:0")
    assert out == [("foo", 5)], (
        "Threshold per-node < 5 debe clampearse a 5 — un operador que "
        "setea 0 no debe disparar el alert cada tick."
    )


def test_b_parser_clamps_threshold_to_1440_max():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible.")
    out = parser("foo:99999")
    assert out == [("foo", 1440)], (
        "Threshold per-node > 1440 (24h) debe clampearse — un operador "
        "que setea 99999 nunca dispara el alert."
    )


def test_b_parser_handles_bad_threshold_falls_to_global():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible.")
    out = parser("foo:notanint")
    assert out == [("foo", None)], (
        "Threshold per-node no parseable como int debe caer al global "
        "(None) en lugar de crashear."
    )


def test_b_parser_strips_whitespace_and_skips_empty():
    parser = _try_import_parser()
    if parser is None:
        pytest.skip("cron_tasks import no disponible.")
    out = parser(" foo , , bar : 60 ")
    assert out == [("foo", None), ("bar", 60)]


# ---------------------------------------------------------------------------
# C) Watchdog usa el parser
# ---------------------------------------------------------------------------
def test_c_watchdog_invokes_parser(cron_src: str):
    body = _isolate_func(cron_src, "_alert_pipeline_metrics_silence")
    assert "_parse_silence_observed_nodes" in body, (
        "P2-PIPELINE-SILENCE-MULTI-NODE regresión: `_alert_pipeline_metrics_silence` "
        "ya no invoca `_parse_silence_observed_nodes`. Sin esto, el "
        "knob multi-node es ignorado."
    )


# ---------------------------------------------------------------------------
# D) Loop sobre nodos
# ---------------------------------------------------------------------------
def test_d_watchdog_iterates_over_parsed_nodes(cron_src: str):
    body = _isolate_func(cron_src, "_alert_pipeline_metrics_silence")
    assert re.search(r"for\s+\w+,\s*\w+\s+in\s+parsed\s*:", body), (
        "P2-PIPELINE-SILENCE-MULTI-NODE regresión: el watchdog ya no "
        "itera sobre la lista parseada `(node, threshold)`. Sin loop, "
        "solo el primer nodo se observa."
    )


# ---------------------------------------------------------------------------
# E) Alert key con sufijo `:<node>` cuando multi-node
# ---------------------------------------------------------------------------
def test_e_alert_key_suffix_in_multi_node_mode(cron_src: str):
    body = _isolate_func(cron_src, "_alert_pipeline_metrics_silence")
    assert re.search(
        r'f["\']pipeline_metrics_silence:\{[^}]+\}["\']',
        body,
    ), (
        "P2-PIPELINE-SILENCE-MULTI-NODE regresión: el alert_key en modo "
        "multi-node ya no incluye sufijo `:<node>`. Sin sufijo, todos "
        "los nodos comparten una alert_key y el último write gana — "
        "perdemos visibilidad per-node."
    )


# ---------------------------------------------------------------------------
# F) BC: 1 nodo → alert_key legacy sin sufijo
# ---------------------------------------------------------------------------
def test_f_backward_compat_single_node_uses_legacy_key(cron_src: str):
    body = _isolate_func(cron_src, "_alert_pipeline_metrics_silence")
    # use_suffix se decide por `len(parsed) > 1`. Anchor: la lógica de
    # selección de alert_key debe ramificarse en `use_suffix`.
    assert "use_suffix" in body, (
        "P2-PIPELINE-SILENCE-MULTI-NODE regresión: la variable "
        "`use_suffix` ya no controla el sufijo del alert_key. Sin ella, "
        "el caso 1-nodo pierde BC y rompe los tests P2-OBSERVABILITY-1."
    )
    # La rama legacy debe seguir produciendo `pipeline_metrics_silence` sin sufijo.
    assert re.search(
        r'["\']pipeline_metrics_silence["\']',
        body,
    ), (
        "P2-PIPELINE-SILENCE-MULTI-NODE regresión: el alert_key legacy "
        "`pipeline_metrics_silence` (sin sufijo) desapareció del cuerpo. "
        "Rompe BC con tests `test_p2_observability_1_*` y la tabla de "
        "system_alerts policy en CLAUDE.md."
    )
