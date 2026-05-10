"""[P1-HIST-LM-WHITELIST · 2026-05-09] Backend sentinel del cross-link
marker `_LAST_KNOWN_PFIX` ↔ tests vitest del frontend + drift detection
backend↔frontend del catálogo de keys de `learning_metrics`.

Contexto:
    El audit Historial 2026-05-09 (gap P1-2) identificó que la
    whitelist `_LM_DISPLAY_KEYS` del tab Métricas tenía 5 keys
    (síntesis/escalación) y ocultaba ~25 keys ricas que
    `cron_tasks.py:_calculate_learning_metrics` y los call sites
    pre/post-pipeline (line ~18348-19651) persisten en cada chunk.

Fix:
    Whitelist categorizada `_LM_DISPLAY_GROUPS` con 4 grupos
    (synthesis / repetition / violations / pantry) y helper
    `_fmtLmValue` con types declarados (bool, pct, severity, preview,
    hours, str). Render por grupo con sub-headers + severity classes
    automáticas según umbrales.

Este test backend cumple DOS roles:

  1. Cross-link con `test_p2_hist_audit_14_marker_test_link.py` —
     `_LAST_KNOWN_PFIX = "P1-HIST-LM-WHITELIST · 2026-05-09"`
     requiere un archivo `tests/test_p1_hist_lm_whitelist*.py`.

  2. Drift detection cross-archivo: cada key emitida por el writer
     del backend (`_calculate_learning_metrics` return + call sites)
     DEBE estar declarada en `_LM_DISPLAY_GROUPS` del frontend. Si
     un futuro fix añade una key al writer sin actualizar la UI, el
     test falla loud antes de mergear.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_HISTORY_CSS = _REPO_ROOT / "frontend" / "src" / "pages" / "History.module.css"
_VITEST_DIR = _REPO_ROOT / "frontend" / "src" / "__tests__"


# ---------------------------------------------------------------------------
# 1. Marker presence + asociación con vitest
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P1-HIST-LM-WHITELIST · 2026-05-09]" in text


def test_marker_present_in_history_css():
    text = _HISTORY_CSS.read_text(encoding="utf-8")
    assert "[P1-HIST-LM-WHITELIST · 2026-05-09]" in text


def test_vitest_file_exists():
    """El test vitest dedicado al P1-HIST-LM-WHITELIST debe existir
    en `frontend/src/__tests__/`. Cubre 51 cases (catálogo, helpers,
    render). Si alguien lo borra por accidente, este test avisa."""
    path = _VITEST_DIR / "History.p1_lm_whitelist_keys.test.js"
    assert path.exists(), (
        f"Falta {path}. Cubre el surface de las ~25 keys de "
        "learning_metrics categorizadas en 4 grupos."
    )


# ---------------------------------------------------------------------------
# 2. Drift detection: keys del writer ↔ keys del catálogo frontend
# ---------------------------------------------------------------------------
def _extract_lm_display_groups_keys() -> set[str]:
    """Parsea `_LM_DISPLAY_GROUPS` en History.jsx y extrae el set de
    todas las keys (1er elemento de cada tuple)."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # Localizar el bloque del array (defensivo: del `_LM_DISPLAY_GROUPS = [`
    # hasta el cierre con `];`).
    start = text.find("_LM_DISPLAY_GROUPS")
    assert start != -1, "_LM_DISPLAY_GROUPS no encontrado en History.jsx"
    end = text.find("];", start)
    assert end != -1, "Cierre `];` no encontrado tras _LM_DISPLAY_GROUPS"
    block = text[start:end]
    # Tuple shape: ['key', 'label', 'type'] — capturamos el primer string.
    pattern = re.compile(
        r"\[\s*['\"]([a-z_]+)['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*['\"][a-z_]+['\"]\s*\]"
    )
    return set(pattern.findall(block))


def _calculate_learning_metrics_return_keys() -> set[str]:
    """Extrae el set de keys que devuelve `_calculate_learning_metrics`
    (cron_tasks.py:~15115-15130). Single source of truth del shape
    canónico per-chunk del aprendizaje."""
    from cron_tasks import _calculate_learning_metrics
    src = inspect.getsource(_calculate_learning_metrics)
    # Match `"key_name":` dentro del último `return { ... }` del cuerpo.
    # Buscamos solo la sección que va de "return {" hasta "}".
    return_match = re.search(r"return\s*\{([\s\S]*?)\}", src)
    assert return_match, (
        "_calculate_learning_metrics no termina con `return { ... }`."
    )
    return_body = return_match.group(1)
    return set(re.findall(r"['\"]([a-z_]+)['\"]\s*:", return_body))


def test_calculate_learning_metrics_keys_in_frontend_catalog():
    """Cada key que `_calculate_learning_metrics` retorna (~14 keys
    canónicas) debe estar declarada en `_LM_DISPLAY_GROUPS` del
    frontend. Si esta aserción falla, alguien añadió una key al
    writer sin categorizarla en la UI."""
    backend_keys = _calculate_learning_metrics_return_keys()
    frontend_keys = _extract_lm_display_groups_keys()
    missing = backend_keys - frontend_keys
    assert not missing, (
        f"Keys persistidas por _calculate_learning_metrics pero ausentes "
        f"en _LM_DISPLAY_GROUPS del frontend: {sorted(missing)}.\n"
        f"Frontend declara: {sorted(frontend_keys)}.\n"
        f"Categorízalas en uno de los 4 grupos (synthesis/repetition/"
        f"violations/pantry) en `frontend/src/pages/History.jsx`."
    )


def test_post_pipeline_keys_in_frontend_catalog():
    """Keys persistidas en el post-pipeline (`cron_tasks.py:18348-19651`):
    `shuffle_*`, `pantry_*`, `inventory_activity_*`, `sparse_logging_*`,
    `learning_signal_strength`, `learning_confidence`, `pipeline_failed`.

    Estas no las devuelve `_calculate_learning_metrics` (se setean por
    la pipeline post-creación) pero llegan al `learning_metrics` jsonb
    final. Si el cron añade una nueva, el frontend debe categorizarla."""
    _POST_PIPELINE_KEYS = {
        "shuffle_learning_applied",
        "shuffle_source",
        "pantry_quantity_violations",
        "sample_pantry_quantity_violations",
        "inventory_activity_proxy_used",
        "inventory_activity_mutations",
        "sparse_logging_proxy_used",
        "learning_signal_strength",
        "pantry_degraded_reason",
        "pantry_snapshot_age_hours_at_pickup",
        "learning_confidence",
        "pipeline_failed",
    }
    frontend_keys = _extract_lm_display_groups_keys()
    missing = _POST_PIPELINE_KEYS - frontend_keys
    assert not missing, (
        f"Keys de la pipeline post-creation ausentes en _LM_DISPLAY_GROUPS: "
        f"{sorted(missing)}.\n"
        f"Categorízalas en grupo 'pantry' o 'synthesis' del frontend."
    )


# ---------------------------------------------------------------------------
# 3. Catálogo legacy de síntesis preservado
# ---------------------------------------------------------------------------
def test_legacy_synth_keys_preserved():
    """Las 5 keys del whitelist legacy (`synth_quality_score`,
    `synthesized_count`, `queue_count`, `recovery_attempts`,
    `escalation_reason`) DEBEN seguir presentes en el catálogo nuevo.
    Cubrir vs. regresión cosmética que las pierda."""
    _LEGACY = {
        "synth_quality_score",
        "synthesized_count",
        "queue_count",
        "recovery_attempts",
        "escalation_reason",
    }
    frontend_keys = _extract_lm_display_groups_keys()
    missing = _LEGACY - frontend_keys
    assert not missing, (
        f"Keys legacy de síntesis perdidas tras la migración a "
        f"_LM_DISPLAY_GROUPS: {sorted(missing)}."
    )


# ---------------------------------------------------------------------------
# 4. Catálogo no incluye internals peligrosos
# ---------------------------------------------------------------------------
def test_pipeline_snapshot_not_in_catalog():
    """`pipeline_snapshot` puede ser MB de jsonb. Render como counter
    rompería UI. NUNCA debe estar en el catálogo."""
    frontend_keys = _extract_lm_display_groups_keys()
    assert "pipeline_snapshot" not in frontend_keys


# ---------------------------------------------------------------------------
# 5. Tipos declarados son válidos
# ---------------------------------------------------------------------------
def test_declared_types_are_valid():
    """Cada tuple en `_LM_DISPLAY_GROUPS` declara un type que el
    helper `_fmtLmValue` debe poder manejar. Type fuera de la
    whitelist => regresión silenciosa (el helper devolvería null
    como default y el chip nunca renderizaría)."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    start = text.find("_LM_DISPLAY_GROUPS")
    end = text.find("];", start)
    block = text[start:end]
    pattern = re.compile(
        r"\[\s*['\"][a-z_]+['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*['\"]([a-z_]+)['\"]\s*\]"
    )
    declared_types = set(pattern.findall(block))
    _VALID_TYPES = {
        "number", "int", "pct", "bool", "preview",
        "severity", "severity_high", "hours", "str",
    }
    invalid = declared_types - _VALID_TYPES
    assert not invalid, (
        f"Types declarados fuera de la whitelist de _fmtLmValue: "
        f"{sorted(invalid)}.\nWhitelist: {sorted(_VALID_TYPES)}."
    )
