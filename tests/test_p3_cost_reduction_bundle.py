"""[P3-COST-REDUCTION-BUNDLE · 2026-05-20] Anchor + regression guards para
el bundle de fixes de "reducir costos directos" del audit
`docs/gaps-audit-2026-05.md` categoría 3:

  - **D4 / P3-TIER-LIMITS-ENV**: tier limits del paywall mensual via knobs
    `MEALFIT_TIER_LIMIT_<TIER>` en lugar de dict literal hardcoded. Permite
    ajustar pricing (e.g. promoción semanal) sin redeploy.

  - **C4 / P3-PLAN-MODEL-KNOBS**: `_PRO_MODEL_NAME` y `_FLASH_MODEL_NAME`
    via knobs `MEALFIT_PRO_MODEL`/`MEALFIT_FLASH_MODEL`. Permite swap a
    `gemini-3.5-flash` (34% cache hit vs 15% del actual `gemini-3-flash-preview`)
    SIN redeploy. Cierra el mismo riesgo R2 que `vision_agent` (D3) para
    el plan-gen pipeline.

  - **C3 / P3-SENTRY-COST-THRESHOLDS**: documentación de thresholds para
    revisar Sentry sampling rates cuando crucemos escala (>500 MAU, >1k MAU).
    Sin enforcement automático — knobs ya son ajustables.

Tooltip-anchor: P3-COST-REDUCTION-BUNDLE.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AUTH_PY = _BACKEND_ROOT / "auth.py"
_GO_PY = _BACKEND_ROOT / "graph_orchestrator.py"
_APP_PY = _BACKEND_ROOT / "app.py"


# ---------------------------------------------------------------------------
# Anchor presence (P2-HIST-AUDIT-14 cross-link).
# ---------------------------------------------------------------------------


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-COST-REDUCTION-BUNDLE" in src


def test_anchors_present_in_each_modified_source():
    """Cada uno de los 3 sub-fixes debe anclar su anchor canonical en el
    source — un futuro reader que vea los knobs sabrá la razón."""
    auth_src = _AUTH_PY.read_text(encoding="utf-8")
    assert "P3-TIER-LIMITS-ENV" in auth_src, (
        "auth.py falta anchor `P3-TIER-LIMITS-ENV` — el bloque del knob "
        "tier limits puede ser 'limpiado' a dict literal sin saber por qué."
    )

    go_src = _GO_PY.read_text(encoding="utf-8")
    assert "P3-PLAN-MODEL-KNOBS" in go_src, (
        "graph_orchestrator.py falta anchor `P3-PLAN-MODEL-KNOBS` — el "
        "helper de modelo via knob puede revertirse a literal sin saber "
        "que cierra riesgo R2 (preview deprecation)."
    )

    app_src = _APP_PY.read_text(encoding="utf-8")
    assert "P3-SENTRY-COST-THRESHOLDS" in app_src, (
        "app.py falta anchor `P3-SENTRY-COST-THRESHOLDS` — los thresholds "
        "de revisión por escala se pierden sin el comment."
    )


# ---------------------------------------------------------------------------
# D4 — Tier limits via env vars.
# ---------------------------------------------------------------------------


def test_tier_limits_use_knobs_not_literal():
    """`auth.py` debe definir `_TIER_LIMITS` via `_env_int(...)` calls
    para los 5 tiers — no dict literal con números hardcoded."""
    src = _AUTH_PY.read_text(encoding="utf-8")

    # Knob names que deben existir
    for env_var in (
        "MEALFIT_TIER_LIMIT_GRATIS",
        "MEALFIT_TIER_LIMIT_BASIC",
        "MEALFIT_TIER_LIMIT_PLUS",
        "MEALFIT_TIER_LIMIT_ULTRA",
        "MEALFIT_TIER_LIMIT_ADMIN",
    ):
        assert env_var in src, (
            f"auth.py no usa env var `{env_var}`. Sin ella, ajustar el "
            f"límite del tier correspondiente requiere redeploy."
        )

    # Debe importar `_env_int` desde knobs
    assert "from knobs import _env_int" in src or "import _env_int" in src, (
        "auth.py no importa `_env_int` — sin auto-registry los knobs no "
        "aparecen en `/health/version`."
    )


def test_tier_limits_default_preserved():
    """Defaults del `_env_int(...)` para cada tier deben preservar
    pricing actual: gratis=15, basic=50, plus=200, ultra=999999, admin=999999."""
    src = _AUTH_PY.read_text(encoding="utf-8")
    expected_defaults = {
        "GRATIS": 15,
        "BASIC": 50,
        "PLUS": 200,
        "ULTRA": 999999,
        "ADMIN": 999999,
    }
    for tier, default_value in expected_defaults.items():
        # Patrón: `_env_int("MEALFIT_TIER_LIMIT_<TIER>", <default>)`
        pattern = rf'_env_int\(\s*"MEALFIT_TIER_LIMIT_{tier}"\s*,\s*{default_value}\s*\)'
        assert re.search(pattern, src), (
            f"auth.py no preserva default `{default_value}` para tier `{tier}`. "
            f"Cambiar el default requiere intención explícita — si fue accidental "
            f"reverter; si fue intencional actualizar este test."
        )


def test_tier_limits_dict_uses_module_const():
    """El callsite `verify_api_quota` debe leer de `_TIER_LIMITS.get(...)`
    no construir un dict literal en cada request (regresión a hardcoded)."""
    src = _AUTH_PY.read_text(encoding="utf-8")
    # Buscar dict literal con 'gratis': 15 dentro de verify_api_quota
    m = re.search(
        r"def verify_api_quota[\s\S]+?(?=\n(?:def |async def |class |\Z))",
        src,
    )
    assert m is not None, "verify_api_quota function not found"
    body = m.group(0)

    # No debe haber dict literal con tier limits dentro
    forbidden = re.search(
        r'\{\s*"gratis"\s*:\s*\d+\s*,\s*"basic"',
        body,
    )
    assert forbidden is None, (
        "verify_api_quota tiene dict literal de tier limits — regresión a "
        "hardcoded. Debe usar `_TIER_LIMITS` module-level."
    )
    assert "_TIER_LIMITS" in body, (
        "verify_api_quota no referencia `_TIER_LIMITS` — el knob no se aplica."
    )


# ---------------------------------------------------------------------------
# C4 — Plan-gen model knobs.
# ---------------------------------------------------------------------------


def test_plan_model_helpers_present():
    """`_plan_pro_model_name()` y `_plan_flash_model_name()` deben existir
    + leer `MEALFIT_PRO_MODEL`/`MEALFIT_FLASH_MODEL` via `_env_str`."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "def _plan_pro_model_name" in src, (
        "graph_orchestrator.py falta helper `_plan_pro_model_name()` — "
        "swap del modelo Pro requiere redeploy."
    )
    assert "def _plan_flash_model_name" in src, (
        "graph_orchestrator.py falta helper `_plan_flash_model_name()` — "
        "swap del modelo Flash requiere redeploy. Este modelo es ~80% del "
        "costo total (gemini-3-flash-preview, $4.83/14d)."
    )
    assert "MEALFIT_PRO_MODEL" in src, "env var MEALFIT_PRO_MODEL missing"
    assert "MEALFIT_FLASH_MODEL" in src, "env var MEALFIT_FLASH_MODEL missing"


def test_plan_model_constants_use_helpers():
    """Los `_PRO_MODEL_NAME` y `_FLASH_MODEL_NAME` module-level (usados
    en 14 callsites del orchestrator) deben asignarse desde los helpers
    via knob, no via string literal."""
    src = _GO_PY.read_text(encoding="utf-8")

    # Buscar las asignaciones module-level
    pro_pattern = re.search(
        r"^_PRO_MODEL_NAME\s*=\s*(.+)$",
        src,
        re.MULTILINE,
    )
    flash_pattern = re.search(
        r"^_FLASH_MODEL_NAME\s*=\s*(.+)$",
        src,
        re.MULTILINE,
    )
    assert pro_pattern is not None, "_PRO_MODEL_NAME assignment not found"
    assert flash_pattern is not None, "_FLASH_MODEL_NAME assignment not found"

    pro_rhs = pro_pattern.group(1).strip()
    flash_rhs = flash_pattern.group(1).strip()

    assert "_plan_pro_model_name()" in pro_rhs, (
        f"_PRO_MODEL_NAME debe asignarse desde `_plan_pro_model_name()`. "
        f"Encontrado: {pro_rhs!r}. Si es literal, regresión a hardcoded."
    )
    assert "_plan_flash_model_name()" in flash_rhs, (
        f"_FLASH_MODEL_NAME debe asignarse desde `_plan_flash_model_name()`. "
        f"Encontrado: {flash_rhs!r}. Si es literal, regresión a hardcoded."
    )


def test_plan_model_defaults_preserved():
    """Defaults del helper deben preservar comportamiento actual:
    Pro=gemini-3.1-pro-preview, Flash=gemini-3-flash-preview."""
    src = _GO_PY.read_text(encoding="utf-8")

    pro_default = re.search(
        r'_env_str\(\s*"MEALFIT_PRO_MODEL"\s*,\s*"([^"]+)"\s*\)',
        src,
    )
    flash_default = re.search(
        r'_env_str\(\s*"MEALFIT_FLASH_MODEL"\s*,\s*"([^"]+)"\s*\)',
        src,
    )
    assert pro_default is not None, "no _env_str call for MEALFIT_PRO_MODEL"
    assert flash_default is not None, "no _env_str call for MEALFIT_FLASH_MODEL"

    assert pro_default.group(1) == "gemini-3.1-pro-preview", (
        f"Default Pro model changed accidentally — got {pro_default.group(1)!r}, "
        f"expected `gemini-3.1-pro-preview` para preservar comportamiento."
    )
    assert flash_default.group(1) == "gemini-3-flash-preview", (
        f"Default Flash model changed accidentally — got {flash_default.group(1)!r}, "
        f"expected `gemini-3-flash-preview` para preservar comportamiento."
    )


# ---------------------------------------------------------------------------
# C3 — Sentry sampling thresholds documentation.
# ---------------------------------------------------------------------------


def test_sentry_threshold_doc_present():
    """app.py debe contener la documentación de cuándo bajar los Sentry
    sample rates (escala > X MAU). Sin esto, futuro reader no sabe el
    plan de capacity Sentry."""
    src = _APP_PY.read_text(encoding="utf-8")
    # El bloque debe mencionar al menos los 3 thresholds documentados
    for keyword in ("500 MAU", "1k MAU", "throttling"):
        assert keyword in src, (
            f"app.py falta keyword `{keyword!r}` en bloque P3-SENTRY-COST-THRESHOLDS. "
            f"La doc previene saturar cuota Sentry sin warning."
        )


# ---------------------------------------------------------------------------
# Functional: knobs registered in _KNOBS_REGISTRY.
# ---------------------------------------------------------------------------


def test_knobs_registered_via_env_str():
    """Los knobs nuevos deben usar `_env_str`/`_env_int` (no `os.environ.get`
    directo). Sin auto-registry el knob no aparece en /health/version."""
    auth_src = _AUTH_PY.read_text(encoding="utf-8")
    go_src = _GO_PY.read_text(encoding="utf-8")

    # auth.py: tier limits via _env_int
    assert "_env_int(" in auth_src and "MEALFIT_TIER_LIMIT_GRATIS" in auth_src

    # graph_orchestrator.py: model helpers via _env_str
    assert "_env_str" in go_src
    # Línea precisa en helper
    pro_helper = re.search(
        r"def _plan_pro_model_name[\s\S]+?return\s+_env_str",
        go_src,
    )
    flash_helper = re.search(
        r"def _plan_flash_model_name[\s\S]+?return\s+_env_str",
        go_src,
    )
    assert pro_helper is not None, (
        "_plan_pro_model_name no usa `_env_str` — sin auto-registry."
    )
    assert flash_helper is not None, (
        "_plan_flash_model_name no usa `_env_str` — sin auto-registry."
    )
