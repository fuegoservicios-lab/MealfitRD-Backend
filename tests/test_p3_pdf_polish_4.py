"""[P3-PDF-POLISH-4 · 2026-05-14] Bundle test post-audit PDF lista-de-compras.

Cierre de 4 P3 defense-in-depth identificados en la re-auditoría 2026-05-14
(post P3-SHOPPING-1/2/3/4):

  P3-PDF-POLISH-4-B   clamp superior `householdSize` (knob `MEALFIT_MAX_HOUSEHOLD_SIZE`)
  P3-PDF-POLISH-4-A   `RateLimiter` en `/recalculate-shopping-list` + `/telemetry/pdf-stale-fallback`
  P3-PDF-POLISH-4-C   doc exemption en CLAUDE.md "Historial-quota-exemption"
  P3-PDF-POLISH-4-D   `logger.exception` en lugar de `logger.error + traceback.print_exc`

Test único para satisfacer la cross-link convention P2-HIST-AUDIT-14
(`p3_pdf_polish_4` slug → `test_p3_pdf_polish_4*.py`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_CONSTANTS_PY = _BACKEND_ROOT / "constants.py"
_CLAUDE_MD = _BACKEND_ROOT.parent / "CLAUDE.md"


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def constants_src() -> str:
    return _CONSTANTS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def claude_md_src() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P3-PDF-POLISH-4-B · clamp superior householdSize
# ---------------------------------------------------------------------------
def test_b_compute_household_multiplier_reads_max_knob(constants_src: str):
    """`compute_household_multiplier` debe leer `MEALFIT_MAX_HOUSEHOLD_SIZE`
    vía `_env_int` (auto-registro en `_KNOBS_REGISTRY`).
    """
    body = _extract_function_body(constants_src, "compute_household_multiplier")
    assert '_env_int("MEALFIT_MAX_HOUSEHOLD_SIZE"' in body, (
        "P3-PDF-POLISH-4-B regresión: `compute_household_multiplier` ya no "
        "lee `MEALFIT_MAX_HOUSEHOLD_SIZE`. Knob requerido para escape hatch "
        "sin redeploy. Ver CLAUDE.md sección 'Knobs operacionales'."
    )
    # Default 20 explícito.
    assert re.search(
        r'_env_int\(\s*"MEALFIT_MAX_HOUSEHOLD_SIZE"\s*,\s*20\s*\)',
        body,
    ), "Default `MEALFIT_MAX_HOUSEHOLD_SIZE=20` perdido."


def test_b_compute_household_multiplier_clamps_composition_and_legacy(constants_src: str):
    """El clamp debe aplicarse a AMBAS ramas (composition + legacy)."""
    body = _extract_function_body(constants_src, "compute_household_multiplier")
    # Buscar `min(raw, float(_max_household))` aplicado dos veces (composition + legacy)
    matches = re.findall(r'min\(raw,\s*float\(_max_household\)\)', body)
    assert len(matches) >= 2, (
        f"P3-PDF-POLISH-4-B regresión: clamp `min(raw, _max_household)` "
        f"debe aparecer en AMBAS ramas (composition + legacy householdSize). "
        f"Encontradas: {len(matches)}."
    )


def test_b_compute_household_multiplier_runtime_clamps_huge_values():
    """Test funcional: `householdSize=999999` → multiplier ≤ 20 (default)."""
    from constants import compute_household_multiplier
    huge = compute_household_multiplier({"householdSize": 999999})
    assert huge == 20.0, f"Esperado 20.0 (clamp), got {huge}"
    # Composition path también clampea.
    huge_comp = compute_household_multiplier({
        "householdComposition": {"adults": 500, "children": 500}
    })
    assert huge_comp == 20.0, f"Esperado 20.0 (clamp composition), got {huge_comp}"


def test_b_compute_household_multiplier_respects_lower_bound():
    """Sanity: 0 personas → 1.0 (mínimo). El clamp superior no rompe el inferior."""
    from constants import compute_household_multiplier
    assert compute_household_multiplier({"householdSize": 0}) == 1.0
    assert compute_household_multiplier(None) == 1.0
    assert compute_household_multiplier({}) == 1.0


def test_b_recalc_endpoint_clamps_household_size_inline(plans_src: str):
    """`api_recalculate_shopping_list` aplica el clamp ANTES de pasar al
    helper, para que `calc_household_size` persistido en plan_data nunca
    pase el cap (defense-in-depth)."""
    handler_body = _extract_function_body(plans_src, "api_recalculate_shopping_list")
    assert "MEALFIT_MAX_HOUSEHOLD_SIZE" in handler_body, (
        "P3-PDF-POLISH-4-B regresión: handler `api_recalculate_shopping_list` "
        "ya no clampea `householdSize`. El POST con valor adversarial podría "
        "persistir `calc_household_size=999999` en plan_data."
    )
    # Buscar la línea del clamp.
    assert re.search(
        r'household_size\s*=\s*max\(\s*1\s*,\s*min\(',
        handler_body,
    ), "Clamp `max(1, min(...))` perdido en `household_size` del recalc handler."


# ---------------------------------------------------------------------------
# P3-PDF-POLISH-4-A · RateLimiter en recalc + telemetry endpoints
# ---------------------------------------------------------------------------
def test_a_recalc_limiter_singleton_defined(plans_src: str):
    """`_RECALC_LIMITER` debe ser singleton a nivel módulo con max_calls=20/60s."""
    assert re.search(
        r'_RECALC_LIMITER\s*=\s*RateLimiter\(\s*max_calls\s*=\s*20\s*,\s*period_seconds\s*=\s*60\s*\)',
        plans_src,
    ), (
        "P3-PDF-POLISH-4-A regresión: `_RECALC_LIMITER = "
        "RateLimiter(max_calls=20, period_seconds=60)` perdido. Sin este "
        "singleton, un cliente autenticado puede spammear /recalculate-"
        "shopping-list (3× compute heavy por call)."
    )


def test_a_pdf_telemetry_limiter_singleton_defined(plans_src: str):
    """`_PDF_TELEMETRY_LIMITER` debe ser singleton a nivel módulo con max_calls=30/60s."""
    assert re.search(
        r'_PDF_TELEMETRY_LIMITER\s*=\s*RateLimiter\(\s*max_calls\s*=\s*30\s*,\s*period_seconds\s*=\s*60\s*\)',
        plans_src,
    ), (
        "P3-PDF-POLISH-4-A regresión: `_PDF_TELEMETRY_LIMITER = "
        "RateLimiter(max_calls=30, period_seconds=60)` perdido. Sin este "
        "singleton, un cliente puede llenar `pipeline_metrics` con eventos "
        "`pdf_stale_inventory_fallback` arbitrarios."
    )


def test_a_recalc_endpoint_uses_recalc_limiter(plans_src: str):
    """El endpoint `/recalculate-shopping-list` debe usar `Depends(_RECALC_LIMITER)`,
    NO `Depends(get_verified_user_id)` plano (que carece de rate limit)."""
    decorator_and_def = _extract_endpoint_decorator_and_signature(
        plans_src, "post", "/recalculate-shopping-list"
    )
    assert "Depends(_RECALC_LIMITER)" in decorator_and_def, (
        "P3-PDF-POLISH-4-A regresión: `/recalculate-shopping-list` ya no usa "
        "`Depends(_RECALC_LIMITER)`. Ver CLAUDE.md 'Historial-quota-exemption' "
        "para el contexto de por qué no usamos `verify_api_quota` aquí."
    )


def test_a_pdf_telemetry_endpoint_uses_pdf_telemetry_limiter(plans_src: str):
    """El endpoint `/telemetry/pdf-stale-fallback` debe usar
    `Depends(_PDF_TELEMETRY_LIMITER)`."""
    decorator_and_def = _extract_endpoint_decorator_and_signature(
        plans_src, "post", "/telemetry/pdf-stale-fallback"
    )
    assert "Depends(_PDF_TELEMETRY_LIMITER)" in decorator_and_def, (
        "P3-PDF-POLISH-4-A regresión: `/telemetry/pdf-stale-fallback` ya no usa "
        "`Depends(_PDF_TELEMETRY_LIMITER)`. Sin él, un atacante puede inflar "
        "`pipeline_metrics`."
    )


# ---------------------------------------------------------------------------
# P3-PDF-POLISH-4-C · doc exemption en CLAUDE.md
# ---------------------------------------------------------------------------
def test_c_claude_md_documents_recalc_exemption(claude_md_src: str):
    """La tabla `Historial-quota-exemption` en CLAUDE.md debe incluir una row
    para `/recalculate-shopping-list` con su razón y mitigación."""
    section = _extract_claude_section(claude_md_src, "### Historial-quota-exemption")
    assert "`/recalculate-shopping-list`" in section, (
        "P3-PDF-POLISH-4-C regresión: row para `/recalculate-shopping-list` "
        "ausente en sección 'Historial-quota-exemption' de CLAUDE.md."
    )
    assert "P3-PDF-POLISH-4-C" in section, "Anchor de marker perdido en la row."
    assert "_RECALC_LIMITER" in section, (
        "La razón debe mencionar `_RECALC_LIMITER` (mitigación del riesgo "
        "de spam — el paywall no es la herramienta correcta)."
    )


def test_c_claude_md_documents_telemetry_exemption(claude_md_src: str):
    """Misma tabla debe incluir row para `/telemetry/pdf-stale-fallback`."""
    section = _extract_claude_section(claude_md_src, "### Historial-quota-exemption")
    assert "`/telemetry/pdf-stale-fallback`" in section, (
        "P3-PDF-POLISH-4-C regresión: row para `/telemetry/pdf-stale-fallback` "
        "ausente en sección 'Historial-quota-exemption' de CLAUDE.md."
    )
    assert "_PDF_TELEMETRY_LIMITER" in section, (
        "La razón debe mencionar `_PDF_TELEMETRY_LIMITER`."
    )


# ---------------------------------------------------------------------------
# P3-PDF-POLISH-4-D · logger.exception en api_recalculate_shopping_list
# ---------------------------------------------------------------------------
def test_d_recalc_handler_uses_logger_exception(plans_src: str):
    """El except branch de `api_recalculate_shopping_list` debe usar
    `logger.exception(...)` (incluye traceback estructurado), NO la pareja
    `logger.error + traceback.print_exc` que escribía a stdout sin pasar
    por el logger handler."""
    handler_body = _extract_function_body(plans_src, "api_recalculate_shopping_list")
    assert "logger.exception" in handler_body, (
        "P3-PDF-POLISH-4-D regresión: `logger.exception` ausente del except "
        "branch. Sin él, el traceback no llega estructurado al sink de logs."
    )
    # Verificar que la pareja legacy fue removida.
    assert "traceback.print_exc()" not in handler_body, (
        "P3-PDF-POLISH-4-D regresión: `traceback.print_exc()` legacy aún presente. "
        "Debe haberse removido junto con `logger.error` redundante."
    )


# ---------------------------------------------------------------------------
# Helpers parser-based
# ---------------------------------------------------------------------------
def _extract_function_body(src: str, fn_name: str) -> str:
    """Devuelve el cuerpo de la función `def <fn_name>(` hasta el siguiente
    `def ` o `@router.` top-level. Cota suficiente para tests parser-based
    de regiones de código de tamaño moderado.
    """
    anchor = re.search(rf'^def\s+{re.escape(fn_name)}\s*\(', src, re.MULTILINE)
    assert anchor is not None, f"No se encontró `def {fn_name}(` en source."
    start = anchor.start()
    # Buscar el siguiente top-level `def ` o `@router.` ANTES del cual termina la fn.
    rest = src[anchor.end():]
    next_top = re.search(r'^(?:def |@router\.|class )', rest, re.MULTILINE)
    end = anchor.end() + (next_top.start() if next_top else len(rest))
    return src[start:end]


def _extract_endpoint_decorator_and_signature(src: str, method: str, endpoint: str) -> str:
    """Devuelve el decorador `@router.<method>("<endpoint>")` + función +
    signature hasta el `):`."""
    pattern = re.compile(
        rf'@router\.{re.escape(method)}\(\s*["\']{re.escape(endpoint)}["\']\s*\).*?def\s+\w+\s*\((.*?)\)\s*:',
        re.DOTALL,
    )
    m = pattern.search(src)
    assert m, f"No se encontró endpoint `{method.upper()} {endpoint}` en routers/plans.py."
    return m.group(0)


def _extract_claude_section(src: str, header: str) -> str:
    """Devuelve el texto desde `header` hasta el siguiente `###` o `---`."""
    start = src.find(header)
    assert start >= 0, f"Sección `{header}` no encontrada en CLAUDE.md."
    rest = src[start + len(header):]
    next_section = re.search(r'^(?:###\s|---)', rest, re.MULTILINE)
    end = start + len(header) + (next_section.start() if next_section else len(rest))
    return src[start:end]
