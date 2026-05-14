"""[P3-PDF-OBS-FU-C · 2026-05-14] Bundle test del follow-up P3 del re-audit
PDF (post P2-PDF-OBS-2).

Cierre de los 3 P3 menores identificados durante el re-audit 2026-05-14:

  P3-PDF-OBS-FU-A   `trackEvent('pdf_render_coherence_block_leak', ...)` en
                    [`shoppingHelpers.js::getActiveShoppingList`](frontend/src/utils/shoppingHelpers.js)
                    complementario al `console.warn` legacy (que se elimina
                    por esbuild `pure: ['console.warn']` en prod). Permite
                    a operadores detectar regresiones en `review_plan_node`
                    que dejen de popear `_shopping_coherence_block`.

  P3-PDF-OBS-FU-B   Clamp superior `[0, 100000]` sobre `fallback_inventory_size`
                    en
                    [`POST /api/plans/telemetry/pdf-stale-fallback`](backend/routers/plans.py).
                    Defense-in-depth contra POST adversarial con payloads
                    grandes que ensucien `pipeline_metrics.metadata`.

  P3-PDF-OBS-FU-C   Hysteresis en `_alert_pdf_stale_inventory_fallback_burst`
                    ([`cron_tasks.py`](backend/cron_tasks.py)): auto-resolve
                    cuando `count < threshold * ratio` (default 0.5, knob
                    `MEALFIT_PDF_STALE_FALLBACK_AUTO_RESOLVE_RATIO`,
                    clamp [0.1, 0.95]). Banda muerta entre ratio y
                    threshold preserva estado para evitar oscilación
                    emit/resolve bajo bursts borderline.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_pdf_obs_fu_c` →
`test_p3_pdf_obs_fu_c*.py`. Test único para el bundle (los 3 sub-fixes
comparten el mismo ciclo de re-audit; marker bundle es el último
alfabético `P3-PDF-OBS-FU-C`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_ROOT = _REPO_ROOT / "backend"
_SHOPPING_HELPERS_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "shoppingHelpers.js"
_ROUTERS_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_CRON_TASKS_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def shopping_helpers_src() -> str:
    return _SHOPPING_HELPERS_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def routers_plans_src() -> str:
    return _ROUTERS_PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_tasks_src() -> str:
    return _CRON_TASKS_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_js_function_body(src: str, fn_name: str) -> str:
    """Extrae el cuerpo de un arrow-fn JS exportado: `export const <fn_name> = (...) => { ... }`.
    Busca hasta la próxima `export const` top-level.

    Tooltip-anchor: P3-PDF-OBS-FU-JS-EXTRACTOR.
    """
    anchor = re.search(
        rf"export\s+const\s+{re.escape(fn_name)}\s*=\s*\([^)]*\)\s*=>\s*\{{",
        src,
    )
    assert anchor is not None, (
        f"No se encontró `export const {fn_name} = (...) => {{` en JS source."
    )
    start = anchor.end()
    rest = src[start:]
    next_export = re.search(r"\nexport\s+(const|function)\s+", rest)
    end = start + (next_export.start() if next_export else len(rest))
    return src[start:end]


def _extract_python_function_body(src: str, fn_name: str) -> str:
    """Extrae cuerpo Python desde `def <fn_name>(` hasta el siguiente top-level
    `def `, `@router.`, o `class `."""
    anchor = re.search(rf"^def\s+{re.escape(fn_name)}\s*\(", src, re.MULTILINE)
    assert anchor is not None, f"No se encontró `def {fn_name}(` en source."
    start = anchor.start()
    rest = src[anchor.end():]
    next_top = re.search(r"^(?:def |@router\.|class |async def )", rest, re.MULTILINE)
    end = anchor.end() + (next_top.start() if next_top else len(rest))
    return src[start:end]


def _extract_endpoint_handler_body(src: str, method: str, endpoint: str) -> str:
    """Extrae el handler completo de `@router.<method>("<endpoint>")` hasta el
    siguiente `\\n@router.` top-level. Tooltip-anchor: P3-PDF-OBS-FU-ENDPOINT-EXTRACTOR."""
    pattern = re.compile(
        rf'@router\.{re.escape(method)}\(\s*["\']{re.escape(endpoint)}["\']\s*\)',
    )
    m = pattern.search(src)
    assert m is not None, f"No se encontró @router.{method}({endpoint!r})."
    start = m.start()
    rest = src[m.end():]
    next_endpoint = rest.find("\n@router.")
    end = m.end() + (next_endpoint if next_endpoint > -1 else len(rest))
    return src[start:end]


# ---------------------------------------------------------------------------
# P3-PDF-OBS-FU-A · trackEvent en getActiveShoppingList
# ---------------------------------------------------------------------------
def test_fu_a_get_active_shopping_emits_track_event(shopping_helpers_src: str):
    """El bloque del `_shopping_coherence_block` violation debe emitir
    `trackEvent('pdf_render_coherence_block_leak', ...)` junto con el
    `console.warn` legacy. Sin el trackEvent, en producción el warn es
    stripped por esbuild → operadores ciegos al contract break."""
    body = _extract_js_function_body(shopping_helpers_src, "getActiveShoppingList")
    assert "pdf_render_coherence_block_leak" in body, (
        "P3-PDF-OBS-FU-A regresión: `trackEvent('pdf_render_coherence_block_leak', "
        "...)` desapareció de `getActiveShoppingList`. Sin él, una regresión en "
        "`review_plan_node` que deje de popear `_shopping_coherence_block` pasa "
        "inadvertida en prod (`console.warn` se elimina por esbuild)."
    )


def test_fu_a_track_event_inside_coherence_block_guard(shopping_helpers_src: str):
    """El trackEvent debe estar DENTRO del `if (Array.isArray(...) && length>0)`
    para que solo se emita cuando el contrato realmente está roto. Si se emitiera
    siempre, los analytics quedan flooded de eventos no-accionables.
    """
    body = _extract_js_function_body(shopping_helpers_src, "getActiveShoppingList")
    guard_idx = body.find("Array.isArray(planData._shopping_coherence_block)")
    next_top_idx = body.find("const keyMap", guard_idx + 1)
    assert guard_idx > -1 and next_top_idx > -1, (
        "Anclas estructurales del bloque guard ausentes — refactor mayor."
    )
    guard_block = body[guard_idx:next_top_idx]
    assert "pdf_render_coherence_block_leak" in guard_block, (
        "P3-PDF-OBS-FU-A regresión: el trackEvent debe estar dentro del bloque "
        "guard que chequea `_shopping_coherence_block` non-empty. Si se emite "
        "incondicionalmente, los analytics quedan flooded."
    )


def test_fu_a_track_event_uses_dynamic_import(shopping_helpers_src: str):
    """Para no encadenar `analytics.js` al bundle de `shoppingHelpers` (que
    se importa eager en muchas rutas), el trackEvent debe leerse vía
    `import('./analytics.js')` dynamic. Usuarios con planes válidos NO pagan
    la carga del módulo."""
    body = _extract_js_function_body(shopping_helpers_src, "getActiveShoppingList")
    assert re.search(r"import\(\s*['\"]\./analytics\.js['\"]\s*\)", body), (
        "P3-PDF-OBS-FU-A regresión: el trackEvent debe usar `import('./analytics.js')` "
        "dinámico para que la carga del módulo de analytics sea lazy (solo el "
        "rare contract-break path la dispara)."
    )


def test_fu_a_track_event_payload_includes_entries_count(shopping_helpers_src: str):
    """Payload del trackEvent debe incluir `entries_count` (cuántas
    divergencias) para que operadores correlen con severidad."""
    body = _extract_js_function_body(shopping_helpers_src, "getActiveShoppingList")
    track_idx = body.find("pdf_render_coherence_block_leak")
    window = body[track_idx:track_idx + 500]
    assert "entries_count" in window, (
        "P3-PDF-OBS-FU-A regresión: payload no incluye `entries_count`. Sin él "
        "operadores no pueden distinguir 1 divergencia leve vs burst de 50."
    )
    assert "plan_id" in window, (
        "P3-PDF-OBS-FU-A regresión: payload no incluye `plan_id` (correlación)."
    )


def test_fu_a_track_event_best_effort(shopping_helpers_src: str):
    """El bloque del trackEvent debe estar en try/catch para no romper el
    render si analytics SDK falla."""
    body = _extract_js_function_body(shopping_helpers_src, "getActiveShoppingList")
    track_idx = body.find("pdf_render_coherence_block_leak")
    # Window backward ~300 chars para capturar el try { ... import().then(({ trackEvent }) => { try { trackEvent(...
    backward = body[max(0, track_idx - 400):track_idx]
    assert backward.count("try") >= 2, (
        "P3-PDF-OBS-FU-A regresión: el trackEvent debe estar en doble try "
        "(outer protege `import()` sync-error, inner protege fallo del SDK). "
        "Un fallo del módulo de analytics NO debe romper el render del PDF."
    )


# ---------------------------------------------------------------------------
# P3-PDF-OBS-FU-B · clamp fallback_inventory_size
# ---------------------------------------------------------------------------
def test_fu_b_telemetry_endpoint_clamps_fallback_size(routers_plans_src: str):
    """El handler `/telemetry/pdf-stale-fallback` debe clampear
    `fallback_size > 100000` al cap. Sin clamp, un POST adversarial con
    `fallback_inventory_size=999999999` se persiste tal cual."""
    handler = _extract_endpoint_handler_body(
        routers_plans_src, "post", "/telemetry/pdf-stale-fallback"
    )
    assert re.search(
        r"fallback_size\s+is\s+not\s+None\s+and\s+fallback_size\s*>\s*100000",
        handler,
    ), (
        "P3-PDF-OBS-FU-B regresión: clamp `fallback_size > 100000` perdido en "
        "el handler `/telemetry/pdf-stale-fallback`. Defense-in-depth contra "
        "POST adversarial."
    )
    # Y el clamp asigna 100000 (no un valor arbitrario).
    assert "fallback_size = 100000" in handler, (
        "P3-PDF-OBS-FU-B regresión: el clamp debe asignar 100000 (cap), no otro "
        "valor. Mantiene la signal sin perderla por completo."
    )


def test_fu_b_clamp_anchored_with_marker(routers_plans_src: str):
    """El bloque del clamp debe llevar anchor `[P3-PDF-OBS-FU-B]` para
    trazabilidad en futuros refactors."""
    handler = _extract_endpoint_handler_body(
        routers_plans_src, "post", "/telemetry/pdf-stale-fallback"
    )
    assert "P3-PDF-OBS-FU-B" in handler, (
        "P3-PDF-OBS-FU-B regresión: anchor del marker perdido en el handler. "
        "Sin anchor, un refactor cosmético puede eliminar el clamp sin alarma."
    )


def test_fu_b_clamp_preserves_existing_validation(routers_plans_src: str):
    """El clamp NO debe romper las validaciones previas (`bool`/`int`/`<0`)."""
    handler = _extract_endpoint_handler_body(
        routers_plans_src, "post", "/telemetry/pdf-stale-fallback"
    )
    # Las validaciones legacy siguen presentes.
    assert "isinstance(fallback_size, bool)" in handler, (
        "P3-PDF-OBS-FU-B regresión: validación `isinstance(bool)` perdida."
    )
    assert "fallback_size < 0" in handler, (
        "P3-PDF-OBS-FU-B regresión: validación `fallback_size < 0` perdida."
    )


# ---------------------------------------------------------------------------
# P3-PDF-OBS-FU-C · hysteresis en burst auto-resolve
# ---------------------------------------------------------------------------
def test_fu_c_cron_reads_auto_resolve_ratio_knob(cron_tasks_src: str):
    """El cron `_alert_pdf_stale_inventory_fallback_burst` debe leer
    `MEALFIT_PDF_STALE_FALLBACK_AUTO_RESOLVE_RATIO` vía `_env_float`."""
    body = _extract_python_function_body(
        cron_tasks_src, "_alert_pdf_stale_inventory_fallback_burst"
    )
    assert "MEALFIT_PDF_STALE_FALLBACK_AUTO_RESOLVE_RATIO" in body, (
        "P3-PDF-OBS-FU-C regresión: knob `MEALFIT_PDF_STALE_FALLBACK_AUTO_RESOLVE_RATIO` "
        "ausente del cron de burst. Sin hysteresis, bursts borderline oscilan "
        "alert emit/resolve cada tick."
    )
    assert "_env_float" in body, (
        "P3-PDF-OBS-FU-C regresión: el knob debe leerse vía `_env_float` "
        "(auto-registra en `_KNOBS_REGISTRY` + validator de rango)."
    )
    # Default 0.5.
    assert re.search(
        r'_env_float\(\s*\n?\s*"MEALFIT_PDF_STALE_FALLBACK_AUTO_RESOLVE_RATIO"\s*,\s*\n?\s*0\.5',
        body,
    ), (
        "P3-PDF-OBS-FU-C regresión: default `0.5` del knob perdido."
    )


def test_fu_c_cron_uses_validator_for_ratio_range(cron_tasks_src: str):
    """El `_env_float` debe llevar `validator=lambda v: 0.1 <= v <= 0.95`.
    0.1 sería ratio absurdo (alert quedaría abierta indefinida),
    0.95 anula el hysteresis (vuelve al comportamiento pre-fix)."""
    body = _extract_python_function_body(
        cron_tasks_src, "_alert_pdf_stale_inventory_fallback_burst"
    )
    assert re.search(r"validator\s*=\s*lambda\s+v\s*:\s*0\.1\s*<=\s*v\s*<=\s*0\.95", body), (
        "P3-PDF-OBS-FU-C regresión: validator del knob debe clamparse `[0.1, 0.95]`. "
        "Sin validator, un operador setea 0.001 y la alert queda abierta para "
        "siempre, o setea 1.5 y el hysteresis se anula (pre-fix behavior)."
    )


def test_fu_c_cron_uses_auto_resolve_threshold(cron_tasks_src: str):
    """El cuerpo del cron debe usar `auto_resolve_threshold` (computed)
    para la rama de resolve, NO el `threshold` raw."""
    body = _extract_python_function_body(
        cron_tasks_src, "_alert_pdf_stale_inventory_fallback_burst"
    )
    assert "auto_resolve_threshold" in body, (
        "P3-PDF-OBS-FU-C regresión: variable `auto_resolve_threshold` perdida."
    )
    # La rama de resolve debe ser `n < auto_resolve_threshold`.
    assert re.search(r"if\s+n\s*<\s*auto_resolve_threshold\s*:", body), (
        "P3-PDF-OBS-FU-C regresión: rama de auto-resolve debe ser "
        "`if n < auto_resolve_threshold:`, NO `n < threshold` (pre-fix)."
    )


def test_fu_c_cron_has_dead_band_branch(cron_tasks_src: str):
    """Entre `auto_resolve_threshold` y `threshold` el cron debe early-return
    sin emit ni resolve. Banda muerta del hysteresis."""
    body = _extract_python_function_body(
        cron_tasks_src, "_alert_pdf_stale_inventory_fallback_burst"
    )
    # Tras el `if n < auto_resolve_threshold:` ... `return`, debe venir
    # `if n < threshold:` ... `return` (dead band).
    auto_resolve_idx = body.find("if n < auto_resolve_threshold")
    assert auto_resolve_idx > -1
    # En los ~800 chars siguientes (cubre el bloque resolve + dead band)
    # debe aparecer el segundo `if n < threshold:` y un `return`.
    window = body[auto_resolve_idx:auto_resolve_idx + 1500]
    assert re.search(r"if\s+n\s*<\s*threshold\s*:\s*\n", window), (
        "P3-PDF-OBS-FU-C regresión: banda muerta del hysteresis ausente. "
        "Tras el auto-resolve, debe haber un `if n < threshold: return` que "
        "preserve el estado del alert cuando el count está entre ratio y "
        "threshold."
    )


def test_fu_c_runtime_validator_clamps_extreme_ratios():
    """Test funcional: el validator del `_env_float` rechaza valores fuera
    de `[0.1, 0.95]`. Usamos el helper canónico del repo."""
    import os
    from knobs import _env_float

    # Caso 1: valor extremo bajo → cae a default
    os.environ["__TEST_RATIO_LOW__"] = "0.001"
    v = _env_float(
        "__TEST_RATIO_LOW__",
        0.5,
        validator=lambda v: 0.1 <= v <= 0.95,
    )
    del os.environ["__TEST_RATIO_LOW__"]
    assert v == 0.5, f"Valor extremo bajo debió caer a default 0.5, got {v}"

    # Caso 2: valor extremo alto → cae a default
    os.environ["__TEST_RATIO_HIGH__"] = "1.5"
    v = _env_float(
        "__TEST_RATIO_HIGH__",
        0.5,
        validator=lambda v: 0.1 <= v <= 0.95,
    )
    del os.environ["__TEST_RATIO_HIGH__"]
    assert v == 0.5, f"Valor extremo alto debió caer a default 0.5, got {v}"

    # Caso 3: valor válido → respetado
    os.environ["__TEST_RATIO_OK__"] = "0.3"
    v = _env_float(
        "__TEST_RATIO_OK__",
        0.5,
        validator=lambda v: 0.1 <= v <= 0.95,
    )
    del os.environ["__TEST_RATIO_OK__"]
    assert v == 0.3, f"Valor válido debió respetarse, got {v}"


def test_fu_c_threshold_computation_handles_small_thresholds(cron_tasks_src: str):
    """Edge case: si threshold=2 y ratio=0.5, naive `int(2*0.5)=1` está OK.
    Pero si threshold=1 y ratio=0.5, `int(0.5)=0` haría que NUNCA auto-resolve
    (count siempre >= 0). El `max(1, ...)` defense protege ese caso."""
    body = _extract_python_function_body(
        cron_tasks_src, "_alert_pdf_stale_inventory_fallback_burst"
    )
    assert re.search(r"auto_resolve_threshold\s*=\s*max\(\s*1\s*,", body), (
        "P3-PDF-OBS-FU-C regresión: `max(1, ...)` perdido. Con threshold pequeño "
        "y ratio bajo, `int(threshold * ratio)` puede ser 0 → alert NUNCA "
        "auto-resuelve."
    )
