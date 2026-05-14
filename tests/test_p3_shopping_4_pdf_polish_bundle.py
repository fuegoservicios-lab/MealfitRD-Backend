"""[P3-SHOPPING-1/2/3/4 · 2026-05-14] Bundle de polish del flujo "lista
de compras PDF" (cierre 4/4 P3 del audit 2026-05-13).

Los 4 P3 son cambios cosméticos/operacionales:

  - P3-SHOPPING-1: nombre del PDF con discriminador único (fecha +
    plan_id[:8]) para evitar colisiones cuando el usuario descarga
    varios PDFs.

  - P3-SHOPPING-2: gate de `_debug_recalc` en `/recalculate-shopping-list`
    — antes se persistía SIEMPRE a `plan_data._debug_recalc`. Ahora
    solo en non-production (o con knob `MEALFIT_PERSIST_DEBUG_RECALC`).

  - P3-SHOPPING-3: NO-OP. Ya cerrado por P3-FRONTEND-1 · 2026-05-12 —
    `vite.config.js` ya tiene `pure: ['console.log', 'console.warn',
    'console.debug', 'console.info']` en mode=production. Este test
    confirma la config sigue válida (regression guard).

  - P3-SHOPPING-4: `trackEvent('pdf_download_success', ...)` +
    `trackEvent('pdf_download_failed', ...)` para discriminar
    "feature no usado" de "feature roto" en analytics.

El test usa el marker P3-SHOPPING-4 (el último alfabético del bundle)
porque solo puede haber un marker activo en `_LAST_KNOWN_PFIX`. El
cross-link test `test_p2_hist_audit_14_marker_test_link` verifica que
el slug `p3_shopping_4` matchea este archivo (glob `test_p3_shopping_4*.py`).

Tooltip-anchor: P3-SHOPPING-1, P3-SHOPPING-2, P3-SHOPPING-3 (no-op),
P3-SHOPPING-4.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASH_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_VITE_CFG = _REPO_ROOT / "frontend" / "vite.config.js"


@pytest.fixture(scope="module")
def dash_src() -> str:
    return _DASH_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def vite_src() -> str:
    return _VITE_CFG.read_text(encoding="utf-8")


def _extract_handler_block(src: str) -> str:
    start = src.find("const handleDownloadShoppingList")
    assert start > 0
    after = src[start + 50:]
    next_match = re.search(r"\n    const handle[A-Z]", after)
    end = (start + 50 + next_match.start()) if next_match else (start + 50 + 6000)
    return src[start:end]


# ===========================================================================
# P3-SHOPPING-1: filename con discriminador único
# ===========================================================================

def test_p3_shopping_1_filename_includes_date(dash_src: str):
    """El filename del PDF incluye fecha (YYYY-MM-DD) como discriminador."""
    body = _extract_handler_block(dash_src)
    # Pattern: el opt.filename usa una variable derivada de `new Date()...`.
    # Verificamos que el helper de fecha aparece cerca de la construcción
    # del filename.
    filename_idx = body.find("filename:")
    assert filename_idx > 0, "P3-SHOPPING-1: opt.filename no encontrado."
    # Window 600 chars hacia atrás cubre la construcción de variables previas.
    window_back = body[max(0, filename_idx - 600):filename_idx]
    assert re.search(r"new\s+Date\(\)\.toISOString\(\)\.slice\(\s*0\s*,\s*10\s*\)", window_back), (
        "P3-SHOPPING-1 regresión: el filename ya no incorpora fecha "
        "(`new Date().toISOString().slice(0, 10)`). Sin discriminador "
        "temporal, re-descargas del mismo plan colisionan en el sistema "
        "de archivos del usuario."
    )


def test_p3_shopping_1_filename_includes_plan_id_prefix(dash_src: str):
    """El filename incluye prefix corto del plan_id como discriminador."""
    body = _extract_handler_block(dash_src)
    filename_idx = body.find("filename:")
    window_back = body[max(0, filename_idx - 600):filename_idx]
    # Pattern: extrae 8 chars del id del plan via .slice(0, 8) sobre
    # effectivePlanData?.id (o equivalente).
    assert re.search(r"\.slice\(\s*0\s*,\s*8\s*\)", window_back), (
        "P3-SHOPPING-1 regresión: el filename ya no usa `.slice(0, 8)` "
        "del plan_id. Sin prefix discriminador, dos planes distintos "
        "descargados el mismo día con la misma duration producen el "
        "mismo nombre y colisionan."
    )


def test_p3_shopping_1_filename_template_uses_discriminators(dash_src: str):
    """El template literal del filename incorpora las variables."""
    body = _extract_handler_block(dash_src)
    filename_idx = body.find("filename:")
    line_end = body.find("\n", filename_idx)
    filename_line = body[filename_idx:line_end if line_end > 0 else filename_idx + 200]
    # Pattern: backtick template con al menos 2 ${...} (durationText + date/id).
    # Contamos ${ apariciones en la línea del filename.
    interpolations = filename_line.count("${")
    assert interpolations >= 3, (
        f"P3-SHOPPING-1 regresión: el template del filename tiene solo "
        f"{interpolations} interpolación(es) `${{...}}`. Esperaba ≥3 "
        f"(durationText + fecha + plan_id_prefix). Si reducir a 1 fue "
        f"intencional, restaurar — re-descargas colisionan sin "
        f"discriminadores múltiples."
    )


def test_p3_shopping_1_anchor_present(dash_src: str):
    """Anchor `[P3-SHOPPING-1 · 2026-05-14]` en comment del handler."""
    body = _extract_handler_block(dash_src)
    assert "P3-SHOPPING-1" in body, (
        "P3-SHOPPING-1 regresión: anchor desapareció del handler. "
        "Restaurar el comment que documenta por qué el filename tiene "
        "discriminadores."
    )


# ===========================================================================
# P3-SHOPPING-2: gate de _debug_recalc en non-production
# ===========================================================================

def test_p3_shopping_2_debug_recalc_gated_by_environment(plans_src: str):
    """`_debug_recalc` se persiste solo si ENVIRONMENT != 'production'
    O si el knob `MEALFIT_PERSIST_DEBUG_RECALC` está activo.
    """
    # Localizar el bloque _debug_recalc. La función completa
    # `api_recalculate_shopping_list` mide ~230 líneas; usamos 30000
    # chars para cubrirla holgadamente sin riesgo de truncar el bloque
    # del fingerprint que está al final.
    fn_idx = plans_src.find("def api_recalculate_shopping_list")
    assert fn_idx > 0
    body = plans_src[fn_idx:fn_idx + 30000]
    debug_idx = body.find('plan_data["_debug_recalc"]')
    assert debug_idx > 0, (
        "P3-SHOPPING-2 regresión: `plan_data[\"_debug_recalc\"] = {...}` "
        "ya no aparece en `api_recalculate_shopping_list`. Si fue "
        "eliminado completamente, eliminar este test."
    )
    # Window backward: 1200 chars antes del bloque cubren el gate +
    # comentarios explicativos. La asignación está dentro de `if
    # _persist_debug:` y el gate se define ~10 líneas arriba.
    window_back = body[max(0, debug_idx - 1200):debug_idx]
    assert re.search(
        r"ENVIRONMENT.*production",
        window_back,
        re.IGNORECASE | re.DOTALL,
    ), (
        "P3-SHOPPING-2 regresión: el gate `ENVIRONMENT != 'production'` "
        "(en code, no solo en comment) desapareció. Sin él, `_debug_recalc` "
        "se persiste en producción ensuciando jsonb sin valor operacional."
    )


def test_p3_shopping_2_knob_escape_hatch_documented(plans_src: str):
    """Knob `MEALFIT_PERSIST_DEBUG_RECALC` permite re-habilitar en prod
    sin redeploy (escape hatch para SRE)."""
    fn_idx = plans_src.find("def api_recalculate_shopping_list")
    body = plans_src[fn_idx:fn_idx + 30000]
    assert "MEALFIT_PERSIST_DEBUG_RECALC" in body, (
        "P3-SHOPPING-2 regresión: knob `MEALFIT_PERSIST_DEBUG_RECALC` "
        "ya no aparece. Sin escape hatch, debugging en producción "
        "requiere redeploy."
    )


def test_p3_shopping_2_anchor_present(plans_src: str):
    """Anchor `[P3-SHOPPING-2 · 2026-05-14]` documenta la razón."""
    fn_idx = plans_src.find("def api_recalculate_shopping_list")
    body = plans_src[fn_idx:fn_idx + 30000]
    assert "P3-SHOPPING-2" in body, (
        "P3-SHOPPING-2 regresión: anchor desapareció. Restaurar el "
        "comment que documenta por qué se gatea el fingerprint."
    )


# ===========================================================================
# P3-SHOPPING-3: NO-OP. Confirmación de que esbuild ya dropea los niveles.
# ===========================================================================

def test_p3_shopping_3_vite_config_drops_console_levels(vite_src: str):
    """`vite.config.js` debe seguir teniendo `pure: ['console.log',
    'console.warn', 'console.debug', 'console.info']` en mode=production.
    Anchor P3-FRONTEND-1 (closure 2026-05-12). Este P3-SHOPPING-3 es
    NO-OP — solo regression guard.
    """
    # Pattern: `pure:` con los 4 niveles canónicos.
    # Tolerante a orden y whitespace.
    assert re.search(r"pure\s*:\s*\[", vite_src), (
        "P3-SHOPPING-3 regresión: `vite.config.js` ya no tiene config "
        "`pure: [...]`. P3-FRONTEND-1 cerraba este gap; sin la config, "
        "console.* sobrevive al bundle prod (ruido + leak menor)."
    )
    for level in ('console.log', 'console.warn', 'console.debug', 'console.info'):
        assert f"'{level}'" in vite_src or f'"{level}"' in vite_src, (
            f"P3-SHOPPING-3 regresión: nivel `{level}` ya no está en el "
            f"array `pure`. Si removiste uno intencionalmente, validar "
            f"que sea con cobertura mantenible — y eliminar el item de "
            f"la lista esperada en este test."
        )


def test_p3_shopping_3_drop_applies_in_production_mode_only(vite_src: str):
    """El drop solo aplica con `mode === 'production'`. En dev/test los
    logs se preservan para debug interactivo + Vitest specs.
    """
    assert re.search(
        r"mode\s*===\s*['\"]production['\"]",
        vite_src,
    ), (
        "P3-SHOPPING-3 regresión: el gate `mode === 'production'` "
        "desapareció — el drop puede estar aplicándose en dev/test, "
        "rompiendo debug interactivo + Vitest specs que inspeccionan "
        "console output."
    )


# ===========================================================================
# P3-SHOPPING-4: trackEvent success + failed
# ===========================================================================

def test_p3_shopping_4_emits_download_success_event(dash_src: str):
    """Después de `html2pdf().save()` exitoso, se emite
    `trackEvent('pdf_download_success', {...})`.
    """
    body = _extract_handler_block(dash_src)
    assert re.search(
        r"trackEvent\(\s*['\"]pdf_download_success['\"]",
        body,
    ), (
        "P3-SHOPPING-4 regresión: no se emite `pdf_download_success` "
        "tras la descarga exitosa. Sin este evento, el operador no "
        "puede distinguir 'feature no usado' de 'feature roto' — ambos "
        "producen 0 success events."
    )


def test_p3_shopping_4_emits_download_failed_event(dash_src: str):
    """En el `catch (error)` se emite `trackEvent('pdf_download_failed',
    {...})` con error_name + error_message.
    """
    body = _extract_handler_block(dash_src)
    assert re.search(
        r"trackEvent\(\s*['\"]pdf_download_failed['\"]",
        body,
    ), (
        "P3-SHOPPING-4 regresión: no se emite `pdf_download_failed` "
        "en el catch. Sin este evento, fallos del render html2pdf "
        "pasan invisibles a analytics."
    )
    # Verificar que se incluye error_name o error_message.
    failed_idx = body.find("'pdf_download_failed'")
    if failed_idx < 0:
        failed_idx = body.find('"pdf_download_failed"')
    assert failed_idx > 0
    window = body[failed_idx:failed_idx + 800]
    assert "error_name" in window or "error_message" in window, (
        "P3-SHOPPING-4 regresión: el payload de `pdf_download_failed` "
        "ya no incluye `error_name` ni `error_message`. Sin esa info, "
        "el evento es contable pero no diagnosticable."
    )


def test_p3_shopping_4_success_event_includes_dimensions(dash_src: str):
    """El evento de éxito incluye al menos 3 dimensiones operacionales
    para hacer drill-down (`total_items`, `duration`, `multi_page`, etc.)."""
    body = _extract_handler_block(dash_src)
    success_idx = body.find("'pdf_download_success'")
    if success_idx < 0:
        success_idx = body.find('"pdf_download_success"')
    assert success_idx > 0
    window = body[success_idx:success_idx + 1200]
    dimensions = ['total_items', 'duration', 'plan_id', 'fresh_inventory_stale', 'is_plan_expired', 'multi_page', 'density']
    present = sum(1 for d in dimensions if d in window)
    assert present >= 4, (
        f"P3-SHOPPING-4 regresión: solo {present} dimensiones presentes "
        f"en el payload de `pdf_download_success` (esperaba ≥4). "
        f"Sin dimensiones operacionales, el evento es contable pero "
        f"no permite drill-down (e.g. ¿multi_page=true falla más?)."
    )


def test_p3_shopping_4_events_wrapped_in_try_catch(dash_src: str):
    """trackEvent calls están en try/catch best-effort — un fallo del
    analytics SDK no debe abortar el handler.
    """
    body = _extract_handler_block(dash_src)
    for event in ("'pdf_download_success'", "'pdf_download_failed'"):
        idx = body.find(event)
        if idx < 0:
            idx = body.find(event.replace("'", '"'))
        assert idx > 0
        window_back = body[max(0, idx - 400):idx]
        assert "try {" in window_back, (
            f"P3-SHOPPING-4 regresión: el call a trackEvent({event}) "
            f"ya no está envuelto en `try {{`. Un fallo del SDK "
            f"abortaría el handler."
        )


def test_p3_shopping_4_anchor_present(dash_src: str):
    """Anchor `[P3-SHOPPING-4 · 2026-05-14]` documenta la razón."""
    body = _extract_handler_block(dash_src)
    assert "P3-SHOPPING-4" in body, (
        "P3-SHOPPING-4 regresión: anchor desapareció. Restaurar el "
        "comment que documenta por qué se emiten los eventos."
    )
