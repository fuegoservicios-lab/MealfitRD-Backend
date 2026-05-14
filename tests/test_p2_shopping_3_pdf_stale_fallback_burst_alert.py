"""[P2-SHOPPING-3 · 2026-05-14] Burst-detector del evento
`pdf_stale_inventory_fallback`.

Ancla 4 piezas del fix:
  1. Endpoint backend `POST /api/plans/telemetry/pdf-stale-fallback`
     en [`routers/plans.py`](backend/routers/plans.py) que persiste el
     evento a `pipeline_metrics` (`node='pdf_stale_inventory_fallback'`).
  2. Cron `_alert_pdf_stale_inventory_fallback_burst` en
     [`cron_tasks.py`](backend/cron_tasks.py) que cuenta filas en la
     ventana lookback y emite/auto-resuelve la alert
     `system_alerts.pdf_stale_inventory_fallback_burst`.
  3. Registración del cron en `register_plan_chunk_scheduler` con knobs
     `MEALFIT_PDF_STALE_FALLBACK_ALERT_*`.
  4. Frontend callsite fire-and-forget en
     [`frontend/src/pages/Dashboard.jsx::handleDownloadShoppingList`](frontend/src/pages/Dashboard.jsx)
     dentro del path `_freshFetchResult.stale`.

Bug pre-fix (audit 2026-05-13):
    `trackEvent('pdf_stale_inventory_fallback', ...)` solo enviaba a
    canales externos (Sentry/PostHog/GA/GTM). El backend NO observaba
    frecuencia → un blip prolongado de Supabase manteniendo a la flota
    en stale fallback pasaba sin alert hasta que alguien miraba Sentry
    manualmente. Operador no podía configurar paging automático sin
    pagar Sentry alerting features.

Tooltip-anchor: P2-SHOPPING-3.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_CRON_PY = _REPO_ROOT / "backend" / "cron_tasks.py"
_DASH_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dash_src() -> str:
    return _DASH_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def claude_md() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Endpoint backend
# ---------------------------------------------------------------------------

def test_endpoint_route_decorator_present(plans_src: str):
    """`@router.post("/telemetry/pdf-stale-fallback")` definido."""
    assert re.search(
        r'@router\.post\(\s*["\']/telemetry/pdf-stale-fallback["\']\s*\)',
        plans_src,
    ), (
        "P2-SHOPPING-3 regresión: el endpoint POST /telemetry/pdf-stale-fallback "
        "ya no está registrado. Sin él, el frontend no tiene sink backend para "
        "el evento `pdf_stale_inventory_fallback` y el cron leería 0 filas."
    )


def test_endpoint_uses_verified_user_id_dependency(plans_src: str):
    """Endpoint exige autenticación via `get_verified_user_id` — no
    `verify_api_quota` (paywall absurdo para telemetría).
    """
    decorator_idx = plans_src.find('@router.post("/telemetry/pdf-stale-fallback")')
    assert decorator_idx > 0
    after = plans_src[decorator_idx:decorator_idx + 4000]
    assert "verified_user_id" in after, (
        "P2-SHOPPING-3 regresión: el endpoint ya no recibe `verified_user_id`. "
        "Sin auth, un atacante puede flood-ear pipeline_metrics y disparar la "
        "alert artificialmente."
    )
    assert "Depends(get_verified_user_id)" in after, (
        "P2-SHOPPING-3 regresión: la dependency `Depends(get_verified_user_id)` "
        "fue removida. Restaurar."
    )
    # `Depends(verify_api_quota)` NO debe aparecer en la signature — pattern
    # más específico que `verify_api_quota` literal (que el docstring
    # menciona para explicar por qué NO se usa).
    handler_end = after.find("\n@router.")
    handler_body = after[:handler_end] if handler_end > 0 else after
    assert "Depends(verify_api_quota)" not in handler_body, (
        "P2-SHOPPING-3 regresión: el endpoint ahora exige `Depends(verify_api_quota)`. "
        "Telemetría no debe consumir paywall — usuarios free que toquen el "
        "fallback no podrían reportar y el cron quedaría ciego para ellos."
    )


def test_endpoint_inserts_into_pipeline_metrics_with_correct_node(plans_src: str):
    """El INSERT persiste `node='pdf_stale_inventory_fallback'`."""
    decorator_idx = plans_src.find('@router.post("/telemetry/pdf-stale-fallback")')
    after = plans_src[decorator_idx:decorator_idx + 3000]
    assert "INSERT INTO pipeline_metrics" in after, (
        "P2-SHOPPING-3 regresión: el handler ya no INSERTa en `pipeline_metrics`. "
        "Sin esa fila, el cron leería 0 y nunca alertaría."
    )
    assert "pdf_stale_inventory_fallback" in after, (
        "P2-SHOPPING-3 regresión: el handler no usa `node='pdf_stale_inventory_fallback'`. "
        "Mismo string esperado por el SELECT del cron — si difieren, los datos "
        "quedan huérfanos."
    )


def test_endpoint_never_raises_on_telemetry_failure(plans_src: str):
    """Best-effort: si el INSERT falla, retornar `{success: False}` en lugar
    de 5xx — un error 500 haría que el frontend muestre toast de error al
    usuario que sí ya descargó el PDF correctamente.
    """
    decorator_idx = plans_src.find('@router.post("/telemetry/pdf-stale-fallback")')
    after = plans_src[decorator_idx:decorator_idx + 4000]
    handler_end = after.find("\n@router.")
    handler_body = after[:handler_end] if handler_end > 0 else after
    # Patrón: except Exception ... return {"success": False, ...}
    assert re.search(r"except\s+Exception", handler_body), (
        "P2-SHOPPING-3 regresión: el handler ya no captura `Exception`. Un "
        "fallo del INSERT propagaría 5xx al cliente."
    )
    # `return {"success": False, ...}` confirma que el path de fallo NO
    # levanta — retorna OK al cliente con flag interno.
    assert re.search(r'return\s*\{\s*["\']success["\']\s*:\s*False', handler_body), (
        "P2-SHOPPING-3 regresión: el except path ya no retorna "
        "`{success: False, ...}`. Si re-raisea, el frontend mostraría "
        "toast de error al usuario que SÍ descargó el PDF — UX peor que "
        "telemetría perdida."
    )


# ---------------------------------------------------------------------------
# 2. Cron _alert_pdf_stale_inventory_fallback_burst
# ---------------------------------------------------------------------------

def test_cron_function_defined(cron_src: str):
    """`_alert_pdf_stale_inventory_fallback_burst` definido en cron_tasks.py."""
    assert "def _alert_pdf_stale_inventory_fallback_burst" in cron_src, (
        "P2-SHOPPING-3 regresión: la función del cron ya no está definida. "
        "Sin ella, no hay observador automático de la métrica."
    )


def test_cron_reads_threshold_and_lookback_knobs(cron_src: str):
    """El cron lee `MEALFIT_PDF_STALE_FALLBACK_ALERT_THRESHOLD` (count) y
    `MEALFIT_PDF_STALE_FALLBACK_ALERT_LOOKBACK_MIN` (ventana de tiempo).
    """
    fn_start = cron_src.find("def _alert_pdf_stale_inventory_fallback_burst")
    assert fn_start > 0
    body = cron_src[fn_start:fn_start + 4000]
    assert "MEALFIT_PDF_STALE_FALLBACK_ALERT_THRESHOLD" in body, (
        "P2-SHOPPING-3 regresión: knob `MEALFIT_PDF_STALE_FALLBACK_ALERT_THRESHOLD` "
        "ya no se lee. Sin él el umbral es hardcoded, imposible ajustar sin "
        "redeploy."
    )
    assert "MEALFIT_PDF_STALE_FALLBACK_ALERT_LOOKBACK_MIN" in body, (
        "P2-SHOPPING-3 regresión: knob `MEALFIT_PDF_STALE_FALLBACK_ALERT_LOOKBACK_MIN` "
        "ya no se lee. Ventana de tiempo hardcoded."
    )


def test_cron_queries_pipeline_metrics_for_correct_node(cron_src: str):
    """El SELECT filtra por `node = 'pdf_stale_inventory_fallback'` —
    mismo string que el endpoint INSERT.
    """
    fn_start = cron_src.find("def _alert_pdf_stale_inventory_fallback_burst")
    body = cron_src[fn_start:fn_start + 4000]
    assert "FROM pipeline_metrics" in body, (
        "P2-SHOPPING-3 regresión: el cron ya no consulta `pipeline_metrics`. "
        "¿Cambió la tabla de sink? Actualizar el test."
    )
    assert "pdf_stale_inventory_fallback" in body, (
        "P2-SHOPPING-3 regresión: el filtro `node = 'pdf_stale_inventory_fallback'` "
        "desapareció — el cron leería todos los nodes y dispararía alerts "
        "espurias por cualquier métrica."
    )


def test_cron_emits_correct_alert_key(cron_src: str):
    """La alert insertada usa `alert_key = 'pdf_stale_inventory_fallback_burst'`
    con `alert_type = 'shopping'` y `severity = 'warning'`.
    """
    fn_start = cron_src.find("def _alert_pdf_stale_inventory_fallback_burst")
    body = cron_src[fn_start:fn_start + 4000]
    assert "pdf_stale_inventory_fallback_burst" in body, (
        "P2-SHOPPING-3 regresión: la alert_key cambió. El SOP y la tabla "
        "system_alerts en CLAUDE.md quedarían desincronizados."
    )
    assert "'shopping'" in body, (
        "P2-SHOPPING-3 regresión: el alert_type ya no es `'shopping'`. La "
        "tabla CLAUDE.md espera esa categoría."
    )
    assert "'warning'" in body, (
        "P2-SHOPPING-3 regresión: la severity ya no es `'warning'`. Bursts "
        "de stale fallback son recoverable — `critical` sería falsa "
        "señal de incidente."
    )


def test_cron_auto_resolves_when_count_drops(cron_src: str):
    """Modelo Auto (explicit): si el count cae bajo umbral, hace
    `UPDATE system_alerts SET resolved_at = NOW() WHERE alert_key =
    'pdf_stale_inventory_fallback_burst' AND resolved_at IS NULL`.
    """
    fn_start = cron_src.find("def _alert_pdf_stale_inventory_fallback_burst")
    body = cron_src[fn_start:fn_start + 4000]
    # Pattern: UPDATE system_alerts SET resolved_at = NOW() WHERE alert_key = '...'
    auto_resolve_re = re.compile(
        r"UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*NOW\(\)\s+"
        r"WHERE\s+alert_key\s*=\s*['\"]pdf_stale_inventory_fallback_burst['\"]",
        re.IGNORECASE | re.DOTALL,
    )
    assert auto_resolve_re.search(body), (
        "P2-SHOPPING-3 regresión: el bloque auto-resolve desapareció. "
        "Sin él, la alert quedaría open eternamente aunque el problema "
        "se haya resuelto — operador tiene que cerrar a mano."
    )


# ---------------------------------------------------------------------------
# 3. Registración en register_plan_chunk_scheduler
# ---------------------------------------------------------------------------

def test_cron_registered_in_scheduler(cron_src: str):
    """El cron está registrado en `register_plan_chunk_scheduler` con el
    knob `MEALFIT_PDF_STALE_FALLBACK_ALERT_INTERVAL_MIN`.
    """
    scheduler_start = cron_src.find("def register_plan_chunk_scheduler")
    assert scheduler_start > 0
    scheduler_body = cron_src[scheduler_start:]
    assert "alert_pdf_stale_inventory_fallback_burst" in scheduler_body, (
        "P2-SHOPPING-3 regresión: el cron no está registrado en "
        "`register_plan_chunk_scheduler`. Sin esa registración, APScheduler "
        "no lo ejecuta — el observador nunca corre."
    )
    assert "MEALFIT_PDF_STALE_FALLBACK_ALERT_INTERVAL_MIN" in scheduler_body, (
        "P2-SHOPPING-3 regresión: el knob de interval ya no se lee en el "
        "scheduler. Sin él, el cron corre con periodo hardcoded."
    )


# ---------------------------------------------------------------------------
# 4. Frontend callsite (fire-and-forget)
# ---------------------------------------------------------------------------

def test_frontend_posts_to_telemetry_endpoint(dash_src: str):
    """`handleDownloadShoppingList` POST best-effort al endpoint en el path
    `_freshFetchResult.stale`.
    """
    start = dash_src.find("const handleDownloadShoppingList")
    assert start > 0
    after = dash_src[start + 50:]
    next_match = re.search(r"\n    const handle[A-Z]", after)
    end = (start + 50 + next_match.start()) if next_match else (start + 50 + 6000)
    body = dash_src[start:end]

    # Pattern: POST a /api/plans/telemetry/pdf-stale-fallback dentro del handler.
    assert "/api/plans/telemetry/pdf-stale-fallback" in body, (
        "P2-SHOPPING-3 regresión: el handler PDF ya NO postea a "
        "`/api/plans/telemetry/pdf-stale-fallback`. Sin la llamada, el cron "
        "leería 0 filas en `pipeline_metrics` y nunca alertaría aunque "
        "TODA la flota esté en stale fallback."
    )


def test_frontend_call_is_fire_and_forget(dash_src: str):
    """El POST tiene `.catch(...)` para silent-fail (best-effort) — un
    fallo del telemetry endpoint NO debe abortar la descarga del PDF.
    """
    start = dash_src.find("const handleDownloadShoppingList")
    after = dash_src[start + 50:]
    next_match = re.search(r"\n    const handle[A-Z]", after)
    end = (start + 50 + next_match.start()) if next_match else (start + 50 + 6000)
    body = dash_src[start:end]

    # Buscar la ventana cercana al POST y verificar `.catch(`.
    post_idx = body.find("/api/plans/telemetry/pdf-stale-fallback")
    assert post_idx > 0
    window = body[post_idx:post_idx + 800]
    assert ".catch(" in window, (
        "P2-SHOPPING-3 regresión: el POST a `/telemetry/pdf-stale-fallback` "
        "ya no tiene `.catch(...)`. Sin él, un fallo del endpoint propagaría "
        "una unhandled promise rejection y posiblemente abortaría el render "
        "del PDF."
    )


def test_frontend_anchor_marker_present(dash_src: str):
    """Anchor `[P2-SHOPPING-3 · 2026-05-14]` en comment del handler."""
    assert "P2-SHOPPING-3" in dash_src, (
        "P2-SHOPPING-3 regresión: marker `[P2-SHOPPING-3 · ...]` ausente "
        "en Dashboard.jsx. Restaurar el comment que documenta por qué se "
        "POSTea al sink backend además de trackEvent."
    )


# ---------------------------------------------------------------------------
# 5. CLAUDE.md system_alerts row documentado
# ---------------------------------------------------------------------------

def test_alert_documented_in_claude_md(claude_md: str):
    """La tabla `system_alerts` de CLAUDE.md tiene una row para
    `pdf_stale_inventory_fallback_burst`. Test `test_p2_audit_4_alert_keys_documented`
    bloquea si falta.
    """
    assert "pdf_stale_inventory_fallback_burst" in claude_md, (
        "P2-SHOPPING-3 regresión: la alert_key no está documentada en la "
        "tabla `system_alerts` de CLAUDE.md. Sin doc, futuros operadores "
        "no sabrán qué significa cuando aparezca en el dashboard. "
        "test_p2_audit_4_alert_keys_documented también fallará."
    )
