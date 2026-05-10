"""[P1-HIST-AUDIT-NEW-1 · 2026-05-10] Read-only y housekeeping del Historial
NO deben gated por `verify_api_quota`.

Bug original (audit 2026-05-10):
    `verify_api_quota` (auth.py:49) bloquea con HTTP 402 cuando
    `get_monthly_api_usage(user_id) >= tier_limit`. La dependencia
    estaba aplicada a TODOS los endpoints de routers/plans.py
    relacionados con el Historial — incluyendo lectura
    (`/history-list`, `/{id}/lessons`, `/{id}/coherence-history`,
    `/{id}/lifetime-lessons`, `/{id}/chunk-metrics`,
    `/{id}/blocked_reasons`, `/{id}/chunk-status`, `/lessons-counts`,
    `/history-status-summary`) y housekeeping no-LLM (`/restore`,
    DELETE `/{id}`, PATCH `/{id}/name`).

    Resultado: usuario tier `gratis` que generó 15 planes este mes
    no podía VER ni MANTENER su historial. Recibía 402
    "Mejora tu plan" al abrir /history. UX regresiva — el contraste
    con el copy "guarda tu progreso" es severo.

Fix:
    Cambiar `Depends(verify_api_quota)` → `Depends(get_verified_user_id)`
    en los 12 endpoints read-only/housekeeping del Historial.
    `get_verified_user_id` solo extrae el `sub` del JWT (auth-only),
    sin tocar `get_monthly_api_usage`. Los endpoints que SÍ consumen
    LLM (analyze, swap-meal, recipe/expand, retry-chunk, regen-degraded,
    regenerate-simplified, restock, consume-inventory, shift-plan,
    analyze/stream) preservan `verify_api_quota`.

Estrategia del test (parser estático, mismo patrón que
`test_p3_b_required_fields_js_parser_added.py`):
    Localizar cada `def <fn>(...)` y inspeccionar la signature
    completa hasta el `):`. Verificar la dependency en `Depends(...)`.

Drift detection bidireccional:
    - Si alguien restaura `verify_api_quota` en un endpoint del
      Historial → falla por elemento en `_HISTORY_ENDPOINTS`.
    - Si alguien quita `verify_api_quota` de un endpoint LLM →
      falla por elemento en `_LLM_ENDPOINTS` (auto-cobertura del
      gate de cuota).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


# Endpoints del Historial que NO consumen LLM. Deben usar get_verified_user_id.
# Tupla: (nombre_funcion, ruta_aproximada_para_diagnostico).
_HISTORY_ENDPOINTS = [
    ("api_chunk_status", "GET /{plan_id}/chunk-status"),
    ("api_blocked_reasons", "GET /{plan_id}/blocked_reasons"),
    ("api_restore_plan", "POST /restore"),
    ("api_delete_plan", "DELETE /{plan_id}"),
    ("api_plans_lessons_counts", "GET /lessons-counts"),
    ("api_plans_history_status_summary", "GET /history-status-summary"),
    ("api_plan_lessons_detail", "GET /{plan_id}/lessons"),
    ("api_plan_coherence_history", "GET /{plan_id}/coherence-history"),
    ("api_plan_lifetime_lessons", "GET /{plan_id}/lifetime-lessons"),
    ("api_plan_chunk_metrics", "GET /{plan_id}/chunk-metrics"),
    ("api_rename_plan", "PATCH /{plan_id}/name"),
    ("api_plans_history_list", "GET /history-list"),
]


# Endpoints que SÍ consumen LLM o crean filas que consumen tokens. Deben
# preservar `verify_api_quota`. Lista NO exhaustiva — solo los más
# representativos. Si alguien agrega un endpoint LLM nuevo, no falla aquí
# (este test no censura ausencias), pero si alguien REMUEVE el gate de
# uno listado, sí falla.
_LLM_ENDPOINTS = [
    ("api_shift_plan", "POST /shift-plan"),
    ("api_analyze", "POST /analyze"),
    ("api_expand_recipe", "POST /recipe/expand"),
    ("api_swap_meal", "POST /swap-meal"),
    ("api_retry_chunk", "POST /{plan_id}/retry-chunk/{chunk_id}"),
    ("api_regen_degraded_chunks", "POST /{plan_id}/regen-degraded"),
]


def _extract_signature(src: str, fn_name: str) -> str:
    """Devuelve el bloque desde `def <fn>(` hasta el cierre `):` con
    paréntesis balanceado. Tolera defaults multi-línea y `Body(...)`
    con paréntesis anidados.
    """
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — el endpoint "
            f"fue renombrado/eliminado y este test necesita actualizarse."
        )
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    if depth != 0:
        raise AssertionError(
            f"Paréntesis no balanceado en signature de `{fn_name}` — "
            f"posible corrupción del fuente."
        )
    return src[m.start():i]


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.mark.parametrize(("fn_name", "route"), _HISTORY_ENDPOINTS)
def test_history_endpoints_use_get_verified_user_id(
    plans_src: str, fn_name: str, route: str
):
    """Los 12 endpoints read-only/housekeeping del Historial deben usar
    `Depends(get_verified_user_id)` — NO `Depends(verify_api_quota)`.
    """
    sig = _extract_signature(plans_src, fn_name)
    assert "Depends(get_verified_user_id)" in sig, (
        f"Endpoint `{fn_name}` ({route}) NO usa "
        f"`Depends(get_verified_user_id)`. Bug P1-HIST-AUDIT-NEW-1: "
        f"un usuario tier gratis al cap de cuota recibirá HTTP 402 al "
        f"acceder a su Historial. Restaurar la dependencia correcta."
    )
    assert "Depends(verify_api_quota)" not in sig, (
        f"Endpoint `{fn_name}` ({route}) tiene `Depends(verify_api_quota)` "
        f"— regresión de P1-HIST-AUDIT-NEW-1. El gate de cuota bloquea "
        f"el Historial cuando el usuario está al cap de su tier; los "
        f"endpoints read-only/housekeeping no consumen LLM y NO deben "
        f"gated por créditos."
    )


@pytest.mark.parametrize(("fn_name", "route"), _LLM_ENDPOINTS)
def test_llm_endpoints_preserve_verify_api_quota(
    plans_src: str, fn_name: str, route: str
):
    """Los endpoints que SÍ consumen LLM deben preservar
    `Depends(verify_api_quota)` para que el paywall funcione. Sin esto,
    un usuario al cap del tier gratis podría seguir generando planes
    sin pagar — P0 monetización.
    """
    sig = _extract_signature(plans_src, fn_name)
    assert "Depends(verify_api_quota)" in sig, (
        f"Endpoint `{fn_name}` ({route}) consume LLM y NO tiene "
        f"`Depends(verify_api_quota)`. Riesgo: bypass del paywall — "
        f"usuario al cap puede seguir generando sin pagar."
    )


def test_marker_anchor_present(plans_src: str):
    """El marker textual `_LAST_KNOWN_PFIX` (app.py) debe correlacionar
    con la fecha del fix. Aquí solo validamos que el anchor del fix
    aparezca en algún sitio del fuente — el test
    `test_p2_hist_audit_14_marker_test_link.py` cubre el cross-link
    completo (slug del marker ↔ archivo `tests/test_<slug>*.py`).
    """
    # El nombre de este archivo es el anchor. Si renombramos, el
    # cross-link de marker se rompería — captura del slug aquí para
    # documentar la convención.
    expected_slug_in_filename = "p1_hist_audit_new_1_quota_gate_history"
    assert expected_slug_in_filename in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo de test debe contener el slug del "
        "P-fix para que `test_p2_hist_audit_14_marker_test_link` lo "
        "matchee con el marker `_LAST_KNOWN_PFIX`."
    )
