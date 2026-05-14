"""[P2-RECALC-GROCERY-DURATION-ENUM · 2026-05-14] Validación enum de
`groceryDuration` en `POST /api/plans/recalculate-shopping-list`.

Motivación (audit 2026-05-14):
    Pre-fix, el handler `api_recalculate_shopping_list` aceptaba
    `data.get("groceryDuration", "weekly")` sin validar el valor. Un
    POST con `groceryDuration="forever"` o un typo (`"weely"`,
    `"month"`) caía silenciosamente al else-weekly del branch
    (línea ~4099 que selecciona `active_list`), pero el valor inválido
    se persistía a `plan_data.calc_grocery_duration` (línea ~4179).
    Un cliente downstream que leyera la key persistida observaría un
    valor nunca generado por el flujo legítimo del frontend — abriendo
    rama de UX incoherente (banner "ciclo desconocido", export PDF
    con label vacío, etc).

Fix:
    Clamp del enum dentro del handler tras leer del body:

        _ALLOWED_GROCERY_DURATIONS = ("weekly", "biweekly", "monthly")
        grocery_duration = data.get("groceryDuration", "weekly")
        if grocery_duration not in _ALLOWED_GROCERY_DURATIONS:
            logger.warning(...)
            grocery_duration = "weekly"

    Defense-in-depth análogo al clamp `_max_household` añadido por
    P3-PDF-POLISH-4-B-RECALC. Normaliza al default sin abortar el
    flujo + emite log warning para captar caller patológico.

Drift detection (parser-based):
    1. El handler define la tupla `_ALLOWED_GROCERY_DURATIONS` con
       exactamente los 3 valores válidos `("weekly", "biweekly",
       "monthly")`.
    2. El handler chequea `if grocery_duration not in _ALLOWED_*` Y
       reasigna a `"weekly"`.
    3. El handler emite `logger.warning("[P2-RECALC-GROCERY-DURATION-ENUM]...")`
       para captar callers patológicos en logs prod.

Whitelist:
    No prevista. Si en el futuro se añade un 4to valor enum (e.g.,
    `"monthly_extended"`), actualizar la tupla `_ALLOWED_*` + este test
    + cualquier consumer downstream que mapee `groceryDuration` a
    días/semanas.

Tooltip-anchor: P2-RECALC-GROCERY-DURATION-ENUM-START | gap audit 2026-05-14
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Aísla `def fn_name(...)` hasta el siguiente top-level `def `,
    `@router.` o `@app.` decorator."""
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    assert m is not None, f"Función `{fn_name}` no encontrada en plans.py"
    start = m.start()
    next_marker = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_marker.start()) if next_marker else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def recalc_body(plans_src: str) -> str:
    return _extract_function_body(plans_src, "api_recalculate_shopping_list")


# ---------------------------------------------------------------------------
# 1. La tupla _ALLOWED_GROCERY_DURATIONS define los 3 valores enum
# ---------------------------------------------------------------------------
def test_allowed_durations_tuple_defined(recalc_body: str):
    """Una tupla local `_ALLOWED_GROCERY_DURATIONS` con exactamente
    `("weekly", "biweekly", "monthly")` debe existir en el handler.

    Drift: si alguien la migra a un set o cambia el orden/contenido,
    revisar consumers downstream (`_build_hybrid_shopping_list`,
    `active_list` branch, frontend que filtra por estos labels).
    """
    pattern = re.compile(
        r"_ALLOWED_GROCERY_DURATIONS\s*=\s*\(\s*"
        r"['\"]weekly['\"]\s*,\s*"
        r"['\"]biweekly['\"]\s*,\s*"
        r"['\"]monthly['\"]\s*,?\s*\)",
    )
    assert pattern.search(recalc_body), (
        "P2-RECALC-GROCERY-DURATION-ENUM regresión: la tupla "
        "`_ALLOWED_GROCERY_DURATIONS = (\"weekly\", \"biweekly\", "
        "\"monthly\")` no está definida en el handler. Sin ella, el "
        "clamp se rompe (NameError) o se reemplaza por una variable "
        "ad-hoc con riesgo de drift. Restaurar la tupla literal."
    )


# ---------------------------------------------------------------------------
# 2. Branch `if grocery_duration not in _ALLOWED_*: ... = "weekly"`
# ---------------------------------------------------------------------------
def test_handler_normalizes_invalid_duration(recalc_body: str):
    """El handler DEBE chequear `if grocery_duration not in _ALLOWED_*`
    Y reasignar `grocery_duration = "weekly"` en la rama true. Sin esto,
    valores inválidos pasan al persist.
    """
    # Match: `if grocery_duration not in _ALLOWED_GROCERY_DURATIONS:` +
    # cualquier código + `grocery_duration = "weekly"` en ≤300 chars.
    pattern = re.compile(
        r"if\s+grocery_duration\s+not\s+in\s+_ALLOWED_GROCERY_DURATIONS\s*:"
        r"[\s\S]{0,400}?"
        r"grocery_duration\s*=\s*['\"]weekly['\"]",
    )
    assert pattern.search(recalc_body), (
        "P2-RECALC-GROCERY-DURATION-ENUM regresión: el handler no "
        "normaliza `groceryDuration` inválido al default `'weekly'`. "
        "Esperado patrón:\n"
        "  `if grocery_duration not in _ALLOWED_GROCERY_DURATIONS:\n"
        "       logger.warning(...)\n"
        "       grocery_duration = 'weekly'`\n"
        "Sin esto, un POST con `groceryDuration=\"forever\"` persiste "
        "ese valor inválido en `plan_data.calc_grocery_duration`."
    )


# ---------------------------------------------------------------------------
# 3. logger.warning con tag P2-RECALC-GROCERY-DURATION-ENUM
# ---------------------------------------------------------------------------
def test_handler_logs_invalid_duration(recalc_body: str):
    """`logger.warning` con el tag `[P2-RECALC-GROCERY-DURATION-ENUM]`
    DEBE acompañar el clamp. Sin esto, SRE no puede correlacionar callers
    patológicos en logs de prod."""
    pattern = re.compile(
        r"logger\.warning\s*\(\s*[\s\S]{0,200}?\[P2-RECALC-GROCERY-DURATION-ENUM\]",
    )
    assert pattern.search(recalc_body), (
        "P2-RECALC-GROCERY-DURATION-ENUM regresión: el clamp no emite "
        "`logger.warning(\"[P2-RECALC-GROCERY-DURATION-ENUM] ...\")`. "
        "Sin este log, no hay forma de detectar en logs prod cuál "
        "frontend/cliente está enviando valores inválidos (typo en "
        "deploy nuevo, mobile app legacy, request adversarial)."
    )


# ---------------------------------------------------------------------------
# 4. El clamp ocurre ANTES de cualquier uso de `grocery_duration`
# ---------------------------------------------------------------------------
def test_clamp_before_usage(recalc_body: str):
    """El clamp DEBE ocurrir ANTES del primer uso de `grocery_duration`
    en lógica de negocio (e.g., `if grocery_duration == "biweekly":` o
    persistencia `calc_grocery_duration`). Si el clamp viene después,
    es no-op.
    """
    clamp_pat = re.compile(
        r"if\s+grocery_duration\s+not\s+in\s+_ALLOWED_GROCERY_DURATIONS",
    )
    # Primer uso post-asignación: tipo `if grocery_duration ==` o
    # `plan_data*["calc_grocery_duration"] = grocery_duration`. Tomamos
    # el segundo uso (el primero es la lectura del body, `data.get(...)`).
    usage_pat = re.compile(
        r"(if\s+grocery_duration\s*==|calc_grocery_duration['\"]\s*\]\s*=\s*grocery_duration)",
    )

    clamp_m = clamp_pat.search(recalc_body)
    assert clamp_m, "Clamp not found (cubre test #2)"

    usage_matches = list(usage_pat.finditer(recalc_body))
    # Filtrar usages que estén DESPUÉS del clamp.
    after_clamp = [m for m in usage_matches if m.start() > clamp_m.end()]
    assert after_clamp, (
        "No se encontraron usos de `grocery_duration` después del clamp. "
        "Si la lógica de negocio movió, este test ya no aplica — "
        "actualizar."
    )

    # Defensive: el clamp DEBE estar antes del primer uso después de su
    # propio scope. Garantiza que normalizemos antes de cualquier branch
    # downstream.
    first_usage_after_clamp = after_clamp[0]
    assert clamp_m.start() < first_usage_after_clamp.start(), (
        f"P2-RECALC-GROCERY-DURATION-ENUM regresión: el clamp ocurre "
        f"en offset {clamp_m.start()} pero el primer uso de "
        f"`grocery_duration` está en offset {first_usage_after_clamp.start()} "
        f"— el orden está invertido. Mover el clamp ANTES del primer "
        f"branch que decida basado en `grocery_duration`."
    )


# ---------------------------------------------------------------------------
# 5. Cross-link slug del marker (P2-HIST-AUDIT-14): este SÍ es el marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Filename DEBE contener `p2_recalc_grocery_duration_enum` para
    matchear el cross-link `test_p2_hist_audit_14_marker_test_link`
    cuando `_LAST_KNOWN_PFIX = P2-RECALC-GROCERY-DURATION-ENUM · 2026-05-14`.
    """
    expected_slug = "p2_recalc_grocery_duration_enum"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        f"El nombre de este archivo debe contener `{expected_slug}` para "
        f"que el cross-link `test_p2_hist_audit_14_marker_test_link` lo "
        f"matchee con el marker activo del bundle "
        f"`P2-RECALC-GROCERY-DURATION-ENUM · YYYY-MM-DD`."
    )
