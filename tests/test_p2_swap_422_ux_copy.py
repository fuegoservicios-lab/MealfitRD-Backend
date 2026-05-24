"""[P2-SWAP-422-UX-COPY · 2026-05-22] Test parser-based del fix UX que
surface el copy honesto del backend cuando el swap-meal rechaza por
strict-pantry sin inventario disponible.

Pre-fix: ``AssessmentContext.jsx::handleSwapMeal`` hacía
``if (!response.ok) throw new Error("Error conectando con la IA")``,
descartando el body del 422 que el backend emite con
``{detail:{code:"swap_strict_pantry_no_inventory", message:"Tu nevera está vacía..."}}``.
El catch downstream caía a ``getAlternativeMeal`` (fallback local genérico)
sin avisar al usuario por qué el LLM rechazó. UX degradado silencioso.

Post-fix: el handler parsea el JSON del 422, propaga ``error.status``,
``error.code`` y ``error.detailMessage``; el catch detecta el código
``swap_strict_pantry_no_inventory`` y dispara ``toast.error`` con el
copy específico + preserva el plato actual (NO degrada a fallback).

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: el slug del
marker ``P2-SWAP-422-UX-COPY`` ↔ filename ``test_p2_swap_422_ux_copy.py``.
"""
import pathlib
import re

FRONTEND_ROOT = pathlib.Path(__file__).parent.parent.parent / "frontend"
CONTEXT_JSX = (FRONTEND_ROOT / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
APP_PY = (pathlib.Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Handler parsea el 422 y propaga estructura
# ---------------------------------------------------------------------------

def test_handler_parses_error_body_when_response_not_ok():
    """El handler debe parsear ``await response.json()`` cuando ``!response.ok``
    para extraer el ``detail.message`` del 422 backend.

    Pre-fix: ``throw new Error("Error conectando con la IA")`` descartaba el body.
    """
    # El bloque del fix debe contener un parse del body
    assert "errorBody = await response.json()" in CONTEXT_JSX, (
        "El handler de !response.ok debe parsear el body para extraer "
        "el detail.message del 422 strict-pantry. Si lo descartas, el "
        "usuario vuelve a ver 'Error conectando con la IA' genérico."
    )
    # Defensivo: catch silencioso si body no es JSON (network truncado, etc.)
    assert re.search(
        r"try\s*\{\s*errorBody\s*=\s*await\s+response\.json\(\)\s*;?\s*\}\s*catch",
        CONTEXT_JSX,
    ), (
        "El parse de errorBody debe estar dentro de un try/catch — algunos "
        "errores del backend (network drop, 502 con HTML, etc.) llegan con "
        "body no-JSON y romperían el handler."
    )


def test_handler_extracts_code_and_message_from_detail():
    """El handler debe extraer ``detail.code`` y ``detail.message`` (shape
    canónico del backend), con fallback al ``detail`` plano si es string."""
    assert "errorBody?.detail?.code" in CONTEXT_JSX, (
        "Falta lectura de errorBody.detail.code — necesario para que el "
        "catch downstream pueda discriminar 'swap_strict_pantry_no_inventory' "
        "de otros 422 sin matching brittle del message."
    )
    assert "errorBody?.detail?.message" in CONTEXT_JSX, (
        "Falta lectura de errorBody.detail.message — el copy honesto "
        "'Tu nevera está vacía...' vive aquí."
    )


def test_handler_attaches_status_code_detail_to_error():
    """El Error throwed debe llevar ``.status``, ``.code`` y ``.detailMessage``
    para que el catch downstream pueda discriminar 422 vs otros."""
    # Buscar la asignación de .status sobre el err
    for prop in ("err.status", "err.code", "err.detailMessage"):
        assert prop in CONTEXT_JSX, (
            f"Falta `{prop}` en el handler — sin esa propiedad el catch "
            f"downstream no puede distinguir 422 strict-pantry de otros errores."
        )


# ---------------------------------------------------------------------------
# Section B — Catch downstream discrimina 422 strict-pantry sin degradar
# ---------------------------------------------------------------------------

def test_catch_discriminates_swap_strict_pantry_no_inventory():
    """El catch debe chequear ``error.status === 422`` Y
    ``error.code === 'swap_strict_pantry_no_inventory'`` (matching exacto
    sobre el code estable, NO sobre el message que podría cambiar)."""
    assert "error?.status === 422" in CONTEXT_JSX, (
        "Catch debe filtrar por error.status === 422 — sin esto un 500/503 "
        "con detail.code falsificado pasaría como strict-pantry no-inventory."
    )
    assert "swap_strict_pantry_no_inventory" in CONTEXT_JSX, (
        "Catch debe matchear contra el code canónico "
        "'swap_strict_pantry_no_inventory' (definido en routers/plans.py:3724). "
        "Si cambias el code en backend, sincroniza este branch."
    )


def test_catch_shows_toast_error_with_user_copy():
    """En el branch 422 strict-pantry el catch debe disparar ``toast.error``
    con el ``detailMessage`` propagado por el handler."""
    # El branch específico debe invocar toast.error con detailMessage
    # Extraemos las 6 líneas que siguen al `if (error?.status === 422 ...) {`
    # (regex de balanceo de braces no es confiable con objetos inline en JSX).
    m = re.search(
        r"if\s*\(\s*error\?\.\s*status\s*===\s*422[^\n]*\{\s*\n((?:[^\n]*\n){1,8})\s*\}",
        CONTEXT_JSX,
    )
    assert m, (
        "No se encontró el bloque `if (error?.status === 422 && ...)` "
        "esperado en el catch."
    )
    branch_body = m.group(0)
    assert "toast.error" in branch_body, (
        "El branch 422 debe invocar toast.error — sin el toast el usuario "
        "ve el plato preservado pero sin entender por qué el LLM rechazó."
    )
    assert "detailMessage" in branch_body, (
        "El toast debe consumir error.detailMessage (copy del backend), "
        "NO un string hardcoded — el copy puede cambiar y debe ser SSOT en backend."
    )


def test_catch_returns_current_name_without_fallback():
    """En el branch 422 strict-pantry NO debe caerse a ``getAlternativeMeal``
    (fallback local genérico) — el contrato es: usuario eligió strict-pantry,
    no hay nevera/lista, mantén el plato como estaba y deja que el usuario
    decida (agregar a nevera o cambiar reason)."""
    # Extraemos las 6 líneas que siguen al `if (error?.status === 422 ...) {`
    # (regex de balanceo de braces no es confiable con objetos inline en JSX).
    m = re.search(
        r"if\s*\(\s*error\?\.\s*status\s*===\s*422[^\n]*\{\s*\n((?:[^\n]*\n){1,8})\s*\}",
        CONTEXT_JSX,
    )
    assert m, "No se encontró el bloque 422 en el catch."
    branch_body = m.group(0)
    assert "getAlternativeMeal" not in branch_body, (
        "El branch 422 strict-pantry NO debe invocar getAlternativeMeal — "
        "produciría un plato genérico ignorando la razón estricta del usuario. "
        "Debe `return currentName` (preservar plato actual)."
    )
    assert "return currentName" in branch_body, (
        "El branch 422 debe `return currentName` para que el caller no "
        "actualice planData con un fallback no deseado."
    )


# ---------------------------------------------------------------------------
# Section C — Backend SSOT del code/message preservado
# ---------------------------------------------------------------------------

def test_backend_emits_canonical_code_for_strict_pantry_422():
    """Sanity check: el backend sigue emitiendo el code canónico que el
    frontend matchea. Si cambias el code en routers/plans.py, este test
    te recuerda actualizar el frontend al unísono.

    [P3-SWAP-SOFT-FAIL-200 · 2026-05-23] Tras la migración a soft-fail
    200, el code aparece como ``error_code`` en el payload de default O
    como ``code`` en el detail del 422 legacy (gateado por knob
    MEALFIT_SWAP_HARD_FAIL_HTTP_422). Aceptamos ambos patrones para
    que el contrato code↔frontend permanezca anclado en cualquier modo.
    """
    plans_py = (pathlib.Path(__file__).parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")
    # Soft-fail path (default post-P3-SWAP-SOFT-FAIL-200)
    soft_match = '"error_code": "swap_strict_pantry_no_inventory"' in plans_py
    # Hard-fail path (legacy 422 gateado por knob)
    hard_match = '"code": "swap_strict_pantry_no_inventory"' in plans_py
    assert soft_match or hard_match, (
        "El backend debe emitir el code canónico `swap_strict_pantry_no_inventory` "
        "en al menos uno de los dos paths (soft-fail 200 con `error_code` o "
        "hard-fail 422 con `code`). Si lo renombras, sincroniza con "
        "AssessmentContext.jsx (handler P2-SWAP-422-UX-COPY / P3-SWAP-SOFT-FAIL-200)."
    )
    assert "status_code=422" in plans_py, (
        "El path 422 hard-fail debe seguir disponible bajo el knob "
        "MEALFIT_SWAP_HARD_FAIL_HTTP_422=true (rollback compat)."
    )


# ---------------------------------------------------------------------------
# Section D — Marker anchor
# ---------------------------------------------------------------------------
# El cross-link `test_p2_hist_audit_14_marker_test_link` solo exige que
# este archivo exista (slug ↔ filename). El "marker bump al valor de este
# P-fix" se pinneaba inicialmente, pero pin-tests se rompen cada P-fix
# siguiente porque el marker avanza. El contract "marker fresco a nivel
# codebase" lo cubre `test_p3_1_last_known_pfix_freshness` (floor check).
# Las 7 secciones A-C de este archivo anclan el CONTENIDO del fix
# (parser-based, no temporales) — esa es la red de seguridad real.
