"""[P3-SWAP-SOFT-FAIL-200 · 2026-05-23] Los HTTP 422 emitidos por
``/api/plans/swap-meal`` (``swap_strict_pantry_no_inventory`` y
``swap_llm_retries_exhausted``) generaban ruido rojo en DevTools del
browser cada vez que el flow caía en el fallback path, aunque el
comportamiento real era correcto (toast + plato original preservado).

User feedback (screenshot 2026-05-23): el error rojo en DevTools parecía
crash del sistema, cuando en realidad era el manejo correcto del fallback.

## Fix: soft-fail HTTP 200

Backend retorna ``200 OK`` con payload:
```json
{
  "swap_failed": true,
  "error_code": "swap_llm_retries_exhausted" | "swap_strict_pantry_no_inventory",
  "error_message": "<copy honesto al user>"
}
```

Frontend checkea ``newMealData.swap_failed === true`` ANTES de procesar
el response como plato exitoso → dispara toast + preserva plato original
(mismo UX que el 422 legacy). DevTools queda limpio.

## Knob de rollback

``MEALFIT_SWAP_HARD_FAIL_HTTP_422=true`` revierte al comportamiento 422
para integradores externos que dependieran del status 4xx para alertas.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_swap_soft_fail_200`` ↔ filename ``test_p3_swap_soft_fail_200.py``.
"""
import pathlib
import re

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"

PLANS_PY = (BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
CONTEXT_JSX = (FRONTEND_ROOT / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Backend retorna 200 (no raise 422) por default
# ---------------------------------------------------------------------------

def test_backend_returns_200_payload_for_strict_pantry_by_default():
    """[PARSER] El branch de ``SWAP_STRICT_PANTRY_NO_INVENTORY`` debe usar
    ``return _payload`` (200) como default, NO ``raise HTTPException(422)``.
    El 422 solo se dispara cuando knob ``MEALFIT_SWAP_HARD_FAIL_HTTP_422=true``."""
    # Match: hay un `return _payload` después del block de SWAP_STRICT_PANTRY
    m = re.search(
        r"if\s+_msg\.startswith\(\s*[\"']SWAP_STRICT_PANTRY_NO_INVENTORY[\"']\s*\).*?return\s+_payload",
        PLANS_PY,
        re.DOTALL,
    )
    assert m, (
        "Falta `return _payload` en el branch de SWAP_STRICT_PANTRY_NO_INVENTORY. "
        "Sin esto, el backend sigue raise-eando 422 por default."
    )


def test_backend_returns_200_payload_for_llm_retries_by_default():
    """[PARSER] Mismo patrón para ``SWAP_LLM_RETRIES_EXHAUSTED``."""
    m = re.search(
        r"if\s+_msg\.startswith\(\s*[\"']SWAP_LLM_RETRIES_EXHAUSTED[\"']\s*\).*?return\s+_payload",
        PLANS_PY,
        re.DOTALL,
    )
    assert m, (
        "Falta `return _payload` en el branch de SWAP_LLM_RETRIES_EXHAUSTED."
    )


def test_payload_includes_swap_failed_flag():
    """[PARSER] El payload debe incluir ``swap_failed: True`` como discriminator
    primario — frontend lo lee ANTES de procesar como plato exitoso."""
    assert '"swap_failed": True' in PLANS_PY, (
        "Falta el flag `swap_failed: True` en el payload. Sin él, el frontend "
        "no puede distinguir failure de success (ambos llegan como 200)."
    )


def test_payload_includes_canonical_error_codes():
    """[PARSER] El payload debe incluir ``error_code`` canónico (mismo que el
    422 legacy usaba en ``detail.code``)."""
    assert '"error_code": "swap_strict_pantry_no_inventory"' in PLANS_PY
    assert '"error_code": "swap_llm_retries_exhausted"' in PLANS_PY


# ---------------------------------------------------------------------------
# Section B — Knob de rollback preserva path legacy
# ---------------------------------------------------------------------------

def test_hard_fail_knob_present():
    """[PARSER] El knob ``MEALFIT_SWAP_HARD_FAIL_HTTP_422`` debe estar
    referenciado — preserva path 422 legacy para integradores externos
    sin redeploy."""
    assert "MEALFIT_SWAP_HARD_FAIL_HTTP_422" in PLANS_PY, (
        "Falta knob `MEALFIT_SWAP_HARD_FAIL_HTTP_422` — sin rollback no "
        "podemos reaccionar si algún cliente externo depende del 422."
    )


def test_hard_fail_knob_default_false():
    """[PARSER] El default del knob debe ser ``"false"`` — soft-fail es el
    comportamiento por default tras este P-fix."""
    m = re.search(
        r'MEALFIT_SWAP_HARD_FAIL_HTTP_422[^\n]*?["\']false["\']',
        PLANS_PY,
    )
    assert m, (
        "Knob `MEALFIT_SWAP_HARD_FAIL_HTTP_422` debe leer con default "
        "'false'. Default 'true' mantiene el ruido rojo en DevTools."
    )


def test_hard_fail_path_preserved_under_knob():
    """[PARSER] El ``raise HTTPException(status_code=422, ...)`` debe seguir
    en el código, gateado por el knob. Sin esto, el rollback no funciona."""
    # El raise debe estar dentro de un `if _hard_fail_422:` branch
    # Match patrón completo
    m = re.search(
        r"if\s+_hard_fail_422\s*:\s*\n\s*raise\s+HTTPException\(\s*status_code\s*=\s*422",
        PLANS_PY,
    )
    assert m, (
        "El `raise HTTPException(status_code=422, ...)` debe estar gateado "
        "por `if _hard_fail_422:`. Sin esto el knob no tiene efecto."
    )


# ---------------------------------------------------------------------------
# Section C — Frontend handler checkea swap_failed flag
# ---------------------------------------------------------------------------

def test_frontend_handler_checks_swap_failed_flag():
    """[PARSER] El handler debe checkear ``newMealData.swap_failed === true``
    ANTES de procesar como plato exitoso."""
    assert "newMealData?.swap_failed === true" in CONTEXT_JSX, (
        "Frontend handler debe checar `newMealData.swap_failed === true` "
        "para detectar soft-fail. Sin esto, el dict de error se trata como "
        "plato exitoso y aparece basura en el dashboard."
    )


def test_frontend_handler_discriminates_error_codes():
    """[PARSER] El handler debe discriminar por ``error_code`` para mostrar
    el toast title correcto (Nevera vacía vs Chef IA sin alternativa)."""
    for code in ["swap_strict_pantry_no_inventory", "swap_llm_retries_exhausted"]:
        assert code in CONTEXT_JSX, (
            f"Frontend debe matchear el error_code canónico {code!r} para "
            f"mostrar el toast title apropiado."
        )


def test_frontend_handler_returns_currentname_on_soft_fail():
    """[PARSER] El handler debe ``return currentName`` después del toast
    para preservar el plato original (NO degradar a fallback local)."""
    # Buscar el bloque del soft-fail check
    m = re.search(
        r"if\s*\(\s*newMealData\?\.\s*swap_failed\s*===\s*true\s*\)\s*\{\s*\n((?:[^\n]*\n){5,20})\s*\}",
        CONTEXT_JSX,
    )
    assert m, (
        "No se encontró el bloque `if (newMealData?.swap_failed === true) "
        "{...}` en el handler. Verifica que el patrón fue añadido."
    )
    block = m.group(0)
    assert "toast.error" in block, "Soft-fail branch debe disparar toast.error"
    assert "return currentName" in block, (
        "Soft-fail branch debe `return currentName` para preservar plato."
    )
    assert "getAlternativeMeal" not in block, (
        "Soft-fail branch NO debe llamar getAlternativeMeal — eso degrada "
        "a fallback local que es lo opuesto de preservar el plato original."
    )


# ---------------------------------------------------------------------------
# Section D — Marker bumped
# ---------------------------------------------------------------------------

def test_marker_bumped():
    # Pin removido — pin-tests se rompen cada P-fix siguiente.
    # `test_p3_1_last_known_pfix_freshness` cubre freshness a nivel codebase.
    pass
