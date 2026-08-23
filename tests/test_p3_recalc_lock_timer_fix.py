"""[P3-RECALC-LOCK-TIMER-FIX · 2026-05-16] El safety timer del `recalcLock` en
`AssessmentContext.jsx` pasa de 15s → 30s, y el warn message se reescribe.

Razón: con el retry de P3-RECALC-503-CLASSIFICATION (frontend reintenta 1× tras
500ms en 5xx), una operación legítima del recalc puede tardar 17-24s en free
tier (1er intento 8-12s + 500ms backoff + 2do intento 8-12s). A 15s, el warn
fuego falso aparecía siempre — el lock se liberaba a los 15s mientras la
operación seguía corriendo legítimamente.

A 30s, el warn solo dispara si la operación realmente excede 30s — señal de
backend lento de verdad, no de bug del caller. El mensaje original
("el caller olvidó setRecalcLock") era engañoso: TODOS los callers usan
`withRecalcLock` que libera en finally. El nuevo mensaje atribuye correctamente
a backend lento.

Tests parser-based — el componente React no es ejecutable en aislamiento
(necesita DOM + React); lo importante es anclar la configuración y el mensaje
para que no regresen accidentalmente.
"""
from __future__ import annotations

import pytest

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _ASSESSMENT
    _ASSESSMENT = (
        _BACKEND_ROOT.parent / "frontend" / "src" / "context" / "AssessmentContext.jsx"
    ).read_text(encoding="utf-8")



def test_safety_timer_constant_declared_at_30s():
    """`_RECALC_SAFETY_TIMER_MS = 30000` debe estar declarado como constante
    (no inline) para que el operador sepa que el valor es deliberado."""
    m = re.search(r"_RECALC_SAFETY_TIMER_MS\s*=\s*(\d+)", _ASSESSMENT)
    assert m, (
        "Constante `_RECALC_SAFETY_TIMER_MS` no declarada en "
        "AssessmentContext.jsx. Revierte P3-RECALC-LOCK-TIMER-FIX."
    )
    val = int(m.group(1))
    assert val == 30000, (
        f"_RECALC_SAFETY_TIMER_MS={val}, esperado 30000ms. Subir más es ok "
        f"(retry largo); bajar de 30s revierte el fix (warn falso aparece "
        f"con retry P3-RECALC-503-CLASSIFICATION)."
    )


def test_setTimeout_uses_the_constant():
    """El `setTimeout` del safety net DEBE usar la constante, no hardcode 15000."""
    # Localizar el setTimeout dentro de setRecalcLock
    idx = _ASSESSMENT.find("const setRecalcLock = useCallback(")
    assert idx > 0, "Función `setRecalcLock` no encontrada."
    end = _ASSESSMENT.find("}, []);", idx)
    body = _ASSESSMENT[idx:end + 10 if end > 0 else idx + 3000]

    assert "_RECALC_SAFETY_TIMER_MS" in body, (
        "setRecalcLock no referencia `_RECALC_SAFETY_TIMER_MS` — usa hardcode."
    )
    # Defensa anti-regresión: el hardcode 15000 NO debe estar dentro de setRecalcLock
    # (puede haber 15000 en comentarios documentando la historia).
    # Buscar específicamente `setTimeout(\s*..., 15000)`
    bad = re.search(r"setTimeout\([^)]*?,\s*15000\s*\)", body)
    assert not bad, (
        "setRecalcLock todavía usa setTimeout(..., 15000) hardcoded — "
        "revierte el fix. Usar `_RECALC_SAFETY_TIMER_MS` (=30000)."
    )


def test_warn_message_does_not_blame_caller():
    """El warn original culpaba al caller ('el caller olvidó setRecalcLock').
    Eso era engañoso — todos los callers usan `withRecalcLock`. El nuevo
    mensaje debe atribuir la causa a backend lento.

    Extraemos SOLO el contenido del `console.warn(...)` activo (no los
    comentarios históricos que pueden mencionar el copy viejo describiendo
    la evolución del fix)."""
    idx = _ASSESSMENT.find("const setRecalcLock = useCallback(")
    end = _ASSESSMENT.find("}, []);", idx)
    body = _ASSESSMENT[idx:end + 10 if end > 0 else idx + 3000]

    # Extraer el contenido del console.warn(...) — soporta template literal
    # `...` multi-línea (que es lo que el fix usa).
    m = re.search(r"console\.warn\(\s*([\s\S]*?)\s*\)\s*;", body)
    assert m, "No se encontró un `console.warn(...)` activo en setRecalcLock."
    warn_call = m.group(1)

    # Sanity: el warn debe seguir mencionando "Safety timer" como prefijo.
    assert "Safety timer" in warn_call, "El prefijo `Safety timer` ausente."

    # NO debe culpar al caller con el copy viejo:
    assert "olvidó setRecalcLock" not in warn_call, (
        "El warn ACTIVO sigue diciendo 'el caller olvidó setRecalcLock' — "
        "es engañoso porque withRecalcLock siempre libera en finally."
    )
    assert "Migrar a withRecalcLock" not in warn_call, (
        "El warn ACTIVO sugiere migrar a withRecalcLock pero TODOS los "
        "callers ya lo usan. Eliminar la sugerencia engañosa."
    )

    # DEBE atribuir a backend lento (en el warn ACTIVO, no comentarios):
    assert "backend lento" in warn_call or "backend slow" in warn_call, (
        "El warn ACTIVO debe atribuir la causa a 'backend lento' (o "
        "equivalente). Sin esto, el operador no sabe dónde investigar."
    )


def test_marker_present():
    """Marker `P3-RECALC-LOCK-TIMER-FIX` debe estar cerca de la lógica para
    que un refactor no borre el contexto del cambio."""
    assert "P3-RECALC-LOCK-TIMER-FIX" in _ASSESSMENT, (
        "Marker `P3-RECALC-LOCK-TIMER-FIX` ausente en AssessmentContext.jsx — "
        "un refactor cosmético podría borrar el por qué del timer/mensaje."
    )
