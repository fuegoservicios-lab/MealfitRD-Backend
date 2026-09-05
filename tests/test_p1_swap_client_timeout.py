"""[P1-SWAP-CLIENT-TIMEOUT · 2026-09-04] «Cambiar plato» medido en prod: 71 s (LLM ~20 s, motor de
macros de S1 sobre TODO el plan ~33 s en `_apply_update_macro_engine`, listas ~15 s). El cliente
cortaba a los 60 s por defecto (nginx 499), pintaba una alternativa LOCAL genérica y el servidor
persistía el plato real 11 s después (P1-SWAP-REGEN-RESUME). Mismo defecto que
P1-DAY-REGEN-CLIENT-TIMEOUT cerró el 03-09 para «actualizar día».

Frontend: tope de 3 min en el fetch del swap; con timeout no hay fallback local ni toast rojo, y el
marker in-flight sobrevive para que el resume/poll aplique el plato persistido. Este ancla vive en
el backend por el contrato marker↔test (P2-HIST-AUDIT-14); el detalle lo prueba
`AssessmentContext.swap_timeout.test.js`.
"""
from __future__ import annotations

from pathlib import Path

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_swap_fetch_has_its_own_timeout_and_honest_timeout_branch():
    src = (_FRONT / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    assert "const SWAP_TIMEOUT_MS = 3 * 60 * 1000;" in src
    i = src.index("fetchWithAuth(API_SWAP_URL, {")
    assert "timeout: SWAP_TIMEOUT_MS," in src[i:i + 200]
    j = src.index("if (error?.code === 'request_timeout') {")
    assert j < src.index("const localFallback = getAlternativeMeal(")
    assert "if (!_swapTimedOut) safeLocalStorageRemove('mealfit_meal_regen_inflight');" in src


def test_marker_present():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "P1-SWAP-CLIENT-TIMEOUT · 2026-09-04" in app
