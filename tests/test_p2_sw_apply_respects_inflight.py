"""[P2-SW-APPLY-RESPECTS-INFLIGHT · 2026-09-04] La auto-aplicación de la versión nueva del frontend
(P1-SW-AUTO-APPLY-SAFE: pestaña oculta + «sin generación en vuelo») no contaba un swap ni un
«actualizar día» como operación en vuelo. Medido en prod (19:36 UTC): el dueño lanzó un swap,
cambió de ventana y la recarga automática cortó la petición a los 9 s (nginx 499); el persist
server-side + el resume lo salvaron, pero era justo la interrupción que 'prompt' quería evitar.
Ahora `_safeToApply` mira también los markers de swap (6 min) y de día (9 min), los mismos que
usan los resumes. Frontend: `SwAutoApply.inflight_guard.test.js`.
"""
from __future__ import annotations

from pathlib import Path

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_safe_to_apply_checks_swap_and_day_markers():
    src = (_FRONT / "main.jsx").read_text(encoding="utf-8")
    i = src.index("const _safeToApply = () => {")
    body = src[i:src.index("const _applyIfSafe", i)]
    assert "['mealfit_meal_regen_inflight', 6 * 60 * 1000]" in body
    assert "['mealfit_day_regen_inflight', 9 * 60 * 1000]" in body
    assert "P2-SW-APPLY-RESPECTS-INFLIGHT" in body


def test_marker_present():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "P2-SW-APPLY-RESPECTS-INFLIGHT · 2026-09-04" in app
