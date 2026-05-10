"""[P2-HIST-NEW-5 · 2026-05-09] Tests del backend `lifetime-lessons`
para el toggle "Ver todos" / "Ver menos" del lifetime history.

Bug original (audit profundo Historial 2026-05-09):
    El bloque "Historial reciente por chunk" del tab Lecciones
    renderizaba `_history.slice(0, 5)` con counter "5 de N" pero sin
    botón para expandir y ver el resto. Surface incompleto para
    planes con >5 entries (tier ultra de 90 días tiene 13+ chunks).

    Fix 100% client-side (state + botón toggle). Este test cierra
    el cross-link del marker (P2-HIST-AUDIT-14 requiere
    `tests/test_p2_hist_new_5*.py`) Y protege el cap del backend
    contra refactors que rompan la longitud del array.

Cobertura backend:
    1. Anchor del marker en History.jsx Y CSS.
    2. Endpoint /lifetime-lessons existe.
    3. Response shape: history es lista capeada (≤50, defensa contra
       payloads inflados).
    4. Counts.history_total presente para el counter "N de M".
"""
from __future__ import annotations

import inspect
import re


# ---------------------------------------------------------------------------
# 1. Anchor del marker — fix vive en History.jsx (client-side)
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists()
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P2-HIST-NEW-5" in text, (
        "Marker `P2-HIST-NEW-5` debe aparecer en History.jsx donde "
        "vive el toggle de expansión del lifetime history."
    )


def test_marker_present_in_css():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    css = repo_root / "frontend" / "src" / "pages" / "History.module.css"
    assert css.exists()
    text = css.read_text(encoding="utf-8")
    assert "[P2-HIST-NEW-5" in text, (
        "Marker `P2-HIST-NEW-5` debe aparecer en History.module.css "
        "donde se declara `.lifetimeHistoryToggle`."
    )


# ---------------------------------------------------------------------------
# 2. Endpoint /lifetime-lessons sigue existiendo
# ---------------------------------------------------------------------------
def test_lifetime_lessons_endpoint_exists():
    """El frontend depende de `/api/plans/{plan_id}/lifetime-lessons`
    para popular `_history`. Si alguien renombra el endpoint sin
    actualizar el frontend, el toggle queda inerte (no hay history)."""
    from routers.plans import api_plan_lifetime_lessons
    assert callable(api_plan_lifetime_lessons)


# ---------------------------------------------------------------------------
# 3. Cap defensivo de history en el endpoint
# ---------------------------------------------------------------------------
def test_lifetime_lessons_caps_history_payload():
    """El handler debe capear el array `history` antes de serializar
    al response (ej. .slice(0, 50) o equivalente). El cap del backend
    es defensa contra _lifetime_lessons_history persistido sin
    truncate en el cron — frontend no debería tener que truncar
    payloads inflados."""
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    # Cap explícito vía slice/limit de algún tipo. Patrones aceptados:
    #   - `[:50]` / `[: 50]` (positive slice)
    #   - `[-_LIFETIME_HISTORY_CAP:]` (negative slice with constant)
    #   - `[-50:]` / `[-N:]` (negative slice numérico)
    #   - `LIMIT N`
    #   - `.slice(...)`
    assert re.search(
        r"\[\s*:\s*\d+\s*\]"            # positive slice numérico
        r"|\[\s*-\s*\w+\s*:\s*\]"       # negative slice con constante
        r"|LIMIT\s+\d+"
        r"|\.slice\(",
        src,
    ), (
        "Endpoint /lifetime-lessons debe capear el array history "
        "para evitar payloads inflados. Patterns aceptados: "
        "`[:N]`, `[-N:]`, `[-CONST:]`, `LIMIT N`, `.slice(...)`."
    )


# ---------------------------------------------------------------------------
# 4. Counts.history_total presente para el counter "N de M"
# ---------------------------------------------------------------------------
def test_lifetime_lessons_response_includes_history_total():
    """El frontend lee `_counts.history_total` para mostrar "5 de 50"
    en el header del bloque. Sin esta key, el counter solo muestra
    el local count (5) y el user no sabe que hay más."""
    from routers.plans import api_plan_lifetime_lessons
    src = inspect.getsource(api_plan_lifetime_lessons)
    # Buscar referencia a `history_total` en el código del handler.
    assert "history_total" in src, (
        "Endpoint debe popular `counts.history_total` para que el "
        "counter del header muestre el total real del lifetime."
    )
