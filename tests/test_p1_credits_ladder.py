"""[P1-CREDITS-LADDER · 2026-07-31] Re-balance comercial del sistema de créditos.

Decisión owner: "cambia el sistema de créditos a como veas más rentable para
uso comercial... el max ilimitado es algo que quiero que cambies".

Escalera final (defaults de código; knobs MEALFIT_TIER_LIMIT_* siguen
permitiendo override sin redeploy):
  gratis 15→10 (hábito sin regalar el producto; múltiplos limpios 5×/20×/50×)
  basic  50    (sin cambio — bajar caps a quien paga es hostil)
  plus   200   (sin cambio)
  ultra  ∞(999999)→500 ("ilimitado" era cola de costo NO acotada + imán de
                abuso; 500/mes ≈ 16/día — inalcanzable para uso humano,
                acotado para scripts. Worst-case LLM ultra ≈ $18/mes sobre
                $49.99 = 36%, antes ∞)
  admin  999999 (sentinel interno, no comercial — se conserva)

Parser-based sobre el source de auth.py: `_TIER_LIMITS` es module-level y se
congela en import-time (reload de auth en tests = contaminación conocida del
repo), así que se ancla el DEFAULT del código, no el valor runtime.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_AUTH_SRC = (_BACKEND / "auth.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")


def _default_of(tier_env: str) -> int:
    m = re.search(rf'_env_int\("MEALFIT_TIER_LIMIT_{tier_env}",\s*(\d+)\)', _AUTH_SRC)
    assert m, f"no encontré el default de MEALFIT_TIER_LIMIT_{tier_env} en auth.py"
    return int(m.group(1))


def test_ladder_defaults():
    assert _default_of("GRATIS") == 10, "gratis bajó de 15 a 10 (P1-CREDITS-LADDER)"
    assert _default_of("BASIC") == 50, "basic sin cambio"
    assert _default_of("PLUS") == 200, "plus sin cambio"
    assert _default_of("ULTRA") == 500, (
        "ultra debe estar ACOTADO (era 999999='ilimitado'): cola de costo LLM "
        "no acotada + imán de abuso. Si necesitas subirlo, knob "
        "MEALFIT_TIER_LIMIT_ULTRA sin redeploy — no restaurar el sentinel."
    )
    # admin conserva el sentinel (cuenta interna).
    assert _default_of("ADMIN") == 999999


def test_ultra_is_not_unlimited_sentinel():
    # Nadie debe re-atar ultra al sentinel de ilimitado.
    assert not re.search(
        r'"ultra":\s*_env_int\("MEALFIT_TIER_LIMIT_ULTRA",\s*999999\)', _AUTH_SRC
    ), "ultra volvió al sentinel 999999 — P1-CREDITS-LADDER lo prohíbe"


def test_marker_bumped():
    assert "P1-CREDITS-LADDER" in _AUTH_SRC
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP_SRC)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-CREDITS-LADDER" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-31"


def test_frontend_synced():
    """El frontend muestra los mismos números (PLAN_LIMIT del context + copy
    de las tarjetas). Sin esta paridad, la UI vendería créditos que el
    backend no honra (o al revés)."""
    _fe = _BACKEND.parent / "frontend" / "src"
    ctx = (_fe / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    assert re.search(r"PLAN_LIMIT\s*=\s*10", ctx), (
        "AssessmentContext.PLAN_LIMIT debe ser 10 (paridad con "
        "MEALFIT_TIER_LIMIT_GRATIS default)"
    )
    upgrade = (_fe / "pages" / "Upgrade.jsx").read_text(encoding="utf-8")
    assert "Ilimitad" not in upgrade, (
        "Upgrade.jsx no debe vender créditos 'ilimitados' (ultra=500 acotado)"
    )
    pricing = (_fe / "components" / "home" / "Pricing.jsx").read_text(encoding="utf-8")
    assert "Créditos Ilimitados" not in pricing, (
        "Pricing.jsx (landing) no debe vender 'Créditos Ilimitados'"
    )
