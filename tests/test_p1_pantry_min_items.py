"""[P1-PANTRY-MIN-ITEMS · 2026-07-11] Piso de alimentos del modo "Desde mi Nevera".

Feedback del owner (screenshot con 1 solo arroz blanco, CTA habilitado): "¿no debería
haber un mínimo para generar un plan mínimamente coherente?". Con 1-2 items el
Zero-Waste no tiene material y el plan resultante es indistinguible del libre — la
queja original que motivó el modo manual-first.

Contrato:
1. Backend SSOT: knob `MEALFIT_PANTRY_MODE_MIN_ITEMS` (default 5, clamp [1, 50]) vía
   `_pantry_mode_min_items()`, expuesto como `min_items` en la respuesta de
   /pantry-feasibility (lazy — ajustable por env sin redeploy).
2. Frontend: QPantryBuilder consume `min_items` con fallback 5, CTA deshabilitado
   bajo el piso con label de progreso `(X/N)`.

tooltip-anchor: P1-PANTRY-MIN-ITEMS
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_QPB = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment"
        / "questions" / "QPantryBuilder.jsx")

_PLANS_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_QPB_SRC = _QPB.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Backend: knob SSOT + respuesta del pre-flight
# ---------------------------------------------------------------------------

def test_min_items_knob_defined_with_clamp():
    m = re.search(
        r'def _pantry_mode_min_items\(\) -> int:.*?_env_int\("MEALFIT_PANTRY_MODE_MIN_ITEMS", 5, validator=lambda v: 1 <= v <= 50\)',
        _PLANS_SRC, re.DOTALL,
    )
    assert m, "knob MEALFIT_PANTRY_MODE_MIN_ITEMS (default 5, clamp [1,50]) desapareció"


def test_min_items_in_feasibility_response():
    assert '"min_items": _pantry_mode_min_items(),' in _PLANS_SRC, (
        "min_items debe viajar en la respuesta de /pantry-feasibility — es el SSOT "
        "que consume el CTA del wizard (sin él, el frontend cae al fallback local)"
    )


def test_min_items_default_value():
    import importlib
    import os
    assert "MEALFIT_PANTRY_MODE_MIN_ITEMS" not in os.environ, (
        "el entorno de test no debe overridear el knob (invalidaría el assert de default)"
    )
    from graph_orchestrator import _env_int
    assert _env_int("MEALFIT_PANTRY_MODE_MIN_ITEMS", 5, validator=lambda v: 1 <= v <= 50) == 5


# ---------------------------------------------------------------------------
# 2. Frontend: gate con progreso
# ---------------------------------------------------------------------------

def test_frontend_consumes_server_min_with_fallback():
    assert "Number(feas?.min_items) >= 1 ? Number(feas.min_items) : 5" in _QPB_SRC, (
        "QPantryBuilder debe leer min_items del server con fallback 5 (el medidor "
        "puede no haber respondido aún)"
    )
    assert "const belowMin = count < minItems;" in _QPB_SRC


def test_frontend_cta_gated_with_progress_label():
    assert "disabled={isSubmitting || belowMin}" in _QPB_SRC
    assert "Agrega al menos ${minItems} alimentos (${count}/${minItems})" in _QPB_SRC, (
        "el label debe mostrar progreso hacia el piso — un disabled mudo frustra"
    )


def test_marker_anchored_in_source():
    assert _PLANS_SRC.count("P1-PANTRY-MIN-ITEMS") >= 2
    assert _QPB_SRC.count("P1-PANTRY-MIN-ITEMS") >= 1
