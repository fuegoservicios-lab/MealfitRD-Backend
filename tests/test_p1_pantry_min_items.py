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

import pytest

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_QPB = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment"
        / "questions" / "QPantryBuilder.jsx")

_PLANS_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _QPB_SRC
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


# [P1-I18N-DASHBOARD · 2026-08-15] El label del CTA dejó de ser un template
# literal (`Agrega al menos ${minItems} alimentos (${count}/${minItems})`) y pasa
# ahora por el traductor:
#     t('Agrega al menos {minimo} alimentos ({actual}/{minimo})',
#       { minimo: minItems, actual: count })
# La PROPIEDAD vigilada NO cambió — el CTA sigue gateado por `belowMin` y el label
# sigue mostrando progreso `(actual/piso)` construido con los mismos dos datos
# (`count` y `minItems`); lo que cambió es la grafía del interpolador. El guard
# acepta AMBAS formas pero, en la forma `t()`, exige además que cada placeholder
# esté BINDEADO a su variable real: el motor (`i18n/index.js::_interpolate`) sólo
# sustituye claves presentes en el objeto de vars, así que un binding perdido deja
# el literal «{minimo}» pintado en el botón. Eso es un bug visible, no cosmética.
_PH_MIN = r"(?:\$\{minItems\}|\{[A-Za-z_]\w*\})"   # `${minItems}` | `{minimo}`
_PH_CUR = r"(?:\$\{count\}|\{[A-Za-z_]\w*\})"      # `${count}`    | `{actual}`
_PROGRESS_LABEL_RE = re.compile(
    r"Agrega al menos (?P<min1>" + _PH_MIN + r") alimentos "
    r"\((?P<cur>" + _PH_CUR + r")/(?P<min2>" + _PH_MIN + r")\)"
)


def _next_button_block() -> str:
    """Slice del CTA final del paso (`<NextButton ... />`)."""
    idx = _QPB_SRC.find("<NextButton")
    assert idx > 0, "El CTA <NextButton> del paso Nevera desapareció"
    end = _QPB_SRC.find("/>", idx)
    assert end > idx, "Cierre del <NextButton> no encontrado"
    return _QPB_SRC[idx:end]


def test_frontend_cta_gated_with_progress_label():
    block = _next_button_block()
    assert "disabled={isSubmitting || belowMin}" in block, (
        "el CTA debe seguir deshabilitado bajo el piso (`belowMin`) — es el gate"
    )
    m = _PROGRESS_LABEL_RE.search(block)
    assert m, (
        "el label debe mostrar progreso hacia el piso — un disabled mudo frustra"
    )
    min1, cur, min2 = m.group("min1"), m.group("cur"), m.group("min2")
    assert min1 == min2, (
        "el piso del texto y el denominador del progreso deben ser el MISMO dato "
        f"(texto={min1!r}, denominador={min2!r})"
    )
    # Forma `t()`: el placeholder debe resolverse a la variable correcta. En el
    # template literal la variable ya viaja dentro del propio `${...}`.
    for placeholder, var in ((min1, "minItems"), (cur, "count")):
        if placeholder.startswith("${"):
            continue
        name = placeholder[1:-1]
        assert re.search(rf"\b{name}\s*:\s*{var}\b", block), (
            f"el placeholder «{placeholder}» del label no está bindeado a `{var}` "
            f"en la llamada a t() — el usuario vería «{placeholder}» literal"
        )


def test_marker_anchored_in_source():
    assert _PLANS_SRC.count("P1-PANTRY-MIN-ITEMS") >= 2
    assert _QPB_SRC.count("P1-PANTRY-MIN-ITEMS") >= 1
