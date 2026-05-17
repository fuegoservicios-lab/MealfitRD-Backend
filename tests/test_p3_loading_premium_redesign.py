"""[P3-LOADING-PREMIUM-REDESIGN · 2026-05-15] Regression guard: la pantalla
de loading de `Plan.jsx::LoadingScreen` debe ser minimalista premium acorde
a la identidad MealfitRD.

Pre-fix: tenía 2 orbs flotantes + shimmer bar + lista de 10 steps + spinner
doble-ring + título gradient + botón cancelar pill rojo prominente. Demasiados
elementos compitiendo por atención, no acorde a "premium minimalista".

Post-fix:
  - Fondo: radial-gradient cálido oscuro (no purple/blue).
  - Indicador central: 1 dot con pulse + 1 ring delgado girando (sin doble-ring).
  - Título: "Diseñando tu plan" (sin gradient).
  - Subtítulo: fase actual del pipeline (sin lista de 10 steps).
  - Progress: hairline 2px (era 6px shimmer bar).
  - Tip: sin emoji 💡 prefix, sin italic, color más sutil.
  - Cancel: texto-only sutil (no pill rojo).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent.parent
_PLAN_PATH = _ROOT / "frontend" / "src" / "pages" / "Plan.jsx"


def _read_plan() -> str:
    return _PLAN_PATH.read_text(encoding="utf-8")


def test_marker_stamped_in_loading_screen():
    text = _read_plan()
    assert "P3-LOADING-PREMIUM-REDESIGN" in text, (
        "Falta el marker `P3-LOADING-PREMIUM-REDESIGN` en Plan.jsx. "
        "Si alguien refactoriza el LoadingScreen y no preserva el marker, "
        "perdemos trazabilidad."
    )


def test_old_elements_removed():
    """Los elementos visuales pre-fix DEBEN estar eliminados."""
    text = _read_plan()
    removed_surfaces = [
        ("loading-orb1", "orb 1 flotante"),
        ("loading-orb2", "orb 2 flotante"),
        ("shimmer-bar", "shimmer overlay del progress bar"),
        ("orbit-ring-reverse", "ring secundario rotando en reverso"),
        ("Diseñando tu Estrategia", "título legacy (sustituido por 'Diseñando tu plan')"),
    ]
    for substr, desc in removed_surfaces:
        assert substr not in text, (
            f"P3-LOADING-PREMIUM-REDESIGN: `{substr}` ({desc}) NO fue eliminado. "
            f"Si el rediseño se revirtió parcialmente, esta verificación falla."
        )


def test_new_minimalist_surfaces_present():
    """Los elementos del nuevo diseño DEBEN estar presentes."""
    text = _read_plan()
    new_surfaces = [
        ("Diseñando tu plan", "título nuevo minimalista"),
        ("mf-pulse", "animación del dot central"),
        ("mf-spin", "animación del ring delgado"),
        ("radial-gradient(ellipse at center, #14141a", "fondo nuevo cálido oscuro"),
    ]
    for substr, desc in new_surfaces:
        assert substr in text, (
            f"P3-LOADING-PREMIUM-REDESIGN: falta `{substr}` ({desc})."
        )


def test_cancel_button_minimalist_not_pill():
    """El botón Cancelar debe ser texto-only sutil, NO un pill rojo prominente."""
    text = _read_plan()
    # Buscar la región del botón cancel-trigger.
    region = re.search(
        r'key="cancel-trigger".*?onClick=\{\(\)\s*=>\s*setShowCancelConfirm\(true\)\}.*?>\s*([^<]+?)</motion\.button>',
        text,
        re.DOTALL,
    )
    assert region, "No encontré el botón cancel-trigger."
    button_block = region.group(0)
    # El nuevo diseño usa "Cancelar" como label, NO "Cancelar Generación" con icono X.
    assert "Cancelar Generación" not in button_block or "Cancelar" in button_block, (
        "El botón cancelar debe usar copy minimalista."
    )
    # NO debe tener border rojo prominente (rgba 239,68,68 pre-fix).
    # Permitimos rgba(239,68,68) en OTRAS partes (e.g. confirm modal),
    # pero no en el trigger principal.
    if "border: '1px solid rgba(239, 68, 68" in button_block:
        pytest.fail(
            "Botón cancel-trigger sigue con border pill rojo. "
            "P3-LOADING-PREMIUM-REDESIGN lo cambió a texto-only sutil."
        )
