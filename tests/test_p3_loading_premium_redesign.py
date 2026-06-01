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
        # [P3-LOADING-PALETTE-ALIGN · 2026-05-16] el radial pasó de #14141a a
        # slate #1E293B; [LOADING-DARK-BG · 2026-05-31] el fondo se movió del
        # inline a la clase .mf-loading-bg en el bloque <style> (ver test
        # dedicado test_loading_dark_bg_matches_dashboard_formulario abajo).
        ("radial-gradient(ellipse at center, #1E293B", "fondo claro slate premium"),
        (".mf-loading-bg", "clase del fondo theme-aware del loading"),
    ]
    for substr, desc in new_surfaces:
        assert substr in text, (
            f"P3-LOADING-PREMIUM-REDESIGN: falta `{substr}` ({desc})."
        )


def test_cancel_button_minimalist_not_pill():
    """El botón Cancelar debe ser texto-only sutil, NO un pill rojo prominente.

    [P3-CANCEL-ONE-CLICK · 2026-05-16] El botón pasó a single-click: se eliminó
    el modal confirm inline (antes `key="cancel-trigger"` + `setShowCancelConfirm`)
    y ahora dispara `onCancel()` + navigate directo. Anclamos al marker del
    rediseño one-click en lugar del key legacy.
    """
    text = _read_plan()
    assert "P3-CANCEL-ONE-CLICK" in text, (
        "Falta el marker P3-CANCEL-ONE-CLICK que ancla el botón cancelar single-click."
    )
    # Localiza el bloque del botón: desde el marker hasta el <motion.button> con
    # label "Cancelar". DOTALL + lazy con backtracking salta los `>` internos de
    # las arrow functions del onClick hasta el `>` real del tag de apertura.
    region = re.search(
        r'P3-CANCEL-ONE-CLICK.*?<motion\.button\b.*?>\s*Cancelar\s*</motion\.button>',
        text,
        re.DOTALL,
    )
    assert region, "No encontré el botón Cancelar (motion.button con label 'Cancelar')."
    button_block = region.group(0)
    # Texto-only sutil: fondo transparente (no pill).
    assert "background: 'transparent'" in button_block, (
        "El botón cancelar debe ser texto-only (background transparent)."
    )
    # NO debe tener border rojo prominente (rgba 239,68,68 pre-fix).
    if "border: '1px solid rgba(239, 68, 68" in button_block:
        pytest.fail(
            "Botón cancelar sigue con border pill rojo. "
            "P3-LOADING-PREMIUM-REDESIGN lo cambió a texto-only sutil."
        )


def test_loading_dark_bg_matches_dashboard_formulario():
    """[LOADING-DARK-BG · 2026-05-31] En modo oscuro la pantalla de loading debe
    usar el MISMO fondo ambiental que el Dashboard y el Formulario
    (P3-DARK-BG-STRIPES): rayas diagonales 45° 1px sutiles + glows indigo/púrpura
    sobre #0B1120. El fondo vive en la clase `.mf-loading-bg` del bloque <style>
    (no inline) para que el override de tema gane especificidad.

    Regression guard: si alguien revierte el fondo del loading a un slate plano
    o lo deja sólo en el estilo inline, este test falla.
    """
    text = _read_plan()
    assert "LOADING-DARK-BG" in text, "Falta el marker LOADING-DARK-BG en Plan.jsx."
    # El contenedor del loading aplica la clase.
    assert 'className="mf-loading-bg"' in text, (
        "El div del LoadingScreen debe tener className='mf-loading-bg'."
    )
    # Override de tema oscuro presente sobre la clase.
    assert 'html[data-theme="dark"] .mf-loading-bg' in text, (
        "Falta el override del fondo oscuro `html[data-theme=\"dark\"] .mf-loading-bg`."
    )
    # Capa de rayas diagonales — firma compartida con Dashboard/Formulario.
    assert "repeating-linear-gradient(45deg, rgba(255,255,255,0.04)" in text, (
        "Falta la capa de rayas diagonales del fondo oscuro (P3-DARK-BG-STRIPES)."
    )
    # Base slate-950 idéntica al --bg-page oscuro del resto del producto.
    assert "background-color: #0B1120" in text, (
        "El fondo oscuro del loading debe partir de #0B1120 (--bg-page dark)."
    )
