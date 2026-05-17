"""[P3-PLAN-FLOW-MINIMALIST · 2026-05-15] Regression guard: el flow de
generación de plan debe ser:
  1. LoadingScreen sin progress bar ni porcentaje (solo dot + ring + mensaje).
  2. LoadingScreen muestra mensaje informativo del comportamiento deep-search.
  3. Tras pipeline complete sin observaciones → redirect DIRECTO a /dashboard,
     NO renderizar PreviewScreen (eliminar flash de 50-300ms visible al usuario).

PreviewScreen sigue existiendo SOLO para casos con observaciones reales
(banners médicos / pantry degradada) — el usuario DEBE poder leer + decidir.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent.parent
_PLAN = _ROOT / "frontend" / "src" / "pages" / "Plan.jsx"


def _read() -> str:
    return _PLAN.read_text(encoding="utf-8")


def test_progress_bar_removed():
    """La hairline progress bar (2px + linear-gradient o solid white) NO
    debe seguir en el código del LoadingScreen."""
    text = _read()
    # Pre-fix tenía `height: '2px'` para la barra hairline en el LoadingScreen.
    assert "width: '100%', height: '2px'" not in text, (
        "P3-PLAN-FLOW-MINIMALIST: la hairline progress bar (2px height) debe "
        "estar removida del LoadingScreen."
    )


def test_percentage_display_removed():
    """El texto `{Math.round(displayProgress)}%` debe estar removido."""
    text = _read()
    assert "Math.round(displayProgress)" not in text, (
        "P3-PLAN-FLOW-MINIMALIST: el porcentaje numérico (`{Math.round(displayProgress)}%`) "
        "debe estar removido. El usuario no debe ver progreso numérico."
    )


def test_deep_search_message_present():
    """El mensaje informativo del comportamiento deep-search debe estar
    presente en el LoadingScreen."""
    text = _read()
    assert "Puedes salir si quieres" in text, (
        "Falta el mensaje 'Puedes salir si quieres' que comunica el "
        "comportamiento deep-search al usuario."
    )
    assert "Te avisamos cuando tu plan esté listo" in text


def test_happy_path_skips_preview_screen():
    """Plan.jsx debe hacer skip del setState('preview') cuando el plan
    llega limpio (sin observaciones). El check se hace en el PARENT antes
    de renderizar PreviewScreen → elimina flash."""
    text = _read()
    # El bloque `_hasObservations` debe existir en el flujo de processPlan.
    assert "_hasObservations" in text, (
        "Falta el check `_hasObservations` en processPlan. Sin esto, "
        "PreviewScreen renderiza brevemente antes del redirect (flash)."
    )
    # Y debe ramificar: si hay observaciones → setStatus('preview'),
    # si no → navigate('/dashboard') directamente.
    assert "if (_hasObservations)" in text


def test_observations_check_includes_3_banner_flags():
    """El check de `_hasObservations` debe combinar los 3 flags de banners
    que justifican mostrar PreviewScreen: critical_rejection, review_failed,
    pantry_degraded. Sin alguno, planes con observaciones se redirigen
    silenciosamente (bug)."""
    text = _read()
    # Buscar el bloque que define _hasObservations.
    region = re.search(
        r"_hasObservations\s*=\s*!!\([^)]+\)",
        text,
        re.DOTALL,
    )
    assert region, "No encontré la definición de `_hasObservations`."
    body = region.group(0)
    for flag in (
        "_critical_rejection",
        "_review_failed_but_delivered",
        "_pantry_degraded_summary",
    ):
        assert flag in body, (
            f"`_hasObservations` debe incluir `{flag}`. Sin él, planes con "
            f"observaciones se skipean a /dashboard sin que el usuario los vea."
        )


def test_preview_screen_component_still_exists():
    """PreviewScreen NO se debe eliminar — sigue siendo necesaria para casos
    con observaciones. Solo cambiamos el FLOW (cuándo se monta), no el componente."""
    text = _read()
    assert "const PreviewScreen" in text, (
        "PreviewScreen debe seguir definida. P3-PLAN-FLOW-MINIMALIST solo "
        "cambia cuándo se renderiza, NO elimina el componente."
    )
    assert "if (status === 'preview')" in text, (
        "El render condicional de PreviewScreen debe seguir existiendo "
        "(solo aplica cuando hay observaciones)."
    )
