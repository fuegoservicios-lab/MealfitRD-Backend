"""[P3-PLAN-AUTO-APPLY-CLEAN · 2026-05-15] Regression guard: la pantalla
intermedia `PreviewScreen` de `Plan.jsx` debe auto-aplicar el plan + ir al
dashboard cuando el plan se aprobó LIMPIAMENTE (sin observaciones que
requieran decisión del usuario).

Razón UX: post-fixes de cost optimization, el plan happy-path llega
`review_passed=True` sin banners. La pantalla "Compara los cambios" + botón
"Aceptar y Aplicar Nuevo Plan" es fricción innecesaria — el usuario ya
disparó la generación, no hay nada nuevo que decidir.

Casos en que SE PRESERVA la pantalla:
  - `showReviewCriticalBanner` (rechazo médico crítico)
  - `showReviewWarningBanner` (observaciones no-críticas)
  - `showPantryBanner` (despensa degradada)

Kill switch sin redeploy: `VITE_PLAN_AUTO_APPLY_ON_CLEAN_REVIEW=false` en
`.env.local`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent.parent
_PLAN_PATH = _ROOT / "frontend" / "src" / "pages" / "Plan.jsx"


def _read_plan() -> str:
    return _PLAN_PATH.read_text(encoding="utf-8")


def test_knob_referenced():
    """El knob VITE_PLAN_AUTO_APPLY_ON_CLEAN_REVIEW debe leerse desde
    `import.meta.env` y servir de kill switch."""
    text = _read_plan()
    assert "VITE_PLAN_AUTO_APPLY_ON_CLEAN_REVIEW" in text, (
        "P3-PLAN-AUTO-APPLY-CLEAN: falta lectura del knob "
        "`VITE_PLAN_AUTO_APPLY_ON_CLEAN_REVIEW` en Plan.jsx."
    )
    # Default debe ser 'true' (skip-by-default cuando hay happy path).
    assert re.search(
        r"VITE_PLAN_AUTO_APPLY_ON_CLEAN_REVIEW\s*\?\?\s*['\"]true['\"]",
        text,
    ), (
        "Default del knob debe ser `'true'` (la mejora UX está activada por "
        "default; el knob solo sirve para desactivarla)."
    )


def test_auto_apply_useeffect_present():
    """Debe existir un useEffect que llame onAccept() cuando el plan está
    limpio y el knob no está deshabilitado."""
    text = _read_plan()
    # No prescriptivo del whitespace exacto — buscamos las firmas clave.
    assert "autoApplyEnabled" in text
    assert "_hasReviewableObservations" in text
    # El useEffect debe llamar onAccept().
    # Buscar la región del useEffect específico.
    # Buscar el useEffect específico que hace el skip. Anchor: la check
    # `if (!autoApplyEnabled)` es única en el archivo.
    auto_apply_idx = text.find("if (!autoApplyEnabled)")
    assert auto_apply_idx > 0, (
        "P3-PLAN-AUTO-APPLY-CLEAN: no encontré `if (!autoApplyEnabled) return`."
    )
    # En las ~500 chars siguientes debe llamar `onAccept()`.
    window = text[auto_apply_idx : auto_apply_idx + 500]
    assert "_hasReviewableObservations" in window, (
        "El useEffect debe checkear `_hasReviewableObservations` para "
        "preservar la pantalla cuando hay observaciones."
    )
    assert "onAccept()" in window, (
        "El useEffect debe llamar `onAccept()` en el happy-path para "
        "disparar el skip al dashboard."
    )


def test_auto_apply_respects_critical_banner():
    """`_hasReviewableObservations` debe combinar los 3 flags de banners
    (pantry + review crítico + review warning). Si alguno está true, NO skip."""
    text = _read_plan()
    region = re.search(
        r"_hasReviewableObservations\s*=\s*\([^)]+\)",
        text,
    )
    assert region, "No encontré la asignación de `_hasReviewableObservations`."
    body = region.group(0)
    for flag in (
        "showPantryBanner",
        "showReviewCriticalBanner",
        "showReviewWarningBanner",
    ):
        assert flag in body, (
            f"`_hasReviewableObservations` debe incluir `{flag}` (banner "
            f"médico/pantry — usuario DEBE poder leer + decidir, no skip)."
        )


def test_preview_screen_buttons_preserved_for_observations():
    """Cuando NO se hace skip, la pantalla debe seguir mostrando los botones
    y banners. Verificación parser-based de presencia de los strings clave."""
    text = _read_plan()
    for surface in (
        "Aceptar y Aplicar Nuevo Plan",
        "¡Plan Generado!",
        "Verificación médica con observaciones",
        "Plan reemplazado por seguridad",
    ):
        assert surface in text, (
            f"P3-PLAN-AUTO-APPLY-CLEAN: el surface `{surface}` debe seguir "
            f"existiendo en Plan.jsx (solo skipeamos el RENDER cuando happy "
            f"path; el componente sigue ahí para casos con observaciones)."
        )
