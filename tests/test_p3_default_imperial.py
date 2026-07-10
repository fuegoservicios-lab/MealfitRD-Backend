"""[P3-DEFAULT-IMPERIAL · 2026-05-20] Test anti-regresión del default
imperial (lb + ft) en lugar de métrico (kg + cm) en TODA la app.

Decisión de producto:
    El user mercado primario prefiere imperial. Pre-fix:
      - `_getDefaultWeightUnit()` en AssessmentContext branched por
        navigator.language: en-US/en-LR/my → 'lb', resto → 'kg'.
      - Wizard `InteractiveQuestions` defaulteaba `_heightInputUnit` a 'cm'.
      - Settings `heightUnit` arrancaba en 'cm' sin pre-conversión.
    Resultado: usuarios DO/LATAM veían inputs en kg/cm aunque el mercado
    objetivo prefiere imperial.

Fix:
    Defaults unificados a imperial (lb + ft) en los 3 callsites. El user
    puede togglear a métrico (kg/cm) y la preferencia persiste — solo el
    default INICIAL cambia.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONTEXT_JSX = _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"
_WIZARD_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "assessment" / "questions" / "QMeasurements.jsx"
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_context_weight_unit_default_lb_unconditional():
    """[P3-DEFAULT-IMPERIAL] `_getDefaultWeightUnit` debe retornar 'lb'
    incondicionalmente. Pre-fix branchaba por navigator.language."""
    src = _read(_CONTEXT_JSX)
    fn_match = re.search(
        r"const\s+_getDefaultWeightUnit\s*=\s*\(\s*\)\s*=>\s*\{(.+?)\};",
        src,
        re.DOTALL,
    )
    assert fn_match, "_getDefaultWeightUnit no encontrada"
    body = fn_match.group(1)
    # Anti-pattern bloqueado: branching por navigator.language.
    assert "navigator.language" not in body, (
        "`_getDefaultWeightUnit` aún branchea por `navigator.language` — "
        "default no es unificado a 'lb'. Ver P3-DEFAULT-IMPERIAL · 2026-05-20."
    )
    # Sanity positiva: retorna 'lb'.
    assert re.search(r"return\s+['\"]lb['\"]", body), (
        "`_getDefaultWeightUnit` no retorna 'lb'."
    )


def test_wizard_height_unit_defaults_to_ft():
    """[P3-DEFAULT-IMPERIAL] Wizard `InteractiveQuestions` defaultea
    `_heightInputUnit` a 'ft' (no 'cm'). Solo aplica si el user no eligió
    explícitamente — el `||` left-side preserva su elección si ya guardó."""
    src = _read(_WIZARD_JSX)
    assert re.search(
        r"formData\._heightInputUnit\s*\|\|\s*['\"]ft['\"]",
        src,
    ), (
        "Wizard NO defaultea `_heightInputUnit` a 'ft'. Ver "
        "P3-DEFAULT-IMPERIAL · 2026-05-20."
    )


def test_settings_height_unit_defaults_to_ft_with_preconversion():
    """[P3-DEFAULT-IMPERIAL] Settings `heightUnit` arranca en 'ft' Y
    pre-convierte la altura cm previa a ft+in para que los inputs
    imperiales arranquen poblados (no vacíos requiriendo toggle manual)."""
    src = _read(_SETTINGS_JSX)
    # Default 'ft' en useState(heightUnit).
    assert re.search(
        r"const\s*\[\s*heightUnit\s*,\s*setHeightUnit\s*\]\s*=\s*useState\(\s*['\"]ft['\"]\s*\)",
        src,
    ), (
        "Settings `heightUnit` no defaultea a 'ft'. Ver "
        "P3-DEFAULT-IMPERIAL · 2026-05-20."
    )
    # Pre-conversión cm → ft+in en initializer.
    assert "_ftInitial" in src, (
        "Pre-conversión `_ftInitial` ausente. Sin esto, los inputs ft "
        "arrancan vacíos aunque haya altura cm previa — UX confusa."
    )
    # heightFeet y heightInches inicializan desde _ftInitial.
    assert re.search(
        r"useState\(\s*\(\s*\)\s*=>\s*_ftInitial\.ft\s*\)",
        src,
    ), (
        "`heightFeet` no inicializa con `_ftInitial.ft` — input ft "
        "arranca vacío."
    )


def test_settings_weight_unit_defaults_to_lb_fallback():
    """[P3-DEFAULT-IMPERIAL] Settings `weightUnit` debe defaultear a 'lb'
    en el fallback final (cuando no hay valor en formData/userProfile)."""
    src = _read(_SETTINGS_JSX)
    # Buscar useState de weightUnit y verificar el fallback final.
    match = re.search(
        r"useState\(\s*\(\s*\)\s*=>\s*formData\?\.weightUnit\s*\|\|\s*userProfile\?\.health_profile\?\.weightUnit\s*\|\|\s*['\"](\w+)['\"]\s*\)",
        src,
    )
    assert match, (
        "useState de `weightUnit` con fallback chain no encontrado en Settings."
    )
    fallback = match.group(1)
    assert fallback == "lb", (
        f"Fallback final de `weightUnit` es '{fallback}', debe ser 'lb'. "
        f"Ver P3-DEFAULT-IMPERIAL · 2026-05-20."
    )


def test_settings_cleanup_reverts_to_ft_not_cm():
    """[P3-DEFAULT-IMPERIAL] `_revertBodyMetricsToOriginal` debe revertir
    `heightUnit` a 'ft' (no 'cm') y pre-poblar ft/in desde la altura cm
    original. Sin esto, tras un cleanup (cambio de sección sin commit),
    el user pierde su elección imperial."""
    src = _read(_SETTINGS_JSX)
    fn_match = re.search(
        r"const\s+_revertBodyMetricsToOriginal\s*=\s*\(\s*\)\s*=>\s*\{(.+?)\};",
        src,
        re.DOTALL,
    )
    assert fn_match, "_revertBodyMetricsToOriginal no encontrada"
    body = fn_match.group(1)
    # Revierte a 'ft', NO a 'cm'.
    assert re.search(r"setHeightUnit\(\s*['\"]ft['\"]\s*\)", body), (
        "`_revertBodyMetricsToOriginal` no setea heightUnit a 'ft' — "
        "tras cleanup vuelve a métrico, perdiendo elección imperial. "
        "Ver P3-DEFAULT-IMPERIAL · 2026-05-20."
    )
    # Pre-popula ft/in desde cm (cálculo presente).
    assert "2.54" in body, (
        "Conversión cm → ft+in ausente del revert (falta divisor 2.54). "
        "Inputs ft/in quedan vacíos tras cleanup."
    )
