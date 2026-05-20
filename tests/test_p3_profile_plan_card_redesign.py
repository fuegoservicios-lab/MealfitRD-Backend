"""[P3-PROFILE-PLAN-CARD-REDESIGN · 2026-05-20] Test anti-regresion del
redesign minimal outline del card "Tu Objetivo Actual" en Settings >
Plan & Objetivo.

Bug pre-fix:
    1. Label en ingles ("Gain Muscle" via `userGoal.replace(/_/g, ' ')`).
       El producto es es-DO permanente (CLAUDE.md > "i18n: es-DO").
    2. Gradient indigo->verde + sombra emerald saturada
       (`rgba(16, 185, 129, 0.4)`) no usaba tokens del design system.
       Mismo issue que P3-RESTOCK-MINIMAL-CTA cerro en Dashboard.
    3. Sombra extensa se clipeaba abajo por `overflow:hidden` del .grid
       wrapper en mobile -> "figura rara" del bottom del card.

Fix:
    Card blanco border slate-200, icon container coloreado por meta
    (_GOAL_META mapping con label es-DO + Icon + accent + tint),
    texto slate-900, CTA slate-900 con ArrowRight + translateX hover,
    kcal con formato es-DO (toLocaleString).

Cubre invariante CLAUDE.md > "i18n: es-DO permanente" para este surface.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_marker_present_as_tooltip_anchor():
    """[P3-PROFILE-PLAN-CARD-REDESIGN] Marker presente como tooltip-anchor."""
    src = _read(_SETTINGS_JSX)
    assert "P3-PROFILE-PLAN-CARD-REDESIGN" in src, (
        "Marker `P3-PROFILE-PLAN-CARD-REDESIGN` ausente en Settings.jsx. "
        "Si quieres remover el fix, primero remueve este test."
    )


def test_goal_meta_mapping_es_do_labels():
    """[P3-PROFILE-PLAN-CARD-REDESIGN] El mapping `_GOAL_META` declara
    los 4 enums backend con labels es-DO."""
    src = _read(_SETTINGS_JSX)
    assert re.search(r"const\s+_GOAL_META\s*=", src), (
        "Mapping `_GOAL_META` no declarado. El render debe usar el "
        "mapping canonical, no `userGoal.replace(/_/g, ' ')`."
    )
    # Cubre los 4 enums del backend (formValidation.js): gain_muscle,
    # lose_fat, maintenance, performance.
    expected_pairs = [
        ("gain_muscle", "Ganar"),
        ("lose_fat", "Perder"),
        ("maintenance", "Mantenimiento"),
        ("performance", "Rendimiento"),
    ]
    for enum_key, es_label_prefix in expected_pairs:
        pattern = rf"{enum_key}\s*:\s*\{{[^}}]*label\s*:\s*['\"]?{es_label_prefix}"
        assert re.search(pattern, src, re.IGNORECASE), (
            f"Mapping para `{enum_key}` no tiene label es-DO empezando "
            f"con `{es_label_prefix}`. Producto es 100% es-DO (CLAUDE.md)."
        )


def test_goal_label_rendered_via_mapping_not_raw_replace():
    """[P3-PROFILE-PLAN-CARD-REDESIGN] El card NO debe renderizar la meta
    via `userGoal.replace(/_/g, ' ')` (renderizaba "gain muscle" capitalizado
    en ingles). Debe usar `_goalMeta.label` del mapping es-DO."""
    src = _read(_SETTINGS_JSX)
    # Localizar el bloque de la seccion 'plan' (~30 lineas alrededor del
    # render del card). Buscar dentro del bloque, no en todo el archivo
    # (otros surfaces de Settings pueden usar replace _ legitimamente).
    section_match = re.search(
        r"activeSection\s*===\s*['\"]plan['\"][\s\S]{0,8000}?(?=activeSection\s*===\s*['\"]subscription['\"])",
        src,
    )
    assert section_match, "No se localizo el bloque de la seccion 'plan'."
    section_src = section_match.group(0)
    assert "userGoal.replace(/_/g" not in section_src, (
        "Render directo `userGoal.replace(/_/g, ' ')` detectado en seccion "
        "Plan & Objetivo -> mostrara 'Gain Muscle' en ingles. Usa "
        "`_goalMeta.label` del mapping `_GOAL_META`."
    )
    assert "_goalMeta.label" in section_src, (
        "Render via `_goalMeta.label` ausente en seccion Plan & Objetivo."
    )
    assert "_goalMeta.Icon" in section_src, (
        "Icon dinamico `<_goalMeta.Icon ... />` ausente en card."
    )


def test_legacy_emerald_gradient_and_shadow_removed():
    """[P3-PROFILE-PLAN-CARD-REDESIGN] El gradient indigo->verde y la sombra
    emerald saturada estan ausentes del bloque del card. El .grid wrapper
    tiene `overflow:hidden` y la sombra de 25px se clipeaba abajo."""
    src = _read(_SETTINGS_JSX)
    section_match = re.search(
        r"activeSection\s*===\s*['\"]plan['\"][\s\S]{0,8000}?(?=activeSection\s*===\s*['\"]subscription['\"])",
        src,
    )
    assert section_match, "No se localizo el bloque de la seccion 'plan'."
    section_src = section_match.group(0)

    # Gradient legacy: `linear-gradient(135deg, var(--primary) 0%, #16a34a 100%)`
    assert "#16a34a" not in section_src, (
        "Gradient legacy indigo->verde (#16a34a) presente en card Plan & "
        "Objetivo. Usa background blanco + border slate-200."
    )
    # Sombra emerald saturada legacy
    assert "rgba(16, 185, 129, 0.4)" not in section_src, (
        "Sombra emerald `rgba(16, 185, 129, 0.4)` presente en card Plan & "
        "Objetivo. Se clipea por overflow:hidden del .grid wrapper en mobile."
    )


def test_cta_uses_minimal_slate_pattern():
    """[P3-PROFILE-PLAN-CARD-REDESIGN] La CTA usa la clase `plan-goal-cta`
    (slate-900 solid + ArrowRight translateX hover), consistente con
    P3-RESTOCK-MINIMAL-CTA del Dashboard."""
    src = _read(_SETTINGS_JSX)
    section_match = re.search(
        r"activeSection\s*===\s*['\"]plan['\"][\s\S]{0,8000}?(?=activeSection\s*===\s*['\"]subscription['\"])",
        src,
    )
    assert section_match
    section_src = section_match.group(0)
    assert 'className="plan-goal-cta"' in section_src, (
        "CTA del card no usa className `plan-goal-cta`. Patron minimal "
        "outline + slate-900 solid esperado."
    )
    assert "<ArrowRight" in section_src, (
        "Icon `ArrowRight` ausente en la CTA. Patron P3-RESTOCK-MINIMAL-CTA "
        "usa flecha con translateX en hover."
    )
    # Hover translateX presente en el <style> block del card
    assert re.search(
        r"\.plan-goal-cta:hover\s+\.plan-goal-arrow\s*\{\s*transform:\s*translateX",
        section_src,
    ), "Microinteraccion `translateX` en hover de la flecha ausente."


def test_prefers_reduced_motion_respected():
    """[P3-PROFILE-PLAN-CARD-REDESIGN] `@media (prefers-reduced-motion)` neutral
    la microinteraccion translateX para a11y (WCAG 2.3.3 baseline)."""
    src = _read(_SETTINGS_JSX)
    section_match = re.search(
        r"activeSection\s*===\s*['\"]plan['\"][\s\S]{0,8000}?(?=activeSection\s*===\s*['\"]subscription['\"])",
        src,
    )
    assert section_match
    section_src = section_match.group(0)
    assert "prefers-reduced-motion" in section_src, (
        "Falta `@media (prefers-reduced-motion: reduce)` para a11y. "
        "Microinteracciones de hover deben respetar la preferencia del SO."
    )


def test_kcal_formatted_for_es_do_locale():
    """[P3-PROFILE-PLAN-CARD-REDESIGN] kcal renderiza con `toLocaleString('es-DO')`
    para separador de miles consistente con el producto (no '2100' raw)."""
    src = _read(_SETTINGS_JSX)
    section_match = re.search(
        r"activeSection\s*===\s*['\"]plan['\"][\s\S]{0,8000}?(?=activeSection\s*===\s*['\"]subscription['\"])",
        src,
    )
    assert section_match
    section_src = section_match.group(0)
    assert "toLocaleString('es-DO')" in section_src, (
        "kcal del card no se formatea con `toLocaleString('es-DO')`. "
        "Para valores >= 1000, el separador de miles mejora legibilidad."
    )
