"""[P3-DEPLETED-CARD-VERTICAL · 2026-05-22] Test del rediseño VERTICAL de
las cards "Agotados" en Pantry.jsx (tercera iteración del mismo día,
reemplazando el layout horizontal de P3-DEPLETED-CARD-BREATHE que el
user rechazó por seguir viéndose desbalanceado).

Pre-fix (P3-DEPLETED-CARD-BREATHE v2):
  - Layout horizontal 2-col (info | acciones).
  - Botón Reponer flotaba TOP-derecha mientras X quedaba BOTTOM-derecha,
    creando ~40px de espacio vacío vertical entre ellos.
  - User: "sigue viéndose igual lo de agotados visualmente".

Fix (P3-DEPLETED-CARD-VERTICAL):
  - Layout vertical clásico de card con CTA principal abajo.
  - Top row: badge AGOTADO (left) + X dismiss (right) — secondary actions.
  - Middle: nombre tachado + "Tenías: X unidad" — info.
  - Bottom: Reponer FULL-WIDTH como CTA principal centrado.
  - Patrón e-commerce / material card; jerarquía visual estándar.

Marker bumped tooltip-anchor: el archivo de test mantiene el slug
`p3_depleted_card_breathe` por compatibilidad con la convención
P2-HIST-AUDIT-14 (un test file por iteración del fix); el anchor del
marker activo es P3-DEPLETED-CARD-VERTICAL.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


@pytest.fixture(scope="module")
def src() -> str:
    return _PANTRY_JSX.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — CSS del card: layout vertical
# ===========================================================================

def test_card_uses_flex_column(src: str):
    """El card debe ser `flex-direction: column` (no horizontal split)."""
    card_re = re.compile(
        r"\.nevera-depleted-card\s*\{([^}]+)\}",
        re.DOTALL,
    )
    m = card_re.search(src)
    assert m is not None, "regla `.nevera-depleted-card` no encontrada."
    body = m.group(1)
    assert "flex-direction: column" in body, (
        "P3-DEPLETED-CARD-VERTICAL regresión: card volvió a layout horizontal. "
        "El user rechazó el flex row porque botones Reponer+X quedaban "
        "verticalmente separados con espacio vacío entre ellos."
    )


def test_card_has_breathing_padding(src: str):
    """Padding ≥1.2rem para que el contenido no quede pegado al borde."""
    card_re = re.compile(
        r"\.nevera-depleted-card\s*\{([^}]+)\}",
        re.DOTALL,
    )
    m = card_re.search(src)
    assert m is not None
    body = m.group(1)
    padding_match = re.search(r"padding:\s*([^;]+);", body)
    assert padding_match is not None
    nums = [float(m) for m in re.findall(r"(\d+\.?\d*)rem", padding_match.group(1))]
    assert nums and min(nums) >= 1.2, (
        f"P3-DEPLETED-CARD-VERTICAL regresión: padding mínimo {min(nums)}rem "
        f"<1.2. El contenido vuelve a verse pegado al borde."
    )


def test_top_row_subclass_present(src: str):
    """Subclase `.nevera-depleted-card__top` para el row badge+dismiss."""
    css = src
    assert ".nevera-depleted-card__top" in css, (
        "P3-DEPLETED-CARD-VERTICAL regresión: subclase `__top` removida. "
        "Esa clase es el row arriba que contiene badge AGOTADO + X dismiss."
    )
    assert re.search(
        r"\.nevera-depleted-card__top\s*\{[^}]*justify-content:\s*space-between",
        css,
        re.DOTALL,
    ), (
        "P3-DEPLETED-CARD-VERTICAL: `__top` debe usar "
        "`justify-content: space-between` para alinear badge izq + X der."
    )


def test_info_subclass_present(src: str):
    """Subclase `.nevera-depleted-card__info` para nombre + tenías."""
    assert ".nevera-depleted-card__info" in src, (
        "P3-DEPLETED-CARD-VERTICAL regresión: subclase `__info` removida."
    )


# ===========================================================================
# Sección 2 — CSS del botón Reponer: full-width CTA
# ===========================================================================

def test_restore_button_full_width(src: str):
    """`width: 100%` en `.nevera-restore-btn` — CTA principal full-width."""
    btn_re = re.compile(
        r"\.nevera-restore-btn\s*\{([^}]+)\}",
        re.DOTALL,
    )
    m = btn_re.search(src)
    assert m is not None
    body = m.group(1)
    assert re.search(r"width:\s*100%", body), (
        "P3-DEPLETED-CARD-VERTICAL regresión: botón Reponer no es full-width. "
        "Pre-fix era inline pill pequeño top-derecha — visualmente "
        "desbalanceado. Debe ser CTA principal width:100% abajo."
    )
    assert "justify-content: center" in body, (
        "P3-DEPLETED-CARD-VERTICAL regresión: botón Reponer perdió "
        "`justify-content: center`. Texto+icono deben centrarse dentro "
        "del botón full-width."
    )


# ===========================================================================
# Sección 3 — JSX usa la nueva estructura
# ===========================================================================

def test_jsx_uses_top_row_class(src: str):
    """JSX debe usar `nevera-depleted-card__top` para el row badge+dismiss."""
    map_idx = src.find("visibleDepletedItems.map")
    block = src[map_idx:map_idx + 3000]
    assert "nevera-depleted-card__top" in block, (
        "P3-DEPLETED-CARD-VERTICAL regresión: JSX no usa la clase `__top`. "
        "Sin esa estructura, el layout horizontal previo regresa."
    )


def test_jsx_dismiss_button_in_top_row(src: str):
    """El botón X dismiss vive dentro del `__top` row (no separado abajo)."""
    map_idx = src.find("visibleDepletedItems.map")
    block = src[map_idx:map_idx + 3000]
    # Buscar el bloque __top y verificar que contiene el botón dismiss.
    top_block_match = re.search(
        r'nevera-depleted-card__top"[\s\S]*?</div>',
        block,
    )
    assert top_block_match is not None, (
        "P3-DEPLETED-CARD-VERTICAL: bloque `__top` no encontrado en JSX."
    )
    top_body = top_block_match.group(0)
    assert "handleDismissDepleted" in top_body, (
        "P3-DEPLETED-CARD-VERTICAL regresión: botón dismiss (X) NO está "
        "dentro del row `__top`. Debe estar arriba-derecha junto al badge."
    )


def test_jsx_restore_button_is_sibling_after_info(src: str):
    """El botón Reponer es sibling del `__info` (no nested), abajo de la card."""
    map_idx = src.find("visibleDepletedItems.map")
    block = src[map_idx:map_idx + 3000]
    info_idx = block.find("nevera-depleted-card__info")
    restore_idx = block.find("handleRestoreDepleted")
    assert info_idx >= 0 and restore_idx >= 0
    assert restore_idx > info_idx, (
        "P3-DEPLETED-CARD-VERTICAL regresión: botón Reponer aparece ANTES "
        "del bloque info en el JSX. Layout vertical espera info middle + "
        "CTA abajo."
    )


def test_grid_minmax_reasonable(src: str):
    """Grid minmax 240-320px — layout vertical ocupa menos ancho, podemos
    apretar más cards por row."""
    map_idx = src.find("visibleDepletedItems.map")
    pre_block = src[max(0, map_idx - 1500):map_idx]
    minmax_matches = list(re.finditer(r"minmax\(\s*(\d+)px\s*,", pre_block))
    assert minmax_matches
    min_px = int(minmax_matches[-1].group(1))
    assert 240 <= min_px <= 340, (
        f"P3-DEPLETED-CARD-VERTICAL: grid minmax={min_px}px fuera de "
        f"rango [240, 340]. Layout vertical permite cards más estrechas "
        f"sin apretar el contenido (Reponer full-width fuerza la card a "
        f"ser al menos ~220px utilizables)."
    )


def test_marker_present_as_tooltip_anchor(src: str):
    """Marker P3-DEPLETED-CARD-VERTICAL presente como anchor."""
    assert "P3-DEPLETED-CARD-VERTICAL" in src, (
        "Anchor del fix actual ausente. Sin él los lectores futuros no "
        "tienen contexto del cambio de layout horizontal → vertical."
    )
