"""[P3-PANTRY-REDESIGN-ANCHORS · 2026-07-28] Consolidación tras la muerte de la nevera visual.

## Qué pasó

El diseño esquiomórfico de la Nevera (`.nevera-fridge-body`, tirador, patas, panel LED…) fue
**reemplazado a propósito** por `cafb430 P3-PANTRY-FRIDGE-REDESIGN` (2026-06-24): sidebar de
escritorio + layout móvil dedicado. Cuatro archivos de test siguieron anclando aquel diseño —
**45 rojos durante un mes** que no protegían nada, incluidos sus propios markers (ya no están en
`Pantry.jsx`):

    test_p3_pantry_fridge_layout.py   20 rojos / 12 verdes
    test_p3_pantry_fridge_polish.py   10 rojos /  3 verdes
    test_p3_pantry_fridge_unit.py      7 rojos /  1 verde
    test_p3_pantry_mobile_polish.py    7 rojos /  0 verdes

Este archivo los reemplaza: **conserva lo que sobrevivió al rediseño** (la lógica de zonas, que
es dominio, no estética) y deja constancia de la supersesión. Los cuerpos originales viven en
git (`tests/test_p3_pantry_fridge_*.py` hasta este commit).

## Lo que sigue vivo y por qué importa

- `CATEGORY_TO_ZONE`: sin él, el render no mapea `master_ingredients.category` a una zona y los
  items caen al fallback `pantry` — UX rota en silencio.
- `ZONE_DEFINITIONS` + colores semánticos: pre-polish todo era cyan monocromático.
- `inventoryByZone` (useMemo): si vuelve el sort alfabético legacy, la agrupación por zonas
  desaparece sin que nada falle.
- Marker `P3-PANTRY-FRIDGE-REDESIGN` en el código: ancla del diseño VIGENTE.

tooltip-anchor: P3-PANTRY-REDESIGN-ANCHORS
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"

# Representantes mínimos por zona (heredado de fridge_layout; el dominio no cambió con la UI).
ZONE_REPRESENTATIVES = {
    "shelf_dairy":    ["LÁCTEOS", "QUESOS", "HUEVOS"],
    "shelf_proteins": ["POLLO", "PESCADO", "CARNES ROJAS"],
    "shelf_ready":    ["PANADERÍA", "DULCES", "FRUTOS SECOS"],
    "door":           ["BEBIDAS", "ACEITES", "CONDIMENTOS"],
    "drawer_fruits":  ["FRUTAS"],
    "drawer_veggies": ["VEGETALES", "VERDURAS", "HIERBAS"],
    "pantry":         ["CEREALES Y GRANOS", "DESPENSA", "LEGUMBRES", "ESPECIAS"],
}


def _read_pantry() -> str:
    assert _PANTRY_JSX.exists(), f"Pantry.jsx no encontrado en {_PANTRY_JSX}"
    return _PANTRY_JSX.read_text(encoding="utf-8")


# ───────────── 1. el diseño vigente está anclado ─────────────

def test_marker_del_rediseno_vigente_presente():
    """Si el marker desaparece, o hubo OTRO rediseño (→ reanclar este archivo) o alguien
    borró historia. En ambos casos hay que mirar, no ignorar."""
    assert "P3-PANTRY-FRIDGE-REDESIGN" in _read_pantry()


def test_el_diseno_muerto_no_resucita_a_medias():
    """La estética vieja (clases nevera-fridge-*) se retiró ENTERA en cafb430. Si reaparece una
    clase suelta, es un merge zombi — media nevera vieja sobre el sidebar nuevo."""
    assert "nevera-fridge" not in _read_pantry()


# ───────────── 2. la lógica de zonas sobrevivió (dominio, no estética) ─────────────

def test_category_to_zone_mapping_exists():
    src = _read_pantry()
    assert re.search(r"const\s+CATEGORY_TO_ZONE\s*=\s*\{", src), (
        "Constante `CATEGORY_TO_ZONE` no declarada en Pantry.jsx. Sin ella, el render no puede "
        "mapear master_ingredients.category a una zona."
    )


@pytest.mark.parametrize("zone,categories", list(ZONE_REPRESENTATIVES.items()))
def test_each_zone_has_representative_categories(zone, categories):
    """Si falta una clave (ej. alguien elimina 'POLLO' sin pensar qué zona la cubre), esos items
    caen al fallback `pantry` y aparecen fuera de su sitio — UX rota silenciosamente."""
    src = _read_pantry()
    match = re.search(r"const\s+CATEGORY_TO_ZONE\s*=\s*\{(.+?)\};", src, re.DOTALL)
    assert match, "Bloque CATEGORY_TO_ZONE no parseable"
    block = match.group(1)
    presentes = [c for c in categories if f"'{c}'" in block or f'"{c}"' in block]
    assert presentes, (
        f"Zona {zone}: ninguna de sus categorías representativas {categories} está en el mapping"
    )


def test_zone_definitions_have_semantic_colors():
    """Cada zona con `color` propio y distinto — pre-polish todo era cyan monocromático."""
    src = _read_pantry()
    match = re.search(r"const\s+ZONE_DEFINITIONS\s*=\s*\[(.+?)\];", src, re.DOTALL)
    assert match, "ZONE_DEFINITIONS no parseable"
    colores = re.findall(r"color:\s*'(#[0-9A-Fa-f]{6})'", match.group(1))
    assert len(colores) >= 3, f"muy pocas zonas con color: {colores}"
    assert len(set(colores)) >= 3, f"colores repetidos (vuelve el monocromo): {colores}"


def test_render_uses_inventory_by_zone_not_legacy_sort():
    """El render agrupa por `inventoryByZone` (useMemo). Si vuelve
    `Object.keys(filteredInventory).sort()`, el rediseño deja de surtir efecto sin fallar."""
    src = _read_pantry()
    assert re.search(r"const\s+inventoryByZone\s*=\s*useMemo", src), (
        "useMemo `inventoryByZone` no encontrado — sin él no hay agrupación por zona"
    )
    assert not re.search(r"Object\.keys\(filteredInventory\)\.sort\(\)", src), (
        "volvió el sort alfabético legacy: los items se renderizan sin estructura de zonas"
    )
