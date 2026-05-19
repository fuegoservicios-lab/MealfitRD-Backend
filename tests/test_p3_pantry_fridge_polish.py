"""[P3-PANTRY-FRIDGE-POLISH · 2026-05-19] Polish visual del rediseño de
la Nevera tras feedback del usuario sobre el primer round:

1. **"GAVETA DE"** redundante en labels de crispers (la metáfora visual
   con asita + radius pronunciado ya comunica que es una gaveta). Labels
   acortados a 'Frutas' y 'Verduras'.

2. **Manija lateral** se veía plana y simple. Mejorada con:
   - Gradiente metálico cromado (7 stops, no 4)
   - Reflejo vertical central tipo "brillo de cromo"
   - Pivotes superior e inferior con sombras 3D propias
   - Anchura 18→22px para presencia visual

3. **Gaveta izquierda corta** se estiraba a la altura de la derecha
   (hueco vacío al fondo cuando Frutas tenía 3 items vs Verduras con 7).
   Fix: `align-items: start` en `.nevera-drawers-row`.

4. **Panel de control superior** añadido — display LED simulado (`❄ 4°C`),
   rejilla de ventilación y power dot verde pulsante. Da identidad
   "nevera moderna real".

5. **Sombra del piso** bajo el marco (radial-gradient blur'd) — sensación
   de objeto pesado apoyado.

6. **Patitas inferiores** más anchas (28→44px) y con gradiente metálico.

7. **Iconos por zona** con colores semánticos (no solo cyan):
   - Lácteos: #0EA5E9 (cyan)
   - Proteínas: #DC2626 (rojo carne)
   - Listos: #F59E0B (dorado)
   - Puerta: #0891B2 (turquesa oscuro)
   - Frutas: #EC4899 (rosa)
   - Verduras: #16A34A (verde)
   - Alacena: #92400E (ámbar)

Este test ancla las decisiones de polish. El test de layout
(`test_p3_pantry_fridge_layout.py`) cubre la estructura general; este
cubre los detalles visuales que pueden regresar si alguien "limpia"
CSS sin entender por qué cada pieza está ahí.

Por qué parser-based:
    Mismas razones que test_p3_pantry_fridge_layout.py — el polish es
    visual puro, no funcional. Validamos anclas estructurales (existen
    las clases, los styles, los labels) no pixels.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


def _read_pantry() -> str:
    assert _PANTRY_JSX.exists(), f"Pantry.jsx no encontrado en {_PANTRY_JSX}"
    return _PANTRY_JSX.read_text(encoding="utf-8")


def test_p3_pantry_fridge_polish_marker_present():
    """Marker textual presente en JSX — sin él, el slug del marker en
    `_LAST_KNOWN_PFIX` no matchea ningún archivo de test y el cross-link
    de `test_p2_hist_audit_14` falla.
    """
    src = _read_pantry()
    assert "P3-PANTRY-FRIDGE-POLISH" in src, (
        "Marker `P3-PANTRY-FRIDGE-POLISH` no encontrado en Pantry.jsx. "
        "Si reviertes el polish, bumpea `_LAST_KNOWN_PFIX` al P-fix de "
        "reversión y elimina este test."
    )


def test_drawer_labels_have_no_gaveta_prefix():
    """Pedido explícito del usuario 2026-05-19: 'borra el texto gaveta'.
    Los labels de drawers deben ser solo 'Frutas' / 'Verduras' (la
    metáfora visual del crisper ya comunica que es una gaveta).
    """
    src = _read_pantry()
    drawer_lines = re.findall(
        r"\{\s*key:\s*'drawer_\w+'.+?kind:\s*'drawer'.+?\}",
        src,
    )
    assert len(drawer_lines) >= 2, (
        "Esperaba ≥2 definiciones de drawer en ZONE_DEFINITIONS."
    )
    for line in drawer_lines:
        assert "Gaveta de" not in line, (
            f"Label de drawer contiene 'Gaveta de' (redundante con la metáfora "
            f"visual). Línea ofensiva: {line[:120]}"
        )


def test_zone_definitions_have_semantic_colors():
    """Cada zona debe tener un campo `color` distinto. Pre-polish todos
    los iconos eran cyan `#0EA5E9` — la nevera se sentía monocromática.
    """
    src = _read_pantry()
    match = re.search(
        r"const\s+ZONE_DEFINITIONS\s*=\s*\[(.+?)\];",
        src,
        re.DOTALL,
    )
    assert match, "ZONE_DEFINITIONS no parseable"
    block = match.group(1)

    color_decls = re.findall(r"color:\s*'(#[0-9A-Fa-f]{6})'", block)
    assert len(color_decls) >= 7, (
        f"Esperaba ≥7 declaraciones `color` (una por zona), encontré "
        f"{len(color_decls)}: {color_decls}"
    )
    assert len(set(color_decls)) >= 5, (
        f"Solo {len(set(color_decls))} colores únicos — la diferenciación "
        f"cromática se pierde. Esperaba ≥5."
    )


def test_drawers_row_uses_align_items_start():
    """Sin `align-items: start` reaparece el bug del hueco vacío bajo
    la gaveta corta (reportado por user 2026-05-19 con screenshot).
    """
    src = _read_pantry()
    match = re.search(
        r"\.nevera-drawers-row\s*\{([^}]+)\}",
        src,
    )
    assert match, ".nevera-drawers-row no encontrada"
    body = match.group(1)
    assert "align-items: start" in body, (
        "`.nevera-drawers-row` debe declarar `align-items: start` para "
        "que las gavetas tengan altura independiente."
    )


def test_render_passes_zone_color_to_icon():
    """El render del Icon debe pasar `zone.color` desde ZONE_DEFINITIONS,
    NO un hardcoded `#0EA5E9`.
    """
    src = _read_pantry()
    icon_color_refs = re.findall(
        r"<Icon[^>]+style=\{\{\s*color:\s*([^,}]+)",
        src,
    )
    assert len(icon_color_refs) >= 2, (
        f"Esperaba ≥2 referencias `<Icon ... color=...>`, encontré "
        f"{len(icon_color_refs)}"
    )
    for ref in icon_color_refs:
        assert "zone.color" in ref.strip(), (
            f"<Icon> debe usar `zone.color`, no hardcoded. Encontré: "
            f"`{ref.strip()}`"
        )


CONTROL_PANEL_CSS_CLASSES = [
    ".nevera-fridge-control-panel",
    ".nevera-fridge-led-display",
    ".nevera-fridge-led-icon",
    ".nevera-fridge-led-temp",
    ".nevera-fridge-vent",
    ".nevera-fridge-power-dot",
]


@pytest.mark.parametrize("css_class", CONTROL_PANEL_CSS_CLASSES)
def test_control_panel_css_classes_defined(css_class):
    """El panel superior tipo "display de control" se compone de 6
    sub-elementos (panel wrapper + LED display + ventilation grid +
    power dot). Si alguno falta, el panel se rompe visualmente.
    """
    src = _read_pantry()
    pattern = rf"{re.escape(css_class)}\s*[\.,\{{>+~:]"
    assert re.search(pattern, src), (
        f"Clase CSS '{css_class}' del panel de control no declarada."
    )


def test_control_panel_rendered_in_jsx():
    """El panel superior debe estar en el JSX render antes del primer
    estante. Si solo está el CSS pero no se renderiza, el polish no
    se ve.
    """
    src = _read_pantry()
    assert 'className="nevera-fridge-control-panel"' in src, (
        "El JSX no renderiza `<div className='nevera-fridge-control-panel'>`. "
        "Sin el render el CSS queda muerto."
    )
    # Sub-elementos esperados dentro del panel
    assert 'className="nevera-fridge-led-display"' in src, (
        "El LED display no se renderiza dentro del panel de control."
    )
    assert 'className="nevera-fridge-power-dot"' in src, (
        "El power dot (LED verde de encendido) no se renderiza."
    )


def test_floor_shadow_under_fridge():
    """`.nevera-fridge-body::after` debe declarar la sombra del piso —
    sensación de objeto pesado apoyado. Sin ella el marco "flota".
    """
    src = _read_pantry()
    # Buscar el bloque del body::after
    match = re.search(
        r"\.nevera-fridge-body::after\s*\{([^}]+)\}",
        src,
        re.DOTALL,
    )
    assert match, "`.nevera-fridge-body::after` no encontrada"
    body = match.group(1)
    assert "radial-gradient" in body, (
        "Sombra del piso debe usar `radial-gradient` para difuminar "
        "natural. Encontré bloque sin radial-gradient: " + body[:200]
    )
    assert "filter:" in body and "blur" in body, (
        "Sombra del piso debe aplicar `filter: blur(...)` para look "
        "suave. Sin blur la sombra se ve dura/artificial."
    )
