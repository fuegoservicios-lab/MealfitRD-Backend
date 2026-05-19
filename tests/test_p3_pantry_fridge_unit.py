"""[P3-PANTRY-FRIDGE-UNIT · 2026-05-19] Refactor estructural del rediseño
de la Nevera tras feedback "que la parte de arriba se parezca también a
una nevera y los bordes fíjate también":

**Problema pre-fix:**
1. El header con búsqueda + botones "Borrar Todos/Añadir" flotaba como
   panel independiente (fondo cyan translúcido + border cyan).
2. El cuerpo `.nevera-fridge-body` tenía SU PROPIO `.nevera-fridge-interior-wrap`
   con border cyan adicional — **doble marco anidado** dentro del
   `.nevera-page-frame` exterior que también tenía border cyan.
3. La alacena (granos secos) quedaba atrapada DENTRO del marco cyan
   exterior, contradiciendo la decisión de producto "alacena externa,
   lo seco no va en nevera".

**Cambios estructurales:**

1. **Nuevo wrapper `.nevera-page-outer`** que envuelve toda la página
   (sin border, solo padding + bg). El estilo "marco de nevera completa"
   ahora vive exclusivamente en `.nevera-page-frame` (interior).

2. **`.nevera-page-frame` = único marco visual de nevera** — bordes
   metálicos grayscale (no cyan), sombras 3D fuertes que dan volumen
   de electrodoméstico, `overflow: hidden` para que el header recto
   se clipee a las esquinas redondeadas.

3. **Header transformado a "puerta del freezer":**
   - Fondo metálico perlado multi-layer (white → slate-50 → slate-100 →
     slate-200) con radial gradient de luz desde arriba-centro.
   - Patrón sutil de "acero cepillado" via `repeating-linear-gradient`.
   - `.nevera-brand-label` centrado arriba con LED verde pulsante —
     etiqueta tipo "logo discreto de electrodoméstico".
   - `.nevera-header-handle` propia (manija del freezer) alineada
     verticalmente con la manija del cuerpo — visual de "nevera de
     dos puertas" top-mounted.
   - Border-bottom groove (línea oscura + línea clara) marca la unión
     entre puerta superior y puerta inferior.

4. **`.nevera-fridge-interior-wrap` pierde border/radius** — el page-frame
   exterior es el marco real. Solo conserva fondo cyan claro tipo
   "interior frío iluminado" + sombras inset.

5. **Alacena externa** — el JSX renderiza `.nevera-pantry-section` FUERA
   del `.nevera-page-frame`, como sibling. La unidad de nevera (page-frame)
   contiene header + body; la alacena vive aparte.

6. **Body alineado con header** — el wrapper del listado pierde
   `padding: 0 1.5rem`, el body llena el ancho del page-frame (mismo
   ancho que el header).

Tests parser-based — el cambio es 100% estructural+visual:
    1. Marker presente
    2. page-outer ≠ page-frame (estructura wrapper externo + marco interno)
    3. Page-frame con border metálico grayscale (no cyan)
    4. Brand label rendered en JSX dentro del header
    5. Header handle rendered (manija propia del freezer)
    6. Interior-wrap SIN border ni border-radius propios
    7. Alacena renderizada como sibling del page-frame (no dentro)
    8. Wrapper del listado SIN padding lateral (alineado con header)
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


def test_p3_pantry_fridge_unit_marker_present():
    """Marker textual presente — cross-link con `test_p2_hist_audit_14`
    requiere que el slug del marker matchee este archivo.
    """
    src = _read_pantry()
    assert "P3-PANTRY-FRIDGE-UNIT" in src, (
        "Marker `P3-PANTRY-FRIDGE-UNIT` no encontrado en Pantry.jsx. "
        "Si reviertes la refactor estructural, bumpea `_LAST_KNOWN_PFIX` "
        "al P-fix de reversión y elimina este test."
    )


def test_page_outer_and_frame_are_separate():
    """`.nevera-page-outer` debe ser el wrapper exterior (sin border),
    `.nevera-page-frame` debe ser el marco interno (con border). Pre-fix
    solo existía `.nevera-page-frame` que actuaba como ambas cosas a la vez,
    atrapando la alacena dentro del marco.
    """
    src = _read_pantry()
    assert 'className="nevera-page-outer"' in src, (
        "Wrapper `.nevera-page-outer` no encontrado en JSX. Necesario para "
        "que la alacena pueda vivir como sibling del page-frame."
    )
    assert 'className="nevera-page-frame"' in src, (
        "Marco `.nevera-page-frame` interior no encontrado."
    )

    # CSS de page-outer DEBE existir
    assert re.search(r"\.nevera-page-outer\s*\{", src), (
        "CSS de `.nevera-page-outer` no declarado."
    )


def test_page_frame_uses_metallic_grayscale_border():
    """Pre-fix el page-frame tenía `border: ... rgba(125, 211, 252, ...)`
    (cyan). Ahora debe usar grayscale (rgba con valores R≈G≈B en gris)
    para coherencia con el look metálico de electrodoméstico.
    """
    src = _read_pantry()
    # Aislar el bloque del page-frame
    match = re.search(
        r"\.nevera-page-frame\s*\{([^}]+)\}",
        src,
        re.DOTALL,
    )
    assert match, "Bloque `.nevera-page-frame` no parseable"
    block = match.group(1)

    # El border principal NO debe ser cyan (125, 211, 252) ni (56, 189, 248)
    cyan_patterns = [
        r"border:[^;]*rgba\(125,\s*211,\s*252",
        r"border:[^;]*rgba\(56,\s*189,\s*248",
    ]
    for pattern in cyan_patterns:
        assert not re.search(pattern, block), (
            f"El border del page-frame sigue usando cyan ({pattern}). "
            f"Debe ser grayscale (slate-300, slate-400) para coherencia con "
            f"el look metálico."
        )

    # Debe haber al menos una declaración de border grayscale
    grayscale_pattern = r"border[^:]*:[^;]*rgba\((1[0-9]{2}|2[0-2][0-9])"
    assert re.search(grayscale_pattern, block), (
        "No se detecta border con paleta grayscale en page-frame."
    )


def test_brand_label_rendered_in_header():
    """La etiqueta de marca debe estar en el JSX como hijo del header —
    elemento decorativo que da identidad de "electrodoméstico marca X".
    """
    src = _read_pantry()
    assert 'className="nevera-brand-label"' in src, (
        "`<div className='nevera-brand-label'>` no se renderiza en JSX. "
        "Sin él el header no tiene la identidad de marca de electrodoméstico."
    )
    assert 'className="nevera-brand-dot"' in src, (
        "El LED verde del brand label (`.nevera-brand-dot`) no se renderiza."
    )
    # Debe haber CSS para ambos
    for css_class in [".nevera-brand-label", ".nevera-brand-dot"]:
        pattern = rf"{re.escape(css_class)}\s*\{{"
        assert re.search(pattern, src), (
            f"CSS de `{css_class}` no declarado."
        )


def test_header_has_own_freezer_handle():
    """Header debe tener su propia manija (manija del freezer) alineada
    con la manija del body — visual de "nevera de dos puertas".
    """
    src = _read_pantry()
    assert 'className="nevera-header-handle"' in src, (
        "`<div className='nevera-header-handle'>` no se renderiza. "
        "Sin él falta la manija del freezer (puerta superior)."
    )
    assert re.search(r"\.nevera-header-handle\s*\{", src), (
        "CSS de `.nevera-header-handle` no declarado."
    )


def test_interior_wrap_has_no_duplicate_border():
    """`.nevera-fridge-interior-wrap` NO debe tener `border:` ni
    `border-radius:` propios — el page-frame externo es el marco. Si
    reaparecen, vuelve el bug de doble marco anidado.
    """
    src = _read_pantry()
    match = re.search(
        r"\.nevera-fridge-interior-wrap\s*\{([^}]+)\}",
        src,
        re.DOTALL,
    )
    assert match, "Bloque `.nevera-fridge-interior-wrap` no parseable"
    block = match.group(1)

    # No debe haber `border:` con valor (declaración de border solid/dashed/etc)
    # Pero box-shadow inset SÍ está permitido.
    border_decl = re.search(
        r"(?<!-)border\s*:\s*[^;]+(solid|dashed|dotted)",
        block,
    )
    assert not border_decl, (
        f"`.nevera-fridge-interior-wrap` tiene declaración de border: "
        f"{border_decl.group(0) if border_decl else '?'}. Debe pasar a "
        f"NO border (el page-frame externo es el marco). Si la traes "
        f"de vuelta, vuelve el doble marco anidado."
    )

    border_radius = re.search(r"border-radius\s*:", block)
    assert not border_radius, (
        "`.nevera-fridge-interior-wrap` tiene `border-radius`. Pre-fix esto "
        "creaba doble esquina redondeada (page-frame outside + wrap inside). "
        "Debe quedar sin radius."
    )


def test_pantry_section_rendered_outside_page_frame():
    """La sección de alacena debe estar en el JSX FUERA del cierre del
    `.nevera-page-frame`. Pre-fix estaba atrapada dentro junto al fridge
    body, contradiciendo "alacena externa".
    """
    src = _read_pantry()

    # Buscar el cierre del page-frame y la posición del pantry-section
    # Como heurística parser-based: el comment del cierre tiene `/nevera-page-frame`
    closing_pattern = re.search(
        r"\{/\*\s*/nevera-page-frame\s*\*/\}",
        src,
    )
    assert closing_pattern, (
        "Comment `{/* /nevera-page-frame */}` no encontrado — sin él no "
        "podemos saber dónde cierra el page-frame. Mantén ese comentario "
        "para que este test pueda anclar la posición del cierre."
    )

    # La pantry-section debe estar DESPUÉS del cierre del page-frame
    pantry_render = src.find('className="nevera-pantry-section"')
    closing_pos = closing_pattern.start()

    assert pantry_render > 0, "`.nevera-pantry-section` no se renderiza en JSX."
    assert pantry_render > closing_pos, (
        f"La sección de alacena (pos {pantry_render}) se renderiza ANTES "
        f"del cierre del page-frame (pos {closing_pos}) — está atrapada "
        f"dentro del marco de nevera. Debe ser sibling, NO hijo."
    )


def test_listado_wrapper_has_no_lateral_padding():
    """El wrapper del listado de inventario debe usar `padding: '0'` (no
    `padding: '0 1.5rem'`) para que el body llene el ancho del page-frame
    alineado con el header arriba.
    """
    src = _read_pantry()

    # Buscar la línea que abre el div del listado
    # Pre-fix: <div style={{ padding: '0 1.5rem' }}>
    # Post-fix: <div style={{ padding: '0' }}>
    legacy = re.search(
        r"\{/\* Listado de Inventario [^*]+\*/\}\s*(?:\{/\*[^*]*\*/\}\s*)*<div style=\{\{\s*padding:\s*'0 1\.5rem'",
        src,
    )
    assert not legacy, (
        "El wrapper del listado todavía tiene `padding: '0 1.5rem'` — body "
        "queda más estrecho que el header. Cambia a `padding: '0'`."
    )
