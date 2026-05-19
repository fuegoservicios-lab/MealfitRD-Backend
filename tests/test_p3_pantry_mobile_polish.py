"""[P3-PANTRY-MOBILE-POLISH · 2026-05-19] Compactación del layout mobile
de la Nevera tras feedback usuario con screenshot iPhone 12 Pro (390x844):
"mejoremos el diseño para móviles ya que en pc me gustó el resultado".

**Issues pre-fix visibles en el screenshot:**
1. **Gap visible entre header y body** — el header (puerta del freezer) y el
   cuerpo terminaban con un espacio en blanco entre el search input y el
   control panel "4°C", rompiendo la metáfora de "una nevera unificada".
2. **Botón "Añadir Alimento" en 2 líneas** — texto cortado por falta de
   ancho en mobile angosto.
3. **Header demasiado alto** — padding vertical generoso + título +
   Snowflake icon + pills "Solo lo que tienes" + "27 items" + botones
   apilados → ~50% del viewport ocupado solo por el header.
4. **Control panel "4°C" muy grande** — el LED display ocupaba todo el
   ancho con padding sobrado.
5. **Cards con mucho padding interno** — espacio desperdiciado en mobile.

**Cambios aplicados:**

- **Header**: padding top 1.6rem→1.4rem, bottom 1.25rem→0.75rem. Título
  2rem→1.7rem. Snowflake icon 30→24. Brand label más arriba (top 0.55rem→0.4rem)
  y más pequeño (font-size 0.55rem, letter-spacing 0.12em).

- **Botones**: `flex: 1` aplicado a AMBOS (no solo add-btn). Padding
  0.75rem 1.4rem → 0.6rem 0.8rem. Font-size 0.85rem. `white-space: nowrap`
  para que el texto NO se rompa en 2 líneas.

- **Búsqueda**: padding reducido + font-size 0.95rem.

- **Body**: padding-top del interior-wrap 1rem→0.75rem para cerrar gap
  con header. Margin-bottom del body 1.5rem→1rem.

- **Control panel 4°C**: padding 0.4rem 0.6rem → 0.35rem 0.55rem. Border-radius
  0.8rem → 0.6rem. Power dot 8px→7px. LED display padding reducido.

- **Zonas y cards**: zone padding 1.1rem→0.85rem, zone-header font 1.05rem→0.78rem,
  item-card padding 1.2rem→0.85rem 0.9rem, item h3 1.05rem→0.95rem.

- **Drawers** (gavetas): gap 0.9rem→0.7rem, padding interno reducido.

- **Alacena**: padding 1.1rem→1rem, subtitle font 0.82rem→0.75rem.

- **Extra-small breakpoint (≤380px)** — nuevo, para iPhone SE y similares:
  header padding 1.4rem→1.3rem, título 1.7rem→1.5rem, botones 0.85rem→0.78rem.

Por qué parser-based:
    El polish es 100% CSS dentro de `@media (max-width: 640px)` y
    `@media (max-width: 380px)`. Validamos anclas estructurales:
    1. Marker presente
    2. Bloque mobile contiene las nuevas reglas (header padding compacto,
       botones flex:1 con nowrap, body padding-top reducido, control panel
       compactado).
    3. Existe segundo breakpoint `@media (max-width: 380px)` (extra-small).
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


def _extract_media_block(src: str, max_width_px: int) -> str:
    """Extrae el contenido de un bloque `@media (max-width: Npx) { ... }`
    rastreando paréntesis para soportar reglas anidadas.
    """
    needle = f"@media (max-width: {max_width_px}px)"
    start = src.find(needle)
    if start < 0:
        return ""
    # Encontrar el `{` que abre y rastrear el `}` que cierra
    brace_start = src.find("{", start)
    if brace_start < 0:
        return ""
    depth = 0
    i = brace_start
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[brace_start + 1:i]
        i += 1
    return ""


def test_p3_pantry_mobile_polish_marker_present():
    """Marker textual presente — cross-link con `test_p2_hist_audit_14`."""
    src = _read_pantry()
    assert "P3-PANTRY-MOBILE-POLISH" in src, (
        "Marker `P3-PANTRY-MOBILE-POLISH` no encontrado en Pantry.jsx."
    )


def test_mobile_breakpoint_exists():
    """Bloque `@media (max-width: 640px)` debe existir con contenido."""
    src = _read_pantry()
    body = _extract_media_block(src, 640)
    assert body.strip(), "Bloque `@media (max-width: 640px)` vacío o no encontrado."


def test_mobile_header_padding_compact():
    """Padding del header mobile reducido para evitar header gigante.
    Pre-fix: `padding: 1.6rem 2.2rem 1.25rem 1rem`. Post-fix: ≤ 1.4rem top,
    ≤ 0.85rem bottom.
    """
    src = _read_pantry()
    body = _extract_media_block(src, 640)
    # Match: .nevera-header { ... padding: <top> <right> <bottom> <left> ... }
    match = re.search(
        r"\.nevera-header\s*\{\s*padding:\s*([\d.]+)rem\s+([\d.]+)rem\s+([\d.]+)rem\s+([\d.]+)rem",
        body,
    )
    assert match, (
        "No se encontró declaración `.nevera-header { padding: ... }` "
        "en el bloque mobile."
    )
    top, _, bottom, _ = map(float, match.groups())
    assert top <= 1.4, (
        f"Mobile header padding-top = {top}rem (debe ser ≤ 1.4rem). "
        f"Pre-fix era 1.6rem que dejaba header gigante."
    )
    assert bottom <= 0.85, (
        f"Mobile header padding-bottom = {bottom}rem (debe ser ≤ 0.85rem). "
        f"Pre-fix era 1.25rem que dejaba gap visible al body."
    )


def test_mobile_buttons_use_flex_and_nowrap():
    """Botones Borrar Todos + Añadir Alimento deben usar `flex: 1` ambos
    y `white-space: nowrap` para que NO se rompan en 2 líneas. Pre-fix
    solo `add-btn` tenía flex:1 y "Añadir Alimento" se cortaba.
    """
    src = _read_pantry()
    body = _extract_media_block(src, 640)
    # Buscar bloque que aplique a ambos botones
    match = re.search(
        r"\.nevera-add-btn,\s*\.nevera-delete-all-btn\s*\{([^}]+)\}",
        body,
    )
    assert match, (
        "No se encontró regla `.nevera-add-btn, .nevera-delete-all-btn { ... }` "
        "aplicada a AMBOS botones. Sin esto el delete-btn queda con padding "
        "diferente al add-btn y se ven inconsistentes."
    )
    block = match.group(1)
    assert "flex: 1" in block, (
        "Los botones del header mobile no usan `flex: 1`. Sin esto, los "
        "botones quedan a tamaño natural y se rompen en 2 líneas."
    )
    assert "white-space: nowrap" in block, (
        "Los botones del header mobile no usan `white-space: nowrap`. Sin "
        "esto, 'Añadir Alimento' se corta a 2 líneas en pantallas <400px."
    )


def test_mobile_interior_wrap_padding_top_reduced():
    """`.nevera-fridge-interior-wrap` con padding-top reducido en mobile
    para cerrar el gap con el header. Pre-fix `1rem`; post-fix ≤ 0.85rem.
    """
    src = _read_pantry()
    body = _extract_media_block(src, 640)
    # Match relajado: aislamos el bloque de `.nevera-fridge-interior-wrap`
    # (incluyendo comentarios) y dentro buscamos la primera línea de padding.
    block_match = re.search(
        r"\.nevera-fridge-interior-wrap\s*\{([^}]+)\}",
        body,
    )
    assert block_match, (
        "Regla `.nevera-fridge-interior-wrap` no encontrada en bloque mobile."
    )
    block = block_match.group(1)
    padding_match = re.search(r"padding:\s*([\d.]+)rem", block)
    assert padding_match, (
        "No se declara `padding:` dentro de `.nevera-fridge-interior-wrap` "
        "del bloque mobile. Debe declararse para cerrar el gap header-body."
    )
    padding_top = float(padding_match.group(1))
    assert padding_top <= 0.85, (
        f"Mobile interior-wrap padding-top = {padding_top}rem (debe ser "
        f"≤ 0.85rem). Pre-fix era 1rem y dejaba gap visible con el header."
    )


def test_mobile_control_panel_compactado():
    """Panel de control LED 4°C debe estar compactado en mobile —
    padding y border-radius reducidos.
    """
    src = _read_pantry()
    body = _extract_media_block(src, 640)
    match = re.search(
        r"\.nevera-fridge-control-panel\s*\{([^}]+)\}",
        body,
    )
    assert match, (
        "No se encontró regla mobile de `.nevera-fridge-control-panel`. "
        "Sin esto el control panel ocupa demasiado espacio vertical."
    )
    block = match.group(1)
    # Debe tener padding declarado más compacto que el desktop (0.5rem 0.8rem)
    padding_match = re.search(r"padding:\s*([\d.]+)rem\s+([\d.]+)rem", block)
    assert padding_match, "Control panel mobile sin padding declarado."
    pad_v, pad_h = map(float, padding_match.groups())
    assert pad_v <= 0.4 and pad_h <= 0.6, (
        f"Control panel mobile padding ({pad_v}rem {pad_h}rem) no es "
        f"suficientemente compacto. Esperado ≤ 0.4rem 0.6rem."
    )


def test_extra_small_breakpoint_exists():
    """Nuevo breakpoint para pantallas tipo iPhone SE (≤380px) — header
    y botones aún más compactos.
    """
    src = _read_pantry()
    body = _extract_media_block(src, 380)
    assert body.strip(), (
        "Bloque `@media (max-width: 380px)` no encontrado. Sin este "
        "breakpoint extra-small, iPhone SE y similares ven el layout "
        "de 640px que sigue siendo demasiado para 360-380px."
    )
    # Debe tener al menos reglas para header padding y title font-size
    assert ".nevera-header" in body, (
        "Bloque ≤380px sin regla para `.nevera-header`."
    )
    assert ".nevera-title" in body, (
        "Bloque ≤380px sin regla para `.nevera-title`."
    )
