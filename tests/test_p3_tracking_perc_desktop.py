"""[P3-TRACKING-PERC-DESKTOP . 2026-05-20] Test anti-regresion del % blanco
dentro del fill aplicado UNIVERSALMENTE (desktop + mobile) en el card
"Progreso en Tiempo Real".

Iteracion del dia (4to+ intento sobre el % display):
    1. P3-TRACKING-PERC-INLINE (rechazado) — pill inline con valor.
    2. P3-TRACKING-PERC-MOBILE-HIDE — pill oculta solo en mobile.
    3. P3-TRACKING-BAR-INLINE-PERC — % blanco dentro del fill, solo mobile.
    4. P3-TRACKING-PERC-DESKTOP (este) — el mismo battery style ahora aplica
       para desktop. Las reglas se movieron de @media a base.

Layout final TODOS los viewports:
    [icon] Label             N / G unit
           ##########50%-------------

La pill `.percRow` queda oculta globalmente (preservada en JSX como
defensa contra revert, pero CSS la oculta en base).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACKING_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.jsx"
_TRACKING_CSS = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.module.css"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_css_comments(src: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", src)


def _extract_media_block(css: str, max_width: str) -> str:
    """Extrae el body del @media (max-width: <max_width>px). Retorna '' si no existe."""
    match = re.search(
        rf"@media\s*\(\s*max-width:\s*{re.escape(max_width)}px\s*\)\s*\{{([\s\S]*?)\n\}}",
        css,
    )
    return match.group(1) if match else ""


def _extract_base_css(css: str) -> str:
    """Retorna el CSS fuera de cualquier @media block."""
    return re.sub(r"@media[\s\S]*?\n\}", "", css)


def test_marker_present_as_tooltip_anchor():
    """[P3-TRACKING-PERC-DESKTOP] Marker presente en CSS."""
    css = _read(_TRACKING_CSS)
    assert "P3-TRACKING-PERC-DESKTOP" in css, (
        "Marker `P3-TRACKING-PERC-DESKTOP` ausente en "
        "TrackingProgress.module.css. Si quieres revertir, primero "
        "remueve este test."
    )


def test_fill_perc_span_rendered_inside_track():
    """[P3-TRACKING-PERC-DESKTOP · 2026-05-20] El <span .fillPerc> existe en JSX.
    [P3-TRACKING-PERC-NARROW-FIX · 2026-05-22] Movido de child del `.fill` a
    child del `.track` (con position: absolute + left: ${fillWidth}%) para
    que pueda salir del fill cuando el % es bajo (proteína 7% caso real
    reportado). Verificamos que sigue siendo descendiente del `.track`."""
    jsx = _read(_TRACKING_JSX)
    # Buscar el bloque del `.track` y su cierre — `.fill` ahora es
    # self-closing y `.fillPerc` es sibling, no child.
    track_idx = jsx.find("className={styles.track}")
    assert track_idx >= 0, "`.track` className no encontrado en JSX."
    # Buscar el primer `</div>` que NO esté precedido por otro tag abierto
    # sin cerrar — heurística simple: tomar el slice de 2500 chars y
    # verificar que `styles.fillPerc` aparece dentro.
    track_slice = jsx[track_idx:track_idx + 2500]
    assert "styles.fillPerc" in track_slice, (
        "`styles.fillPerc` no esta dentro del bloque `.track`. Post-fix narrow "
        "debe vivir aquí (no en `.fill`) para poder salir del fill cuando "
        "el % es bajo."
    )
    # El `.fill` debe ser self-closing (no contener body).
    fill_self_closing = re.search(
        r"className=\{styles\.fill\}[^/]*?/>",
        track_slice,
    )
    assert fill_self_closing, (
        "`.fill` debe ser self-closing (`<div ... />`). Si re-aparece como "
        "`<div ...>...</div>` con body, el bug narrow puede regresar si alguien "
        "movió el span de vuelta dentro."
    )


def test_fill_perc_visible_in_base_css():
    """[P3-TRACKING-PERC-DESKTOP] `.fillPerc { display: flex }` ahora vive
    en BASE (no gated por media query). Esto la hace visible en desktop."""
    css = _strip_css_comments(_read(_TRACKING_CSS))
    base = _extract_base_css(css)
    # Buscar .fillPerc en base + verificar display: flex
    fill_perc_match = re.search(
        r"\.fillPerc\s*\{([^}]*)\}",
        base,
    )
    assert fill_perc_match, (
        "Definicion base `.fillPerc {` no encontrada. Pre-fix vivia "
        "dentro del @media — debe moverse a base para aplicar en desktop."
    )
    body = fill_perc_match.group(1)
    assert "display: flex" in body, (
        "`.fillPerc` base no tiene `display: flex`. Pre-fix era `display: none` "
        "(oculto desktop) — el fix lo activa universalmente."
    )
    # [P3-TRACKING-PERC-NARROW-FIX · 2026-05-22] El `color: white` se movió a
    # inline style del JSX porque ahora alterna entre white (inside fill) y
    # el color del macro (outside fill, fill estrecho). El text-shadow sigue
    # en base CSS pero la variante `.fillPercOutside` lo overridea a none.
    assert "text-shadow" in body, (
        "`.fillPerc` base debe preservar `text-shadow` (legibilidad sobre fill)."
    )
    # Estructura post-fix: position absolute + transform translateX(-100%).
    assert "position: absolute" in body, (
        "`.fillPerc` base debe tener `position: absolute` para posicionarse "
        "al borde del fill via `left: ${fillWidth}%`."
    )


def test_track_height_bumped_in_base():
    """[P3-TRACKING-PERC-DESKTOP] `.track { height: 22px !important }` en
    BASE (antes solo dentro del @media mobile). Sin esto, en desktop la
    barra mantiene la altura inline del JSX (10-12px) — muy fina para texto."""
    css = _strip_css_comments(_read(_TRACKING_CSS))
    base = _extract_base_css(css)
    track_match = re.search(
        r"\.track\s*\{([^}]*)\}",
        base,
    )
    assert track_match, "Definicion base `.track {` no encontrada."
    body = track_match.group(1)
    assert re.search(r"height:\s*22px\s*!important", body), (
        "`.track` base debe tener `height: 22px !important`. Sin ello, "
        "desktop conserva la altura inline del JSX (10-12px) y el % "
        "blanco no tiene espacio vertical para leerse."
    )


def test_perc_row_hidden_in_base():
    """[P3-TRACKING-PERC-DESKTOP] La pill `.percRow` (que antes vivia debajo
    de la barra en desktop) se oculta globalmente — el % dentro del fill
    la reemplaza para todos los viewports."""
    css = _strip_css_comments(_read(_TRACKING_CSS))
    base = _extract_base_css(css)
    perc_row_match = re.search(
        r"\.percRow\s*\{([^}]*)\}",
        base,
    )
    assert perc_row_match, "Definicion base `.percRow {` no encontrada."
    body = perc_row_match.group(1)
    assert "display: none" in body, (
        "`.percRow` base debe ser `display: none` — el % dentro del fill "
        "es ahora el unico indicador. Si revives la pill debajo, tendras "
        "doble indicador (redundancia visual)."
    )


def test_mobile_media_no_longer_duplicates_base_rules():
    """[P3-TRACKING-PERC-DESKTOP] Las reglas `.percRow display:none`,
    `.track height: 22px`, `.fillPerc display:flex` ya NO se duplican en
    el @media mobile — viven solo en base. Si alguien las re-introduce en
    el media query, indica que estaba pensando que era un cambio mobile-only."""
    css = _strip_css_comments(_read(_TRACKING_CSS))
    mobile_body = _extract_media_block(css, "768")
    assert mobile_body, "Bloque `@media (max-width: 768px)` no encontrado."
    # `.percRow display:none` ya no debe estar duplicado en mobile
    assert not re.search(
        r"\.percRow\s*\{[^}]*display:\s*none",
        mobile_body,
    ), (
        "`.percRow { display: none }` ahora vive en base — duplicarlo en "
        "el @media mobile sugiere que el dev no sabia que se aplicaba "
        "universalmente. Remueve la regla redundante del media query."
    )
    assert not re.search(
        r"\.track\s*\{[^}]*height:\s*22px",
        mobile_body,
    ), (
        "`.track { height: 22px }` ahora vive en base — duplicacion en mobile "
        "es señal de un revert parcial. Limpia el media query."
    )
