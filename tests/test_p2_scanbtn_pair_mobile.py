"""[P2-SCANBTN-PAIR-MOBILE · 2026-08-17] El botón de registro no desborda en móvil.

Bug (reporte del dueño con captura): en el dashboard móvil (modo seguimiento) el botón
de escanear comida quedaba decapitado en el borde derecho de la tarjeta y el conjunto
se leía como roto. Causa: `.scanBtn { width: 100% }` en el media query móvil nació
cuando ese botón era ÚNICO (P2-DIARY-SCAN-MACROS · 2026-05-30); P1-MANUAL-FOOD-LOG
(2026-08-11) lo convirtió en un PAR dentro de `.logButtons` (inline-flex, fila), y dos
botones de 100% con el `flex-shrink: 0` de base desbordan la fila — el
`overflow: hidden` de `.card` recorta al segundo. Reproducido a 390px reales: la
cámara medía 216px y terminaba 90px fuera de la tarjeta.

[RECONVERTIDO por P1-PLAN-LOTE-85 · 2026-09-17] El par se deshizo a propósito: la tarjeta
vuelve a tener UN solo botón y las dos vías (buscar o escribir / escanear con foto) se eligen
dentro del componedor. Lo que sigue siendo cierto y este test conserva:

1. Ningún bloque de media query vuelve a declarar `width: 100%` (ni width fijo) sobre
   `.scanBtn`: el botón llena la fila con `flex: 1` dentro de `.logButtons` a ancho completo,
   así que si alguien vuelve a montar un par, no desborda.
2. `.scanBtnSecondary` ya no existe en ningún sitio (ni CSS ni JSX): el icono de cámara
   murió con el par. Si reaparece un segundo botón, re-anclar el reparto flex del par
   (la versión anterior de este test lo describe) en vez de heredarlo a ciegas.
3. El JSX monta exactamente UN `styles.scanBtn` dentro de `.logButtons`.

tooltip-anchor: P2-SCANBTN-PAIR-MOBILE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"
_CSS = _FRONT / "components" / "dashboard" / "TrackingProgress.module.css"
_JSX = _FRONT / "components" / "dashboard" / "TrackingProgress.jsx"


def _css_src() -> str:
    if not _CSS.exists():
        pytest.skip("sin el repo del frontend al lado")
    return _CSS.read_text(encoding="utf-8")


def _media_blocks(src: str) -> list[str]:
    """Bloques `@media (...) { ... }` completos, por conteo de llaves."""
    blocks = []
    for m in re.finditer(r"@media[^{]*\{", src):
        depth, i = 1, m.end()
        while depth and i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        blocks.append(src[m.start():i])
    return blocks


def _rule_body(block: str, selector: str) -> str:
    """Cuerpo de la primera regla cuyo grupo selector contiene `selector`."""
    for m in re.finditer(r"([^{}/]+)\{([^{}]*)\}", block):
        if re.search(rf"\.{selector}(?![\w-])", m.group(1)):
            return m.group(2)
    return ""


def test_no_media_block_gives_scanbtn_full_width():
    """Regla 1: `width` fijo sobre `.scanBtn` en un media query = desbordamiento
    garantizado en cuanto el botón vuelva a vivir en un par. Fue exactamente el bug."""
    for block in _media_blocks(_css_src()):
        body = _rule_body(block, "scanBtn")
        m = re.search(r"(?<!-)width\s*:\s*([^;]+);", body)
        assert not (m and m.group(1).strip() != "auto"), (
            f"un media query declara width:{m.group(1).strip()!r} sobre .scanBtn — "
            "con dos botones en fila eso decapita al segundo contra el overflow:hidden "
            "de la tarjeta (P2-SCANBTN-PAIR-MOBILE)"
        )


def test_single_button_fills_the_row_in_mobile_block():
    """Regla 2: el botón único llena la fila con flex, no con width; el icono de cámara no vuelve."""
    src = _css_src()
    mobile = next((b for b in _media_blocks(src) if "logButtons" in b), "")
    assert mobile, "el bloque móvil con .logButtons desapareció — re-anclar el contrato"
    assert re.search(r"\.logButtons\s*\{[^}]*width\s*:\s*100%", mobile), (
        ".logButtons perdió su width:100% móvil — el botón ya no llena la fila"
    )
    assert re.search(r"flex\s*:\s*1", _rule_body(mobile, "scanBtn")), (
        ".scanBtn perdió su flex:1 móvil — vuelve a depender de un width que ya desbordó una vez"
    )
    assert "scanBtnSecondary" not in src, (
        "reapareció .scanBtnSecondary — el par se deshizo en P1-PLAN-LOTE-85; si vuelve un "
        "segundo botón, re-anclar su reparto flex (flex:1 / flex:0) en vez de heredarlo a ciegas"
    )


def test_jsx_mounts_exactly_one_button_inside_logbuttons():
    """Regla 3: un solo botón dentro de .logButtons; las dos vías viven en el componedor."""
    if not _JSX.exists():
        pytest.skip("sin el repo del frontend al lado")
    jsx = _JSX.read_text(encoding="utf-8")
    m = re.search(r"styles\.logButtons.*?</div>", jsx, re.S)
    assert m, "el contenedor styles.logButtons desapareció del JSX — re-evaluar contrato"
    assert m.group(0).count("styles.scanBtn") == 1, (
        "hay más de un botón dentro de .logButtons — el par se deshizo en P1-PLAN-LOTE-85 "
        "(las dos vías se eligen dentro del componedor); si vuelve, revisar el reparto flex del CSS"
    )
    assert "scanBtnSecondary" not in jsx and "t('Escanear comida con la cámara')" not in jsx


def test_marker_anchored_in_css():
    assert "P2-SCANBTN-PAIR-MOBILE" in _css_src(), (
        "el marcador desapareció del CSS — sin él, el próximo que lea el bloque móvil "
        "no sabe que width:100% sobre .scanBtn ya rompió producción una vez"
    )
