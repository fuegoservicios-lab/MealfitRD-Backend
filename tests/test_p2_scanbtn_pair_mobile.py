"""[P2-SCANBTN-PAIR-MOBILE · 2026-08-17] El par de botones de registro no desborda en móvil.

Bug (reporte del dueño con captura): en el dashboard móvil (modo seguimiento) el botón
de escanear comida quedaba decapitado en el borde derecho de la tarjeta y el conjunto
se leía como roto. Causa: `.scanBtn { width: 100% }` en el media query móvil nació
cuando ese botón era ÚNICO (P2-DIARY-SCAN-MACROS · 2026-05-30); P1-MANUAL-FOOD-LOG
(2026-08-11) lo convirtió en un PAR dentro de `.logButtons` (inline-flex, fila), y dos
botones de 100% con el `flex-shrink: 0` de base desbordan la fila — el
`overflow: hidden` de `.card` recorta al segundo. Reproducido a 390px reales: la
cámara medía 216px y terminaba 90px fuera de la tarjeta.

Contrato que ancla este test (la PROPIEDAD que rompió, no la grafía del fix):
1. Ningún bloque de media query vuelve a declarar `width: 100%` (ni width fijo) sobre
   `.scanBtn` — con DOS botones en fila eso garantiza el desbordamiento.
2. El reparto del par existe: `.scanBtn` estira (`flex: 1 ...`) y `.scanBtnSecondary`
   no (`flex: 0 ...`), dentro de un `.logButtons` a ancho completo.
3. El JSX sigue montando el par dentro de `.logButtons` (si un refactor lo separa,
   re-evaluar el contrato en vez de heredarlo a ciegas).

tooltip-anchor: P2-SCANBTN-PAIR-MOBILE
"""
from __future__ import annotations

import re
from pathlib import Path

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"
_CSS = _FRONT / "components" / "dashboard" / "TrackingProgress.module.css"
_JSX = _FRONT / "components" / "dashboard" / "TrackingProgress.jsx"


def _css_src() -> str:
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
    garantizado mientras el botón viva en un PAR. Fue exactamente el bug."""
    for block in _media_blocks(_css_src()):
        body = _rule_body(block, "scanBtn")
        m = re.search(r"(?<!-)width\s*:\s*([^;]+);", body)
        assert not (m and m.group(1).strip() != "auto"), (
            f"un media query declara width:{m.group(1).strip()!r} sobre .scanBtn — "
            "con dos botones en fila (P1-MANUAL-FOOD-LOG) eso decapita la cámara "
            "contra el overflow:hidden de la tarjeta (P2-SCANBTN-PAIR-MOBILE)"
        )


def test_pair_split_exists_in_mobile_block():
    """Regla 2: el reparto del par (composer estira, cámara compacta) sigue en pie."""
    src = _css_src()
    mobile = next((b for b in _media_blocks(src) if "logButtons" in b), "")
    assert mobile, "el bloque móvil con .logButtons desapareció — re-anclar el contrato"
    assert re.search(r"\.logButtons\s*\{[^}]*width\s*:\s*100%", mobile), (
        ".logButtons perdió su width:100% móvil — el par ya no llena la fila"
    )
    scan = _rule_body(mobile, "scanBtn")
    assert re.search(r"flex\s*:\s*1", scan), (
        ".scanBtn perdió su flex:1 móvil — el compositor deja de estirar y el par "
        "vuelve a repartirse mal la fila"
    )
    secondary = _rule_body(mobile, "scanBtnSecondary")
    assert re.search(r"flex\s*:\s*0", secondary), (
        ".scanBtnSecondary perdió su flex:0 móvil — la cámara vuelve a competir por "
        "el ancho del compositor"
    )


def test_jsx_still_mounts_pair_inside_logbuttons():
    """Regla 3: el contrato asume DOS botones dentro de .logButtons."""
    jsx = _JSX.read_text(encoding="utf-8")
    m = re.search(r"styles\.logButtons.*?</div>", jsx, re.S)
    assert m, "el contenedor styles.logButtons desapareció del JSX — re-evaluar contrato"
    assert m.group(0).count("styles.scanBtn") >= 2, (
        "ya no hay dos botones dentro de .logButtons — si el par se deshizo, este "
        "contrato y el reparto flex del CSS deben revisarse juntos"
    )


def test_marker_anchored_in_css():
    assert "P2-SCANBTN-PAIR-MOBILE" in _css_src(), (
        "el marcador desapareció del CSS — sin él, el próximo que lea el bloque móvil "
        "no sabe que width:100% sobre .scanBtn ya rompió producción una vez"
    )
