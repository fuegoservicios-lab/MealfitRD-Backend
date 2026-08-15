"""[P1-VIEWPORT-ZOOM-LOCK · documentado 2026-08-15] El pinch-zoom bloqueado es una
DECISIÓN, no deuda.

POR QUÉ EXISTE ESTE TEST. La auditoría de accesibilidad del 2026-08-15 (Lighthouse
móvil sobre bioboros.com desplegado) dio 91/100, y el fallo que más pesa es
`meta-viewport`: «user-scalable=no disables zooming on mobile devices». Es un fallo
WCAG 1.4.4 real, y la reacción natural de cualquiera que lo vea —incluido quien
escribe esto, que estuvo a un paso de hacerlo— es quitarlo.

Ya se hizo una vez. `P2-A11Y-VIEWPORT-ZOOM` lo quitó por accesibilidad y **se
revirtió con el dueño** (feel de app nativa en el PWA standalone), y el comentario
de `index.css` lo dice literalmente: «Reversión de P2-A11Y-VIEWPORT-ZOOM confirmada
con el dueño».

El problema era que esa decisión vivía SÓLO en dos comentarios de código, y no en
la sección «Decisiones de producto» de CLAUDE.md, que existe exactamente para que
un auditor técnico no confunda una decisión con deuda. Una auditoría que corre cada
pocos meses volvería a proponerlo cada vez.

QUÉ ANCLA. No que el bloqueo exista —eso es del dueño, y puede cambiar de opinión—
sino que **si cambia, cambien las dos cosas a la vez**: el código y su
documentación. Si alguien abre el zoom sin retirar la entrada de CLAUDE.md, el
siguiente lector creerá que sigue bloqueado y razonará sobre un producto que ya no
existe. Y si alguien retira la entrada dejando el bloqueo, la próxima auditoría lo
reabre desde cero.

El trade-off aceptado, para no tener que reconstruirlo: la vía real de
accesibilidad no es el pinch, es la escala de fuente del SISTEMA (Dynamic Type en
iOS, tamaño de fuente en Android), que sigue funcionando porque
`-webkit-text-size-adjust: 100%` no la desactiva — sólo impide que el navegador
infle el texto por su cuenta al rotar.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_HTML = _REPO_ROOT / "frontend" / "index.html"
_INDEX_CSS = _REPO_ROOT / "frontend" / "src" / "index.css"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"

_MARKER = "P1-VIEWPORT-ZOOM-LOCK"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def _zoom_bloqueado(html: str) -> bool:
    m = re.search(r'<meta\s+name="viewport"\s+content="([^"]+)"', html)
    assert m, "No encuentro el <meta name=viewport> en index.html."
    contenido = m.group(1)
    return "user-scalable=no" in contenido or re.search(r"maximum-scale=\s*1(\.0)?\b", contenido) is not None


def test_codigo_y_documentacion_no_pueden_divergir() -> None:
    """El bloqueo y su entrada en CLAUDE.md viven o mueren juntos.

    Es un `if and only if` a propósito: los dos estados son legítimos, lo que no lo
    es es que discrepen.
    """
    bloqueado = _zoom_bloqueado(_leer(_INDEX_HTML))
    documentado = _MARKER in _leer(_CLAUDE_MD)

    if bloqueado and not documentado:
        pytest.fail(
            "El zoom sigue bloqueado (`user-scalable=no` / `maximum-scale=1`) pero "
            f"`{_MARKER}` ya no está en CLAUDE.md. Sin esa entrada, la próxima "
            "auditoría de accesibilidad volverá a reportarlo como defecto y alguien "
            "lo 'arreglará' — deshaciendo una decisión que el dueño ya revirtió una "
            "vez (ver P2-A11Y-VIEWPORT-ZOOM)."
        )
    if documentado and not bloqueado:
        pytest.fail(
            "CLAUDE.md sigue documentando el bloqueo del zoom como decisión activa, "
            "pero el viewport ya NO lo bloquea. Si se abrió el pinch a propósito "
            "—que es una decisión perfectamente válida— retira también la entrada: "
            "una decisión documentada que no se cumple es peor que ninguna, porque "
            "el siguiente lector razona sobre un producto que no existe."
        )


def test_la_razon_sigue_junto_al_codigo() -> None:
    """El porqué vive donde está el efecto, no sólo en un doc lejano."""
    if not _zoom_bloqueado(_leer(_INDEX_HTML)):
        pytest.skip("el zoom ya no está bloqueado; este ancla no aplica")

    html = _leer(_INDEX_HTML)
    css = _leer(_INDEX_CSS)
    assert "user-scalable" in html and re.search(r"Trade-off WCAG|WCAG", html), (
        "El `<meta viewport>` perdió el comentario que explica el trade-off WCAG. "
        "Sin él, la línea parece un descuido."
    )
    assert _MARKER in css and "P2-A11Y-VIEWPORT-ZOOM" in css, (
        "`index.css` perdió la referencia a la reversión. Ese comentario es la "
        "única prueba, junto al código, de que quitarlo YA se intentó y se revirtió "
        "con el dueño."
    )


def test_la_escala_de_fuente_del_so_sigue_viva() -> None:
    """La vía de accesibilidad que justifica el trade-off no puede desaparecer.

    El argumento aceptado no es «el zoom no hace falta», es «hay otra vía». Si
    `text-size-adjust` desapareciera o se pusiera en `none`, el trade-off se
    quedaría sin su mitad compensatoria y la decisión habría que rehacerla.
    """
    css = _leer(_INDEX_CSS)
    assert re.search(r"text-size-adjust:\s*100%", css), (
        "`text-size-adjust: 100%` desapareció de index.css. Con el pinch bloqueado, "
        "la escala de fuente del SO es la ÚNICA vía de accesibilidad que queda: es "
        "la mitad que hace defendible el trade-off."
    )
    assert not re.search(r"text-size-adjust:\s*none", css), (
        "`text-size-adjust: none` desactiva la escala de fuente del sistema. Con el "
        "pinch ya bloqueado, eso deja a un usuario con baja visión sin NINGUNA vía."
    )
