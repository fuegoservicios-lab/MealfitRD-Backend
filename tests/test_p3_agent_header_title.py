"""[P3-AGENT-HEADER-TITLE · 2026-05-19 · P2-WORDMARK-BIOBOROS 2026-07-30] Test
parser-based: el título del header del chat (`AgentPage.jsx`, dentro de
`<span className="agent-header-title">`) es **wordmark + versión** — NO el
literal legacy `MealfitRD`.

Por qué este test:
    Versioning visible para el usuario en el header del chat. Decisión de
    producto. Si alguien refactoriza el header o copy y revierte
    accidentalmente al string legacy "MealfitRD", este test lo flagea antes
    de mergear.

[P2-WORDMARK-BIOBOROS · 2026-07-30] El header decía `Mealfit V1.0` hardcodeado
y el rebrand lo cambió a `<Wordmark /> 1` (ver el comentario en AgentPage.jsx:
la marca pasa a componente para que ninguna renombrada futura la deje atrás, y
la versión va en cifra desnuda porque el prefijo "V" es jerga de release). Este
test se quedó anclado al literal viejo y llevaba rojo desde ese commit — el
propio mensaje que emitía pedía actualizarlo "en el mismo commit".

Ahora ancla el CONTRATO (marca por componente + versión visible), no la cadena:
el número de versión puede cambiar sin poner rojo el test, pero borrar el
wordmark o la versión sí.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_agent_header_title`
matchea este archivo `test_p3_agent_header_title.py`.

Tooltip-anchor: P3-AGENT-HEADER-TITLE-START | user request 2026-05-19
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = (
    _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
)


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


def _contenido_del_titulo(src: str) -> str:
    """Contenido del `<span className="agent-header-title">` hasta su cierre.

    Acotado por ORDEN RELATIVO (apertura → `</span>` de cierre del bloque), no por
    una ventana de bytes fija: las ventanas fijas caducan solas en cuanto alguien
    añade un estilo inline al span.
    """
    open_re = re.compile(r'className\s*=\s*["\']agent-header-title["\'][\s\S]*?>')
    m_open = open_re.search(src)
    assert m_open is not None, (
        "P3-AGENT-HEADER-TITLE regresión: `<span className=\"agent-header-title\">` "
        "no encontrado en AgentPage.jsx."
    )
    resto = src[m_open.end():]
    # El título contiene un <span> anidado (la versión); su `</span>` NO es el cierre
    # del título. Cerramos en el segundo.
    cierres = [m.start() for m in re.finditer(r"</span>", resto)]
    assert cierres, "P3-AGENT-HEADER-TITLE regresión: el span del título no cierra."
    fin = cierres[1] if len(cierres) > 1 else cierres[0]
    return resto[:fin]


def test_header_title_lleva_wordmark_y_version(agent_page_src: str):
    """El título es marca-por-componente + versión visible.

    Anclado al contrato, no a la cadena: cambiar el número de versión NO debe poner
    esto rojo (el test viejo se anclaba al literal `Mealfit V1.0` y llevaba rojo
    desde el rebrand). Borrar el wordmark o quitar la versión SÍ.
    """
    titulo = _contenido_del_titulo(agent_page_src)

    assert re.search(r"<\s*Wordmark\b", titulo), (
        "P3-AGENT-HEADER-TITLE regresión: el título del header ya no renderiza "
        "`<Wordmark />`. La marca va por componente a propósito (P2-WORDMARK-BIOBOROS): "
        "hardcodearla es justo lo que hizo que el rebrand se dejara este header atrás."
    )
    assert re.search(r"^import\s+Wordmark\s+from", agent_page_src, re.M), (
        "P3-AGENT-HEADER-TITLE regresión: `<Wordmark />` se usa en el título pero no "
        "está importado — el header reventaría en runtime."
    )

    # Versión visible al usuario: cifra desnuda ("1"), con o sin prefijo/patch.
    version = re.search(r">\s*(V?\d+(?:\.\d+)*)\s*<", titulo)
    assert version, (
        "P3-AGENT-HEADER-TITLE regresión: el título ya no muestra número de versión. "
        "Es decisión de producto (versioning visible al usuario). Cambiar el número "
        "es OK; quitarlo requiere actualizar esta decisión, no solo el test."
    )


def test_legacy_mealfitrd_not_in_header_title(agent_page_src: str):
    """`MealfitRD` (legacy) NO debe aparecer DENTRO del span del título.

    Antes recortaba `>\\s*([^<]+?)\\s*<` — el texto ANTES del primer `<`. Desde que el
    título empieza por `<Wordmark />` ese grupo captura solo el salto de línea, así que
    la aserción comparaba `""` contra `"MealfitRD"` y pasaba **en vacío**: verde por una
    línea distinta a la que dice mirar. Ahora mira todo el contenido del span.
    """
    titulo = _contenido_del_titulo(agent_page_src)
    assert "MealfitRD" not in titulo, (
        f"P3-AGENT-HEADER-TITLE regresión: el título del header volvió a incluir el "
        f"literal legacy 'MealfitRD'. Contenido actual: {titulo.strip()[:200]!r}"
    )


def test_anchor_present(agent_page_src: str):
    assert "P3-AGENT-HEADER-TITLE" in agent_page_src, (
        "P3-AGENT-HEADER-TITLE regresión: anchor textual perdido en AgentPage.jsx."
    )
