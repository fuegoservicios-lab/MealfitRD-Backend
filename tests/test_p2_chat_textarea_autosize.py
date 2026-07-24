"""[P2-CHAT-TEXTAREA-AUTOSIZE · 2026-07-24] Anchor parser-based: el alto del
textarea del chat se deriva del ESTADO, no del evento `onInput`.

Bug original (reportado por el owner 2026-07-24, con screenshot del input del
AgentPage inflado mostrando el placeholder):

    "El chat del agente su tamaño se buguea a veces y se pone ancho, tengo que
     refrescar la página web para que vuelva a su tamaño normal."

Causa raíz:
    `AgentPage.jsx` auto-dimensionaba el textarea SOLO desde el handler
    `onInput` del DOM::

        onInput={(e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
        }}

    React no es dueño de ese `height` inline (no viaja en el prop `style`), así
    que NO lo revierte al re-renderizar. Todo cambio de `input` que no venía de
    una tecla dejaba pegado el alto del mensaje anterior:
      - `handleSend`      → `setInput('')`  ← el caso del screenshot
      - `handleNewChat`   → `setInput('')`
      - pill de sugerencia → `setInput(suggestion.text)`
      - prefill del dashboard (P3-AGENT-PREFILL)
    Y como AgentPage es **keep-alive** (App.jsx lo oculta con `display:none` en
    vez de desmontarlo, P1-AGENT-KEEP-ALIVE), el alto stale sobrevivía a la
    navegación entre tabs del dashboard — solo un reload de la página lo
    reseteaba. Exactamente el workaround que el owner reportaba.

Fix:
    SSOT `frontend/src/utils/autosizeTextarea.js`:
      - `autosizeTextarea(el, max)` — pura, idempotente, no escribe cuando el
        elemento está oculto (`scrollHeight === 0` bajo el keep-alive; escribir
        `0px` dejaría el input colapsado al volver).
      - `useAutosizeTextarea(ref, signature, max)` — `useLayoutEffect` sobre la
        firma (valor + lo que cambia el ancho disponible) + listener `resize`.
    AgentPage consume el hook y NO conserva escrituras imperativas de altura.

Tests conductuales (jsdom, con `scrollHeight` modelado):
    `frontend/src/__tests__/utils/autosizeTextarea.test.jsx`.

Este archivo ancla el CONTRATO ESTRUCTURAL (que nadie reintroduzca el
auto-resize por evento) y cumple el cross-link de P2-HIST-AUDIT-14 con el bump
de `_LAST_KNOWN_PFIX`.

Tooltip-anchor: P2-CHAT-TEXTAREA-AUTOSIZE-START
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"
_AGENT_PAGE = _FRONTEND_SRC / "pages" / "AgentPage.jsx"
_UTIL = _FRONTEND_SRC / "utils" / "autosizeTextarea.js"
_BEHAVIOR_TEST = _FRONTEND_SRC / "__tests__" / "utils" / "autosizeTextarea.test.jsx"

# El backend se despliega solo en el VPS (sin el repo hermano del frontend):
# skip en vez de fallar cuando el árbol del frontend no está presente.
_SKIP_NO_FRONTEND = pytest.mark.skipif(
    not _FRONTEND_SRC.is_dir(),
    reason="frontend/src no presente en este checkout (repos hermanos)",
)

_MARKER = re.compile(r"\[P2-CHAT-TEXTAREA-AUTOSIZE\s*·\s*2026-07-24\]")

# Escritura imperativa de altura: `<algo>.style.height = ...`.
_STYLE_HEIGHT_WRITE = re.compile(r"\.style\.height\s*=")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_js_comments(src: str) -> str:
    """Quita `/* ... */` y el tramo desde `//` hasta EOL.

    Heurístico (mismo trade-off documentado en P1-NEW-A): no cubre `//` dentro
    de un string literal. Suficiente aquí — sirve para no contar como código
    los comentarios que CITAN el patrón viejo en prosa (los hay, a propósito,
    documentando el bug cerrado).
    """
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", src)
    return "\n".join(line.split("//", 1)[0] for line in no_block.splitlines())


# ---------------------------------------------------------------------------
# 1. El SSOT existe y tiene la defensa del keep-alive
# ---------------------------------------------------------------------------
@_SKIP_NO_FRONTEND
def test_util_ssot_existe_con_marker():
    assert _UTIL.is_file(), f"falta el SSOT del autosize: {_UTIL}"
    src = _read(_UTIL)
    assert _MARKER.search(src), "marker P2-CHAT-TEXTAREA-AUTOSIZE ausente en el SSOT"
    assert "export function autosizeTextarea" in src
    assert "export function useAutosizeTextarea" in src
    assert "CHAT_TEXTAREA_MAX_HEIGHT_PX" in src


@_SKIP_NO_FRONTEND
def test_util_no_escribe_altura_cuando_el_elemento_esta_oculto():
    """Regresión del propio fix: AgentPage vive oculto con `display:none`
    (keep-alive). Medir ahí devuelve `scrollHeight === 0`; escribir `0px`
    dejaría el input colapsado al volver al chat."""
    src = _strip_js_comments(_read(_UTIL))
    assert re.search(r"measured\s*<=\s*0", src), (
        "el SSOT debe cortocircuitar cuando la medición es 0 (elemento oculto)"
    )
    # Y debe restaurar el valor previo en vez de dejar `auto`.
    assert re.search(r"el\.style\.height\s*=\s*prev", src), (
        "al cortocircuitar hay que restaurar la altura previa"
    )


@_SKIP_NO_FRONTEND
def test_util_resetea_a_auto_antes_de_medir():
    """Sin `height='auto'` previo, `scrollHeight` nunca baja de la altura ya
    fijada → la caja podría crecer pero jamás encoger."""
    src = _strip_js_comments(_read(_UTIL))
    auto_idx = src.find("el.style.height = 'auto'")
    # La medición que importa es la que se guarda para decidir el alto final
    # (el `typeof el.scrollHeight` del guard de entrada no cuenta).
    measure_idx = src.find("const measured = el.scrollHeight")
    assert auto_idx != -1, "falta el reset a 'auto' antes de medir"
    assert measure_idx != -1, "falta `const measured = el.scrollHeight`"
    assert auto_idx < measure_idx, "el reset a 'auto' debe preceder a la medición"


# ---------------------------------------------------------------------------
# 2. AgentPage consume el hook y NO auto-dimensiona por evento
# ---------------------------------------------------------------------------
@_SKIP_NO_FRONTEND
def test_agentpage_usa_el_hook():
    src = _read(_AGENT_PAGE)
    assert _MARKER.search(src), "marker P2-CHAT-TEXTAREA-AUTOSIZE ausente en AgentPage.jsx"
    assert re.search(
        r"import\s*\{[^}]*useAutosizeTextarea[^}]*\}\s*from\s*['\"]\.\./utils/autosizeTextarea['\"]",
        src,
    ), "AgentPage debe importar el hook desde el SSOT"
    assert re.search(
        r"useAutosizeTextarea\s*\(\s*chatInputRef\s*,", src
    ), "AgentPage debe cablear el hook al ref del textarea del chat"


@_SKIP_NO_FRONTEND
def test_agentpage_firma_incluye_valor_y_ancho_disponible():
    """La firma debe reaccionar al CONTENIDO (`input`) y a lo que cambia el
    ANCHO disponible: el mismo texto ocupa distintas líneas según el ancho."""
    src = _read(_AGENT_PAGE)
    m = re.search(r"useAutosizeTextarea\s*\(\s*chatInputRef\s*,([^;]*?)\)\s*;", src, re.DOTALL)
    assert m is not None, "no se encontró el callsite del hook"
    signature = m.group(1)
    for token in ("input", "isMobile", "showSidebar", "previewUrl"):
        assert token in signature, (
            f"la firma del autosize debe incluir `{token}` "
            f"(cambia contenido o ancho disponible); firma actual: {signature.strip()}"
        )


@_SKIP_NO_FRONTEND
def test_agentpage_sin_autoresize_por_evento():
    """El patrón `onInput={... .style.height = ...}` es el bug original.

    Cualquier escritura imperativa de altura en AgentPage reintroduce la clase
    de fallo (React no revierte inline styles que no le pertenecen).
    """
    code = _strip_js_comments(_read(_AGENT_PAGE))
    offenders = [
        (i, line.strip())
        for i, line in enumerate(code.splitlines(), start=1)
        if _STYLE_HEIGHT_WRITE.search(line)
    ]
    assert not offenders, (
        "AgentPage.jsx no debe escribir `.style.height` a mano — usa "
        "`useAutosizeTextarea` (SSOT utils/autosizeTextarea.js). Violaciones:\n"
        + "\n".join(f"  L{n}: {snippet}" for n, snippet in offenders)
    )


@_SKIP_NO_FRONTEND
def test_agentpage_no_hardcodea_el_cap_del_textarea():
    """El `maxHeight` del textarea debe salir de la constante del SSOT: si el
    cap se toca en un sitio y no en el otro, la caja hace scroll interno a una
    altura distinta de la que el autosize permite."""
    src = _read(_AGENT_PAGE)
    m = re.search(r"maxHeight:\s*([^,\n]+)", src)
    assert m is not None, "no se encontró el maxHeight del textarea"
    assert "CHAT_TEXTAREA_MAX_HEIGHT_PX" in m.group(1), (
        f"maxHeight debe derivar de CHAT_TEXTAREA_MAX_HEIGHT_PX, no de un literal: {m.group(1)}"
    )


# ---------------------------------------------------------------------------
# 3. Cross-link con los tests conductuales (vitest)
# ---------------------------------------------------------------------------
@_SKIP_NO_FRONTEND
def test_existe_el_test_conductual_del_bug_reportado():
    assert _BEHAVIOR_TEST.is_file(), f"falta el test conductual: {_BEHAVIOR_TEST}"
    src = _read(_BEHAVIOR_TEST)
    assert "BUG REPORTADO" in src, (
        "el test conductual debe cubrir explícitamente el caso reportado "
        "(limpiar el valor programáticamente al enviar)"
    )
