"""[P1-CHAT-CANCEL-A11Y · 2026-05-19] Test parser-based: bundle de los 2
P1 livianos del audit prod-readiness del Agente (2026-05-19):

  (1) BACKEND — `chat_with_agent_stream` captura `GeneratorExit` cuando el
      cliente aborta el SSE stream (tab-close, AbortController, network
      drop). Pre-fix `except Exception` outer NO atrapaba GeneratorExit
      (hereda de BaseException, no Exception); el iterator interno de
      LangGraph seguía invocando LLM en threads internos → costo LLM
      desperdiciado. Fix cierra el iterator explícitamente con
      `stream_iter.close()` antes de re-raise.

  (2) FRONTEND — `.messages-container` en AgentPage.jsx tiene
      `role="log" aria-live="polite" aria-relevant="additions text"
      aria-label=<es-DO>` para anunciar mensajes nuevos a screen readers.
      MessageBubble.jsx marca el bubble como `aria-busy="true"` mientras
      `msg.isStreaming===true` para suprimir announcements parciales
      durante el stream chunked.

Por qué este test:
    Ambos gaps estaban explícitamente flagueados como PENDIENTES en la
    memoria del audit 2026-05-19 (ver
    `~/.claude/projects/.../memory/project_p1_chat_cb_sanitize_2026_05_19.md`).
    Este test ancla ambos cierres para que un refactor futuro no los
    reabra silenciosamente.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p1_chat_cancel_a11y`
matchea este archivo `test_p1_chat_cancel_a11y.py`.

Tooltip-anchor: P1-CHAT-CANCEL | P1-CHAT-A11Y-LIVE | audit 2026-05-19
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PY = _REPO_ROOT / "backend" / "agent.py"
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_MESSAGE_BUBBLE_JSX = (
    _REPO_ROOT / "frontend" / "src" / "components" / "agent" / "MessageBubble.jsx"
)


@pytest.fixture(scope="module")
def agent_py_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def message_bubble_src() -> str:
    return _MESSAGE_BUBBLE_JSX.read_text(encoding="utf-8")


def _extract_chat_stream_body(src: str) -> str:
    """Devuelve el cuerpo de `chat_with_agent_stream` para asserts focalizados.

    Match desde `def chat_with_agent_stream(...)` hasta el próximo `^def `
    al inicio de línea (o EOF).
    """
    m = re.search(
        r"(?ms)^def\s+chat_with_agent_stream\b.*?(?=\n^def\s|\Z)",
        src,
    )
    assert m, "P1-CHAT-CANCEL: función `chat_with_agent_stream` no encontrada en agent.py"
    return m.group(0)


def _extract_generator_exit_block(body: str) -> str:
    """Devuelve el bloque entre `except GeneratorExit:` y el siguiente
    `except Exception as e:` outer (handler del mismo try del stream).

    Nota: el handler GeneratorExit puede tener un `try/except Exception:`
    interno (cleanup best-effort de `stream_iter.close()`); por eso
    delimitamos por `except Exception AS e:` (con `as e`) que es la firma
    del handler outer original — el interno usa `except Exception:` solo."""
    m = re.search(
        r"except\s+GeneratorExit\s*:\s*\n([\s\S]*?)(?=^\s*except\s+Exception\s+as\s+e\b)",
        body,
        re.MULTILINE,
    )
    assert m, (
        "P1-CHAT-CANCEL: bloque `except GeneratorExit:` ... "
        "`except Exception as e:` no encontrado. El handler debe declararse "
        "INMEDIATAMENTE antes del except Exception outer del stream."
    )
    return m.group(1)


# =============================================================================
# (1) BACKEND — GeneratorExit handler + stream_iter.close()
# =============================================================================


def test_anchor_present_backend(agent_py_src: str):
    body = _extract_chat_stream_body(agent_py_src)
    assert "P1-CHAT-CANCEL" in body, (
        "P1-CHAT-CANCEL-A11Y regresión: anchor textual perdido dentro de "
        "`chat_with_agent_stream` en agent.py."
    )


def test_stream_iter_assigned_before_for_loop(agent_py_src: str):
    """`stream_iter = chat_graph_app.stream(...)` debe asignarse a una
    variable ANTES del `for event in ...` (necesario para poder llamar
    `stream_iter.close()` desde el except handler)."""
    body = _extract_chat_stream_body(agent_py_src)
    pattern = re.compile(
        r"stream_iter\s*=\s*chat_graph_app\.stream\([\s\S]*?\)[\s\S]*?"
        r"for\s+event\s+in\s+stream_iter",
    )
    assert pattern.search(body), (
        "P1-CHAT-CANCEL regresión: `chat_graph_app.stream(...)` ya no se "
        "asigna a `stream_iter` antes del `for`. Sin la referencia, el "
        "except GeneratorExit no puede cerrar el iterator interno y los "
        "workers de LangGraph siguen consumiendo LLM tras tab-close."
    )


def test_generator_exit_handler_present(agent_py_src: str):
    """Debe existir un `except GeneratorExit:` declarado INMEDIATAMENTE
    antes del `except Exception` outer del stream — es la firma del fix."""
    body = _extract_chat_stream_body(agent_py_src)
    # Match: except GeneratorExit ... (cuerpo) ... except Exception, todo dentro
    # del cuerpo de chat_with_agent_stream. El except Exception outer debe
    # venir RIGHT AFTER (sin otro handler de por medio).
    pattern = re.compile(
        r"except\s+GeneratorExit\s*:\s*\n[\s\S]*?^\s*except\s+Exception\b",
        re.MULTILINE,
    )
    assert pattern.search(body), (
        "P1-CHAT-CANCEL regresión: `except GeneratorExit:` debe declararse "
        "INMEDIATAMENTE antes del `except Exception:` outer del stream. "
        "Sin él, GeneratorExit (BaseException) no se atrapa y el LangGraph "
        "sigue ejecutando tools tras tab-close."
    )


def test_handler_calls_stream_iter_close(agent_py_src: str):
    """El handler `except GeneratorExit:` debe llamar `stream_iter.close()`
    para propagar la cancelación a los workers internos de LangGraph."""
    body = _extract_chat_stream_body(agent_py_src)
    block = _extract_generator_exit_block(body)
    assert "stream_iter.close()" in block, (
        "P1-CHAT-CANCEL regresión: el handler GeneratorExit ya no invoca "
        "`stream_iter.close()`. Sin ese close, los workers de LangGraph "
        "siguen invocando LLM aunque el cliente cerró la conexión."
    )


def test_handler_reraises_generator_exit(agent_py_src: str):
    """El handler DEBE re-elevar (`raise`) — Python requiere que
    GeneratorExit propague para cleanup correcto del generator."""
    body = _extract_chat_stream_body(agent_py_src)
    block = _extract_generator_exit_block(body)
    # Match bare `raise` (no `raise SomeException`). MULTILINE para que ^/$
    # apliquen a cada línea del bloque.
    assert re.search(r"(?m)^\s*raise\s*(?:#.*)?$", block), (
        "P1-CHAT-CANCEL regresión: el handler GeneratorExit no re-eleva "
        "con `raise`. Suprimir GeneratorExit deja el generator en estado "
        "indefinido y rompe el cleanup del SSE."
    )


def test_handler_does_not_yield(agent_py_src: str):
    """El handler NO debe emitir más yields — la conexión está cerrada y
    el write fallaría con BrokenPipeError, mascarando la causa real."""
    body = _extract_chat_stream_body(agent_py_src)
    block = _extract_generator_exit_block(body)
    # `yield` puede aparecer como sub-string dentro de comentarios; nos
    # interesa solo el statement real → match a inicio de línea (whitespace
    # + 'yield ' o 'yield\n').
    yield_stmt = re.search(r"(?m)^\s*yield\b", block)
    assert yield_stmt is None, (
        "P1-CHAT-CANCEL regresión: el handler GeneratorExit contiene un "
        "`yield` statement. La conexión SSE ya está cerrada — escribir "
        "más data fallaría con BrokenPipeError y enmascararía el motivo "
        "real del cierre."
    )


def test_handler_logs_warning_with_marker(agent_py_src: str):
    """El handler debe loguear con `logger.warning` (NO error — es flujo
    legítimo) y el marker `[P1-CHAT-CANCEL]` para grep en producción."""
    body = _extract_chat_stream_body(agent_py_src)
    block = _extract_generator_exit_block(body)
    assert "logger.warning" in block, (
        "P1-CHAT-CANCEL regresión: el handler no usa `logger.warning(...)`. "
        "Abortar el SSE NO es error — es UX legítima. Usar warning permite "
        "filtrar el incidente sin contaminar el sink de errores."
    )
    assert "[P1-CHAT-CANCEL]" in block, (
        "P1-CHAT-CANCEL regresión: el log warning ya no incluye el marker "
        "textual `[P1-CHAT-CANCEL]`. Sin él, grep en logs de prod no "
        "diferencia cancelaciones del usuario de otros warnings."
    )


# =============================================================================
# (2) FRONTEND a11y — role="log" + aria-live + aria-busy on streaming
# =============================================================================


def test_anchor_present_frontend(agent_page_src: str):
    assert "P1-CHAT-A11Y-LIVE" in agent_page_src, (
        "P1-CHAT-CANCEL-A11Y regresión: anchor textual perdido en AgentPage.jsx "
        "(se espera marker P1-CHAT-A11Y-LIVE como tooltip del contenedor)."
    )


# -----------------------------------------------------------------------------
# [P1-I18N-DASHBOARD · 2026-08-15] Reanclaje del bloque a11y: la GRAFÍA cambió,
# la CONDUCTA no — y al mutar salió a la luz un hueco PREEXISTENTE.
#
# Grafía: el dashboard pasó a multiidioma y el atributo se escribe ahora
# `aria-label={t('Historial de conversación con el asistente')}`. En este motor
# **la clave ES el texto español** (es-DO no lleva catálogo: es el fallback), así
# que la cadena es literalmente la que lee un lector de pantalla en es-DO. Se
# aceptan las dos formas (literal y envuelta) para que un revert tampoco ciegue
# el guard.
#
# El hueco: estos cuatro tests anclaban por PROXIMIDAD AL COMENTARIO
# (`P1-CHAT-A11Y-LIVE[\s\S]{0,3000}?aria-label=…`). Al verificar por mutación
# —borrar el `aria-label` de los DOS contenedores— el test siguió VERDE: lo
# satisfacía el `aria-label` de un spinner `<Loader2>` 18 líneas más abajo, que
# cae dentro de la misma ventana. Un guard que un elemento vecino puede
# satisfacer no vigila al contenedor.
#
# Ahora se ancla al ELEMENTO: se localiza cada etiqueta de apertura que declara
# `role="log"` y se exige el juego completo de atributos EN ESA etiqueta. Aplica
# a TODAS (hay dos caminos de render, virtualizado y simple: ambos son el
# contenedor de mensajes y ambos deben anunciarse igual).
# -----------------------------------------------------------------------------

_ROLE_LOG_RE = re.compile(r"role\s*=\s*[\"']log[\"']")

# Valor de atributo JSX: literal "…" | envuelto {t('…')}.
_ATTR_VALUE = r"(?:[\"']([^\"']+)[\"']|\{\s*t\(\s*['\"]([^'\"]+)['\"])"


def _enclosing_jsx_open_tag(src: str, idx: int) -> str | None:
    """Etiqueta de apertura JSX que CONTIENE la posición `idx`.

    Escanea hacia atrás hasta el `<` y hacia delante hasta el `>` de cierre a
    profundidad 0 de llaves (así `style={{…}}` o `onX={(a) => b}` no lo cortan).
    Devuelve None si la etiqueta cierra ANTES de `idx` — que es como se
    descartan las apariciones dentro de comentarios (este archivo documenta
    `role="log"` en el comentario del marker, y esa no es una etiqueta real).
    """
    start = src.rfind("<", 0, idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(src)):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ">" and depth == 0:
            return src[start : i + 1] if i >= idx else None
    return None


def _log_container_tags(src: str) -> list[str]:
    """Todas las etiquetas de apertura que declaran `role="log"`."""
    tags = []
    for m in _ROLE_LOG_RE.finditer(src):
        tag = _enclosing_jsx_open_tag(src, m.start())
        if tag is not None:
            tags.append(tag)
    return tags


def test_messages_container_has_role_log(agent_page_src: str):
    """El contenedor de mensajes debe tener `role="log"` — la role estándar
    ARIA para chat conversations / activity logs.

    Se exige además que el marker `P1-CHAT-A11Y-LIVE` documente el bloque
    (proximidad al primer contenedor), que es lo que ata este guard a ESTE
    contenedor y no a un `role="log"` cualquiera de otra pantalla.
    """
    tags = _log_container_tags(agent_page_src)
    assert tags, (
        "P1-CHAT-CANCEL-A11Y regresión: el contenedor de mensajes ya no "
        "tiene `role=\"log\"`. Sin él, screen readers no identifican el "
        "área como un log de conversación."
    )
    documented = re.search(
        r"P1-CHAT-A11Y-LIVE[\s\S]{0,3000}?role\s*=\s*[\"']log[\"']",
        agent_page_src,
    )
    assert documented, (
        "P1-CHAT-CANCEL-A11Y regresión: hay `role=\"log\"` pero ya no bajo el "
        "marker `P1-CHAT-A11Y-LIVE`. El marker es el tooltip-anchor que hace "
        "que un renombre falle el test antes de cambiar producción."
    )


def test_messages_container_has_aria_live_polite(agent_page_src: str):
    """`aria-live="polite"` anuncia nuevos mensajes en pausa (NO interrumpe
    al usuario). NO usar `assertive` — sería invasivo en cada chunk."""
    tags = _log_container_tags(agent_page_src)
    assert tags, "P1-CHAT-CANCEL-A11Y: ningún contenedor con role=\"log\"."
    for tag in tags:
        assert re.search(r"aria-live\s*=\s*[\"']polite[\"']", tag), (
            "P1-CHAT-CANCEL-A11Y regresión: `aria-live=\"polite\"` ya no está "
            "en el contenedor `role=\"log\"`. Sin él, screen readers no "
            f"anuncian mensajes nuevos del asistente. Tag: {tag[:200]!r}"
        )
        assert not re.search(r"aria-live\s*=\s*[\"']assertive[\"']", tag), (
            "P1-CHAT-CANCEL-A11Y regresión: `aria-live=\"assertive\"` "
            "interrumpiría al usuario en CADA chunk del stream. Usar "
            f"`polite`. Tag: {tag[:200]!r}"
        )


def test_messages_container_has_aria_label_es_do(agent_page_src: str):
    """`aria-label` da contexto al log — debe estar en español dominicano
    (es-DO), suelto o envuelto en `t()`.

    [P1-I18N-DASHBOARD · 2026-08-15] Reanclaje: la GRAFÍA cambió, la CONDUCTA
    no. El dashboard pasó a multiidioma y el atributo se escribe ahora
    `aria-label={t('Historial de conversación con el asistente')}`. En este
    motor **la clave ES el texto español** (es-DO no lleva catálogo: es el
    fallback), así que la cadena capturada sigue siendo LITERALMENTE la que
    lee un lector de pantalla en es-DO — el mismo valor que este test validaba
    antes. Se aceptan las dos formas para que un revert del i18n tampoco deje
    el guard ciego, y se conservan intactas las dos comprobaciones de fondo:
    que el label exista con contenido y que no sea un placeholder en inglés.

    Nota sobre `P3-I18N-DEFERRED`: quedó SUPERSEDED por `P1-I18N-DASHBOARD`.
    Lo que se traduce es la INTERFAZ; el requisito de que este label nazca en
    es-DO no se relaja, porque el español es la clave de la que cuelgan las
    otras 4 traducciones.
    """
    tags = _log_container_tags(agent_page_src)
    assert tags, "P1-CHAT-CANCEL-A11Y: ningún contenedor con role=\"log\"."
    for tag in tags:
        # El atributo debe vivir EN la etiqueta del contenedor — no en un
        # vecino: un `<Loader2 aria-label={t('Cargando mensajes')} />` cercano
        # satisfacía la versión anterior de este guard con los DOS contenedores
        # ya sin nombre accesible.
        m = re.search(r"aria-label\s*=\s*" + _ATTR_VALUE, tag)
        assert m, (
            "P1-CHAT-CANCEL-A11Y regresión: `aria-label` ya no está en el "
            "contenedor `role=\"log\"`. Sin él, el log queda sin nombre "
            "accesible. Se aceptan `aria-label=\"…\"` y `aria-label={t('…')}` "
            f"(P1-I18N-DASHBOARD). Tag: {tag[:200]!r}"
        )
        label = (m.group(1) or m.group(2)).strip()
        assert len(label) >= 10, (
            f"P1-CHAT-CANCEL-A11Y: aria-label muy corto ({label!r}); usar "
            f"frase descriptiva es-DO."
        )
        # Detectar copy obviamente inglés/placeholder
        forbidden_en = {"chat log", "conversation log", "messages", "TODO", "log"}
        assert label.lower() not in forbidden_en, (
            f"P1-CHAT-CANCEL-A11Y: aria-label parece placeholder/inglés "
            f"({label!r}). La clave del i18n es el texto es-DO "
            f"(P1-I18N-DASHBOARD, supersede P3-I18N-DEFERRED)."
        )


def test_messages_container_has_aria_relevant(agent_page_src: str):
    """`aria-relevant` controla qué cambios anuncia el live region. Debe
    incluir `additions` (bubbles nuevos) y `text` (streaming chunks)."""
    tags = _log_container_tags(agent_page_src)
    assert tags, "P1-CHAT-CANCEL-A11Y: ningún contenedor con role=\"log\"."
    for tag in tags:
        m = re.search(r"aria-relevant\s*=\s*" + _ATTR_VALUE, tag)
        assert m, (
            "P1-CHAT-CANCEL-A11Y regresión: `aria-relevant` no encontrado en "
            "el contenedor `role=\"log\"`. Default del browser (`additions "
            "text`) puede variar; declararlo explícito hace el comportamiento "
            f"predictible. Tag: {tag[:200]!r}"
        )
        relevant = m.group(1) or m.group(2)
        assert "additions" in relevant, (
            f"P1-CHAT-CANCEL-A11Y: aria-relevant={relevant!r} no incluye "
            f"`additions` — bubbles nuevos no se anunciarían."
        )


# -- MessageBubble aria-busy on streaming --


def test_streaming_bubble_marked_aria_busy(message_bubble_src: str):
    """Mientras `msg.isStreaming === true`, el bubble debe spread
    `aria-busy: true`. Esto suprime announcements parciales durante el
    stream chunked — el screen reader anuncia el mensaje completo recién
    cuando `isStreaming===false` (aria-busy=false dispara el announcement)."""
    pattern = re.compile(
        r"msg\.role\s*===\s*['\"]model['\"]\s*&&\s*msg\.isStreaming"
        r"[\s\S]{0,200}?['\"]aria-busy['\"]\s*:\s*true"
    )
    assert pattern.search(message_bubble_src), (
        "P1-CHAT-CANCEL-A11Y regresión: bubbles streaming ya no spread "
        "`{'aria-busy': true}`. Sin ese flag, cada chunk parcial del LLM "
        "se anunciaría al screen reader, generando lectura entrecortada y "
        "frustrante para usuarios de tecnología asistiva."
    )
