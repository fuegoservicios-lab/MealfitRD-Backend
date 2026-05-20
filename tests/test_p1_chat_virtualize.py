"""[P1-CHAT-VIRTUALIZE · 2026-05-19] Test parser-based: el chat virtualiza
el listado de mensajes con `react-virtuoso` cuando
`messages.length > VIRTUALIZE_THRESHOLD` (default 100).

Cierre del último P1 pendiente del audit prod-readiness del Agente
(2026-05-19): pre-fix, `messages.map((msg, i) => <MessageBubble />)`
renderizaba TODOS los mensajes en DOM aún cuando la sesión tenía 500+
mensajes — re-renders caros, scroll janky, memoria creciente. Fix:
`<VirtualizedMessageList>` (Virtuoso) mide alturas con ResizeObserver,
sólo monta los items en viewport + buffer, y soporta follow-tail
("stick to bottom unless user scrolled up") nativamente.

Por qué Virtuoso y no react-window:
    Mensajes tienen alturas variables (texto / markdown / imágenes) y
    crecen durante streaming. react-window requiere conocer alturas;
    Virtuoso las mide. Bundle ~28KB gzip — accepted trade-off para
    cerrar el P1 sin reinventar scroll-anchoring.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p1_chat_virtualize`
matchea este archivo `test_p1_chat_virtualize.py`.

Tooltip-anchor: P1-CHAT-VIRTUALIZE | audit 2026-05-19 P1 cierre final
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_VIRT_LIST_JSX = (
    _REPO_ROOT / "frontend" / "src" / "components" / "agent" / "VirtualizedMessageList.jsx"
)
_PACKAGE_JSON = _REPO_ROOT / "frontend" / "package.json"


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def virt_list_src() -> str:
    return _VIRT_LIST_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def package_json() -> dict:
    return json.loads(_PACKAGE_JSON.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Dep + archivo del componente
# -----------------------------------------------------------------------------


def test_react_virtuoso_in_dependencies(package_json: dict):
    """`react-virtuoso` debe estar en `dependencies` (no devDependencies —
    se usa en runtime)."""
    deps = package_json.get("dependencies", {})
    assert "react-virtuoso" in deps, (
        "P1-CHAT-VIRTUALIZE regresión: `react-virtuoso` no está en "
        "frontend/package.json#dependencies. Sin la dep el componente "
        "VirtualizedMessageList rompe el build."
    )


def test_virtualized_component_file_exists():
    assert _VIRT_LIST_JSX.exists(), (
        "P1-CHAT-VIRTUALIZE regresión: "
        f"{_VIRT_LIST_JSX} no existe."
    )


def test_anchor_present_virt_list(virt_list_src: str):
    assert "P1-CHAT-VIRTUALIZE" in virt_list_src, (
        "P1-CHAT-VIRTUALIZE regresión: anchor textual perdido en "
        "VirtualizedMessageList.jsx."
    )


def test_virt_list_imports_virtuoso(virt_list_src: str):
    """El componente debe importar `Virtuoso` desde `react-virtuoso`."""
    pattern = re.compile(
        r"import\s*\{\s*Virtuoso\s*\}\s*from\s*['\"]react-virtuoso['\"]"
    )
    assert pattern.search(virt_list_src), (
        "P1-CHAT-VIRTUALIZE regresión: import `{ Virtuoso }` desde "
        "`react-virtuoso` no encontrado en VirtualizedMessageList.jsx."
    )


def test_virtualize_threshold_constant_exported(virt_list_src: str):
    """`VIRTUALIZE_THRESHOLD` debe declararse y exportarse para que
    AgentPage haga el switch.

    El valor debe ser un entero positivo razonable (>=50). Valores muy
    bajos virtualizan sesiones cortas sin beneficio; muy altos retrasan
    el cierre del P1."""
    pattern = re.compile(
        r"export\s+const\s+VIRTUALIZE_THRESHOLD\s*=\s*(\d+)\s*;"
    )
    m = pattern.search(virt_list_src)
    assert m, (
        "P1-CHAT-VIRTUALIZE regresión: `export const VIRTUALIZE_THRESHOLD = N` "
        "no encontrado."
    )
    value = int(m.group(1))
    assert 50 <= value <= 500, (
        f"P1-CHAT-VIRTUALIZE: VIRTUALIZE_THRESHOLD={value} fuera del rango "
        f"razonable [50,500]. El audit pedía >200 — usar 100-200 es óptimo."
    )


def test_virt_list_renders_message_bubble(virt_list_src: str):
    """El item de Virtuoso debe renderizar `MemoizedMessageBubble` —
    NO un componente custom que pierda el wiring de actions / error UI."""
    assert "MemoizedMessageBubble" in virt_list_src, (
        "P1-CHAT-VIRTUALIZE regresión: el item del Virtuoso ya no usa "
        "MemoizedMessageBubble. El componente unifica streaming bubble, "
        "error bubble con role=\"alert\", y MessageActions — duplicar la "
        "lógica acá rompería P1-CHAT-ERROR-DIFF y P1-CHAT-CANCEL-A11Y."
    )


def test_virt_list_uses_follow_output(virt_list_src: str):
    """Virtuoso debe usar `followOutput` para auto-scroll a mensajes
    nuevos cuando el user está cerca del bottom. Sin él, sesiones largas
    pierden la UX de "stick to bottom"."""
    pattern = re.compile(r"followOutput\s*=\s*[\"'][^\"']+[\"']")
    assert pattern.search(virt_list_src), (
        "P1-CHAT-VIRTUALIZE regresión: prop `followOutput` no encontrada en "
        "el Virtuoso. Sin él, mensajes nuevos no auto-scrollean."
    )


# -----------------------------------------------------------------------------
# AgentPage.jsx — switch threshold + import
# -----------------------------------------------------------------------------


def test_anchor_present_agent_page(agent_page_src: str):
    assert "P1-CHAT-VIRTUALIZE" in agent_page_src, (
        "P1-CHAT-VIRTUALIZE regresión: anchor textual perdido en AgentPage.jsx."
    )


def test_agent_page_imports_virtualize_threshold(agent_page_src: str):
    """AgentPage debe importar `VirtualizedMessageList` y `VIRTUALIZE_THRESHOLD`
    desde el módulo VirtualizedMessageList."""
    pattern = re.compile(
        r"import\s*\{\s*(?:VirtualizedMessageList\s*,\s*VIRTUALIZE_THRESHOLD"
        r"|VIRTUALIZE_THRESHOLD\s*,\s*VirtualizedMessageList)\s*\}"
        r"\s*from\s*['\"]\.\./components/agent/VirtualizedMessageList['\"]"
    )
    assert pattern.search(agent_page_src), (
        "P1-CHAT-VIRTUALIZE regresión: el import de VirtualizedMessageList + "
        "VIRTUALIZE_THRESHOLD no se encuentra en AgentPage.jsx. Sin él, el "
        "switch del path virtualizado no funciona."
    )


def test_threshold_switch_in_render(agent_page_src: str):
    """En el render, debe existir un check `messages.length > VIRTUALIZE_THRESHOLD`
    que decida entre `<VirtualizedMessageList>` y el `messages.map(...)` simple."""
    pattern = re.compile(
        r"messages\.length\s*>\s*VIRTUALIZE_THRESHOLD"
    )
    matches = pattern.findall(agent_page_src)
    assert len(matches) >= 1, (
        "P1-CHAT-VIRTUALIZE regresión: comparación "
        "`messages.length > VIRTUALIZE_THRESHOLD` no encontrada en el "
        "render. Sin ella el path simple corre para todas las sesiones."
    )


def test_virtualized_path_uses_component(agent_page_src: str):
    """El branch `messages.length > VIRTUALIZE_THRESHOLD` debe renderizar
    `<VirtualizedMessageList`."""
    pattern = re.compile(
        r"messages\.length\s*>\s*VIRTUALIZE_THRESHOLD[\s\S]{0,500}?<VirtualizedMessageList"
    )
    assert pattern.search(agent_page_src), (
        "P1-CHAT-VIRTUALIZE regresión: el branch threshold no renderiza "
        "<VirtualizedMessageList> en los próximos 500 chars. ¿Renombraste "
        "el componente?"
    )


def test_messages_map_path_preserved(agent_page_src: str):
    """El path simple `messages.map(...)` con `MemoizedMessageBubble` debe
    seguir presente — ES el fallback para sesiones cortas (99% del uso)."""
    pattern = re.compile(
        r"messages\.map\([\s\S]{0,400}?<MemoizedMessageBubble"
    )
    assert pattern.search(agent_page_src), (
        "P1-CHAT-VIRTUALIZE regresión: el path simple "
        "`messages.map(...) <MemoizedMessageBubble>` ya no está. Romperías "
        "todas las sesiones <= VIRTUALIZE_THRESHOLD. El path simple es el "
        "default — el virtualizado es la excepción para sesiones largas."
    )


def test_container_overflow_adapts_when_virtualized(agent_page_src: str):
    """El `.messages-container` padre debe ceder el scroll a Virtuoso
    cuando virtualizado: `overflowY: 'hidden'` si threshold cruzado,
    `'auto'` en otro caso. Sin esto, doble-scroll rompe el follow-tail."""
    pattern = re.compile(
        r"overflowY\s*:\s*messages\.length\s*>\s*VIRTUALIZE_THRESHOLD"
        r"\s*\?\s*['\"]hidden['\"]\s*:\s*['\"]auto['\"]"
    )
    assert pattern.search(agent_page_src), (
        "P1-CHAT-VIRTUALIZE regresión: el `.messages-container` ya no "
        "alterna `overflowY` entre 'hidden' (virtualizado) y 'auto' "
        "(simple). Sin esto Virtuoso tendría doble scroll y rompería el "
        "follow-tail."
    )


def test_role_log_present_in_both_paths(agent_page_src: str):
    """Ambos paths (virtualizado + simple) deben envolver el render en
    un `<div role="log" aria-live="polite">` — preservar P1-CHAT-A11Y-LIVE."""
    role_log_count = len(re.findall(r"role\s*=\s*[\"']log[\"']", agent_page_src))
    aria_live_count = len(re.findall(r"aria-live\s*=\s*[\"']polite[\"']", agent_page_src))
    assert role_log_count >= 2, (
        f"P1-CHAT-VIRTUALIZE regresión: solo {role_log_count} `role=\"log\"` "
        f"encontrado(s) — deberían ser 2 (uno por path: virtualizado + simple). "
        f"Sin ambos, screen readers pierden el contexto en sesiones largas."
    )
    assert aria_live_count >= 2, (
        f"P1-CHAT-VIRTUALIZE regresión: solo {aria_live_count} `aria-live="
        f"\"polite\"` encontrado(s) — deberían ser 2 (uno por path)."
    )
