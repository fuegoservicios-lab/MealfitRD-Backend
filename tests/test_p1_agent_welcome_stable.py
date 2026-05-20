"""[P1-AGENT-WELCOME-STABLE · 2026-05-20] Anti-regresión del welcome screen
estable — no regenerar la hora con cada re-fetch innecesario.

Bug observado:
    `fetchSessionMessages` (useCallback) tenía deps en `userProfile`,
    `formData`, `planData`. Esos values cambian con polling del
    AssessmentContext → callback se recrea → useEffect dispara fetch →
    al ver sesión vacía, llamaba `setMessages([{ welcome con hora
    actualizada }])`. Resultado UX: el welcome screen se "refrescaba
    varias veces" — la hora cambiaba visiblemente (04:25 → 04:26 → ...).
    Reportado 2026-05-20.

Fix:
    Helper `_setWelcomeIfAbsent` que solo setea welcome si NO hay uno
    ya activo. Usa `setMessages(prev => ...)` con guard:
        if (prev.length === 1 && prev[0]?.isWelcome) return prev;
    React detecta misma referencia → no-op → no rerender → no flash.

    Los 3 callsites en `fetchSessionMessages` que regeneraban welcome
    (sesión vacía, response no-ok tras retries, catch network) usan el
    helper en lugar de `setMessages([{welcome ...}])` literal.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_set_welcome_if_absent_helper_exists():
    """[P1-AGENT-WELCOME-STABLE] Helper `_setWelcomeIfAbsent` debe existir
    y usar el guard `prev[0]?.isWelcome` para preservar el welcome existente."""
    src = _read(_AGENT_PAGE_JSX)
    assert "_setWelcomeIfAbsent" in src, (
        "Helper `_setWelcomeIfAbsent` ausente. Sin él, los callsites de "
        "fetchSessionMessages regeneran welcome con hora actualizada en cada "
        "re-fetch. Ver P1-AGENT-WELCOME-STABLE · 2026-05-20."
    )
    # Anchor: el helper usa setMessages(prev => ...) con guard isWelcome.
    helper_match = re.search(
        r"_setWelcomeIfAbsent\s*=\s*useCallback\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*,",
        src,
        re.DOTALL,
    )
    assert helper_match, "`_setWelcomeIfAbsent = useCallback(...)` no encontrado."
    body = helper_match.group(1)
    # Guard: si ya hay welcome, retornar prev (misma referencia).
    assert re.search(
        r"prev\[\s*0\s*\]\??\.isWelcome",
        body,
    ), (
        "Guard `prev[0]?.isWelcome` ausente. Sin él, el helper SIEMPRE genera "
        "welcome nuevo → mismo bug del refresh visible."
    )
    # El return cuando hay welcome es `prev` (misma referencia → no-op de React).
    assert re.search(r"return\s+prev\s*;", body), (
        "El helper no retorna `prev` cuando hay welcome activo. Debe devolver "
        "la misma referencia para que React skipee el rerender."
    )


def test_fetch_session_messages_uses_helper():
    """[P1-AGENT-WELCOME-STABLE] Los 3 callsites de welcome regeneration
    en `fetchSessionMessages` deben usar `_setWelcomeIfAbsent()` en lugar
    de `setMessages([{welcome ...}])` literal."""
    src = _read(_AGENT_PAGE_JSX)
    # Buscar el cuerpo de fetchSessionMessages.
    fn_match = re.search(
        r"const\s+fetchSessionMessages\s*=\s*useCallback\(.*?\}\s*,\s*\[",
        src,
        re.DOTALL,
    )
    assert fn_match
    body = fn_match.group(0)
    # Anti-pattern: setMessages([{...generateIntelligentWelcome...}]) inline.
    inline_welcome_calls = re.findall(
        r"setMessages\(\s*\[\s*\{[^}]*generateIntelligentWelcome",
        body,
    )
    assert not inline_welcome_calls, (
        f"{len(inline_welcome_calls)} callsite(s) inline de "
        f"`setMessages([{{...generateIntelligentWelcome...}}])` en "
        f"fetchSessionMessages — deben usar `_setWelcomeIfAbsent()`. Ver "
        f"P1-AGENT-WELCOME-STABLE · 2026-05-20."
    )
    # Sanity positiva: al menos 2 invocaciones de _setWelcomeIfAbsent en el cuerpo.
    helper_calls = re.findall(r"_setWelcomeIfAbsent\s*\(\s*\)", body)
    assert len(helper_calls) >= 2, (
        f"Solo {len(helper_calls)} invocaciones de `_setWelcomeIfAbsent()` en "
        f"fetchSessionMessages — esperaba >=2 (sesión vacía + response no-ok). "
        f"El callsite del catch puede estar fuera del match. Ver "
        f"P1-AGENT-WELCOME-STABLE."
    )


def test_fetch_session_messages_deps_minimal():
    """[P1-AGENT-WELCOME-STABLE] El useCallback de `fetchSessionMessages`
    NO debe tener `userProfile`/`formData`/`planData` en sus deps — esas
    cambian con polling y hacen que la callback se re-cree constantemente,
    disparando el useEffect que la invoca → re-fetch innecesario.

    Las deps deben ser solo `setMessages`, `setIsLoadingHistory`,
    `_setWelcomeIfAbsent` (que YA encapsula esos values internamente)."""
    src = _read(_AGENT_PAGE_JSX)
    # Buscar el dep array final del useCallback de fetchSessionMessages.
    # Pattern: `}, [setMessages, setIsLoadingHistory, _setWelcomeIfAbsent]);`
    deps_match = re.search(
        r"const\s+fetchSessionMessages\s*=\s*useCallback\(.*?\}\s*,\s*\[([^\]]+)\]\s*\)\s*;",
        src,
        re.DOTALL,
    )
    assert deps_match, "Dep array de fetchSessionMessages no encontrado."
    deps_str = deps_match.group(1)
    forbidden = ["userProfile", "formData", "planData"]
    for dep in forbidden:
        assert dep not in deps_str, (
            f"Dep `{dep}` en fetchSessionMessages useCallback — causa re-creates "
            f"con cada cambio del AssessmentContext y dispara re-fetch ciclic. "
            f"Encapsular en `_setWelcomeIfAbsent`. Ver "
            f"P1-AGENT-WELCOME-STABLE · 2026-05-20."
        )
    # Sanity: las deps actuales deben ser las 3 esperadas.
    assert "_setWelcomeIfAbsent" in deps_str, (
        "`_setWelcomeIfAbsent` debe estar en deps (helper encapsula los datos "
        "del profile/plan que el welcome usa)."
    )
