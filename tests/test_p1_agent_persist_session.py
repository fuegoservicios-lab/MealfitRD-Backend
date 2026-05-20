"""[P1-AGENT-PERSIST-SESSION · 2026-05-20] Test anti-regresión de la
preservación del `currentSessionId` al re-montar `AgentPage.jsx`.

Bug observado:
    El `useState(() => { const newId = crypto.randomUUID(); ...; return newId })`
    de `currentSessionId` ejecutaba el initializer en cada mount del
    componente. Como React Router DESMONTA el componente al navegar
    (Nevera/Plan/Recetas → Agente), cada navegación generaba un UUID
    nuevo, lo persistía sobrescribiendo `mealfit_current_session`, y
    perdía el chat activo del user. UX visible: "se refresca y molesta"
    (reportado 2026-05-20) — el chat en curso desaparece, vuelve al
    welcome screen, fetchSessionMessages no encuentra mensajes para la
    sesión recién-creada.

Fix:
    El initializer del useState DEBE leer `mealfit_current_session`
    de localStorage primero. Solo si no existe (o no es un UUID válido)
    genera uno nuevo. Patron clásico de "hydrate-from-storage".

Test parser-based: anchor en el initializer literal.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_agent_page_reads_session_from_local_storage_at_mount():
    """[P1-AGENT-PERSIST-SESSION] El initializer de `currentSessionId`
    DEBE leer `mealfit_current_session` de localStorage ANTES de generar
    UUID nuevo.

    Anti-pattern bloqueado: `useState(() => { const newId = crypto.randomUUID();
    ...; return newId })` SIN read previo de localStorage.
    """
    src = _read(_AGENT_PAGE_JSX)
    # Extraer el bloque del initializer de currentSessionId.
    match = re.search(
        r"const\s*\[\s*currentSessionId\s*,\s*_setCurrentSessionId\s*\]\s*=\s*useState\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*\)",
        src,
        re.DOTALL,
    )
    assert match, (
        "useState initializer de `currentSessionId` no encontrado — "
        "refactor inesperado."
    )
    body = match.group(1)
    # Anchor: debe haber un read de localStorage (via safeLocalStorageGet
    # o cualquier otro method) ANTES de cualquier crypto.randomUUID().
    storage_read_idx = -1
    uuid_gen_idx = -1
    storage_match = re.search(
        r"safeLocalStorageGet\s*\(\s*['\"]mealfit_current_session",
        body,
    )
    if storage_match:
        storage_read_idx = storage_match.start()
    uuid_match = re.search(r"crypto\.randomUUID\s*\(", body)
    if uuid_match:
        uuid_gen_idx = uuid_match.start()

    assert storage_read_idx >= 0, (
        "El initializer NO lee `mealfit_current_session` de localStorage. "
        "Cada re-mount sobrescribe la sesión activa con un UUID nuevo. Ver "
        "P1-AGENT-PERSIST-SESSION · 2026-05-20."
    )
    if uuid_gen_idx >= 0:
        assert storage_read_idx < uuid_gen_idx, (
            f"`safeLocalStorageGet` en pos {storage_read_idx} viene DESPUÉS "
            f"de `crypto.randomUUID()` en pos {uuid_gen_idx}. El read debe ir "
            f"PRIMERO (fast-path), generar UUID solo si no hay sesión "
            f"persistida. Ver P1-AGENT-PERSIST-SESSION."
        )


def test_safe_local_storage_get_imported():
    """[P1-AGENT-PERSIST-SESSION] El helper `safeLocalStorageGet` debe
    estar importado desde utils/safeLocalStorage para que el initializer
    pueda usarlo."""
    src = _read(_AGENT_PAGE_JSX)
    assert "safeLocalStorageGet" in src, (
        "Import de `safeLocalStorageGet` ausente — el initializer no puede "
        "leer de localStorage. Ver P1-AGENT-PERSIST-SESSION · 2026-05-20."
    )
    # Sanity: el import debe venir del helper SSOT (no inline localStorage.getItem).
    assert re.search(
        r"import\s*\{[^}]*safeLocalStorageGet[^}]*\}\s*from\s*['\"][^'\"]*safeLocalStorage",
        src,
    ), "El helper debe importarse desde utils/safeLocalStorage (SSOT)."
