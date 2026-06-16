"""[P3-DOC-1 · 2026-05-11] Plan.jsx::savePlanToHistory dead code removed.

Cierre del audit 2026-05-11 (P3 polish):
    Audit cross-codebase confirmó que `savePlanToHistory` en Plan.jsx
    era DEAD CODE (0 callers via `grep -r savePlanToHistory frontend/src`).
    Era la única excepción whitelisted a la invariante I6 (frontend NO
    escribe direct a `meal_plans`) que quedaba activa. El backend ya
    persiste vía `services._save_plan_and_track_background` post-SSE
    (comentario explícito en `AssessmentContext.jsx:1467`).

Cierre:
    1. Eliminada la función completa de Plan.jsx (líneas 366-428 pre-fix).
    2. Tooltip-anchor `P3-DOC-1-DEAD-CODE-REMOVED` documenta el cierre +
       guía para futuros devs (si necesitan fallback frontend-side, crear
       endpoint backend `POST /api/plans/persist-from-stream`, NO
       restaurar el INSERT directo).
    3. La señal `mealfit_history_dirty_at` se movió a
       `AssessmentContext.jsx::saveGeneratedPlan` (callsite real
       post-SSE-success, donde el backend ya persistió). El handshake
       con `History.jsx` se preserva intacto — sólo cambió quién emite.
    4. Whitelist de Plan.jsx:398 removida de CLAUDE.md "Anti-patrones
       de frontend prohibidos > Operaciones permitidas". Ahora **cero
       excepciones whitelisted sobre `meal_plans` desde el frontend**.

Tests funcionales del handshake P0-HIST-NEW-2 viven en
`test_p0_hist_new_2_savePlanToHistory_signal.py` (actualizado para
apuntar al nuevo callsite). Este archivo es el **marker anchor** para
el cross-link bidireccional `P2-HIST-AUDIT-14` (slug `p3_doc_1`
derivado del marker `P3-DOC-1 · 2026-05-11`).

Tooltip-anchor: P3-DOC-1-MARKER-ANCHOR
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAN_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Plan.jsx"
_ASSESSMENT_CTX = _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"


@pytest.fixture(scope="module")
def plan_src() -> str:
    return _PLAN_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def context_src() -> str:
    return _ASSESSMENT_CTX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def claude_md() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


def test_dead_code_removed_anchor_present(plan_src: str):
    """Plan.jsx contiene el tooltip-anchor `P3-DOC-1-DEAD-CODE-REMOVED`
    documentando la decisión + guía para fallback futuro."""
    assert "P3-DOC-1-DEAD-CODE-REMOVED" in plan_src, (
        "P3-DOC-1 regresión: tooltip-anchor `P3-DOC-1-DEAD-CODE-REMOVED` "
        "desapareció de Plan.jsx. Sin él, un dev futuro puede re-introducir "
        "`savePlanToHistory` sin contexto de por qué se eliminó. Restaurar "
        "el bloque de comentario que documenta:\n"
        "  - dead code (0 callers via grep);\n"
        "  - backend ya persiste;\n"
        "  - señal movida a AssessmentContext;\n"
        "  - guía: si necesitas fallback, endpoint backend, no INSERT directo."
    )


def test_save_plan_to_history_function_gone(plan_src: str):
    """La función `savePlanToHistory` está eliminada (no `const ... = async`)."""
    assert "const savePlanToHistory = async" not in plan_src, (
        "P3-DOC-1 regresión: la función `savePlanToHistory` reaparecio. "
        "Era dead code + violaba I6. Anti-regression catch — eliminar de "
        "nuevo o, si es necesario, migrar a endpoint backend."
    )


def test_signal_moved_to_assessment_context(context_src: str):
    """`mealfit_history_dirty_at` setItem vive en AssessmentContext, no
    en Plan.jsx."""
    assert "mealfit_history_dirty_at" in context_src, (
        "P3-DOC-1 regresión: la señal `mealfit_history_dirty_at` no aparece "
        "en AssessmentContext.jsx. El handshake con History.jsx se rompe — "
        "restaurar el setItem dentro de `saveGeneratedPlan`."
    )
    # [P2-AUDIT-3 · 2026-05-15] El raw `localStorage.setItem(...)` fue migrado
    # al helper SSOT `safeLocalStorageSet(...)` (frontend/src/utils/safeLocalStorage.js)
    # que atrapa SecurityError/QuotaExceededError; internamente llama a
    # `window.localStorage.setItem(key, serialized)` (línea 73), así que el
    # handshake con History.jsx se preserva. Aceptamos AMBAS formas.
    pattern = re.compile(
        r"(?:localStorage\s*\.\s*setItem|safeLocalStorageSet)\s*\(\s*['\"]mealfit_history_dirty_at['\"]"
    )
    assert pattern.search(context_src), (
        "P3-DOC-1: `mealfit_history_dirty_at` aparece en AssessmentContext "
        "pero NO como `localStorage.setItem` ni `safeLocalStorageSet`. "
        "Necesario para el handshake."
    )


def test_claude_md_whitelist_entry_removed(claude_md: str):
    """La entry `Plan.jsx:398` whitelisted desapareció de CLAUDE.md
    'Operaciones permitidas'."""
    # Buscar la sección "Operaciones permitidas" y verificar que no menciona
    # Plan.jsx:398 como whitelist.
    section_idx = claude_md.find("### Operaciones permitidas (whitelist documentada)")
    assert section_idx > 0, "sección 'Operaciones permitidas' no encontrada en CLAUDE.md"
    # Sección ~hasta el siguiente ###/---
    next_section = re.search(r"\n###|\n---", claude_md[section_idx + 60:])
    end = section_idx + 60 + (next_section.start() if next_section else 2000)
    section = claude_md[section_idx: end]
    assert "Plan.jsx:398" not in section, (
        "P3-DOC-1 regresión: la whitelist entry `Plan.jsx:398` reaparecio en "
        "CLAUDE.md 'Operaciones permitidas'. La función fue eliminada — la "
        "entry queda inconsistente con el código real. Removerla de la tabla."
    )
    # Y debe haber nota explicativa del cierre
    assert "P3-DOC-1" in section, (
        "P3-DOC-1: la sección no menciona el cierre P3-DOC-1. Documentar "
        "explícitamente que la última excepción whitelisted sobre meal_plans "
        "fue eliminada (ayuda a auditores futuros a confirmar el estado)."
    )


def test_zero_meal_plans_writes_in_plan_jsx_code(plan_src: str):
    """Cero líneas de código (no comentarios) con `supabase.from('meal_plans')`
    en Plan.jsx — la función eliminada era la única ocurrencia."""
    code_lines = [
        ln for ln in plan_src.split("\n")
        if "supabase.from('meal_plans')" in ln
        and not ln.strip().startswith("//")
        and not ln.strip().startswith("*")
    ]
    assert not code_lines, (
        f"P3-DOC-1 regresión: encontradas {len(code_lines)} líneas de código "
        f"con `supabase.from('meal_plans')` en Plan.jsx (excluyendo comments). "
        f"Líneas:\n  " + "\n  ".join(code_lines)
    )
