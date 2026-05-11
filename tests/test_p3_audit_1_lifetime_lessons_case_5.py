"""[P3-AUDIT-1 · 2026-05-10] Docstring de `/lifetime-lessons` enumera
5 casos legítimos de `last_chunk_learning = null`.

Bug original (audit 2026-05-10):
    Docstring P3-NEW-2 documentaba 4 casos legítimos pero NO mencionaba
    el caso "seed fallido sin rebuild automático". Si
    `_seed_last_chunk_learning` falla `_seed_attempts=3` en T1 pre-merge,
    el campo persiste `null` Y NO hay cron de retry automático. Operador
    interpretaba "null = espera cron" cuando no había rescate posible
    → silent degradation indefinida.

Fix:
    Añadido caso 5 al docstring con triple info:
    - Síntoma observable (`null` indefinido).
    - Causa raíz (`_seed_attempts=3` agotado pre-merge).
    - Recuperación operacional (trigger manual via `/regenerate-simplified`).
    Sección "BUG real" actualizada para excluir este caso del set
    accionable (evita false alarm si SRE observa drift).

Tests (parser-based sobre routers/plans.py):
    1. Docstring de `api_plan_lifetime_lessons` menciona el caso 5
       (P3-AUDIT-1 anchor + texto "seed fallido").
    2. El número de casos enumerados en la sección "Cuándo esperar `null`"
       es ≥5 (regex sobre items numerados).
    3. La sección "SÍ ES BUG" excluye explícitamente el caso seed-fallido
       (mención de `_seed_attempts`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


def _extract_lifetime_lessons_docstring(src: str) -> str:
    """Localiza el docstring de la función `api_plan_lifetime_lessons`."""
    pattern = re.compile(
        r'def\s+api_plan_lifetime_lessons\s*\([^)]*\)[^:]*:\s*\n\s*"""(.*?)"""',
        re.DOTALL,
    )
    m = pattern.search(src)
    assert m, (
        "P3-AUDIT-1: no se encontró el docstring de `api_plan_lifetime_lessons` "
        "en routers/plans.py — ¿se refactorizó?"
    )
    return m.group(1)


def test_docstring_mentions_p3_audit_1_anchor(plans_src: str):
    """El docstring debe tener anchor `[P3-AUDIT-1 · 2026-05-10]` que
    permita a un futuro auditor encontrar el contexto de POR QUÉ se
    añadió el caso 5."""
    docstring = _extract_lifetime_lessons_docstring(plans_src)
    assert "P3-AUDIT-1" in docstring, (
        "P3-AUDIT-1 regresión: docstring perdió el anchor "
        "`[P3-AUDIT-1 · 2026-05-10]`. Sin el anchor, el siguiente "
        "mantenedor no entiende POR QUÉ se documentó el caso seed-fallido."
    )


def test_docstring_mentions_seed_failure_recovery(plans_src: str):
    """El docstring debe mencionar la recuperación operacional
    (`/regenerate-simplified`) para el caso seed-fallido. Sin esto,
    el operador ve "null indefinido" sin saber qué hacer."""
    docstring = _extract_lifetime_lessons_docstring(plans_src)
    assert "regenerate-simplified" in docstring or "_seed_attempts" in docstring, (
        "P3-AUDIT-1 regresión: docstring no menciona `/regenerate-simplified` "
        "ni `_seed_attempts`. La recuperación operacional para el caso "
        "seed-fallido quedó sin documentar — operador no sabe qué hacer."
    )


def test_docstring_lists_at_least_5_null_cases(plans_src: str):
    """La sección "Cuándo esperar `null`" debe enumerar ≥5 casos
    (numerados `1.`, `2.`, ...). Si alguien borra el caso 5 en un
    refactor cosmético, este test falla."""
    docstring = _extract_lifetime_lessons_docstring(plans_src)
    # Localizar el bloque "Cuándo esperar null en estos campos".
    section_match = re.search(
        r"Cu[aá]ndo\s+esperar\s+`null`\s+en\s+estos\s+campos.*?"
        r"(?=Cu[aá]ndo\s+esperar\s+`null`\s+y\s+S[IÍ]|$)",
        docstring,
        re.DOTALL | re.IGNORECASE,
    )
    assert section_match, (
        "P3-AUDIT-1: no se encontró la sección 'Cuándo esperar `null` en "
        "estos campos' en el docstring."
    )
    section = section_match.group(0)
    # Contar items numerados al inicio de línea.
    items = re.findall(r"^\s*(\d+)\.\s", section, re.MULTILINE)
    distinct = {int(n) for n in items}
    assert len(distinct) >= 5, (
        f"P3-AUDIT-1 regresión: la sección lista solo "
        f"{len(distinct)} casos numerados ({sorted(distinct)}). "
        f"Debe haber ≥5 incluyendo el caso `seed fallido sin rebuild` "
        f"añadido por P3-AUDIT-1."
    )


def test_docstring_bug_section_excludes_seed_failed_case(plans_src: str):
    """La sección "SÍ ES BUG" debe excluir el caso seed-fallido (mencionar
    `_seed_attempts` como qualifier). Sin esta exclusión, SRE puede
    confundir un seed agotado con un bug del cron."""
    docstring = _extract_lifetime_lessons_docstring(plans_src)
    bug_section = re.search(
        r"S[IÍ]\s+ES\s+BUG.*?(?=Raises|$)",
        docstring,
        re.DOTALL | re.IGNORECASE,
    )
    assert bug_section, (
        "P3-AUDIT-1: no se encontró la sección 'SÍ ES BUG' en el docstring."
    )
    body = bug_section.group(0)
    assert "_seed_attempts" in body or "seed" in body.lower(), (
        "P3-AUDIT-1 regresión: la sección 'SÍ ES BUG' no excluye el caso "
        "seed-fallido (mención de `_seed_attempts`). SRE puede confundir "
        "un seed agotado con un bug del cron de merge."
    )
