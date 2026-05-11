"""[P3-NEW-A · 2026-05-11] Test ancla: la sección "Anti-patrones de
frontend prohibidos" DEBE estar presente en CLAUDE.md con la tabla de
operaciones-prohibidas/reemplazo + cross-link al test blanket P1-NEW-A.

Motivación:
    La invariante I6 ("mutaciones a plan_data desde frontend prohibidas")
    está documentada en la tabla del lifecycle. Pero un desarrollador
    nuevo que aterrice en el repo y quiera entender QUÉ operaciones
    cliente están permitidas vs prohibidas, necesita una sección
    dedicada con el contrato explícito y el endpoint de reemplazo para
    cada anti-patrón. Sin esta sección:

      - I6 documenta la regla, pero no enumera los reemplazos.
      - El test blanket `test_p1_new_a_*` enforza el contrato, pero un
        fallo no incluye el endpoint backend de reemplazo en su mensaje.
      - El operador tendría que leer 6 P-fixes individuales (P0-NEW-A,
        P0-NEW-B, P1-HIST-5, etc.) para reconstruir la tabla.

    P3-NEW-A consolida toda esa información en una sección dedicada de
    CLAUDE.md, y este test la ancla: si el bloque se elimina o pierde
    referencias clave, CI falla.

Tests:
    1. Sección `## Anti-patrones de frontend prohibidos` existe.
    2. Tabla de operaciones prohibidas menciona los 6 endpoints de
       reemplazo (P0-NEW-A, P0-NEW-B, P1-HIST-5, /restore, recipe/expand).
    3. Tabla de operaciones permitidas (whitelist) menciona los 3 casos
       canónicos (Plan.jsx INSERT inicial, Pantry delete, increment_inventory).
    4. Cross-link a `test_p1_new_a_frontend_no_direct_meal_plans_write.py`.
    5. Cross-link slug del marker.

Tooltip-anchor: P3-NEW-A-START | gap P3 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_CLAUDE_MD = Path(__file__).resolve().parent.parent.parent / "CLAUDE.md"


@pytest.fixture(scope="module")
def claude_md_text() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def anti_patterns_section(claude_md_text: str) -> str:
    """Extrae el bloque desde el header `## Anti-patrones de frontend
    prohibidos` hasta el siguiente `## ` (cierre de sección)."""
    marker = "## Anti-patrones de frontend prohibidos"
    idx = claude_md_text.find(marker)
    assert idx >= 0, (
        "P3-NEW-A regresión: el header `## Anti-patrones de frontend "
        "prohibidos` no existe en CLAUDE.md. Si fue eliminado por un "
        "refactor, restaurarlo después de `## Advisors aceptados` con la "
        "tabla de operaciones prohibidas + permitidas + cross-link al "
        "test blanket P1-NEW-A."
    )
    rest = claude_md_text[idx:]
    # Cortar en el próximo `## ` (header del mismo nivel).
    cut = re.search(r"\n## ", rest[len(marker):])
    if cut:
        return rest[: len(marker) + cut.start()]
    return rest


# ---------------------------------------------------------------------------
# 1. Sección existe y referencia I6
# ---------------------------------------------------------------------------
def test_section_present_with_i6_reference(anti_patterns_section: str):
    """La sección DEBE referenciar la invariante I6 del lifecycle (que
    documenta la regla raíz). Sin esta referencia, la sección queda
    descontextualizada del contrato arquitectónico.
    """
    assert "I6" in anti_patterns_section, (
        "P3-NEW-A regresión: la sección no menciona la invariante `I6`. "
        "El test parser-based P1-NEW-A enforza esa invariante; la sección "
        "P3-NEW-A debe cross-linkearla para que el operador entienda el "
        "contrato arquitectónico (no solo la lista enumerada)."
    )


# ---------------------------------------------------------------------------
# 2. Tabla de operaciones prohibidas menciona los reemplazos canónicos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "required_endpoint",
    [
        "/swap-meal/persist",       # P0-NEW-A
        "/grocery-start-date",       # P0-NEW-B
        "/{plan_id}/name",           # P1-HIST-5 rename
        "/recipe/expand",            # P1-HIST-RECIPE-1
        "/restore",                  # P0-HIST-1
    ],
)
def test_prohibited_operations_mention_replacement_endpoint(
    anti_patterns_section: str, required_endpoint: str
):
    """Cada anti-patrón cliente DEBE tener su endpoint de reemplazo
    backend documentado en la tabla. Sin esto, un dev que vea el
    anti-patrón en el código no sabe qué endpoint usar para migrar.
    """
    assert required_endpoint in anti_patterns_section, (
        f"P3-NEW-A regresión: la tabla de operaciones prohibidas no "
        f"menciona el endpoint backend de reemplazo `{required_endpoint}`. "
        f"Sin este cross-link, el dev que vea el anti-patrón legacy no "
        f"sabe a qué endpoint migrar."
    )


# ---------------------------------------------------------------------------
# 3. Whitelist documentada (operaciones permitidas)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "required_whitelist_token",
    [
        "Plan.jsx",                          # INSERT inicial
        "Pantry.jsx",                        # delete user_inventory
        "increment_inventory_quantity",      # RPC SECURITY DEFINER
    ],
)
def test_whitelist_operations_documented(
    anti_patterns_section: str, required_whitelist_token: str
):
    """Cada operación cliente permitida DEBE estar listada con su
    razón. Sin esto, un dev podría asumir que TODA escritura cliente
    está prohibida y migrar innecesariamente operaciones legítimas
    (e.g., INSERT inicial de Plan.jsx que ocurre antes de que el plan
    exista en DB).
    """
    assert required_whitelist_token in anti_patterns_section, (
        f"P3-NEW-A regresión: la tabla de operaciones permitidas no "
        f"menciona `{required_whitelist_token}`. La whitelist explícita "
        f"protege contra migraciones innecesarias de operaciones "
        f"legítimas (e.g., INSERT inicial donde no hay plan_data que "
        f"pisar)."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link al test blanket P1-NEW-A
# ---------------------------------------------------------------------------
def test_cross_link_to_p1_new_a_blanket_test(anti_patterns_section: str):
    """La sección DEBE referenciar el test blanket
    `test_p1_new_a_frontend_no_direct_meal_plans_write.py` que enforza
    el contrato en CI. Sin cross-link, el operador no sabe qué test
    correr para verificar.
    """
    assert "test_p1_new_a_frontend_no_direct_meal_plans_write" in anti_patterns_section, (
        "P3-NEW-A regresión: la sección no referencia el test blanket "
        "`test_p1_new_a_frontend_no_direct_meal_plans_write.py`. Restaurar "
        "el cross-link para que el operador pueda ejecutar el enforcer "
        "directamente."
    )


# ---------------------------------------------------------------------------
# 5. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p3_new_a"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p3_new_a`)."
    )
