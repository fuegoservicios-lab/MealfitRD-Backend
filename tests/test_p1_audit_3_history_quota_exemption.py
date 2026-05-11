"""[P1-AUDIT-3 · 2026-05-10] Convención: GET polling endpoints del
Historial usan `get_verified_user_id` (NO `verify_api_quota`).

Bug original (audit 2026-05-10 flagged como gap P1):
    El audit observó que 3 endpoints GET del Historial usaban
    `Depends(get_verified_user_id)` mientras el resto de endpoints
    mutadores usaban `Depends(verify_api_quota)`. Aparente inconsistencia.

Decisión tomada (documentada en CLAUDE.md "Historial-quota-exemption"):
    Intencional. `verify_api_quota` es el paywall mensual (gratis=15,
    basic=50, etc.) que devuelve HTTP 402 cuando se excede. En GETs
    read-only de visualización del Historial, aplicar paywall negaría
    al usuario VER SU PROPIO historial tras alcanzar el cap mensual.
    UX inaceptable.

    Rate-limiting por hammering es un concern ortogonal — el tool
    correcto sería un middleware per-IP/user (slowapi), NO el paywall
    mensual.

Este test ancla la decisión:
    Si alguien "arregla" estos 3 endpoints reemplazando con
    `verify_api_quota`, el test falla con copy explicativo apuntando
    a CLAUDE.md. El gap del audit queda cerrado como "intencional",
    análogo al patrón de "Advisors aceptados".
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


def _extract_endpoint_decorator_and_signature(src: str, endpoint: str) -> str:
    """Localiza `@router.get("<endpoint>")` + función + signature
    completa hasta el cierre de `)` del def."""
    pattern = re.compile(
        rf'@router\.get\(\s*["\']{re.escape(endpoint)}["\']\s*\).*?def\s+\w+\s*\((.*?)\)\s*:',
        re.DOTALL,
    )
    m = pattern.search(src)
    assert m, f"No se encontró endpoint `{endpoint}` en routers/plans.py."
    return m.group(1)


@pytest.mark.parametrize("endpoint", [
    "/lessons-counts",
    "/history-status-summary",
    "/history-list",
])
def test_endpoint_uses_get_verified_user_id(plans_src: str, endpoint: str):
    """Cada uno de los 3 endpoints debe usar `Depends(get_verified_user_id)`
    (no `verify_api_quota`).

    Si este test falla porque alguien metió `verify_api_quota`, LEER
    CLAUDE.md sección "Historial-quota-exemption" — la decisión es
    intencional. Si la convención cambia, actualizar este test Y la
    doc en el mismo commit.
    """
    sig = _extract_endpoint_decorator_and_signature(plans_src, endpoint)
    assert "get_verified_user_id" in sig, (
        f"P1-AUDIT-3 regresión: endpoint `{endpoint}` ya no usa "
        f"`get_verified_user_id`. Ver CLAUDE.md "
        f"'Historial-quota-exemption' para entender por qué es "
        f"intencional. Si cambias la convención, actualiza CLAUDE.md "
        f"Y este test."
    )


@pytest.mark.parametrize("endpoint", [
    "/lessons-counts",
    "/history-status-summary",
    "/history-list",
])
def test_endpoint_does_not_use_verify_api_quota(plans_src: str, endpoint: str):
    """Defense-in-depth: estos 3 endpoints NO deben usar
    `verify_api_quota`. Aplicar el paywall en GET read-only del
    Historial bloquearía al usuario ver SU PROPIO data tras alcanzar
    el cap mensual → UX inaceptable (HTTP 402 en visualización).

    Si alguien añade `verify_api_quota` "para hardening", revertir y
    leer CLAUDE.md "Historial-quota-exemption". El concern de
    rate-limiting por hammering es ortogonal — usar middleware
    per-IP/user (slowapi), NO el paywall mensual.
    """
    sig = _extract_endpoint_decorator_and_signature(plans_src, endpoint)
    # Buscar `Depends(verify_api_quota)` específicamente — el string
    # `verify_api_quota` puede aparecer en comments justificando la
    # exención sin que el endpoint LO USE realmente.
    pattern = re.compile(r"Depends\(\s*verify_api_quota\s*\)")
    assert not pattern.search(sig), (
        f"P1-AUDIT-3 regresión: endpoint `{endpoint}` ahora usa "
        f"`Depends(verify_api_quota)`. Esto bloquea al usuario tras "
        f"alcanzar el cap mensual (HTTP 402) impidiendo ver SU "
        f"PROPIO Historial. Revertir a `Depends(get_verified_user_id)`. "
        f"Ver CLAUDE.md sección 'Historial-quota-exemption' para el "
        f"contexto completo de la decisión."
    )


@pytest.mark.parametrize("endpoint", [
    "/lessons-counts",
    "/history-status-summary",
    "/history-list",
])
def test_endpoint_has_inline_p1_audit_3_anchor(plans_src: str, endpoint: str):
    """Cada endpoint debe tener comentario inline `[P1-AUDIT-3]`
    explicando la exención. Sin el anchor, el siguiente operador no
    entiende POR QUÉ no usa `verify_api_quota` y podría "arreglarlo"
    rompiendo la UX.
    """
    sig = _extract_endpoint_decorator_and_signature(plans_src, endpoint)
    assert "P1-AUDIT-3" in sig, (
        f"P1-AUDIT-3 regresión: endpoint `{endpoint}` perdió el "
        f"comentario inline `[P1-AUDIT-3 · 2026-05-10]` que justifica "
        f"el uso de `get_verified_user_id`. Sin el anchor, un futuro "
        f"refactor cosmético borra la convención. Restaurar el "
        f"comentario apuntando a CLAUDE.md."
    )


def test_claude_md_documents_exemption():
    """CLAUDE.md debe tener la sección "Historial-quota-exemption"
    enumerando los 3 endpoints exentos y la razón. Si la sección
    desaparece (refactor de doc), este test falla loud."""
    claude_md = _BACKEND_ROOT.parent / "CLAUDE.md"
    text = claude_md.read_text(encoding="utf-8")
    assert "Historial-quota-exemption" in text, (
        "P1-AUDIT-3 regresión: CLAUDE.md ya NO documenta la sección "
        "`Historial-quota-exemption`. Sin la doc, el comment inline "
        "en los 3 endpoints queda huérfano — futuro mantenedor no "
        "entiende la convención."
    )
    # Mencionar al menos uno de los 3 endpoints + el razonamiento.
    assert "/history-list" in text, (
        "CLAUDE.md sección Historial-quota-exemption no lista "
        "`/history-list` — incompleto."
    )
    assert "/lessons-counts" in text, (
        "CLAUDE.md sección Historial-quota-exemption no lista "
        "`/lessons-counts` — incompleto."
    )
    assert "/history-status-summary" in text, (
        "CLAUDE.md sección Historial-quota-exemption no lista "
        "`/history-status-summary` — incompleto."
    )
