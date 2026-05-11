"""[P3-NEW-1 · 2026-05-10] `api_expand_recipe` DEBE no tocar
`plan_data._plan_modified_at`. Contract: la expansión de receta es
cosmética (idempotente vía `isExpanded`), no merece bumpear el sort
semántico del Historial.

Decisión documentada (audit 2026-05-10):
    Argumento "Sí bumpear" (rechazado):
        La expansión cambia el contenido visible (placeholder genérico
        → pasos detallados de chef), semánticamente es modificación.

    Argumento "No bumpear" (aplicado):
        1. Idempotente — `isExpanded=True` previene re-expansión.
        2. Cada cook-click del usuario invoca expand. Si bumpeáramos
           `_plan_modified_at`, el Historial (sort client-side por ese
           path) reordenaría al tope con cada cook-click → ruidoso y
           engañoso (no es mod del plan, es interacción CON el plan).
        3. `meal_plans.updated_at` (columna física, trigger P0-2) SÍ
           se actualiza por el UPDATE — observabilidad operacional
           preservada sin contaminar el sort semántico.

    Si el Historial migra a ordenar por `updated_at` físico (no jsonb),
    esta decisión se debe revisitar. Este test enforza el contrato
    actual; un futuro fix que añada bump aquí debe actualizar AMBOS
    lados al mismo tiempo y marcar este test obsoleto.

Estrategia (parser estático):
    1. Localizar el cuerpo de `api_expand_recipe` en plans.py.
    2. Verificar que NO contiene `_plan_modified_at` como string.
    3. Verificar que el comentario P3-NEW-1 está presente (anchor para
       que un futuro autor lea la decisión antes de bumpear).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py. Si fue "
            f"renombrado/eliminado, actualizar este test."
        )
    start = m.start()
    next_def = re.search(
        r"\n(?:@router\.|@app\.|def\s)",
        src[start + 1:],
    )
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def expand_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_expand_recipe")


def test_no_plan_modified_at_string_in_handler(expand_body: str):
    """El cuerpo de `api_expand_recipe` NO debe contener
    `_plan_modified_at` en NINGUNA forma (asignación, jsonb_set,
    referencia). Si aparece, alguien volvió a bumpear el sort
    semántico desde recipe/expand, contra la decisión P3-NEW-1.
    """
    # Permitimos la mención dentro del comentario explicativo P3-NEW-1
    # (el bloque que DOCUMENTA la decisión). Excluimos esa región antes
    # de buscar el string.
    decision_block = re.search(
        r"\[P3-NEW-1.*?Si en el futuro el Historial.*?al mismo tiempo\.",
        expand_body,
        re.DOTALL,
    )
    if decision_block:
        # Reemplazamos el bloque por placeholder para que su mención de
        # `_plan_modified_at` no falsifique el match.
        scrubbed = (
            expand_body[: decision_block.start()]
            + "<<DECISION_BLOCK_REDACTED>>"
            + expand_body[decision_block.end() :]
        )
    else:
        scrubbed = expand_body

    forbidden = re.search(r"_plan_modified_at", scrubbed)
    assert not forbidden, (
        "P3-NEW-1 regresión: `api_expand_recipe` ahora contiene una "
        "referencia activa a `_plan_modified_at` (fuera del comentario "
        "de decisión). Si la decisión cambió, actualizar el comentario "
        "P3-NEW-1 + retirar este test. Si fue accidente, eliminar la "
        "referencia para preservar el contrato (no contaminar sort del "
        "Historial con cook-clicks)."
    )


def test_decision_block_present(expand_body: str):
    """El comentario `[P3-NEW-1 ... DECISIÓN]` debe estar presente para
    que un futuro autor entienda POR QUÉ no bumpear antes de añadir
    código que lo haga."""
    pattern = re.compile(
        r"\[P3-NEW-1.*?\]\s+DECISIÓN.*?_plan_modified_at",
        re.DOTALL,
    )
    assert pattern.search(expand_body), (
        "P3-NEW-1 regresión: el bloque de decisión documentado "
        "desapareció de `api_expand_recipe`. Sin ese anchor, un futuro "
        "autor podría añadir `jsonb_set(_plan_modified_at, ...)` sin "
        "saber que la decisión fue NO bumpear."
    )


def test_anchor_present(expand_body: str):
    """Anchor `P3-NEW-1` para grep rápido."""
    assert "P3-NEW-1" in expand_body, (
        "P3-NEW-1 regresión: el anchor textual desapareció. Restaurar "
        "para `grep -r P3-NEW-1` en routers/plans.py."
    )
