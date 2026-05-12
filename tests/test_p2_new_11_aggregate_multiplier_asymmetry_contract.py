"""[P2-NEW-11 · 2026-05-11] Contrato de asimetría de `multiplier` en
`aggregate_and_deduct_shopping_list`.

Contrato (ver bloque inline P2-NEW-11 en shopping_calculator.py):
    plan_ingredients:     `qty * multiplier`  (escalado por household)
    consumed_ingredients: `qty`              (sin escalado — cantidad real)

Por qué importa:
    Si un refactor descuidado aplica `* multiplier` a `consumed`, el
    sistema dice "tienes excedente" cuando en realidad el usuario tiene
    una fracción del consumo necesario. Resultado: nunca compra. Bug
    masivo invisible (la lista de compras parece OK, solo falta arroz/
    pollo/etc).

Estrategia del test (parser-based):
    1. El loop sobre `plan_ingredients` DEBE contener `* multiplier`.
    2. El loop sobre `consumed_ingredients` NO debe contener `* multiplier`
       en la línea de mutación de `aggregated[name][unit]`.
    3. El docstring/comment del bloque DEBE mencionar P2-NEW-11 y la
       palabra "asimetría"/"asymmetry" para que un revisor entienda
       que NO es un bug.
    4. Los inline markers `P2-NEW-11` deben aparecer en las DOS líneas
       de mutación (la del plan que sí escala y la del consumed que NO).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SC_FP = _REPO_ROOT / "backend" / "shopping_calculator.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _SC_FP.read_text(encoding="utf-8")


def test_plan_loop_multiplies_by_multiplier(src: str):
    """El loop sobre plan_ingredients DEBE escalar."""
    func_start = src.find("def aggregate_and_deduct_shopping_list(")
    assert func_start > 0, "aggregate_and_deduct_shopping_list no encontrado"
    # Boundary: el siguiente def OR el inicio del bloque "RESOLUCIÓN DE FRICCIÓN".
    body_end = src.find("RESOLUCIÓN DE FRICCIÓN", func_start)
    body = src[func_start:body_end]

    # Patrón canónico: `aggregated[name][unit] += float(qty) * float(multiplier)`
    pattern = re.compile(
        r"aggregated\[name\]\[unit\]\s*\+=\s*float\(qty\)\s*\*\s*float\(multiplier\)",
    )
    assert pattern.search(body), (
        "P2-NEW-11 regresión: el loop de plan_ingredients ya no escala "
        "por `multiplier`. Sin escalado, planes de N personas pedirán "
        "solo 1 porción base."
    )


def test_consumed_loop_does_not_multiply_by_multiplier(src: str):
    """El loop sobre consumed_ingredients NO debe escalar."""
    func_start = src.find("def aggregate_and_deduct_shopping_list(")
    assert func_start > 0
    body_end = src.find("RESOLUCIÓN DE FRICCIÓN", func_start)
    body = src[func_start:body_end]

    # Buscar la línea de mutación del consumed loop:
    # `aggregated[name][unit] -= float(qty)` SIN `* multiplier`.
    pattern_correct = re.compile(
        r"aggregated\[name\]\[unit\]\s*-=\s*float\(qty\)(?!\s*\*\s*float\(multiplier\))",
    )
    assert pattern_correct.search(body), (
        "P2-NEW-11 regresión: el loop de consumed_ingredients ya no usa "
        "`aggregated[name][unit] -= float(qty)` sin multiplier. Si se "
        "añadió `* multiplier` al consumed, la lista de compras dirá "
        "`tienes excedente` cuando realmente el usuario NO compra lo "
        "necesario. Bug masivo invisible."
    )

    # Defensa redundante: el patrón buggy explícito NO debe aparecer.
    pattern_buggy = re.compile(
        r"aggregated\[name\]\[unit\]\s*-=\s*float\(qty\)\s*\*\s*float\(multiplier\)",
    )
    assert not pattern_buggy.search(body), (
        "P2-NEW-11 regresión: detectado patrón buggy "
        "`aggregated[name][unit] -= float(qty) * float(multiplier)`. "
        "Ver bloque CONTRATO P2-NEW-11 en aggregate_and_deduct_shopping_list "
        "antes de revertir."
    )


def test_contract_documented_inline(src: str):
    """El bloque inline P2-NEW-11 debe documentar la asimetría con WHY."""
    func_start = src.find("def aggregate_and_deduct_shopping_list(")
    body_end = src.find("RESOLUCIÓN DE FRICCIÓN", func_start)
    body = src[func_start:body_end]

    # Tokens esperados en el bloque comment (defensa contra refactor que
    # remueva el contrato pensando que es ruido).
    required_tokens = [
        "P2-NEW-11",
        "ASIMETRÍA",
        "plan_ingredients",
        "consumed_ingredients",
        "household",
    ]
    for tok in required_tokens:
        assert tok in body, (
            f"P2-NEW-11 regresión: el bloque CONTRATO ya no menciona "
            f"`{tok}`. Sin el comment completo, un revisor puede creer "
            f"que la asimetría es un bug y aplicar `* multiplier` al "
            f"consumed."
        )


def test_inline_markers_on_mutation_lines(src: str):
    """Las dos líneas de mutación tienen marker inline `P2-NEW-11`."""
    func_start = src.find("def aggregate_and_deduct_shopping_list(")
    body_end = src.find("RESOLUCIÓN DE FRICCIÓN", func_start)
    body = src[func_start:body_end]

    # Marker en la línea del plan (escalado intencional).
    plan_pattern = re.compile(
        r"aggregated\[name\]\[unit\]\s*\+=\s*float\(qty\)\s*\*\s*float\(multiplier\).*?P2-NEW-11",
    )
    assert plan_pattern.search(body), (
        "P2-NEW-11 regresión: la línea de mutación del plan_ingredients "
        "ya no menciona el marker inline. Sin él, un grep rápido pierde "
        "el anchor."
    )

    # Marker en la línea del consumed (SIN escalado intencional).
    consumed_pattern = re.compile(
        r"aggregated\[name\]\[unit\]\s*-=\s*float\(qty\).*?P2-NEW-11",
    )
    assert consumed_pattern.search(body), (
        "P2-NEW-11 regresión: la línea de mutación del consumed_ingredients "
        "ya no menciona el marker inline."
    )
