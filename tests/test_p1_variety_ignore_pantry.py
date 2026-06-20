"""[P1-VARIETY-IGNORE-PANTRY · 2026-06-20] "Renovar Plan Actual" (variety)
ignora la despensa por diseño.

Contexto del bug:
  "Renovar Plan Actual" en el frontend dispara una regeneración con
  `update_reason='variety'` cuya intención es generar un plan NUEVO con
  alimentos DIFERENTES al plan anterior. Sin embargo el cliente seguía
  enviando `current_pantry_ingredients`, y DOS surfaces del backend lo
  consumían contradiciendo esa intención:

    1. `build_pantry_context` (prompts/plan_generator.py) inyectaba el bloque
       Zero-Waste ("ESTRATEGIA ZERO-WASTE: agota estos ingredientes ANTES de
       comprar nuevos") → el LLM anclaba el plan a la nevera → platos casi
       idénticos al anterior, contradiciendo el hint de "MAYOR VARIEDAD" que
       ai_helpers añade para la misma reason.

    2. `review_plan_node` (graph_orchestrator.py) validaba los ingredientes
       del plan contra la despensa → como los platos nuevos NO están en la
       nevera, la validación fallaba → retries → plan degradado.

Fix (backend-only, robusto a clientes cacheados que aún envían la nevera):
  ambos surfaces se saltan cuando `update_reason == 'variety'`. La supresión
  es ESPECÍFICA de 'variety' — `pantry_first` y la primera generación (sin
  update_reason) siguen respetando la despensa.

Contrato verificado aquí:
  - build_pantry_context('variety') == "" aunque la nevera esté llena.
  - build_pantry_context sin update_reason / con 'pantry_first' SÍ emite el
    bloque Zero-Waste (la compuerta no es un apagón global).
  - review_plan_node desactiva needs_pantry_validation para 'variety'
    (parser-anchor sobre la fuente de prod).
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompts.plan_generator import build_pantry_context


# Nevera llena de perecederos que NORMALMENTE dispara el bloque Zero-Waste.
_FULL_PANTRY = [
    "pollo (2 lb)", "pescado fresco", "tomate", "lechuga", "espinaca",
    "queso", "leche", "huevos", "cebolla", "zanahoria",
]


# --------------------------------------------------------------------------
# build_pantry_context — supresión específica de 'variety'
# --------------------------------------------------------------------------

def test_pantry_context_suppressed_for_variety():
    """Con update_reason='variety' el bloque Zero-Waste NO se emite, aun con
    una nevera llena de perecederos."""
    out = build_pantry_context({
        "current_pantry_ingredients": list(_FULL_PANTRY),
        "update_reason": "variety",
    })
    assert out == "", (
        "build_pantry_context emitió el bloque de despensa para "
        "update_reason='variety'. 'Renovar Plan Actual' debe ignorar la nevera "
        "para generar un plan con alimentos diferentes."
    )


def test_pantry_context_emitted_without_update_reason():
    """Sanity / control: la MISMA nevera SÍ emite el bloque Zero-Waste cuando
    no hay update_reason (primera generación). Prueba que la compuerta es
    específica de 'variety' y no un apagón global."""
    out = build_pantry_context({
        "current_pantry_ingredients": list(_FULL_PANTRY),
    })
    assert "ZERO-WASTE" in out and "RECICLAJE DE DESPENSA" in out, (
        "El bloque Zero-Waste desapareció para la generación normal — la "
        "compuerta de 'variety' no debe afectar a otros paths."
    )


def test_pantry_context_emitted_for_pantry_first():
    """`pantry_first` es la reason cuya intención EXPLÍCITA es exprimir la
    despensa — NUNCA debe suprimirse el bloque Zero-Waste."""
    out = build_pantry_context({
        "current_pantry_ingredients": list(_FULL_PANTRY),
        "update_reason": "pantry_first",
    })
    assert "ZERO-WASTE" in out, (
        "build_pantry_context suprimió el bloque para 'pantry_first' — esa "
        "reason debe respetar la despensa."
    )


def test_pantry_context_variety_with_empty_pantry_is_noop():
    """Edge: nevera vacía + variety → "" (igual que el path normal de vacío)."""
    assert build_pantry_context({"current_pantry_ingredients": [], "update_reason": "variety"}) == ""


# --------------------------------------------------------------------------
# review_plan_node — parser-anchor del skip de validación para 'variety'
# --------------------------------------------------------------------------

def _read_orchestrator_source() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_review_plan_node_defines_variety_flag():
    """La fuente de prod DEBE derivar `is_variety_regen` desde
    `update_reason == 'variety'` (anchor: un renombre rompe el test antes que
    producción)."""
    src = _read_orchestrator_source()
    assert re.search(
        r'is_variety_regen\s*=\s*form_data\.get\(\s*["\']update_reason["\']\s*\)\s*==\s*["\']variety["\']',
        src,
    ), (
        "No se encontró la derivación `is_variety_regen = "
        "form_data.get('update_reason') == 'variety'` en review_plan_node."
    )


def test_review_plan_node_excludes_variety_from_pantry_validation():
    """`needs_pantry_validation` DEBE incluir `not is_variety_regen` para que
    el Revisor Médico NO valide los platos nuevos contra la despensa en una
    regeneración de variedad."""
    src = _read_orchestrator_source()
    m = re.search(
        r"needs_pantry_validation\s*=\s*\((?P<body>.*?)\)",
        src,
        re.DOTALL,
    )
    assert m, "No se encontró la asignación de `needs_pantry_validation`."
    body = m.group("body")
    assert "not is_variety_regen" in body, (
        "`needs_pantry_validation` no excluye `is_variety_regen`. Sin esto, "
        "los planes de variedad fallarían la validación de despensa → retries "
        "→ plan degradado."
    )
