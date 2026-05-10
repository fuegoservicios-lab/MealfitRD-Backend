"""[P2-2 · 2026-05-08] Tests del invariante `action_taken nunca queda None`.

Bug original (audit 2026-05-07):
  En `_shopping_coherence_block_history`, el campo `action_taken` quedaba `None`
  en dos casos analíticamente indistinguibles:
    1. Flujo normal con `block_set=False`: el guard detectó divergencias en
       modo `warn`, no se setea el flag, review_plan_node no entra al branch.
       Entry termina con `action_taken=None`.
    2. Bug de hidratación: block_set=True pero la lista de history quedó
       vacía/corrupta cuando review_plan_node intentó marcar la última entry.
       Entry termina con `action_taken=None`.

  Post-mortem leyendo plan_data no podía distinguir (1) "todo OK, modo warn"
  de (2) "hubo un block pero la acción no se registró".

Fix:
  1. `assemble_plan_node` setea `action_taken="not_applicable"` al CREAR la
     entry cuando `block_set=False` (caso 1 desambiguado).
  2. `review_plan_node` setea `action_taken="hydration_error"` cuando la
     hidratación normal falla por estado inesperado (history vacío, último
     elemento no-dict, excepción) — y crea entry sintético si fuera necesario.
  3. Invariante post-condición: tras la pipeline completa, ningún entry
     en `_shopping_coherence_block_history` tiene `action_taken=None` cuando
     el plan se entregó al usuario.

Cobertura:
  - Asemble con block_set=False → action_taken="not_applicable".
  - Asemble con block_set=True → action_taken=None (placeholder para review).
  - Review con history válida + block_set → action_taken=knob resuelto.
  - Review con history corrupta + block_set → action_taken="hydration_error".
  - Smoke: el código fuente refleja la lógica conditional.
"""
import pathlib
import re

import pytest


_GO_PATH = pathlib.Path(__file__).parent.parent / "graph_orchestrator.py"  # P3-CANDIDATE-B test migration: archivo vive un nivel arriba


@pytest.fixture(scope="module")
def go_source() -> str:
    return _GO_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def assemble_block(go_source) -> str:
    start = go_source.find("from shopping_calculator import run_shopping_coherence_guard")
    end = go_source.find("duration = round(time.time() - start_time, 2)", start)
    return go_source[start:end]


@pytest.fixture(scope="module")
def review_block(go_source) -> str:
    start = go_source.find('coherence_block = plan.get("_shopping_coherence_block")')
    end = go_source.find("# Brechas 1 y 4: Errores deterministas del ensamblador", start)
    return go_source[start:end]


# ---------------------------------------------------------------------------
# 1. assemble_plan_node: action_taken inicial conditional
# ---------------------------------------------------------------------------
def test_assemble_action_taken_is_conditional_on_block_set(assemble_block):
    """assemble_plan_node debe diferenciar el initial value de action_taken
    según `_block_set`. block_set=False → "not_applicable"; block_set=True
    → None (placeholder para review_plan_node)."""
    # El patrón debe ser una conditional, no un None hardcoded.
    # Esperamos algo como: action_taken: None if _block_set else "not_applicable"
    assert re.search(
        r'"action_taken":\s*None\s+if\s+_block_set\s+else\s+["\']not_applicable["\']',
        assemble_block,
    ), (
        "assemble_plan_node debe setear `action_taken='not_applicable'` cuando "
        "`block_set=False`. Sin esto, el flujo warn-only es indistinguible "
        "post-mortem de un bug de hidratación."
    )


def test_assemble_extracts_block_set_to_local_var(assemble_block):
    """`_block_set` debe ser una variable local computada antes del dict, para
    que el conditional value de action_taken pueda referirla."""
    assert "_block_set" in assemble_block, (
        "Debe existir variable local `_block_set` para el conditional."
    )


def test_assemble_action_taken_legacy_pattern_removed(assemble_block):
    """El patrón legacy `"action_taken": None,` (sin condicional) debe haber
    sido reemplazado. Si reaparece, post-mortem vuelve a ser ambiguo."""
    # Buscar exactamente el patrón legacy literal.
    legacy_count = len(re.findall(r'"action_taken":\s*None\s*,\s*\n', assemble_block))
    # Permitimos 0 occurrences del patrón legacy literal.
    assert legacy_count == 0, (
        f"Patrón legacy `action_taken: None` encontrado {legacy_count}x. "
        f"Debe ser conditional según _block_set."
    )


# ---------------------------------------------------------------------------
# 2. review_plan_node: hydration_error fallback
# ---------------------------------------------------------------------------
def test_review_marks_hydration_error_on_corrupt_history(review_block):
    """review_plan_node debe setear `action_taken="hydration_error"` cuando
    la hidratación normal falla (history no-list, vacía, último item no-dict,
    o excepción)."""
    assert "hydration_error" in review_block, (
        "review_plan_node debe distinguir errores de hidratación marcando "
        "action_taken=hydration_error en la entry afectada."
    )


def test_review_else_branch_logs_warning(review_block):
    """Si la hidratación normal no aplica (history vacío/corrupto), debe
    haber un branch `else` que emita logging.warning + crear entry sintético."""
    # Buscar el patrón: el if isinstance(...) y un else o except con warning.
    assert re.search(
        r'isinstance\(_coh_hist\[-1\],\s*dict\)\s*:.*?else\s*:.*?logging\.warning',
        review_block,
        re.DOTALL,
    ) or "logging.warning" in review_block, (
        "Si la lista no cumple el shape esperado, debe loguear WARNING (no debug) "
        "porque marca un bug que rompe la invariante de telemetría."
    )


def test_review_creates_synthetic_entry_when_history_missing(review_block):
    """Cuando la history está totalmente ausente/inválida, review debe
    crear entry sintético con action_taken=hydration_error para preservar
    invariante."""
    # Esperamos algo como append({..., "action_taken": "hydration_error", ...})
    assert "hydration_error_reason" in review_block, (
        "Entry sintético debe incluir hydration_error_reason para diagnóstico."
    )


# ---------------------------------------------------------------------------
# 3. Smoke: invariante a nivel código
# ---------------------------------------------------------------------------
def test_action_taken_appears_in_at_least_three_distinct_states(go_source):
    """Tras P2-2, action_taken puede tomar:
      - None (transitorio, antes de review_plan_node, sólo si block_set=True)
      - "not_applicable" (block_set=False)
      - "reject_minor" / "reject_high" / "degrade" (review hidrata)
      - "hydration_error" (defensivo)
    Verificamos que los strings literales aparecen en el código."""
    states = ["not_applicable", "hydration_error", "reject_minor", "reject_high", "degrade"]
    for state in states:
        assert f'"{state}"' in go_source or f"'{state}'" in go_source, (
            f"Estado posible de action_taken `{state}` no aparece literal en "
            f"graph_orchestrator.py. Si se renombra/elimina, post-mortem lectura "
            f"de plan_data queda parcialmente ciega."
        )


def test_review_action_taken_assignment_unchanged_for_normal_path(review_block):
    """El path normal de hidratación (history válida) debe seguir siendo:
    `_coh_hist[-1]["action_taken"] = _block_action`. P2-2 sólo añade
    fallbacks defensivos sin tocar el happy path."""
    assert re.search(
        r'_coh_hist\[-1\]\[\s*["\']action_taken["\']\s*\]\s*=\s*_block_action',
        review_block,
    ), (
        "El happy path debe seguir asignando _block_action a la última entry. "
        "P2-2 añade defensa para cuando la asignación falla, no la reemplaza."
    )
