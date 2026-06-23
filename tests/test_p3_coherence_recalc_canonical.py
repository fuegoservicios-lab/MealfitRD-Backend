"""[P3-COHERENCE-RECALC-CANONICAL · 2026-06-23] El coherence guard de
`/recalculate-shopping-list` DEBE evaluar la lista CANÓNICA de 1 semana
(`scaled_7`), NO el `active_list` (que para biweekly/monthly es el híbrido
escalado ×N y restock-deducido).

Bug que cierra: el usuario veía el toast amarillo "Tu lista de compras tuvo N
revisiones automáticas recientes" SIN razón real al cambiar la duración
(7/15/30). Causa: con duración monthly + restock, `active_list` =
`scaled_30_hybrid` (×4 y con los items ya comprados DEDUCIDOS). El guard comparaba
las recetas (scope 1 semana) contra esa lista → marcaba los items restockeados como
`cap_swallowed_modifier` ("faltan de la lista") + magnitudes falsas vs la lista ×4.
`scaled_7` es la lista canónica de 1 semana a multiplier base → comparación correcta.

Test PARSER-BASED sobre el cuerpo de `api_recalculate_shopping_list`. El
tooltip-anchor en el source hace que un refactor que vuelva a apuntar el guard a
`active_list`/`aggregated_shopping_list` deducido falle ESTE test antes de
re-introducir el ruido en producción.
"""
import re
from pathlib import Path

import pytest

_PLANS_PATH = Path(__file__).resolve().parent.parent / "routers" / "plans.py"
_PLANS_SRC = _PLANS_PATH.read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    """Extrae el cuerpo de una función top-level por indentación."""
    m = re.search(rf"^def {re.escape(fn_name)}\(", src, re.MULTILINE)
    assert m, f"función {fn_name} no encontrada en plans.py"
    start = m.start()
    # Buscar el inicio de la SIGUIENTE def/@router top-level (col 0).
    nxt = re.search(r"^(def |@router\.)", src[start + 1:], re.MULTILINE)
    end = start + 1 + nxt.start() if nxt else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def recalc_body() -> str:
    return _extract_function_body(_PLANS_SRC, "api_recalculate_shopping_list")


def test_tooltip_anchor_present(recalc_body: str):
    assert "P3-COHERENCE-RECALC-CANONICAL" in recalc_body, (
        "P3-COHERENCE-RECALC-CANONICAL: tooltip-anchor ausente del recalc handler."
    )


def test_guard_evaluates_scaled_7_not_active_list(recalc_body: str):
    """Antes del call al guard, `aggregated_shopping_list` se setea a `scaled_7`
    (o fallback), NO al `active_list` deducido."""
    # El swap: plan_data_fresh["aggregated_shopping_list"] = <scaled_7 ...>
    swap_re = re.compile(
        r'plan_data_fresh\[\s*["\']aggregated_shopping_list["\']\s*\]\s*=\s*_list_for_guard'
    )
    guard_re = re.compile(r"(run_shopping_coherence_guard_and_append_history|_coh_recalc)\s*\(")
    assert "scaled_7" in recalc_body, "scaled_7 no referenciado en el handler."
    # `_list_for_guard` deriva de scaled_7.
    assert re.search(r"_list_for_guard\s*=\s*scaled_7", recalc_body), (
        "`_list_for_guard` debe derivar de `scaled_7` (lista canónica de 1 semana)."
    )
    swap_m = swap_re.search(recalc_body)
    assert swap_m, "el swap a `_list_for_guard` antes del guard no se encontró."
    # El swap ocurre ANTES del call al guard.
    guard_calls = [m for m in guard_re.finditer(recalc_body)
                   if "import" not in recalc_body[max(0, m.start() - 80):m.start()]]
    assert guard_calls, "call al coherence guard ausente."
    assert swap_m.start() < guard_calls[0].start(), (
        "el swap de `aggregated_shopping_list` a la lista canónica debe ocurrir "
        "ANTES del call al guard."
    )


def test_active_list_restored_after_guard(recalc_body: str):
    """Tras el guard, `aggregated_shopping_list` se restaura al `active_list`
    (`_active_for_persist`) para persistir la lista correcta en la columna."""
    restore_re = re.compile(
        r'plan_data_fresh\[\s*["\']aggregated_shopping_list["\']\s*\]\s*=\s*_active_for_persist'
    )
    assert restore_re.search(recalc_body), (
        "el restore de `aggregated_shopping_list` = `_active_for_persist` (en finally) "
        "no se encontró — sin él se persistiría la lista canónica en vez del active."
    )
    # Debe estar en un `finally:` (garantiza restore aunque el guard lance).
    assert re.search(r"finally:\s*\n\s*plan_data_fresh\[\s*[\"']aggregated_shopping_list", recalc_body), (
        "el restore debe vivir en un bloque `finally:`."
    )
