"""[P1-AGENT-HINT · 2026-05-22] Tests del hint que `log_consumed_meal` adjunta
al ToolMessage cuando `deduct_consumed_meal_from_inventory` reporta items
no procesables.

Red de seguridad sobre P1-PANTRY-INFER (#1) y P1-FAILED-DEDUCT-RETRY (#3):
si la inferencia de porción típica retorna None (caso raro: name vacío post
`_parse_quantity`) o `add_or_update_inventory_item` falla por master
mismatch, el item queda en `failed_inventory_deductions`. Con este hint la
LLM ve la lista en su próximo turn y puede pedir al usuario que confirme
la cantidad, en vez de fallar silenciosamente.

Contrato:
  - `deduct_consumed_meal_from_inventory` retorna dict
    `{succeeded, inferred, failed_to_deduct}` (cambio backward-compat —
    callers legacy ignoran el return).
  - `log_consumed_meal` examina el dict y, si `failed_to_deduct` no vacío,
    anexa hint al msg con preview de items (max 5).

Cross-link convention (P2-HIST-AUDIT-14): slug `p1_agent_hint_failed_items`
matchea este archivo.

Tooltip-anchor: P1-AGENT-HINT (vive en db_inventory.py + tools.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_INVENTORY_PY = _BACKEND_ROOT / "db_inventory.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"


@pytest.fixture(scope="module")
def db_inv_src() -> str:
    return _DB_INVENTORY_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_PY.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — return value contract de deduct_consumed_meal_from_inventory
# ===========================================================================

def test_deduct_returns_summary_dict(db_inv_src: str):
    """`deduct_consumed_meal_from_inventory` debe retornar dict con keys
    `succeeded`, `inferred`, `failed_to_deduct`. Pre-fix retornaba None
    siempre — el caller no podía construir el hint."""
    fn_re = re.compile(
        r"def\s+deduct_consumed_meal_from_inventory\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(db_inv_src)
    assert m is not None
    body = m.group(0)
    # Buscar el return dict que tiene las 3 keys canónicas.
    assert '"succeeded"' in body, (
        "P1-AGENT-HINT regresión: return value de "
        "`deduct_consumed_meal_from_inventory` no incluye key `succeeded`. "
        "El caller necesita esta info para distinguir 'no hubo items' "
        "vs 'todos se procesaron'."
    )
    assert '"inferred"' in body, (
        "P1-AGENT-HINT regresión: key `inferred` faltante. Es telemetría "
        "útil para audit de qué items pasaron por el path P1-PANTRY-INFER."
    )
    assert '"failed_to_deduct"' in body, (
        "P1-AGENT-HINT regresión: key `failed_to_deduct` faltante. Es la "
        "señal que `log_consumed_meal` usa para decidir si añadir el hint."
    )


def test_deduct_tracks_strs_per_outcome(db_inv_src: str):
    """Tres listas internas (`succeeded_strs`, `inferred_strs`, `failed_strs`)
    deben existir para acumular los strings originales del input."""
    fn_re = re.compile(
        r"def\s+deduct_consumed_meal_from_inventory\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(db_inv_src)
    assert m is not None
    body = m.group(0)
    for varname in ("succeeded_strs", "inferred_strs", "failed_strs"):
        assert varname in body, (
            f"P1-AGENT-HINT regresión: variable de tracking `{varname}` "
            f"removida del cuerpo de `deduct_consumed_meal_from_inventory`. "
            f"Sin estas listas el dict de return queda vacío."
        )


def test_deduct_legacy_callers_unaffected(db_inv_src: str):
    """[P0-5 backward compat] El INSERT en `failed_inventory_deductions` y los
    3 reasons canónicos (parse_failed_or_invalid_qty / deduction_returned_false
    / exception) siguen presentes — el cambio de return value NO debe haber
    eliminado el lado P0-5 de la función."""
    fn_re = re.compile(
        r"def\s+deduct_consumed_meal_from_inventory\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(db_inv_src)
    assert m is not None
    body = m.group(0)
    assert "_persist_failed_inventory_deductions" in body, (
        "P0-5 regresión: helper de persistencia de fallos removido. El "
        "cambio P1-AGENT-HINT NO debe haber roto el lado P0-5."
    )
    for reason in ("parse_failed_or_invalid_qty", "deduction_returned_false", "exception"):
        assert reason in body, (
            f"P0-5 regresión: reason `{reason}` removido del body. El "
            f"cambio P1-AGENT-HINT NO debe haber reducido los 3 reasons "
            f"canónicos que `_alert_failed_inventory_deductions_backlog` "
            f"esperaba."
        )


# ===========================================================================
# Sección 2 — log_consumed_meal consume el dict y construye el hint
# ===========================================================================

def test_log_consumed_meal_captures_deduct_summary(tools_src: str):
    """`log_consumed_meal` debe capturar el return value de
    `deduct_consumed_meal_from_inventory` en una variable (ej. `deduct_summary`).
    Pre-fix descartaba el return."""
    fn_re = re.compile(
        r"def\s+log_consumed_meal\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(tools_src)
    assert m is not None
    body = m.group(0)
    # Buscar asignación que capture el return de la deducción.
    assert re.search(
        r"=\s*db_inventory\.deduct_consumed_meal_from_inventory\(",
        body,
    ), (
        "P1-AGENT-HINT regresión: `log_consumed_meal` no captura el return "
        "value de `deduct_consumed_meal_from_inventory`. Sin esto, no puede "
        "construir el hint a la LLM."
    )


def test_log_consumed_meal_builds_hint_on_failed_items(tools_src: str):
    """El msg de éxito debe extenderse con un hint cuando hay items en
    `failed_to_deduct`. La LLM lo lee en el ToolMessage y puede pedir
    cantidades al usuario en su próximo turn."""
    fn_re = re.compile(
        r"def\s+log_consumed_meal\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(tools_src)
    assert m is not None
    body = m.group(0)
    assert "failed_to_deduct" in body, (
        "P1-AGENT-HINT regresión: `log_consumed_meal` no inspecciona "
        "`failed_to_deduct`. El hint nunca se construye."
    )
    # El hint debe mencionar 'cantidad' (instrucción a la LLM de cómo
    # remediar) y debe ser visible en el msg (concatenación con msg).
    assert re.search(r"cantidad", body, re.IGNORECASE), (
        "P1-AGENT-HINT regresión: el hint no menciona 'cantidad'. Sin esa "
        "palabra la LLM no entiende qué pedirle al usuario."
    )


def test_hint_truncates_long_lists(tools_src: str):
    """Si hay >5 items fallidos, mostrar preview de 5 + `+N más`. Sin
    truncate, un meal con 20 ingredientes inflaría el ToolMessage."""
    fn_re = re.compile(
        r"def\s+log_consumed_meal\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(tools_src)
    assert m is not None
    body = m.group(0)
    # Buscar slice `[:5]` o equivalente que limite el preview.
    assert "[:5]" in body or "preview" in body.lower(), (
        "P1-AGENT-HINT regresión: el hint no trunca la lista de items "
        "fallidos. Comidas con muchos ingredientes podrían inflar el "
        "ToolMessage y consumir tokens."
    )


def test_hint_safe_when_summary_is_none(tools_src: str):
    """Si `deduct_consumed_meal_from_inventory` no fue llamada (has_ingredients
    es False), el summary queda None — el código NO debe crashear intentando
    inspeccionarlo."""
    fn_re = re.compile(
        r"def\s+log_consumed_meal\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(tools_src)
    assert m is not None
    body = m.group(0)
    # El branch que inspecciona el summary debe gatear con `isinstance(...,
    # dict)` o `if deduct_summary:` para no fallar con None.
    assert (
        "isinstance(deduct_summary, dict)" in body
        or "deduct_summary is not None" in body
        or "if deduct_summary" in body
    ), (
        "P1-AGENT-HINT regresión: la inspección del summary no es None-safe. "
        "Cuando `has_ingredients=False` el summary es None — sin guard, "
        "`summary.get(...)` lanza AttributeError."
    )
