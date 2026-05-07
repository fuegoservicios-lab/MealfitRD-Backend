"""[P0-5] Tests para garantizar que `should_retry` se comporta determinísticamente
cuando `_rejection_severity` viene como `None` o ausente del state.

Bug original (audit P0-5):
  En Python `dict.get(k, default)` devuelve `None` cuando la key existe con
  valor `None` (NO el default). El `initial_state` seteaba
  `_rejection_severity=None`, y `should_retry` leía
  `severity = state.get("_rejection_severity", "minor")` esperando "minor"
  como default. Si `review_plan_node` lanzaba excepción sin escribir
  severity, `severity == None` no matcheaba `"critical"` ni `"high"`,
  cayendo a la rama "retry" — loop silencioso de regeneración hasta agotar
  MAX_ATTEMPTS o GLOBAL_PIPELINE_TIMEOUT_S.

Fix:
  1. `initial_state` ahora setea `_rejection_severity="minor"` por default
     (no None) → elimina la trampa en el path normal.
  2. `should_retry` normaliza con `severity = state.get(...) or "minor"`,
     defendiendo contra nodos intermedios que escriban None explícito.
  3. Invariant log a nivel ERROR si severity=None y review_passed=False,
     para que SRE detecte bugs upstream que dejen el state inconsistente.

Cobertura:
  - test_should_retry_with_none_severity_does_not_loop_when_budget_low
  - test_should_retry_with_none_severity_normalizes_to_minor
  - test_should_retry_critical_aborts_no_retry
  - test_should_retry_high_aborts_no_retry
  - test_should_retry_minor_with_budget_returns_retry
  - test_should_retry_review_passed_returns_end
  - test_initial_state_default_severity_is_minor_not_none
  - test_invariant_log_fires_when_severity_is_none
"""
import logging
import time
import re

import pytest

import graph_orchestrator
from graph_orchestrator import should_retry, GLOBAL_PIPELINE_TIMEOUT_S


def _state_with_budget(*, severity, review_passed=False, attempt=1, elapsed_s=0.0):
    """Construye un state mínimo con budget completo para should_retry."""
    return {
        "review_passed": review_passed,
        "_rejection_severity": severity,
        "attempt": attempt,
        "rejection_reasons": ["test rejection"],
        "pipeline_start": time.time() - elapsed_s,
    }


# ---------------------------------------------------------------------------
# 1. Comportamiento de severity=None — NO debe causar retry loop.
# ---------------------------------------------------------------------------
def test_should_retry_with_none_severity_does_not_loop_when_budget_exhausted():
    """severity=None + budget agotado → 'end', no 'retry'. Garantiza que el
    bug del retry indefinido está cerrado: incluso si severity llega como
    None y por tanto NO matchea 'critical'/'high', el budget guard funciona."""
    state = _state_with_budget(severity=None, elapsed_s=GLOBAL_PIPELINE_TIMEOUT_S - 1)
    decision = should_retry(state)
    assert decision == "end", \
        f"con budget agotado y severity=None debe devolver 'end', got {decision!r}"


def test_should_retry_with_none_severity_with_budget_returns_retry():
    """severity=None + budget OK → 'retry' (porque None se normaliza a 'minor').
    Antes del fix: ambiguo (caía a retry siempre por el comentario "fall through").
    Después del fix: normalizado explícitamente a 'minor', retry es la decisión
    correcta y consistente."""
    state = _state_with_budget(severity=None, elapsed_s=0.0)
    decision = should_retry(state)
    assert decision == "retry", \
        f"con budget OK y severity=None (→minor), retry es esperado, got {decision!r}"


# ---------------------------------------------------------------------------
# 2. Severidades explícitas — comportamiento documentado.
# ---------------------------------------------------------------------------
def test_should_retry_critical_aborts_no_retry():
    state = _state_with_budget(severity="critical", elapsed_s=0.0)
    assert should_retry(state) == "end"


def test_should_retry_high_contextual_aborts_no_retry():
    """[P1-RETRY-CLASSIFY] HIGH 'contextual' (despensa, alergia, condición)
    aborta sin retry — comportamiento original. Las restricciones del usuario
    no cambian entre intentos así que regenerar produciría el mismo error."""
    state = _state_with_budget(severity="high", elapsed_s=0.0)
    state["rejection_reasons"] = ["Violación de despensa estricta: ingrediente fuera de inventario"]
    assert should_retry(state) == "end"


def test_should_retry_high_regenerable_allows_retry():
    """[P1-RETRY-CLASSIFY] HIGH 'regenerable' (skeleton fidelity, repetición,
    falta variedad) PERMITE retry — son fallos de calidad del LLM que un
    segundo intento puede arreglar. Política nueva post-incidente 2026-05-05.
    """
    state = _state_with_budget(severity="high", elapsed_s=0.0)
    state["rejection_reasons"] = [
        "Día 1 omitió múltiples proteínas clave asignadas (skeleton fidelity)"
    ]
    assert should_retry(state) == "retry"


def test_should_retry_minor_with_budget_returns_retry():
    state = _state_with_budget(severity="minor", elapsed_s=0.0)
    assert should_retry(state) == "retry"


def test_should_retry_review_passed_returns_end_regardless_of_severity():
    """`review_passed=True` corto-circuita antes de leer severity."""
    state = _state_with_budget(severity=None, review_passed=True, elapsed_s=0.0)
    assert should_retry(state) == "end"


# ---------------------------------------------------------------------------
# 3. Invariante en initial_state.
# ---------------------------------------------------------------------------
def test_initial_state_default_severity_is_minor_not_none():
    """El default explícito de `_rejection_severity` en `initial_state` debe
    ser 'minor', NO None. Verificamos vía source-grep para evitar tener que
    instrumentar `arun_plan_pipeline` completo."""
    src = open(graph_orchestrator.__file__, encoding="utf-8").read()
    # Match exacto la línea de assignment en initial_state.
    pattern = re.compile(r'"_rejection_severity"\s*:\s*([^\s,]+)\s*,')
    matches = pattern.findall(src)
    assert matches, "no se encontró asignación de _rejection_severity en initial_state"
    # La PRIMERA occurrence en el archivo es la del initial_state (PlanState
    # TypedDict no tiene asignaciones inline). Aceptamos múltiples ocurrencias
    # mientras que ninguna sea None bare.
    for val in matches:
        assert val.lower() != "none", \
            f"P0-5 regression: _rejection_severity asignado a {val!r}; debe ser un string como 'minor'"


# ---------------------------------------------------------------------------
# 4. Invariant log — para detectar bugs upstream sin perder seguridad runtime.
# ---------------------------------------------------------------------------
def test_invariant_log_fires_when_severity_is_none(caplog):
    """Cuando severity=None (estado inconsistente), should_retry debe loggear
    en nivel ERROR para que operadores detecten bugs upstream — pero el
    pipeline NO debe romperse: severity se normaliza a 'minor' y el flow
    continúa."""
    state = _state_with_budget(severity=None, elapsed_s=0.0)
    with caplog.at_level(logging.ERROR, logger=graph_orchestrator.logger.name):
        should_retry(state)
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("INVARIANT" in m and "_rejection_severity=None" in m for m in error_messages), \
        f"esperado log ERROR sobre invariant violation, got: {error_messages}"


def test_invariant_log_does_not_fire_when_severity_is_minor(caplog):
    """Caso normal: severity='minor' (default explícito) → SIN log de invariant."""
    state = _state_with_budget(severity="minor", elapsed_s=0.0)
    with caplog.at_level(logging.ERROR, logger=graph_orchestrator.logger.name):
        should_retry(state)
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert not any("INVARIANT" in m for m in error_messages), \
        f"NO esperaba log de invariant cuando severity es válido, got: {error_messages}"


# ---------------------------------------------------------------------------
# 5. Defensa contra el patrón roto.
# ---------------------------------------------------------------------------
def test_should_retry_does_not_use_get_with_default_anymore_for_severity():
    """Defensa textual: el patrón `state.get("_rejection_severity", "minor")`
    NO debe reaparecer en código activo (era el bug). La normalización debe
    usar `state.get("_rejection_severity") or "minor"` o equivalente."""
    src = open(graph_orchestrator.__file__, encoding="utf-8").read()
    # Buscamos el patrón roto, ignorando líneas que sean comentarios.
    bad_pattern = re.compile(r'state\.get\(\s*["\']_rejection_severity["\']\s*,\s*["\']minor["\']\s*\)')
    offending = []
    for i, line in enumerate(src.splitlines(), start=1):
        if not bad_pattern.search(line):
            continue
        if line.strip().startswith("#"):
            continue
        offending.append((i, line.strip()))
    assert not offending, \
        f"P0-5 regression: el patrón `state.get('_rejection_severity', 'minor')` reapareció: {offending}"
