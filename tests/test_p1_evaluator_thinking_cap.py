"""[P1-EVALUATOR-THINKING-CAP · 2026-06-01] Regresión del cap de thinking del
EVALUATOR de self_critique (+ rider: el cap del planner, nodo de generación que
faltaba en el set original P1-COST-THINKING-CAP).

Contexto (audit de coste Gemini, 2ª pasada):
  - La decisión P1-COST-THINKING-CAP (2026-05-28) dejó los evaluadores SIN cap
    ("necesitan razonar"). La telemetría de prod (`llm_usage_events`: self_critique
    avg_out 6426 / max 14350 tok para un schema de 5 scores + bool + string) reveló
    reasoning runaway — las señales duras (slot/staples/monotonía) ya van
    pre-calculadas por código, así que el reasoning extra es desperdicio.
  - Decisión revisada (consenso 2026-06-01): cap DEDICADO y GENEROSO del evaluator
    (`EVALUATOR_THINKING_BUDGET`, default 4096 = 2x day-gen) que recorta el tail
    patológico sin tocar el razonamiento normal. Redes: FLOOR determinista de días +
    knob `EVALUATOR_USE_PRO`.
  - El planner (genera el skeleton = nombres+slots) es un nodo de GENERACIÓN y debió
    estar capado desde P1-COST-THINKING-CAP; se cierra el gap reusando
    `_thinking_budget_kwargs(planner_model)` (no-op en el default flash-lite).

Tests parser-based con tooltip-anchors en el código de prod: un renombre falla el
test ANTES de cambiar producción. El behavioral del helper está guardado por import
(graph_orchestrator es pesado de importar) — skip si no se puede importar.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (_BACKEND / name).read_text(encoding="utf-8")


def _strip_comments(text: str) -> str:
    """Quita comentarios `#...` (línea completa o inline). Las regiones objetivo no
    tienen `#` dentro de literales."""
    out = []
    for line in text.splitlines():
        if "#" in line:
            line = line[: line.index("#")]
        out.append(line)
    return "\n".join(out)


_GO_SRC = _read("graph_orchestrator.py")
_GO_CODE = _strip_comments(_GO_SRC)


# ---------------------------------------------------------------------------
# 1. Knob dedicado del evaluator (default 4096 = 2x day-gen)
# ---------------------------------------------------------------------------
def test_evaluator_thinking_budget_knob_default_4096():
    m = re.search(
        r'EVALUATOR_THINKING_BUDGET\s*=\s*_env_int\s*\(\s*"MEALFIT_EVALUATOR_THINKING_BUDGET"\s*,\s*(\d+)',
        _GO_CODE,
    )
    assert m, (
        "Falta el knob EVALUATOR_THINKING_BUDGET = _env_int('MEALFIT_EVALUATOR_THINKING_BUDGET', 4096, ...). "
        "tooltip-anchor: P1-EVALUATOR-THINKING-CAP"
    )
    assert m.group(1) == "4096", (
        f"El default del evaluator debe ser 4096 (2x day-gen, margen de razonamiento). Hallado {m.group(1)}."
    )
    # Mismo clamp [-1, 32768] que day-gen (sentinela -1 = rollback sin redeploy).
    assert "MEALFIT_EVALUATOR_THINKING_BUDGET" in _GO_CODE
    assert re.search(r"EVALUATOR_THINKING_BUDGET[\s\S]{0,160}-1 <= v <= 32768", _GO_CODE), (
        "El knob del evaluator debe clampear a [-1, 32768] (sentinela -1=sin cap)."
    )


# ---------------------------------------------------------------------------
# 2. Helper dedicado del evaluator (gemelo del de day-gen)
# ---------------------------------------------------------------------------
def test_evaluator_thinking_helper_exists_and_gates_lite():
    assert "def _evaluator_thinking_budget_kwargs(model_name: str) -> dict:" in _GO_CODE, (
        "Falta el helper _evaluator_thinking_budget_kwargs(model_name). tooltip-anchor: P1-EVALUATOR-THINKING-CAP"
    )
    # Usa el presupuesto DEDICADO, no el de day-gen.
    assert '"thinking_budget": EVALUATOR_THINKING_BUDGET' in _GO_CODE
    # Aislar el cuerpo del helper (hasta el siguiente `def ` a nivel módulo).
    m = re.search(
        r"def _evaluator_thinking_budget_kwargs\(model_name: str\) -> dict:(?P<body>[\s\S]*?)\n(?:def |# \[P1-PROMPT-CACHE)",
        _GO_CODE,
    )
    assert m, "No se aisló el cuerpo de _evaluator_thinking_budget_kwargs."
    body = m.group("body")
    # Sentinela -1 → sin cap; flash-lite → dict vacío (no soporta thinking_config).
    assert "EVALUATOR_THINKING_BUDGET < 0" in body, "El helper debe gatear el sentinela -1 (sin cap)."
    assert '"lite" in model_name.lower()' in body, "El helper debe excluir flash-lite (no soporta thinking)."
    # El helper canónico de day-gen NO debe haberse tocado en su firma (anclado por
    # test_p0_orch_audit_impl.py); lo verificamos aquí también por cercanía.
    assert "def _thinking_budget_kwargs(model_name: str) -> dict:" in _GO_CODE


# ---------------------------------------------------------------------------
# 3. El EVALUATOR de self_critique aplica el cap dedicado (P1)
# ---------------------------------------------------------------------------
def test_evaluator_llm_has_dedicated_thinking_cap():
    # El constructor del evaluator_llm debe spread el helper dedicado.
    assert "**_evaluator_thinking_budget_kwargs(_evaluator_model)" in _GO_CODE, (
        "evaluator_llm (self_critique_node) debe capar reasoning vía "
        "**_evaluator_thinking_budget_kwargs(_evaluator_model). tooltip-anchor: P1-EVALUATOR-THINKING-CAP"
    )
    # Debe estar en el MISMO constructor que produce CritiqueEvaluation (no en otro nodo).
    m = re.search(
        r"evaluator_llm\s*=\s*ChatGoogleGenerativeAI\((?P<body>[\s\S]{0,400}?)\)\.with_structured_output\(CritiqueEvaluation\)",
        _GO_CODE,
    )
    assert m, "No se halló el constructor de evaluator_llm con .with_structured_output(CritiqueEvaluation)."
    assert "_evaluator_thinking_budget_kwargs(_evaluator_model)" in m.group("body"), (
        "El cap dedicado debe estar DENTRO del constructor de evaluator_llm, no en otro sitio."
    )


# ---------------------------------------------------------------------------
# 4. El PLANNER (nodo de generación) ahora también está capado (rider)
# ---------------------------------------------------------------------------
def test_planner_llm_has_thinking_cap():
    m = re.search(
        r"planner_llm\s*=\s*ChatGoogleGenerativeAI\((?P<body>[\s\S]{0,400}?)\)\.with_structured_output\(PlanSkeletonModel\)",
        _GO_CODE,
    )
    assert m, "No se halló el constructor de planner_llm con .with_structured_output(PlanSkeletonModel)."
    assert "**_thinking_budget_kwargs(planner_model)" in m.group("body"), (
        "planner_llm debe reusar **_thinking_budget_kwargs(planner_model) (planner = generador). "
        "No-op en el default flash-lite; endurece los paths override. tooltip-anchor: P1-COST-THINKING-CAP"
    )


# ---------------------------------------------------------------------------
# 5. La decisión documentada refleja la revisión (no se override en silencio)
# ---------------------------------------------------------------------------
def test_documented_decision_revised_in_comment():
    assert "P1-EVALUATOR-THINKING-CAP" in _GO_SRC, (
        "El comentario de la decisión P1-COST-THINKING-CAP debe documentar la revisión "
        "P1-EVALUATOR-THINKING-CAP (la decisión 2026-05-28 NO debe revertirse en silencio)."
    )


# ---------------------------------------------------------------------------
# 6. Marker bumpeado + cross-link (este archivo ES el target del cross-link)
# ---------------------------------------------------------------------------
def test_marker_bumped_to_evaluator_cap():
    src = _read("app.py")
    assert "P1-EVALUATOR-THINKING-CAP" in src, (
        "_LAST_KNOWN_PFIX debe bumpearse a P1-EVALUATOR-THINKING-CAP al cerrar este bundle."
    )


# ---------------------------------------------------------------------------
# 7. Behavioral (guardado por import — graph_orchestrator es pesado)
# ---------------------------------------------------------------------------
def test_evaluator_helper_behavioral(monkeypatch):
    try:
        import importlib

        _GO = importlib.import_module("graph_orchestrator")
    except Exception as e:  # pragma: no cover - entornos sin deps de prod
        pytest.skip(f"graph_orchestrator no importable en este entorno: {e!r}")

    b = _GO.EVALUATOR_THINKING_BUDGET
    kw = _GO._evaluator_thinking_budget_kwargs("gemini-3.5-flash")
    if b >= 0:
        assert kw == {"thinking_budget": b}, f"esperaba budget {b}, vino {kw}"
    else:
        assert kw == {}, "sentinela -1 → sin kwarg"
    # flash-lite y vacío → sin kwarg (no soporta thinking_config).
    assert _GO._evaluator_thinking_budget_kwargs("gemini-3.1-flash-lite") == {}
    assert _GO._evaluator_thinking_budget_kwargs("") == {}
    # El evaluator usa un presupuesto >= day-gen (más generoso o igual).
    assert _GO.EVALUATOR_THINKING_BUDGET >= _GO.DAYGEN_THINKING_BUDGET or _GO.EVALUATOR_THINKING_BUDGET < 0
