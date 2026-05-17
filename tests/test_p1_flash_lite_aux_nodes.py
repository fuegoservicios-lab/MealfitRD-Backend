"""[P1-FLASH-LITE-AUX-NODES · 2026-05-15] Regression guards para el switch
a `gemini-3.1-flash-lite-preview` en los 3 nodos auxiliares de schema-strict
(adversarial judge / fact-checker clínico / reviewer holístico).

Pre-fix: cada nodo invocaba `_route_model(...)` que en perfiles clínicos
complejos escalaba a `gemini-3.1-pro-preview` (~10x más caro que Flash-Lite).
Para tareas binarias bajo `with_structured_output(...)`, Pro no aporta
diferencia medible vs Lite.

Fix: 3 helpers (`_judge_model_name`, `_fact_checker_model_name`,
`_reviewer_model_name`) con default `gemini-3.1-flash-lite-preview` y
knob de override por nodo (`MEALFIT_JUDGE_MODEL` / `MEALFIT_FACT_CHECKER_MODEL` /
`MEALFIT_REVIEWER_MODEL`) — patrón canónico P3-PREVIEW-MODEL-KNOB.

Trade-off documentado: REMUEVE el escalado automático a Pro para estos
3 nodos. Mitigación: override por nodo sin redeploy si calidad degrada.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_graph() -> str:
    return _GRAPH_PATH.read_text(encoding="utf-8")


# ----- Helpers definidos top-level --------------------------------------

@pytest.mark.parametrize(
    "func_name, knob_name",
    [
        ("_judge_model_name", "MEALFIT_JUDGE_MODEL"),
        ("_fact_checker_model_name", "MEALFIT_FACT_CHECKER_MODEL"),
        ("_reviewer_model_name", "MEALFIT_REVIEWER_MODEL"),
    ],
)
def test_helper_defined_and_reads_knob(func_name, knob_name):
    text = _read_graph()
    assert re.search(rf"^def {re.escape(func_name)}\(", text, re.MULTILINE), (
        f"P1-FLASH-LITE-AUX-NODES: helper `{func_name}` debe existir top-level."
    )
    # Body del helper debe leer el knob.
    body_match = re.search(
        rf"def {re.escape(func_name)}\(.*?(?=\n(?:def |class |[A-Z_]+\s*=))",
        text,
        re.DOTALL,
    )
    assert body_match, f"No pude extraer body de `{func_name}`."
    body = body_match.group(0)
    assert knob_name in body, (
        f"Helper `{func_name}` debe leer del knob `{knob_name}` "
        f"(convención P3-PREVIEW-MODEL-KNOB)."
    )
    assert "_env_str" in body, (
        f"Helper `{func_name}` debe usar `_env_str` (auto-registro en "
        f"_KNOBS_REGISTRY para `/api/system/admin/knobs`)."
    )


def test_default_flash_lite_constant_present():
    """Una sola constante `_FLASH_LITE_DEFAULT` que los 3 helpers reusan —
    evita drift entre defaults si Google deprecara la versión preview."""
    text = _read_graph()
    assert re.search(
        r'_FLASH_LITE_DEFAULT\s*=\s*"gemini-3\.1-flash-lite-preview"', text
    ), (
        "P1-FLASH-LITE-AUX-NODES: constante `_FLASH_LITE_DEFAULT` con valor "
        "`'gemini-3.1-flash-lite-preview'` debe existir como SSOT del default."
    )


# ----- Callsites: cada nodo usa su helper, no `_route_model` -----------

def test_judge_callsite_uses_helper():
    """`judge_llm` debe instanciarse con `_judge_model_name()`, NO con
    `_route_model(form_data, force_fast=False)` (legacy con escalada Pro)."""
    text = _read_graph()
    # Localizar la región de `judge_llm = ChatGoogleGenerativeAI(...)` y
    # las ~6 líneas previas que asignan `_judge_model`.
    m = re.search(
        r"_judge_model\s*=\s*(?P<rhs>[^\n]+)\n.*?judge_llm = ChatGoogleGenerativeAI\(\s*\n\s*model=_judge_model,",
        text,
        re.DOTALL,
    )
    assert m, "No encontré la asignación de `_judge_model` antes de `judge_llm`."
    rhs = m.group("rhs").strip()
    assert "_judge_model_name()" in rhs, (
        f"Callsite del judge debe usar `_judge_model_name()`, encontré: {rhs!r}. "
        f"Pre-fix usaba `_route_model(form_data, force_fast=False)`."
    )
    assert "_route_model(" not in rhs, (
        "Callsite del judge NO debe seguir llamando a `_route_model` — "
        "P1-FLASH-LITE-AUX-NODES unificó el routing a Lite via knob."
    )


def test_fact_checker_callsite_uses_helper():
    text = _read_graph()
    m = re.search(
        r"_fact_checker_model\s*=\s*(?P<rhs>[^\n]+)\n.*?fact_checker_llm = ChatGoogleGenerativeAI",
        text,
        re.DOTALL,
    )
    assert m, "No encontré la asignación de `_fact_checker_model`."
    rhs = m.group("rhs").strip()
    assert "_fact_checker_model_name()" in rhs, (
        f"Callsite del fact-checker debe usar `_fact_checker_model_name()`, "
        f"encontré: {rhs!r}."
    )
    assert "_route_model(" not in rhs


def test_reviewer_callsite_uses_helper():
    text = _read_graph()
    m = re.search(
        r"_reviewer_model\s*=\s*(?P<rhs>[^\n]+)\n.*?reviewer_llm = ChatGoogleGenerativeAI",
        text,
        re.DOTALL,
    )
    assert m, "No encontré la asignación de `_reviewer_model`."
    rhs = m.group("rhs").strip()
    assert "_reviewer_model_name()" in rhs, (
        f"Callsite del reviewer debe usar `_reviewer_model_name()`, "
        f"encontré: {rhs!r}."
    )
    assert "_route_model(" not in rhs


# ----- Pricing dict ya incluye Flash-Lite (P1-COST-INSTRUMENTATION) -----

def test_pricing_dict_includes_flash_lite():
    """Cross-link: el dict de pricing en db_profiles.py debe incluir
    `gemini-3.1-flash-lite-preview` para que `compute_gemini_cost_micros`
    pueda calcular el costo. Sin esta entry, `llm_usage_events.cost_usd_micros`
    queda NULL y el ROI de este P-fix es invisible."""
    db_path = _BACKEND_ROOT / "db_profiles.py"
    text = db_path.read_text(encoding="utf-8")
    assert '"gemini-3.1-flash-lite-preview"' in text, (
        "P1-COST-INSTRUMENTATION pricing dict debe incluir entry para "
        "`gemini-3.1-flash-lite-preview`. Sin esta entry el costo no se calcula."
    )
    # Pricing esperado (en micros/M tokens): $0.10 / $0.40 / $0.025
    m = re.search(
        r'"gemini-3\.1-flash-lite-preview"\s*:\s*\{\s*"input"\s*:\s*100_000\s*,\s*"output"\s*:\s*400_000\s*,\s*"cached"\s*:\s*25_000\s*\}',
        text,
    )
    assert m, (
        "Pricing de Flash-Lite debe ser input=100_000 / output=400_000 / "
        "cached=25_000 micros/M (= $0.10/$0.40/$0.025 per M tokens)."
    )
