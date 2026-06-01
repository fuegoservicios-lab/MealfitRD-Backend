"""[P1-BUDGET-CUSTOM · 2026-05-31] Opción "Personalizar" del presupuesto:
el usuario define un monto total (RD$) que el LLM usa para ajustar ingredientes.

Pedido del usuario: "quiero que haya una opción que diga 'personalizar'... para
poder agregar de cuánto es mi presupuesto total."

Contexto (lo que cambió el diseño): pre-fix `budget` se recolectaba y validaba
contra un enum de 4 valores (low/medium/high/unlimited) pero NUNCA llegaba al
prompt del LLM — solo invalidaba caché. La app no tiene precios por ingrediente,
así que el ajuste es CUALITATIVO (guía al LLM), no un cálculo exacto.

Cadena completa anclada por estos tests:
  1. Backend acepta `budget='custom'` (enum + label).
  2. `build_budget_context` existe y maneja el monto custom + fallback.
  3. Se importa + inyecta (`budget_context`) + se usa en el prompt del day-generator.
  4. Frontend: QBudget tiene la opción "Personalizar" + input `budgetAmount`.
  5. Frontend flow: el step de budget exige `budgetAmount` cuando custom (validateExtra).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_PLANS_PY = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_PLANGEN_PY = (_BACKEND_ROOT / "prompts" / "plan_generator.py").read_text(encoding="utf-8")
_ORCH_PY = (_BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")
_IQ_JSX = (_REPO_ROOT / "frontend" / "src" / "components" / "assessment" / "questions" / "InteractiveQuestions.jsx").read_text(encoding="utf-8")
_FLOW_JSX = (_REPO_ROOT / "frontend" / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx").read_text(encoding="utf-8")
_CTX_JSX = (_REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Backend: el enum acepta 'custom'
# ---------------------------------------------------------------------------
def test_budget_enum_includes_custom():
    assert re.search(r'_BUDGET_ENUM\s*=\s*frozenset\(\{[^}]*"custom"[^}]*\}\)', _PLANS_PY), (
        "`_BUDGET_ENUM` no incluye 'custom' — el backend rechazaría budget='custom' con 422."
    )
    # El label del 422 debe mencionar custom.
    assert re.search(r'"budget",\s*_BUDGET_ENUM,\s*True,\s*"[^"]*custom[^"]*"', _PLANS_PY), (
        "El label de validación de `budget` no menciona 'custom'."
    )


# ---------------------------------------------------------------------------
# 2. build_budget_context: existe + maneja el monto custom + fail-soft
# ---------------------------------------------------------------------------
def test_build_budget_context_exists_and_handles_custom():
    assert "def build_budget_context(form_data: dict) -> str:" in _PLANGEN_PY, (
        "`build_budget_context` ausente en prompts/plan_generator.py."
    )
    # Aísla el cuerpo de la función.
    m = re.search(r"def build_budget_context\(.*?(?=\ndef )", _PLANGEN_PY, re.DOTALL)
    assert m, "No se pudo aislar build_budget_context."
    body = m.group(0)
    # Lee el monto custom + la duración.
    assert 'form_data.get("budgetAmount")' in body, (
        "build_budget_context no lee `budgetAmount`."
    )
    assert 'budget == "custom"' in body, "No hay rama para budget=='custom'."
    # Fail-soft: '' cuando no hay budget.
    assert 'return ""' in body, "Falta el fail-soft (return '')."
    # Inyecta el monto en RD$.
    assert "RD$" in body, "El contexto custom no formatea el monto en RD$."


# ---------------------------------------------------------------------------
# 3. graph_orchestrator: import + inyección + uso en el prompt
# ---------------------------------------------------------------------------
def test_budget_context_imported_injected_and_used():
    assert "build_budget_context," in _ORCH_PY, (
        "graph_orchestrator no importa build_budget_context."
    )
    assert '"budget_context": build_budget_context(form_data),' in _ORCH_PY, (
        "graph_orchestrator no inyecta `budget_context` al dict de contextos."
    )
    assert "ctx['budget_context']" in _ORCH_PY, (
        "El prompt del day-generator no usa `ctx['budget_context']` — el "
        "presupuesto no llegaría al LLM."
    )


# ---------------------------------------------------------------------------
# 4. Frontend: opción "Personalizar" + input del monto
# ---------------------------------------------------------------------------
def test_frontend_qbudget_has_personalizar_and_amount_input():
    assert 'value="custom"' in _IQ_JSX and "Personalizar" in _IQ_JSX, (
        "QBudget no tiene la opción 'Personalizar' (value=custom)."
    )
    assert 'updateData(\'budgetAmount\'' in _IQ_JSX or "updateData(\"budgetAmount\"" in _IQ_JSX, (
        "QBudget no captura `budgetAmount` desde el input."
    )
    # El input de monto existe (id budgetAmount).
    assert 'id="budgetAmount"' in _IQ_JSX, "Falta el input id='budgetAmount'."
    # NO auto-avanza al elegir custom (el onChange de custom no llama onAutoAdvance).
    assert "onChange={() => updateData('budget', 'custom')}" in _IQ_JSX, (
        "La card 'Personalizar' debe NO auto-avanzar (onChange solo setea budget=custom)."
    )


# ---------------------------------------------------------------------------
# 5. Frontend flow: validateExtra exige budgetAmount cuando custom
# ---------------------------------------------------------------------------
def test_flow_gates_next_on_budget_amount_for_custom():
    # El step de budget declara validateExtra.
    assert "validateExtra:" in _FLOW_JSX and "budgetAmount" in _FLOW_JSX, (
        "El flow no tiene `validateExtra` exigiendo budgetAmount para custom."
    )
    # `stepFieldsFilled` incorpora validateExtra (scoped por-step).
    assert "currentStepConfig.validateExtra" in _FLOW_JSX, (
        "`stepFieldsFilled` no incorpora `validateExtra` — el botón 'Siguiente "
        "Paso' no se gatearía con el monto."
    )


def test_initial_form_data_has_budget_amount():
    assert re.search(r"budgetAmount:\s*''", _CTX_JSX), (
        "initialFormData no declara `budgetAmount: ''`."
    )


# ---------------------------------------------------------------------------
# 6. [BUDGET-CURRENCY · 2026-05-31] Toggle de moneda RD$/US$ (default DOP/peso)
# ---------------------------------------------------------------------------
def test_budget_currency_toggle_defaults_to_dop():
    # initialFormData: default 'DOP' (peso dominicano) — pedido del usuario.
    assert re.search(r"budgetCurrency:\s*'DOP'", _CTX_JSX), (
        "initialFormData no declara `budgetCurrency: 'DOP'` (default peso dominicano)."
    )
    # Frontend QBudget: toggle RD$/US$ que setea budgetCurrency.
    assert "updateData('budgetCurrency', 'DOP')" in _IQ_JSX, (
        "Falta el botón RD$ (setea budgetCurrency='DOP')."
    )
    assert "updateData('budgetCurrency', 'USD')" in _IQ_JSX, (
        "Falta el botón US$ (setea budgetCurrency='USD')."
    )
    # El default visible es RD$ (peso) cuando el campo no se ha tocado.
    assert "formData.budgetCurrency || 'DOP'" in _IQ_JSX, (
        "El default de la moneda debe ser 'DOP' (peso dominicano)."
    )


def test_build_budget_context_uses_currency():
    m = re.search(r"def build_budget_context\(.*?(?=\ndef )", _PLANGEN_PY, re.DOTALL)
    assert m
    body = m.group(0)
    assert 'form_data.get("budgetCurrency")' in body, (
        "build_budget_context no lee `budgetCurrency` — el LLM no sabría si el "
        "monto es en RD$ o US$ (escalas muy distintas)."
    )
    # Sanitiza a {DOP, USD} (anti-injection: símbolo/nombre de un mapping fijo).
    assert 'currency not in ("DOP", "USD")' in body, (
        "build_budget_context no sanitiza `budgetCurrency` a {DOP, USD}."
    )
    assert "US$" in body and "RD$" in body, (
        "build_budget_context no formatea ambos símbolos (RD$ / US$)."
    )


# ---------------------------------------------------------------------------
# 7. [BUDGET-MIN · 2026-05-31] Mínimo viable escalado por duración + moneda
# ---------------------------------------------------------------------------
def test_budget_minimum_enforced_and_shared_ssot():
    _FORMVAL = (_REPO_ROOT / "frontend" / "src" / "config" / "formValidation.js").read_text(encoding="utf-8")
    # SSOT del mínimo en formValidation.
    assert "export const minBudgetFor" in _FORMVAL and "BUDGET_MIN_PER_DAY" in _FORMVAL, (
        "Falta el helper SSOT `minBudgetFor` / `BUDGET_MIN_PER_DAY` en formValidation.js."
    )
    # El flow gatea "Siguiente Paso" con el MÍNIMO (no solo > 0).
    assert "minBudgetFor(fd.budgetCurrency" in _FLOW_JSX, (
        "El validateExtra del flow no exige el mínimo (minBudgetFor) — solo > 0."
    )
    # QBudget usa el mismo helper (SSOT) para el hint + el input min + la advertencia.
    assert "minBudgetFor(budgetCurrency, formData.groceryDuration)" in _IQ_JSX, (
        "QBudget no calcula el mínimo viable con `minBudgetFor` (drift vs el flow)."
    )
    assert "belowMin" in _IQ_JSX, (
        "QBudget no advierte cuando el monto está por debajo del mínimo."
    )
