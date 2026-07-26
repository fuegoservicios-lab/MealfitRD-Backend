"""[P1-RECIPE-EXPAND-MODEL-PROVIDER · 2026-07-26] El segundo sitio donde un modelo mejor se paga.

`MEALFIT_RECIPE_EXPAND_MODEL` existe desde P1-RECIPE-EXPAND-FAILSIGNAL, pero el cliente se
construía con `ChatDeepSeek(model=<knob>)` **a secas**: apuntar el knob a un modelo OpenAI lo mandaba
al `base_url` de DeepSeek con la key equivocada. Mismo fallo que P1-LUNA-USAGE-BLIND cerró en el
day-gen, aquí latente desde que el knob existe.

## Por qué ESTE nodo y no otro

    · lo dispara el USUARIO sobre UN plato ("regenera para más detalle"), no corre en cada plan
    · es exactamente donde sale el badge `_dish_quality_degraded` — la receta que quedó pobre
      (medido 2026-07-26: 5 de 72 recetas con ≤2 pasos; el «Vasito Frío» del plan 1070ceb1 tenía
      Mise en place y Montaje, sin Toque de Fuego)
    · la llamada es diminuta (una receta), así que el premium se mide en céntimos

Frente al day-gen, donde el modelo caro se paga en el 100% de los planes y los datos no muestran que
lo valga (ver `P1-CANARY-RETRY-ONLY`), aquí el gasto es opt-in del usuario y cae justo sobre lo que
está mirando.

## Sigue apagado

El default del knob es `deepseek-v4-flash`. Esto habilita el cambio, no lo hace.
"""
import os

import pytest

import ai_helpers as ah


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_el_default_sigue_siendo_el_barato():
    assert ah._recipe_expand_model_name() == "deepseek-v4-flash"


@pytest.mark.parametrize("modelo,esperado", [
    ("deepseek-v4-flash", "ChatDeepSeek"),
    ("deepseek-v4-pro", "ChatDeepSeek"),
    ("gpt-5.6-luna", "ChatOpenAIInstrumented"),
    ("gpt-5.6-terra", "ChatOpenAIInstrumented"),
])
def test_el_proveedor_se_elige_por_prefijo(modelo, esperado):
    c = ah._build_expand_llm(modelo, temperature=0.7, timeout=30)
    assert type(c).__name__ == esperado


def test_el_cliente_openai_esta_INSTRUMENTADO():
    """No basta con acertar el proveedor: si no lleva el mixin, la llamada no aparece en
    `llm_usage_events` y el costo del nodo se vuelve invisible — exactamente lo que pasó con el
    day-gen (P1-LUNA-USAGE-BLIND)."""
    import graph_orchestrator as go
    c = ah._build_expand_llm("gpt-5.6-luna", temperature=0.7, timeout=30)
    assert isinstance(c, go._LLMBackpressureCostMixin)
    assert c.stream_usage is True


@pytest.mark.parametrize("modelo", ["deepseek-v4-flash", "gpt-5.6-luna"])
def test_structured_output_funciona_en_los_dos(modelo):
    """`ChatDeepSeek` override-a `with_structured_output` para las rarezas de DeepSeek
    (`function_calling` en vez de `json_schema`); OpenAI quiere el default de langchain. Por eso el
    builder NO lo aplica y lo deja al caller."""
    from schemas import ExpandedRecipeModel
    ah._build_expand_llm(modelo, temperature=0.7, timeout=30).with_structured_output(ExpandedRecipeModel)


def test_el_builder_no_aplica_structured_output():
    """Se mira el CÓDIGO, no el docstring: el docstring nombra `with_structured_output` justo para
    explicar por qué no se aplica, y un `in` sobre el fuente entero lo confundía con una llamada."""
    import ast, inspect, textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(ah._build_expand_llm))).body[0]
    llamadas = {n.func.attr for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "with_structured_output" not in llamadas


def test_fail_cheap_ante_cualquier_error(monkeypatch):
    """Si el dispatch falla, se cae al proveedor barato de siempre — nunca se rompe la expansión ni
    se escala el gasto por accidente."""
    import llm_provider
    monkeypatch.setattr(llm_provider, "is_openai_model",
                        lambda m: (_ for _ in ()).throw(RuntimeError("boom")))
    c = ah._build_expand_llm("gpt-5.6-luna", temperature=0.7, timeout=30)
    assert type(c).__name__ == "ChatDeepSeek"


def test_el_callsite_usa_el_builder():
    import inspect
    src = inspect.getsource(ah.expand_recipe_agent)
    assert "_build_expand_llm(" in src
    assert "ChatDeepSeek(" not in src, "con ChatDeepSeek directo el knob no puede apuntar a OpenAI"
