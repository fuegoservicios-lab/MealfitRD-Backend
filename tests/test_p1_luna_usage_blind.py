"""[P1-LUNA-USAGE-BLIND · 2026-07-26] El day-gen dejó de aparecer en `llm_usage_events`.

Primera corrida real del canario Luna (plan 2fedd27e, 2026-07-26 07:09). El plan salió bien:
banda 1.00 entregada, cero reintentos, 4 minutos. Y `llm_usage_events` registró `planner`,
`compressor`, `self_critique`, `self_critique_correction`… y **cero filas de `day_generator`** —
el nodo más caro del pipeline, el 62% del costo.

La telemetría no dijo "falta el modelo nuevo". Dijo que el day-gen **no existió**.

## La causa

`graph_orchestrator` define su PROPIA `ChatGLM` que subclasea la de `llm_provider` para
añadir dos cosas: backpressure (rate limit per-user + slot global) y la captura de
`usage_metadata` que llena `llm_usage_events`. `P1-DAYGEN-LUNA-CANARY` cambió `_build_day_llm`
para construir el cliente vía `llm_provider.build_chat_llm`, que devuelve las clases **base**.

El agujero no era sólo del camino OpenAI: con la fábrica, un día que cayera al fallback GLM
también perdía instrumentación **y** rate limit.

## El arreglo

Los overrides pasan a `_LLMBackpressureCostMixin`, aplicado a los dos clientes. Un cliente nuevo
se instrumenta heredando del mixin, no copiando métodos.

`ChatOpenAIInstrumented` fuerza `stream_usage=True` en su `__init__` en vez de confiar en el
callsite: el day-gen llama por `.astream()` y sin esa bandera el stream no trae `usage_metadata`,
que es justo el modo de fallo que este cliente existe para cerrar.
"""
import inspect

import pytest

import graph_orchestrator as go
import llm_provider as lp


_METODOS = ("invoke", "stream", "generate", "ainvoke", "astream", "agenerate")


# ───────────── 1. los dos clientes comparten el mixin ─────────────

@pytest.mark.parametrize("cliente", ["ChatGLM", "ChatOpenAIInstrumented"])
def test_hereda_del_mixin(cliente):
    assert issubclass(getattr(go, cliente), go._LLMBackpressureCostMixin)


@pytest.mark.parametrize("cliente", ["ChatGLM", "ChatOpenAIInstrumented"])
@pytest.mark.parametrize("metodo", _METODOS)
def test_los_seis_caminos_pasan_por_el_mixin(cliente, metodo):
    """`.ainvoke()` directo, `.astream()` (el que usa el day-gen), structured-output y
    bind_tools desembocan en estos seis. Si uno se escapa, ese camino no se contabiliza."""
    cls = getattr(go, cliente)
    assert getattr(cls, metodo) is getattr(go._LLMBackpressureCostMixin, metodo)


@pytest.mark.parametrize("metodo", _METODOS)
def test_backpressure_en_los_seis(metodo):
    src = inspect.getsource(getattr(go._LLMBackpressureCostMixin, metodo))
    assert "acquire_user_and_global" in src, f"{metodo} sin backpressure"


@pytest.mark.parametrize("metodo", ["ainvoke", "astream", "agenerate"])
def test_contabilidad_en_los_caminos_ASYNC(metodo):
    """Sólo los async contabilizan, y es suficiente: el pipeline de generación es async de punta
    a punta (`arun_plan_pipeline`), el day-gen entra por `.astream()`.

    ⚠️ Hueco PREEXISTENTE, anotado a propósito en vez de asumirlo: los tres síncronos
    (`invoke`/`stream`/`generate`) toman el semáforo pero NO emiten a `llm_usage_events`. Si algún
    día un camino de producción llama sincrónicamente, su costo no se registra. No se cierra aquí
    porque hoy no hay ninguno y cambiarlo sin un caso que lo ejerza es código no medido."""
    src = inspect.getsource(getattr(go._LLMBackpressureCostMixin, metodo))
    assert "_emit_llm_usage_event_best_effort" in src, f"{metodo} no contabiliza"


# ───────────── 2. stream_usage no se deja al callsite ─────────────

def test_openai_fuerza_stream_usage(monkeypatch):
    """El day-gen usa `.astream()`; sin `stream_usage` el último chunk no trae
    `usage_metadata` y la llamada no se registra — el bug exacto de esta corrida."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert go.ChatOpenAIInstrumented(model="gpt-5.6-luna").stream_usage is True


def test_openai_key_del_entorno_no_del_callsite(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        go.ChatOpenAIInstrumented(model="gpt-5.6-luna")


# ───────────── 3. el day-gen NO usa la fábrica sin instrumentar ─────────────

def _cuerpo_de(nombre: str) -> str:
    """Fuente EXACTA de una función por AST.

    Nada de `src[i:i+N]`: una ventana de bytes fija caduca en cuanto el archivo crece y entonces
    el test lee código de otra función. Ya me pasó cinco veces en una sesión."""
    import ast
    from pathlib import Path
    ruta = Path(go.__file__).resolve().parent / "graph_orchestrator.py"
    src = ruta.read_text(encoding="utf-8")
    for nodo in ast.walk(ast.parse(src)):
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)) and nodo.name == nombre:
            return ast.get_source_segment(src, nodo) or ""
    raise AssertionError(f"no existe la función {nombre}")


def test_el_daygen_construye_las_clases_locales():
    """La comprobación va por AST y no por subcadena: el nombre de la fábrica aparece
    legítimamente en el comentario que explica por qué NO se usa, y un `in` lo confundiría con
    una llamada — el mismo error de subcadena que P1-SEASONING-WORD-BOUNDARY."""
    import ast, textwrap
    cuerpo = _cuerpo_de("_build_day_llm")
    # El AST no conserva comentarios, así que todo `Name` que aparezca aquí es código de verdad.
    nombres = {n.id for n in ast.walk(ast.parse(textwrap.dedent(cuerpo)))
               if isinstance(n, ast.Name)}
    assert {"ChatOpenAIInstrumented", "ChatGLM"} <= nombres, nombres
    assert "is_openai_model" in nombres, "el proveedor debe elegirse por prefijo del modelo"
    assert "build_chat_llm" not in nombres, \
        "la fábrica devuelve las clases BASE: sin costo ni backpressure"


def test_la_fabrica_avisa_de_su_limitacion():
    """`build_chat_llm` sigue siendo útil para scripts y pruebas; lo que no puede es entrar al
    pipeline sin que quede dicho por qué."""
    doc = lp.build_chat_llm.__doc__ or ""
    assert "P1-LUNA-USAGE-BLIND" in doc
    assert "day_generator" in doc


# ───────────── 4. la clase de bug ─────────────

def test_todo_cliente_chat_de_produccion_esta_instrumentado():
    """El test que habría atrapado esto: enumerar las subclases de cliente chat declaradas en el
    orquestador y exigir el mixin a todas — en vez de recordar añadirlo a mano."""
    faltan = []
    for nombre, obj in vars(go).items():
        if not inspect.isclass(obj) or not issubclass(obj, lp.ChatOpenAI):
            continue
        if obj is lp.ChatOpenAI or obj is lp.ChatGLM:
            continue          # las BASES de llm_provider, no clientes de este módulo
        if not issubclass(obj, go._LLMBackpressureCostMixin):
            faltan.append(nombre)
    assert not faltan, f"clientes chat sin instrumentar: {faltan}"
