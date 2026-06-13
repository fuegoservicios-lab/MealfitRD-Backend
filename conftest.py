"""[P3-NEW-E · 2026-05-08] Conftest a nivel raíz de `backend/`.

Por qué existe:
  ANTES, la lógica de eager-import de `langgraph` y `langchain_google_genai`
  vivía sólo en `backend/tests/conftest.py`. Pytest aplica conftest a su
  directorio Y a subdirectorios, no hacia ARRIBA. Por eso los ~181 tests
  que viven en `backend/` (raíz) no se beneficiaban del primer del stub
  fragility — y por eso P3-C 2026-05-07 documentó "mover los 178 raíz
  restantes queda OUT OF SCOPE hasta arreglar stub langgraph fragility".

  Hoist-eando el bloque de eager-imports a este archivo (que es padre
  TANTO de `backend/test_*.py` raíz COMO de `backend/tests/test_*.py`),
  todos los tests reciben el primer ANTES de que cualquier módulo de
  test importe. Cierre del bloqueador estructural.

  `backend/tests/conftest.py` mantiene su bloque por idempotencia
  (try/except que sólo stubea si el import real falla) — la duplicación
  es intencional: si alguien ejecuta los tests con cwd=tests/, el conftest
  raíz no se carga, y el local sigue resolviendo. Defensa-en-profundidad.

NO añadir fixtures de e2e aquí. Las fixtures que tocan DB
(`seeded_user_profile` etc.) viven en `backend/tests/conftest.py` porque
sólo aplican al subset E2E.
"""

# [P0-5 · hoisted P3-NEW-E] Eager-resolve `langgraph` antes de que cualquier
# módulo de test cargue. Algunos test files hacen
# `sys.modules.setdefault('langgraph', MagicMock())` para entornos sin el
# paquete; `setdefault` no distingue MagicMock previamente instalado de paquete
# real, así que si un test alfabéticamente-temprano corría su `setdefault`
# primero, instalaba MagicMock y rompía a todos los siguientes. Importarlo
# acá puebla `sys.modules` con el paquete real → siguientes `setdefault`s son
# no-ops. Sólo stubeamos si el import real falla (CI sin la dependencia).
try:
    import langgraph  # noqa: F401
    import langgraph.graph  # noqa: F401
    import langgraph.graph.message  # noqa: F401
    import langgraph.checkpoint.memory  # noqa: F401
except Exception:
    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault("langgraph", MagicMock())
    sys.modules.setdefault("langgraph.graph", MagicMock())
    sys.modules.setdefault("langgraph.graph.message", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint.memory", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint.postgres", MagicMock())

# [P0-5 · hoisted P3-NEW-E · P0-DEEPSEEK-MIGRATION 2026-06-12] Mismo
# eager-import para `langchain_openai` (cliente base del provider DeepSeek,
# ver `llm_provider.py`). Si un test file instala un stub parcial primero,
# un import posterior del surface real (e.g. via cron_tasks → ai_helpers →
# llm_provider) levanta `ImportError`. Importar el paquete real puebla
# sys.modules con el surface completo y los stubs subsecuentes quedan como
# no-op. Sólo stubeamos si el import real falla (CI sin la dependencia).
try:
    import langchain_openai  # noqa: F401
    from langchain_openai import (  # noqa: F401
        ChatOpenAI,
        OpenAIEmbeddings,
    )
except Exception:
    import sys
    from unittest.mock import MagicMock
    if "langchain_openai" not in sys.modules:
        _stub = MagicMock()
        _stub.ChatOpenAI = object
        _stub.OpenAIEmbeddings = object
        sys.modules["langchain_openai"] = _stub
