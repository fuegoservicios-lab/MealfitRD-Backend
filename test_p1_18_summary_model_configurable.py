"""[P1-18] Tests para que `MEMORY_SUMMARY_MODEL` sea configurable y tenga
default a un modelo válido de Gemini.

Bug original (audit P1-18):
  `memory_manager.summarize_and_prune` instanciaba `ChatGoogleGenerativeAI`
  con `model="gemini-3.1-flash-lite-preview"` (hardcoded en 2 sitios).
  Ese ID NO corresponde a ningún modelo público conocido de Gemini (la
  familia 3.x no existe; los modelos públicos son 1.5/2.0/2.5). Si el
  SDK lo rechazaba con 404:
    - El `with_structured_output(EvolutionaryState)` fallaba.
    - El `except Exception` lo loggeaba como `print warning silencioso`
      filtrando solo "Server disconnected" — el resto se descartaba.
    - El historial seguía creciendo sin podarse → ventana de tokens
      saturada → costos disparados → degradación silenciosa de la
      memoria del agente.

Fix:
  1. `MEMORY_SUMMARY_MODEL = os.environ.get('MEMORY_SUMMARY_MODEL', 'gemini-2.5-flash')`
     (modelo válido y económico, configurable vía env var).
  2. Las 2 instancias hardcodeadas usan la constante.
  3. `_summarize_failures = {count, last_error}` contador in-memory.
  4. `except` promovido a `logger.error` (el primer fallo + cada 10) +
     `logger.warning` (intermedios). Reset al primer éxito.

Cobertura:
  - test_summary_model_default_is_valid_gemini_id
  - test_summary_model_overridable_via_env_var
  - test_summarize_failures_counter_initialized
  - test_summary_model_constant_used_in_summarize_and_prune
  - test_no_hardcoded_invalid_gemini_id_in_source
  - test_logger_used_instead_of_print_for_errors
  - test_documentation_p1_18_present
"""
import importlib
import inspect
import os

import pytest

import memory_manager


# ---------------------------------------------------------------------------
# 1. Default y override via env var.
# ---------------------------------------------------------------------------
def test_summary_model_default_is_valid_gemini_id():
    """Default sin env var → un ID que pertenece a la familia válida
    (1.5/2.0/2.5). Por especificación P1-18 el default es `gemini-2.5-flash`."""
    # Limpiamos cualquier override del entorno actual.
    if os.environ.get("MEMORY_SUMMARY_MODEL"):
        pytest.skip("MEMORY_SUMMARY_MODEL ya está seteado, no podemos verificar default")
    assert memory_manager.MEMORY_SUMMARY_MODEL == "gemini-2.5-flash"


def test_summary_model_overridable_via_env_var(monkeypatch):
    """Si el operador setea `MEMORY_SUMMARY_MODEL`, se usa ese valor al
    importar el módulo."""
    monkeypatch.setenv("MEMORY_SUMMARY_MODEL", "gemini-2.0-flash-001")
    importlib.reload(memory_manager)
    assert memory_manager.MEMORY_SUMMARY_MODEL == "gemini-2.0-flash-001"
    # Restaurar al default tras el test.
    monkeypatch.delenv("MEMORY_SUMMARY_MODEL", raising=False)
    importlib.reload(memory_manager)


def test_summary_model_strips_whitespace(monkeypatch):
    """`MEMORY_SUMMARY_MODEL = '  gemini-X  '` → strip a `'gemini-X'`."""
    monkeypatch.setenv("MEMORY_SUMMARY_MODEL", "  gemini-1.5-flash-latest  ")
    importlib.reload(memory_manager)
    assert memory_manager.MEMORY_SUMMARY_MODEL == "gemini-1.5-flash-latest"
    monkeypatch.delenv("MEMORY_SUMMARY_MODEL", raising=False)
    importlib.reload(memory_manager)


def test_summary_model_falls_back_when_env_var_empty(monkeypatch):
    """Env var = '' (vacío después de strip) → fallback al default."""
    monkeypatch.setenv("MEMORY_SUMMARY_MODEL", "")
    importlib.reload(memory_manager)
    # `"" or "gemini-2.5-flash"` → `"gemini-2.5-flash"` por la cláusula
    # `or` defensiva en el módulo.
    assert memory_manager.MEMORY_SUMMARY_MODEL == "gemini-2.5-flash"
    monkeypatch.delenv("MEMORY_SUMMARY_MODEL", raising=False)
    importlib.reload(memory_manager)


# ---------------------------------------------------------------------------
# 2. Contador de fallos.
# ---------------------------------------------------------------------------
def test_summarize_failures_counter_initialized():
    """`_summarize_failures` debe ser un dict con keys count y last_error."""
    assert isinstance(memory_manager._summarize_failures, dict)
    assert "count" in memory_manager._summarize_failures
    assert "last_error" in memory_manager._summarize_failures
    # Default values.
    assert memory_manager._summarize_failures["count"] == 0
    assert memory_manager._summarize_failures["last_error"] is None


# ---------------------------------------------------------------------------
# 3. Source-level: las 2 instancias hardcodeadas se reemplazaron.
# ---------------------------------------------------------------------------
def test_summary_model_constant_used_in_summarize_and_prune():
    """`summarize_and_prune` usa la constante `MEMORY_SUMMARY_MODEL` en
    lugar de hardcodear el modelo."""
    src = inspect.getsource(memory_manager.summarize_and_prune)
    assert "MEMORY_SUMMARY_MODEL" in src, (
        "P1-18: summarize_and_prune debe usar la constante configurable"
    )


def test_no_hardcoded_invalid_gemini_id_in_source():
    """El ID inválido `gemini-3.1-flash-lite-preview` NO debe aparecer en
    código activo. Permitimos su mención en comentarios (documentando
    el bug histórico)."""
    src = inspect.getsource(memory_manager)
    # Filtramos líneas-comentario.
    code_lines = [ln for ln in src.split('\n') if not ln.strip().startswith('#')]
    code_only = '\n'.join(code_lines)
    assert "gemini-3.1-flash-lite-preview" not in code_only, (
        "P1-18 regression: el ID inválido reapareció en código activo"
    )


def test_summary_model_constant_is_at_module_level():
    """La constante debe estar exportada a nivel módulo (no inline en una
    función) para que sea testeable y configurable."""
    assert hasattr(memory_manager, "MEMORY_SUMMARY_MODEL")
    assert isinstance(memory_manager.MEMORY_SUMMARY_MODEL, str)
    assert len(memory_manager.MEMORY_SUMMARY_MODEL) > 0


# ---------------------------------------------------------------------------
# 4. Logger en lugar de print silencioso.
# ---------------------------------------------------------------------------
def test_logger_used_instead_of_print_for_errors():
    """El except de `summarize_and_prune` debe usar `logger.error` en
    lugar del `print warning silencioso` original. Sin esto, una racha
    de fallos del modelo queda invisible en producción."""
    src = inspect.getsource(memory_manager.summarize_and_prune)
    assert "logger.error" in src, (
        "P1-18: el except debe usar logger.error para SRE alerting"
    )
    # Y la string del marker para alerting.
    assert "P1-18" in src or "summarize_and_prune falló" in src


def test_failures_counter_incremented_on_exception():
    """Si fuerzo una excepción dentro de `summarize_and_prune` (mockeando
    `acquire_summarizing_lock` para que retorne True pero get_memory para
    que lance), el contador debe incrementar."""
    from unittest.mock import patch
    # Resetear contador.
    memory_manager._summarize_failures["count"] = 0
    memory_manager._summarize_failures["last_error"] = None

    with patch.object(memory_manager, "acquire_summarizing_lock", return_value=True), \
         patch.object(memory_manager, "release_summarizing_lock"), \
         patch.object(memory_manager, "get_memory", side_effect=RuntimeError("simulated")):
        memory_manager.summarize_and_prune("test-session")

    assert memory_manager._summarize_failures["count"] >= 1, (
        "P1-18: el contador de fallos debe incrementar tras excepción"
    )
    assert memory_manager._summarize_failures["last_error"] is not None


# ---------------------------------------------------------------------------
# 5. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_18_present():
    """Comentario `[P1-18]` documenta el rationale en el módulo."""
    src = inspect.getsource(memory_manager)
    assert "[P1-18]" in src
