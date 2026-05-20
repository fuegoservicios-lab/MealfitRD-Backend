"""[P1-18 + UNIFICATION 2026-05-14] Tests para que `MEMORY_SUMMARY_MODEL`
sea configurable y tenga default a un modelo válido de Gemini.

Cronología:
  - Pre-P1-18: `memory_manager.summarize_and_prune` instanciaba
    `ChatGoogleGenerativeAI` con `model="gemini-3.1-flash-lite"`
    (hardcoded en 2 sitios). En ese momento la familia 3.x NO estaba
    publicada (los IDs válidos eran 1.5/2.0/2.5) y el SDK lo rechazaba
    con 404. El except lo loggeaba como print warning silencioso →
    historial sin podar → ventana saturada → degradación silenciosa.
  - P1-18: default cambiado a `'gemini-2.5-flash'` (stable, no-preview),
    configurable vía env var `MEMORY_SUMMARY_MODEL`, las 2 instancias
    hardcoded usan la constante, except promovido a `logger.error`,
    contador `_summarize_failures` para SRE alerting.
  - 2026-05-14: la familia 3.x ya está publicada y el resto del stack
    (chat_llm, fact_extractor, sentiment_classifier, ai_helpers,
    proactive_agent) usa `gemini-3.1-flash-lite` con éxito.
    Default unificado a ese ID; el knob sigue habilitando rollback a
    `gemini-2.5-flash` (stable) sin redeploy si Google deprecara el
    preview.

Cobertura:
  - test_summary_model_default_is_valid_gemini_id
  - test_summary_model_overridable_via_env_var
  - test_summary_model_strips_whitespace
  - test_summary_model_falls_back_when_env_var_empty
  - test_summarize_failures_counter_initialized
  - test_summary_model_constant_used_in_summarize_and_prune
  - test_no_inline_model_literal_in_summarize_and_prune
  - test_summary_model_constant_is_at_module_level
  - test_logger_used_instead_of_print_for_errors
  - test_failures_counter_incremented_on_exception
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
    """Default sin env var → el ID unificado del stack
    (`gemini-3.1-flash-lite`, 2026-05-14)."""
    # Limpiamos cualquier override del entorno actual.
    if os.environ.get("MEMORY_SUMMARY_MODEL"):
        pytest.skip("MEMORY_SUMMARY_MODEL ya está seteado, no podemos verificar default")
    assert memory_manager.MEMORY_SUMMARY_MODEL == "gemini-3.1-flash-lite"


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
    # `"" or "gemini-3.1-flash-lite"` → default por la cláusula
    # `or` defensiva en el módulo.
    assert memory_manager.MEMORY_SUMMARY_MODEL == "gemini-3.1-flash-lite"
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


def test_no_inline_model_literal_in_summarize_and_prune():
    """Ninguna instancia de `model="gemini-..."` literal dentro de
    `summarize_and_prune`. Toda llamada a `ChatGoogleGenerativeAI` debe
    consumir la constante `MEMORY_SUMMARY_MODEL` para mantener el knob
    operativo (rollback sin redeploy). El positivo lo cubre
    `test_summary_model_constant_used_in_summarize_and_prune`; este es
    el negativo equivalente — regression guard contra reintroducir
    hardcode inline tras unificación 2026-05-14."""
    import re
    src = inspect.getsource(memory_manager.summarize_and_prune)
    # Buscamos `model="gemini-..."` o `model='gemini-...'` literal.
    hardcoded = re.findall(r'model\s*=\s*[\'"]gemini-[\w\.\-]+[\'"]', src)
    assert not hardcoded, (
        "Regression: `summarize_and_prune` reintrodujo un model="
        f"\"gemini-...\" hardcoded en lugar de MEMORY_SUMMARY_MODEL. "
        f"Matches: {hardcoded}"
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
