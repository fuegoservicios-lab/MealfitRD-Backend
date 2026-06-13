"""[P1-20] Tests para garantizar el orden correcto save_summary →
purge_langgraph_checkpoint(raise_on_failure=True) → delete_old_messages
en `summarize_and_prune`.

Bug original (audit P1-20):
  El flujo previo era:
    1. save_summary(...)              → COMMIT A (Supabase)
    2. delete_old_messages(...)       → COMMIT A' (Supabase)
    3. purge_langgraph_checkpoint(...) → COMMIT B (PostgresSaver, otro pool)
  Si A/A' commiteaban pero B fallaba (DB blip, schema mismatch, SDK
  rejection), los mensajes estaban borrados de Supabase PERO LangGraph
  aún los retenía → re-inyección al LLM en el siguiente turno, fugando
  contexto privado ya supuestamente resumido. El `purge` además
  silenciaba TODOS los errores con un `print warning`, así que no había
  forma de detectar la racha.

Fix:
  1. `purge_langgraph_checkpoint` acepta `raise_on_failure: bool = False`
     (kw-only). Default preserva comportamiento histórico.
  2. `summarize_and_prune` reordena el flujo:
     - save_summary (idempotente sobre messages_end timestamp).
     - purge_langgraph_checkpoint(raise_on_failure=True) ANTES del delete.
     - delete_old_messages SOLO si purge tuvo éxito.
  3. Si purge falla → propaga al except global → log error (P1-18) +
     alert (P1-19) → delete NUNCA se ejecuta → consistencia garantizada
     (peor caso: redundancia).

Cobertura:
  - test_purge_accepts_raise_on_failure_kwonly
  - test_purge_default_swallows_errors
  - test_purge_with_raise_propagates_no_pool_error
  - test_summarize_calls_purge_with_raise_on_failure_true
  - test_summarize_order_purge_before_delete (defensa estructural source)
  - test_purge_failure_aborts_delete_old_messages
  - test_documentation_p1_20_present
"""
import inspect
from unittest.mock import patch, MagicMock

import pytest

import memory_manager


# ---------------------------------------------------------------------------
# 1. Signature + contrato del flag.
# ---------------------------------------------------------------------------
def test_purge_accepts_raise_on_failure_kwonly():
    """`purge_langgraph_checkpoint` debe aceptar `raise_on_failure` como
    kw-only para evitar passing posicional accidental."""
    sig = inspect.signature(memory_manager.purge_langgraph_checkpoint)
    assert "raise_on_failure" in sig.parameters
    param = sig.parameters["raise_on_failure"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY
    assert param.default is False, (
        "P1-20: default debe ser False para preservar contrato histórico"
    )


# ---------------------------------------------------------------------------
# 2. Comportamiento default (legacy): traga errores.
# ---------------------------------------------------------------------------
def test_purge_default_swallows_errors():
    """Sin `raise_on_failure=True`, los errores del SDK NO propagan
    (preserva el comportamiento legacy de los callers no-summarize)."""
    # Forzar fallo: connection_pool=None es el path canónico.
    with patch.object(memory_manager, "connection_pool", None):
        # No debe lanzar.
        memory_manager.purge_langgraph_checkpoint("test-session")


def test_purge_default_swallows_helper_failure():
    """Si `_get_dummy_purge_graph` retorna None inesperadamente o el
    grafo lanza, sin `raise_on_failure` el caller NO ve la excepción."""
    fake_pool = object()
    with patch.object(memory_manager, "connection_pool", fake_pool), \
         patch.object(memory_manager, "_get_dummy_purge_graph", return_value=None):
        # No debe lanzar.
        memory_manager.purge_langgraph_checkpoint("test-session")


# ---------------------------------------------------------------------------
# 3. Comportamiento con raise_on_failure=True: propaga.
# ---------------------------------------------------------------------------
def test_purge_with_raise_propagates_no_pool_error():
    """Con `raise_on_failure=True`, si no hay pool → RuntimeError."""
    with patch.object(memory_manager, "connection_pool", None):
        with pytest.raises(RuntimeError, match="P1-20"):
            memory_manager.purge_langgraph_checkpoint(
                "test-session", raise_on_failure=True
            )


def test_purge_with_raise_propagates_graph_exception():
    """Con `raise_on_failure=True`, si el get_state lanza, propaga
    la excepción original (no se traga con print warning)."""
    fake_pool = object()
    fake_graph = MagicMock()
    fake_graph.get_state.side_effect = RuntimeError("simulated SDK failure")
    with patch.object(memory_manager, "connection_pool", fake_pool), \
         patch.object(memory_manager, "_get_dummy_purge_graph", return_value=fake_graph):
        with pytest.raises(RuntimeError, match="simulated SDK failure"):
            memory_manager.purge_langgraph_checkpoint(
                "test-session", raise_on_failure=True
            )


# ---------------------------------------------------------------------------
# 4. summarize_and_prune usa el orden correcto.
# ---------------------------------------------------------------------------
def test_summarize_calls_purge_with_raise_on_failure_true():
    """`summarize_and_prune` debe invocar `purge_langgraph_checkpoint`
    con `raise_on_failure=True` para que un fallo aborte el delete."""
    src = inspect.getsource(memory_manager.summarize_and_prune)
    assert "raise_on_failure=True" in src, (
        "P1-20: summarize_and_prune debe invocar purge con raise_on_failure=True"
    )


def test_summarize_order_purge_before_delete():
    """En el source de `summarize_and_prune`, `purge_langgraph_checkpoint`
    debe aparecer ANTES de `delete_old_messages`. Defensa estructural
    contra reordenamiento accidental.

    Filtramos líneas-comentario del source para que las menciones de las
    funciones en docstrings/comments no produzcan falsos positivos."""
    raw_src = inspect.getsource(memory_manager.summarize_and_prune)
    code_only = "\n".join(
        ln for ln in raw_src.splitlines() if not ln.strip().startswith("#")
    )
    # Buscamos las INVOCACIONES (con paréntesis) en código activo.
    purge_idx = code_only.find("purge_langgraph_checkpoint(")
    delete_idx = code_only.find("delete_old_messages(")
    save_idx = code_only.find("save_summary(")
    assert purge_idx > -1, "purge_langgraph_checkpoint(...) no encontrado en código activo"
    assert delete_idx > -1, "delete_old_messages(...) no encontrado en código activo"
    assert save_idx > -1, "save_summary(...) no encontrado en código activo"
    assert save_idx < purge_idx, "P1-20: save_summary debe ir ANTES de purge"
    assert purge_idx < delete_idx, (
        "P1-20: purge_langgraph_checkpoint debe ir ANTES de delete_old_messages "
        "para garantizar consistencia Supabase ↔ LangGraph"
    )


# ---------------------------------------------------------------------------
# 5. Test funcional: si purge falla, delete NUNCA se ejecuta.
# ---------------------------------------------------------------------------
def test_purge_failure_aborts_delete_old_messages():
    """Si `purge_langgraph_checkpoint` lanza dentro de `summarize_and_prune`,
    `delete_old_messages` NO debe ser invocado. Sin esto, el bug original
    re-aparece: Supabase pierde mensajes mientras LangGraph los retiene."""
    # Construir un escenario donde el flujo entra al path de
    # save_summary → purge → delete: necesitamos > KEEP_RECENT mensajes
    # con suficientes caracteres para superar MAX_CHAR_THRESHOLD.
    fake_messages = [
        {"role": "user" if i % 2 == 0 else "model",
         "content": "lorem ipsum dolor sit amet " * 50,
         "created_at": f"2025-01-01T00:{i:02d}:00"}
        for i in range(memory_manager.KEEP_RECENT + 5)
    ]

    delete_calls = []
    save_calls = []

    def _fake_save(*a, **kw):
        save_calls.append((a, kw))

    def _fake_delete(*a, **kw):
        delete_calls.append((a, kw))

    with patch.object(memory_manager, "acquire_summarizing_lock", return_value=True), \
         patch.object(memory_manager, "release_summarizing_lock"), \
         patch.object(memory_manager, "get_memory", return_value=fake_messages), \
         patch.object(memory_manager, "save_summary", side_effect=_fake_save), \
         patch.object(memory_manager, "delete_old_messages", side_effect=_fake_delete), \
         patch.object(memory_manager, "purge_langgraph_checkpoint",
                      side_effect=RuntimeError("simulated purge failure")), \
         patch.object(memory_manager.ChatDeepSeek, "invoke",
                      return_value=MagicMock(content="resumen fake")):
        # No debe lanzar (el except global de summarize_and_prune captura).
        memory_manager.summarize_and_prune("test-session")

    # save_summary SÍ se llamó (paso 1).
    assert len(save_calls) == 1, "save_summary debe ejecutarse en paso 1"
    # delete_old_messages NO debe haberse llamado (paso 3 abortado).
    assert len(delete_calls) == 0, (
        "P1-20: delete_old_messages NO debe ejecutarse si purge falló"
    )


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_20_present():
    """Comentario `[P1-20]` debe documentar el rationale del orden."""
    src = inspect.getsource(memory_manager)
    assert "[P1-20]" in src
