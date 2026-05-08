"""[P1-31] Tests para que `FACT_CHECK_POOL_SIZE` sea configurable vía env var
y el `_FACT_CHECK_EXECUTOR` lo respete.

Bug original (audit P1-31):
  El pool dedicado de fact-checking (`_FACT_CHECK_EXECUTOR`) tenía
  `max_workers=2` HARDCODED. Otros executors y knobs del módulo (LLM
  semaphore, fact-check timeout, shutdown drain) eran env-configurables,
  pero el tamaño del pool no.

  Bajo carga (5+ pipelines concurrentes), el fact-checker generaba
  backlog: los primeros 2 fact-checks corrían y los demás se encolaban
  hasta que un slot se liberase (5-15s/call). Un operador que detectaba
  el backlog en métricas no podía subir la concurrencia sin redeploy
  con código modificado.

Fix:
  1. Knob `FACT_CHECK_POOL_SIZE` (env var
     `MEALFIT_FACT_CHECK_POOL_SIZE`) con default 2 (preserva
     comportamiento previo).
  2. Clamp [1, 16]: 1 deshabilita paralelismo (debugging / providers
     con rate limit estricto); 16 evita config errors absurdos.
  3. `_FACT_CHECK_EXECUTOR` se instancia con `max_workers=
     FACT_CHECK_POOL_SIZE`.
  4. Knob expuesto en el diagnostics dict (`get_concurrency_knobs`)
     para que SRE lo vea en logs de startup.

Cobertura:
  - test_pool_size_knob_exposed
  - test_pool_size_default_is_two
  - test_executor_max_workers_matches_knob
  - test_pool_size_clamped_to_minimum_1
  - test_pool_size_clamped_to_maximum_16
  - test_pool_size_in_concurrency_diagnostics_dict
  - test_pool_size_uses_correct_env_var_name
  - test_documentation_p1_31_present
"""
import importlib
import inspect
import os

import pytest

import graph_orchestrator


# ---------------------------------------------------------------------------
# 1. Knob expuesto y default.
# ---------------------------------------------------------------------------
def test_pool_size_knob_exposed():
    """`FACT_CHECK_POOL_SIZE` debe estar exportado a nivel módulo y ser int."""
    assert hasattr(graph_orchestrator, "FACT_CHECK_POOL_SIZE")
    val = graph_orchestrator.FACT_CHECK_POOL_SIZE
    assert isinstance(val, int)
    assert val > 0


def test_pool_size_default_is_two():
    """Default sin env var = 2 (preserva comportamiento histórico).

    Si el usuario tiene MEALFIT_FACT_CHECK_POOL_SIZE en env, este test
    se skipea (no podemos verificar el default real)."""
    if os.environ.get("MEALFIT_FACT_CHECK_POOL_SIZE"):
        pytest.skip("MEALFIT_FACT_CHECK_POOL_SIZE seteado, no podemos verificar default")
    assert graph_orchestrator.FACT_CHECK_POOL_SIZE == 2, (
        f"P1-31: default debería ser 2 para preservar comportamiento "
        f"previo, vio {graph_orchestrator.FACT_CHECK_POOL_SIZE}"
    )


def test_executor_max_workers_matches_knob():
    """El `_FACT_CHECK_EXECUTOR` debe tener `_max_workers == FACT_CHECK_POOL_SIZE`."""
    expected = graph_orchestrator.FACT_CHECK_POOL_SIZE
    actual = graph_orchestrator._FACT_CHECK_EXECUTOR._max_workers
    assert actual == expected, (
        f"P1-31: pool size desincrónico — knob={expected}, executor={actual}"
    )


# ---------------------------------------------------------------------------
# 2. Clamp del knob.
# ---------------------------------------------------------------------------
def test_pool_size_clamped_to_minimum_1(monkeypatch):
    """Setear MEALFIT_FACT_CHECK_POOL_SIZE=0 (o negativo) debe clamp a 1."""
    monkeypatch.setenv("MEALFIT_FACT_CHECK_POOL_SIZE", "0")
    importlib.reload(graph_orchestrator)
    try:
        assert graph_orchestrator.FACT_CHECK_POOL_SIZE == 1, (
            f"P1-31: 0 debe clamp a 1, vio {graph_orchestrator.FACT_CHECK_POOL_SIZE}"
        )
    finally:
        monkeypatch.delenv("MEALFIT_FACT_CHECK_POOL_SIZE", raising=False)
        importlib.reload(graph_orchestrator)


def test_pool_size_clamped_to_maximum_16(monkeypatch):
    """Setear MEALFIT_FACT_CHECK_POOL_SIZE=200 (absurdo) debe clamp a 16."""
    monkeypatch.setenv("MEALFIT_FACT_CHECK_POOL_SIZE", "200")
    importlib.reload(graph_orchestrator)
    try:
        assert graph_orchestrator.FACT_CHECK_POOL_SIZE == 16, (
            f"P1-31: 200 debe clamp a 16, vio {graph_orchestrator.FACT_CHECK_POOL_SIZE}"
        )
    finally:
        monkeypatch.delenv("MEALFIT_FACT_CHECK_POOL_SIZE", raising=False)
        importlib.reload(graph_orchestrator)


def test_pool_size_within_range_passes_through(monkeypatch):
    """Valores dentro del rango [1, 16] se respetan sin clamp."""
    for valid in (1, 4, 8, 16):
        monkeypatch.setenv("MEALFIT_FACT_CHECK_POOL_SIZE", str(valid))
        importlib.reload(graph_orchestrator)
        try:
            assert graph_orchestrator.FACT_CHECK_POOL_SIZE == valid, (
                f"P1-31: valor válido {valid} debería pasar sin clamp, "
                f"vio {graph_orchestrator.FACT_CHECK_POOL_SIZE}"
            )
        finally:
            monkeypatch.delenv("MEALFIT_FACT_CHECK_POOL_SIZE", raising=False)
    importlib.reload(graph_orchestrator)


def test_executor_respects_overridden_pool_size(monkeypatch):
    """Tras override + reload, el executor debe re-instanciarse con el
    nuevo `max_workers`."""
    monkeypatch.setenv("MEALFIT_FACT_CHECK_POOL_SIZE", "5")
    importlib.reload(graph_orchestrator)
    try:
        assert graph_orchestrator._FACT_CHECK_EXECUTOR._max_workers == 5, (
            f"P1-31: tras override env=5 + reload, executor max_workers "
            f"debe ser 5, vio {graph_orchestrator._FACT_CHECK_EXECUTOR._max_workers}"
        )
    finally:
        monkeypatch.delenv("MEALFIT_FACT_CHECK_POOL_SIZE", raising=False)
        importlib.reload(graph_orchestrator)


# ---------------------------------------------------------------------------
# 3. Visibilidad del knob en diagnostics.
# ---------------------------------------------------------------------------
def test_pool_size_in_concurrency_diagnostics_dict():
    """El knob debe aparecer en el registry auto-poblado de `_log_active_knobs`
    para que SRE pueda verificar el valor activo en logs de startup sin
    inspeccionar código.

    [P3-NEW-D · 2026-05-08] Antes este test verificaba que la key literal
    `"FACT_CHECK_POOL_SIZE"` apareciera en el source del dict hardcoded.
    Tras P3-NEW-D, el dict se reemplazó por iteración sobre `_KNOBS_REGISTRY`
    (auto-poblado por `_env_int/_env_float/_env_bool`). El contrato sigue
    siendo el mismo (knob visible en logs de startup), pero ahora se valida
    via registry presence en lugar de source pattern del dict.
    """
    snap = graph_orchestrator.get_knobs_registry_snapshot()
    assert "MEALFIT_FACT_CHECK_POOL_SIZE" in snap, (
        "P1-31: MEALFIT_FACT_CHECK_POOL_SIZE debe registrarse en "
        "_KNOBS_REGISTRY (auto-poblado por _env_int) para visibilidad en "
        "logs de startup vía _log_active_knobs."
    )
    info = snap["MEALFIT_FACT_CHECK_POOL_SIZE"]
    assert info["type"] == "int", (
        "El knob debe ser de tipo int (resuelto vía _env_int)."
    )


# ---------------------------------------------------------------------------
# 4. Naming convention del env var.
# ---------------------------------------------------------------------------
def test_pool_size_uses_correct_env_var_name():
    """El env var debe llamarse `MEALFIT_FACT_CHECK_POOL_SIZE` (consistente
    con `MEALFIT_FACT_CHECK_TOOL_TIMEOUT_S`, `MEALFIT_FACT_CHECK_SHUTDOWN_DRAIN_S`).
    Verifica vía source pattern."""
    src = inspect.getsource(graph_orchestrator)
    assert "MEALFIT_FACT_CHECK_POOL_SIZE" in src, (
        "P1-31: env var debe ser `MEALFIT_FACT_CHECK_POOL_SIZE` "
        "(consistente con otros knobs MEALFIT_FACT_CHECK_*)."
    )


# ---------------------------------------------------------------------------
# 5. Defensa estructural: hardcoded "max_workers=2" desapareció.
# ---------------------------------------------------------------------------
def test_executor_no_longer_hardcodes_max_workers_2():
    """El source NO debe contener `max_workers=2` literal en la
    instanciación de `_FACT_CHECK_EXECUTOR` (el knob ya lo cubre).

    Defensa contra reintroducir el hardcode por refactor."""
    src = inspect.getsource(graph_orchestrator)
    # Buscar el bloque de instanciación de _FACT_CHECK_EXECUTOR.
    idx = src.find("_FACT_CHECK_EXECUTOR = _DrainableThreadPoolExecutor(")
    assert idx > -1, "No se encontró la instanciación de _FACT_CHECK_EXECUTOR"
    block = src[idx : idx + 400]
    # Debe usar el knob.
    assert "FACT_CHECK_POOL_SIZE" in block, (
        f"P1-31: la instanciación debe usar `FACT_CHECK_POOL_SIZE`, no "
        f"un literal. Bloque: {block[:200]!r}"
    )
    # NO debe usar `max_workers=2` literal.
    assert "max_workers=2," not in block and "max_workers=2)" not in block, (
        f"P1-31 regression: `max_workers=2` literal reapareció. "
        f"Bloque: {block[:200]!r}"
    )


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_31_present():
    """Comentario `[P1-31]` debe documentar el knob y rationale."""
    src = inspect.getsource(graph_orchestrator)
    assert "[P1-31]" in src, (
        "P1-31: falta marker que documente la introducción del knob."
    )


def test_documentation_mentions_clamp_or_backlog():
    """El comentario debe explicar el rationale: backlog bajo carga,
    clamp [1, 16], default 2 preserva comportamiento previo."""
    src = inspect.getsource(graph_orchestrator)
    p131_idx = src.find("[P1-31]")
    window = src[p131_idx : p131_idx + 2000]
    needles = ["backlog", "clamp", "default 2", "comportamiento previo",
               "redeploy", "concurrencia", "tunear"]
    found = any(n.lower() in window.lower() for n in needles)
    assert found, (
        f"P1-31: el comentario debe explicar backlog / clamp / default. "
        f"Encontrado: {window[:300]!r}"
    )
