"""[P6-EVALUATOR-USE-PRO] Tests para el knob que escala el evaluator del
self-critique a Pro.

Contexto:
  En la corrida 2026-05-05 19:13 el evaluator (Flash) omitió Día 2 en su
  `critique.suggestions` aunque el detector determinístico lo había listado.
  P6-CRITIQUE-DAY-FLOOR resuelve el síntoma (recupera el día), pero no la
  causa: Flash a veces no tiene capacidad para reasonar sobre N días simul-
  tánea y mantener todos en su narrativa.

  El knob `MEALFIT_EVALUATOR_USE_PRO` escala SOLO el evaluator (1 call) a
  Pro. Day generators (3 paralelos) y corrector siguen en Flash. Trade:
  ~+5-10s en self-critique, +$0.001-0.003 por plan, mejor cobertura.

Cobertura:
  - Knob OFF (default) → routing original (Flash via _route_model force_fast)
  - Knob ON → Pro hardcoded
  - Knob ON → timeout bumpeado a 60s (Pro structured output más lento)
  - Knob OFF → timeout original 30s
  - Sanity: marker en source, knob registrado vía _env_bool
  - Default = False (no afecta producción al merge sin override)
"""
import os
import sys
import importlib
import pytest


@pytest.fixture
def reload_with_env(monkeypatch):
    """Helper para reimport el módulo bajo distintos valores del knob."""
    def _reload(value: str | None):
        if value is None:
            monkeypatch.delenv("MEALFIT_EVALUATOR_USE_PRO", raising=False)
        else:
            monkeypatch.setenv("MEALFIT_EVALUATOR_USE_PRO", value)
        # Forzar reimport
        if "graph_orchestrator" in sys.modules:
            del sys.modules["graph_orchestrator"]
        import graph_orchestrator  # noqa: F401
        return sys.modules["graph_orchestrator"]
    return _reload


# ===========================================================================
# 1. Default = False (knob no establecido)
# ===========================================================================
def test_knob_default_off(reload_with_env):
    """Sin env var, EVALUATOR_USE_PRO debe ser False — comportamiento
    backwards-compatible al merge."""
    go = reload_with_env(None)
    assert go.EVALUATOR_USE_PRO is False, (
        "Default DEBE ser OFF para no cambiar producción al merge sin opt-in"
    )


# ===========================================================================
# 2. Knob ON → True
# ===========================================================================
@pytest.mark.parametrize("env_val", ["1", "true", "True", "TRUE", "yes"])
def test_knob_on_truthy_values(reload_with_env, env_val):
    go = reload_with_env(env_val)
    assert go.EVALUATOR_USE_PRO is True, f"'{env_val}' debe ser interpretado como True"


# ===========================================================================
# 3. Knob OFF explícito → False
# ===========================================================================
@pytest.mark.parametrize("env_val", ["0", "false", "False", "no"])
def test_knob_off_falsy_values(reload_with_env, env_val):
    go = reload_with_env(env_val)
    assert go.EVALUATOR_USE_PRO is False


# ===========================================================================
# 4. Sanity: source code refleja la lógica del knob
# ===========================================================================
def test_source_uses_pro_when_knob_on(reload_with_env):
    """Sanity: el self_critique_node debe consultar EVALUATOR_USE_PRO y usar
    _PRO_MODEL_NAME cuando esté ON."""
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.self_critique_node)
    assert "EVALUATOR_USE_PRO" in src, (
        "self_critique_node debe leer el knob"
    )
    assert "_PRO_MODEL_NAME" in src, (
        "self_critique_node debe usar _PRO_MODEL_NAME cuando knob ON"
    )
    assert "P6-EVALUATOR-USE-PRO" in src, (
        "Marker debe existir para alertar regresión"
    )


def test_source_bumps_timeout_when_knob_on(reload_with_env):
    """Pro structured output ~20-40s — el timeout default 30s causaría
    timeouts. Source debe bumpear cuando knob ON."""
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.self_critique_node)
    # Bump explícito visible
    assert "60.0" in src and "EVALUATOR_USE_PRO" in src, (
        "Timeout debe bumpearse a >30s cuando EVALUATOR_USE_PRO está ON"
    )


# ===========================================================================
# 5. Knob registrado correctamente (env_bool, no env var raw)
# ===========================================================================
def test_knob_registered_via_env_bool(reload_with_env):
    """Sanity: knob debe ir vía _env_bool (no os.environ.get raw) para tener
    parsing consistente con el resto del módulo."""
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go)
    # Buscar la definición exacta
    assert 'EVALUATOR_USE_PRO = _env_bool("MEALFIT_EVALUATOR_USE_PRO"' in src, (
        "Knob debe usar _env_bool con nombre canónico MEALFIT_EVALUATOR_USE_PRO"
    )


# ===========================================================================
# 6. Constantes del módulo siguen presentes (no breaking change)
# ===========================================================================
def test_pro_and_flash_model_names_unchanged(reload_with_env):
    """[P0-DEEPSEEK-MIGRATION · 2026-06-12] Flash = tier gratis
    (`deepseek-v4-flash`); Pro = tiers pagados (`deepseek-v4-pro`)."""
    go = reload_with_env(None)
    assert go._PRO_MODEL_NAME == "deepseek-v4-pro"
    assert go._FLASH_MODEL_NAME == "deepseek-v4-flash"


# ===========================================================================
# 7. Knob OFF: ruta original via _route_model(force_fast=True) sigue intacta
# ===========================================================================
def test_knob_off_keeps_original_routing_path(reload_with_env):
    """Cuando OFF, el código debe mantener el path original — el evaluator se
    routea como antes via `_route_model(..., force_fast=True)`."""
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.self_critique_node)
    # Path original todavía presente en el branch else
    assert "force_fast=True" in src, (
        "Path original (force_fast=True) debe seguir presente en branch OFF"
    )


# ===========================================================================
# 8. P6-CRITIQUE-DAY-FLOOR sigue activo (no se removió por este fix)
# ===========================================================================
def test_critique_day_floor_still_active(reload_with_env):
    """Sanity: P6-EVALUATOR-USE-PRO complementa, no reemplaza, P6-CRITIQUE-DAY-FLOOR.
    Aún con Pro evaluator, el floor determinístico debe seguir como red de
    seguridad — Pro también puede omitir días en casos extremos."""
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.self_critique_node)
    assert "P6-CRITIQUE-DAY-FLOOR" in src, (
        "Day-floor debe seguir activo — defensa-en-profundidad complementaria"
    )
