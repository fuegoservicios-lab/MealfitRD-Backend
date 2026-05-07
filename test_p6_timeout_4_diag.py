"""[P6-TIMEOUT-4 / P6-TIMEOUT-DIAG] Tests para el bump de timeout y el
logging diagnóstico de día que timeoutea.

Contexto (corrida 2026-05-05 19:58 [c6eaf808]):
  - Self-critique: 3/3 días timeoutean a 150s
  - P5-MARKER-REGEN: Día 2 timeoutea OTRA VEZ a 150s (segundo intento)
  - Pipeline 486s (vs 225s normal) — 2× degradación
  - Causa inmediata: Gemini 504 DEADLINE_EXCEEDED visto 2×
  - Patrón sospechoso: Día 2 falla consistentemente

Fixes:
  - P6-TIMEOUT-4: bump CRITIQUE_FIX_TIMEOUT_S 150→180s (env knob existente)
  - P6-TIMEOUT-DIAG: logging del prompt/target/ingredients size en cada
    timeout para detectar si Día N es estructuralmente más denso

Cobertura:
  - Default timeout = 180.0
  - Knob env override sigue funcionando (no regresión)
  - Source contiene markers TIMEOUT-4 y TIMEOUT-DIAG
  - Logging incluye prompt size, target size, suggestion size, ingredients count
  - Logging activo en self_critique_node Y P5-MARKER-REGEN (los 2 puntos timeout)
"""
import os
import sys
import importlib
import pytest


@pytest.fixture
def reload_with_env(monkeypatch):
    def _reload(env_vars: dict | None = None):
        if env_vars:
            for k, v in env_vars.items():
                monkeypatch.setenv(k, v)
        else:
            monkeypatch.delenv("MEALFIT_CRITIQUE_FIX_TIMEOUT_S", raising=False)
        if "graph_orchestrator" in sys.modules:
            del sys.modules["graph_orchestrator"]
        import graph_orchestrator
        return sys.modules["graph_orchestrator"]
    return _reload


# ===========================================================================
# 1. Default timeout bumpeado a 180s
# ===========================================================================
def test_default_timeout_is_180s(reload_with_env):
    """P6-TIMEOUT-4: default debe ser 180.0 (vs 150.0 anterior)."""
    go = reload_with_env(None)
    assert go.CRITIQUE_FIX_TIMEOUT_S == 180.0, (
        f"Default debe ser 180s, recibido {go.CRITIQUE_FIX_TIMEOUT_S}s"
    )


# ===========================================================================
# 2. Knob env override sigue funcionando
# ===========================================================================
@pytest.mark.parametrize("env_value,expected", [
    ("60", 60.0),
    ("120.5", 120.5),
    ("240", 240.0),
    ("90", 90.0),
])
def test_env_override(reload_with_env, env_value, expected):
    go = reload_with_env({"MEALFIT_CRITIQUE_FIX_TIMEOUT_S": env_value})
    assert go.CRITIQUE_FIX_TIMEOUT_S == expected


# ===========================================================================
# 3. Marker P6-TIMEOUT-4 en source
# ===========================================================================
def test_source_has_timeout_4_marker(reload_with_env):
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go)
    assert "P6-TIMEOUT-4" in src, "Marker P6-TIMEOUT-4 debe existir"
    # El comment debe explicar el por qué
    assert "180" in src and "150" in src, (
        "Source debe documentar el cambio 150→180"
    )


# ===========================================================================
# 4. Logging P6-TIMEOUT-DIAG en self_critique_node
# ===========================================================================
def test_self_critique_has_timeout_diag_logging(reload_with_env):
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.self_critique_node)
    assert "P6-TIMEOUT-DIAG" in src, "Marker debe existir en self_critique_node"
    # Las 4 piezas de info que loguea
    assert "_prompt_chars" in src
    assert "_target_chars" in src
    assert "_suggestion_chars" in src
    assert "_ingredients_count" in src
    # Print del log
    assert "📐" in src, "Emoji marker debe ser único para grep"


# ===========================================================================
# 5. Logging P6-TIMEOUT-DIAG también en P5-MARKER-REGEN
# ===========================================================================
def test_marker_regen_has_timeout_diag_logging(reload_with_env):
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.surgical_marker_regen_node)
    assert "P6-TIMEOUT-DIAG" in src, "Marker debe existir en surgical_marker_regen_node"
    assert "_prompt_chars" in src
    assert "_ingredients_count" in src
    # Reusa target_day en scope (no re-lookup)
    assert "len(json.dumps(target_day" in src or "json.dumps(target_day" in src, (
        "Debe medir target_day directamente"
    )


# ===========================================================================
# 6. Logging solo se ejecuta en branch except (no en happy path)
# ===========================================================================
def test_timeout_diag_only_in_except_branch(reload_with_env):
    """Sanity: el logging diagnóstico debe estar en `except asyncio.TimeoutError`,
    no en el path normal (no logueamos size en cada corrección exitosa, eso
    sería spam y revelaría el contenido del plan en producción)."""
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.self_critique_node)
    # Busca el bloque except y verifica que P6-TIMEOUT-DIAG está después
    except_idx = src.find("except asyncio.TimeoutError")
    diag_idx = src.find("P6-TIMEOUT-DIAG")
    assert except_idx >= 0, "Branch except debe existir"
    assert diag_idx > except_idx, (
        "P6-TIMEOUT-DIAG debe estar DENTRO del bloque except, no en happy path"
    )


# ===========================================================================
# 7. No-regresión: P4-TIMEOUT-2 circuit breaker sigue activo
# ===========================================================================
def test_p4_timeout_2_circuit_breaker_still_active(reload_with_env):
    """Sanity: bumpear el cap individual no removió el CB cascade del
    P4-TIMEOUT-2. Los dos coexisten — CB es "abort si N días timeoutean
    en paralelo", el bump individual da margen al primer timeout."""
    go = reload_with_env(None)
    import inspect
    src = inspect.getsource(go.self_critique_node)
    assert "P4-TIMEOUT-2" in src, "CB cascade-overload debe seguir presente"
    assert "CRITIQUE_TIMEOUT_ABORT_THRESHOLD" in src, "Knob del CB debe seguir"


# ===========================================================================
# 8. Sanity: el bump no rompió la lectura del knob env
# ===========================================================================
def test_env_bool_helpers_intact(reload_with_env):
    """Sanity: _env_float existe y se usa para el knob."""
    go = reload_with_env(None)
    assert hasattr(go, "_env_float")
    import inspect
    src = inspect.getsource(go)
    assert '_env_float("MEALFIT_CRITIQUE_FIX_TIMEOUT_S"' in src, (
        "Knob debe seguir leyéndose vía _env_float (consistencia)"
    )
