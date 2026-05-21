"""[P1-HEDGE-THRESHOLD-RAISE · 2026-05-21] Verifica los defaults de hedge
tras el cost-reduction bundle del 2026-05-21.

Bug productivo del chunk 713ff43a 2026-05-21 00:08:
  Los 3 day_generators tardaron 74s, 89s, 104s (rangos normales bajo
  throttling de gemini-3-flash). Con threshold previo de 45s, los 3 días
  dispararon hedge especulativo → "3/3 hedges en flight" (saturación del
  limiter) + 3 llamadas Gemini innecesarias (primaries terminaron <30s
  después del hedge fire).

Fix:
  Subir HEDGE_AFTER_BASE_S default 45.0 → 90.0. En el mismo chunk solo el
  día 3 (104s) dispararía hedge → ahorro ~2 llamadas Gemini por chunk en
  condiciones normales. HEDGE_MAX_CONCURRENT se mantiene en 3 — bajarlo
  reintroduciría el bug P2-HEDGE-LIMITER-RAISE 2026-05-16 (3er día sin slot
  → CB OPEN → self_critique bloqueado).

Cobertura:
  - Default actual de HEDGE_AFTER_BASE_S es 90.0
  - Default actual de HEDGE_MAX_CONCURRENT es 3 (no se tocó)
  - Default actual de HARD_CEILING_S sigue siendo 170.0 (sanity)
  - Override vía env var sigue funcionando (rollback path)
  - Tooltip-anchor del marker para que un revert futuro caiga este test
"""
import os
import importlib
from pathlib import Path

import pytest


_GRAPH_ORCH = Path(__file__).parent.parent / "graph_orchestrator.py"


# ---------------------------------------------------------------------------
# Tests estructurales (parser-based, sin imports pesados)
# ---------------------------------------------------------------------------

def test_hedge_after_base_default_is_90_in_source():
    """El default literal debe ser 90.0 en source. Si alguien baja a 45
    en un revert, este test falla antes de regresar el comportamiento
    costoso. Si quieres bajarlo (e.g., a 60), actualiza también este test
    con justificación en CLAUDE.md."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert 'HEDGE_AFTER_BASE_S          = _env_float("MEALFIT_HEDGE_AFTER_BASE_S",          90.0)' in src, (
        "HEDGE_AFTER_BASE_S default debe ser 90.0 (P1-HEDGE-THRESHOLD-RAISE)."
    )


def test_hedge_max_concurrent_default_preserved_at_3():
    """HEDGE_MAX_CONCURRENT debe mantenerse en 3 — bajarlo reintroduce el
    bug P2-HEDGE-LIMITER-RAISE 2026-05-16 (3er día sin slot → CB OPEN).
    Aunque el cost-reduction bundle 2026-05-21 evaluó bajarlo a 1, lo
    rechazamos porque comprometía la fix simétrica de hace solo 5 días."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert 'HEDGE_MAX_CONCURRENT_KNOB   = _env_int  ("MEALFIT_HEDGE_MAX_CONCURRENT",        3)' in src, (
        "HEDGE_MAX_CONCURRENT debe mantenerse en 3. Ver P2-HEDGE-LIMITER-RAISE memoria."
    )


def test_hard_ceiling_unchanged_at_170():
    """Sanity check: HARD_CEILING_S no debe haberse modificado por el
    bundle de threshold (es ortogonal — kill-switch para chunks colgados)."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert 'HARD_CEILING_S              = _env_float("MEALFIT_HARD_CEILING_S",              170.0)' in src


def test_tooltip_anchor_for_threshold_raise_present():
    """El marker `P1-HEDGE-THRESHOLD-RAISE` debe estar en el comentario
    justificativo. Un revert que borre el comentario debe fallar este test."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert "P1-HEDGE-THRESHOLD-RAISE" in src
    # El comentario debe explicar la razón observable (no solo el qué).
    assert "713ff43a" in src or "74s, 89s, 104s" in src, (
        "El comentario debe citar el incidente real que motivó el raise — sin "
        "eso, un dev futuro que vea un chunk lento podría revertir sin saber por qué."
    )


# ---------------------------------------------------------------------------
# Test funcional: env var override sigue funcionando (rollback path)
# ---------------------------------------------------------------------------

def test_env_var_override_rollback_to_45():
    """Si producción quiere rollback sin redeploy: `MEALFIT_HEDGE_AFTER_BASE_S=45`
    debe restaurar el comportamiento pre-fix. Verificado parseando el módulo
    `knobs.py` con la env var seteada."""
    # Probar el helper subyacente directamente sin importar graph_orchestrator
    # (que requiere langgraph/supabase). El contrato testeable es:
    # `_env_float("MEALFIT_HEDGE_AFTER_BASE_S", 90.0)` debe devolver 45.0
    # cuando la env var está seteada a "45".
    try:
        from knobs import _env_float
    except ImportError:
        pytest.skip("knobs module not importable in this env (missing deps).")

    os.environ["MEALFIT_HEDGE_AFTER_BASE_S"] = "45"
    try:
        result = _env_float("MEALFIT_HEDGE_AFTER_BASE_S", 90.0)
        assert result == 45.0, f"Env var override roto: esperado 45.0, recibido {result}"
    finally:
        del os.environ["MEALFIT_HEDGE_AFTER_BASE_S"]

    # Sin env var, debe usar default 90.0
    assert _env_float("MEALFIT_HEDGE_AFTER_BASE_S", 90.0) == 90.0
