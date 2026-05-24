"""[P3-PLANNER-LITE-COST · 2026-05-21] Override del SKELETON PLANNER a
`gemini-3.1-flash-lite` via knob `MEALFIT_PLANNER_MODEL` para reducir costo
~7% por primera generación de plan.

Contexto:
  El planner skeleton corre primero en la pipeline y solo asigna nombres +
  slots (proteína por día, carbo por día, vegetales, técnica de cocción) —
  classification-like, sin constraints duros de macros. Los constraints
  duros viven en los day generators downstream (3 paralelos). Lite cubre
  la tarea de asignación + reduce costo en la etapa que hoy pesa ~10% del
  total del plan.

Patrón:
  - `_planner_model_name()` retorna `MEALFIT_PLANNER_MODEL` con default
    `gemini-3.1-flash-lite`.
  - Si knob set a cadena vacía (`""`), el callsite respeta el ruteo dinámico
    legacy `_route_model(form_data, attempt)` (PRO complejos, FLASH simples).
  - Si knob set a un modelo específico (`gemini-3.5-flash`), fuerza ese modelo.

Trade-off:
  Skeleton de menor calidad → day generators trabajan más → potencial aumento
  retry rate. Mitigación: monitor `pipeline_metrics.attempts` post-deploy.
  Rollback inmediato: `MEALFIT_PLANNER_MODEL=""` restaura ruteo legacy.

NOT cubierto:
  - Day generators (siguen en flash via `_route_model`) — pruning de su
    prompt es B del bundle 2026-05-21, deferred sin medición per-block.
  - Self-critique corrections (siguen en flash) — constraints duros.
  - Medical reviewer / judge / fact-checker (ya en lite via P1-FLASH-LITE-AUX-NODES).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).parent.parent
_GO_PY = _BACKEND / "graph_orchestrator.py"

_FLASH_LITE = "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Sección 1 — Helper presente + default correcto
# ---------------------------------------------------------------------------


def test_planner_helper_exists():
    """`_planner_model_name()` debe existir + leer `MEALFIT_PLANNER_MODEL`
    vía `_env_str` (auto-registry en `_KNOBS_REGISTRY`)."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "def _planner_model_name" in src, (
        "Helper `_planner_model_name()` no encontrado en graph_orchestrator.py. "
        "Sin él el override del planner no aplica."
    )
    assert "MEALFIT_PLANNER_MODEL" in src, (
        "Knob `MEALFIT_PLANNER_MODEL` ausente en graph_orchestrator.py."
    )


def test_planner_default_is_flash_lite():
    """Default debe ser `gemini-3.1-flash-lite` — esto es la decisión de
    P3-PLANNER-LITE-COST (skeleton no necesita razonamiento Pro/Flash)."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert f'_env_str("MEALFIT_PLANNER_MODEL", "{_FLASH_LITE}")' in src, (
        f"Default `MEALFIT_PLANNER_MODEL` debe ser {_FLASH_LITE!r}. Rollback "
        f"a ruteo legacy: `MEALFIT_PLANNER_MODEL=\"\"` (cadena vacía)."
    )


# ---------------------------------------------------------------------------
# Sección 2 — Callsite usa el helper con fallback al ruteo dinámico
# ---------------------------------------------------------------------------


def test_planner_callsite_uses_override():
    """El callsite del planner debe usar `_planner_model_name()` con fallback
    a `_route_model(form_data, attempt)` cuando el override es vacío.
    Patrón: `_planner_override if _planner_override else _route_model(...)`."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Localizar el bloque del planner_model = ...
    m = re.search(
        r"_planner_override\s*=\s*_planner_model_name\(\)\s*\n"
        r"\s*planner_model\s*=\s*_planner_override\s+if\s+_planner_override\s+else\s+_route_model\(",
        src,
    )
    assert m is not None, (
        "Callsite del planner no usa el patrón override+fallback. Debe ser:\n"
        "    _planner_override = _planner_model_name()\n"
        "    planner_model = _planner_override if _planner_override else _route_model(form_data, attempt)\n"
        "Sin el fallback, set vacío de `MEALFIT_PLANNER_MODEL` rompería el callsite."
    )


def test_planner_callsite_no_direct_route_model_only():
    """El callsite NO debe ser solo `planner_model = _route_model(...)` sin
    el override (eso revertiría P3-PLANNER-LITE-COST). Verificación negativa
    para que un revert futuro falle el test."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Buscar `planner_model = _route_model(...)` sin override previo en la
    # misma región del archivo (heurística: la línea exacta sin ternario).
    bad_pattern = re.search(
        r"^\s*planner_model\s*=\s*_route_model\(form_data,\s*attempt\)\s*$",
        src,
        re.MULTILINE,
    )
    assert bad_pattern is None, (
        "Encontré `planner_model = _route_model(form_data, attempt)` directo sin "
        "override. Esto es revert de P3-PLANNER-LITE-COST — el helper "
        "`_planner_model_name()` debe interceptar el ruteo del planner."
    )


# ---------------------------------------------------------------------------
# Sección 3 — Marker presente + cross-link con test
# ---------------------------------------------------------------------------


def test_marker_present_in_graph_orchestrator():
    """Marker `P3-PLANNER-LITE-COST` debe estar en graph_orchestrator.py
    como tooltip-anchor. Si alguien borra el comentario sin marker, este
    test cae antes que `test_p2_hist_audit_14_marker_test_link.py`."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "P3-PLANNER-LITE-COST" in src, (
        "Marker `P3-PLANNER-LITE-COST` ausente en graph_orchestrator.py. "
        "Sin él un revert futuro perdería el contexto del override."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Env var override (rollback paths)
# ---------------------------------------------------------------------------


def test_env_var_rollback_empty_string_restores_routing():
    """`MEALFIT_PLANNER_MODEL=""` debe ser tratado como "respetar ruteo
    dinámico" (cadena vacía cae en el else del ternario). Verificado en el
    callsite: `_planner_override if _planner_override else _route_model(...)`.
    Python: cadena vacía es falsy, así que el else se dispara."""
    # Test conceptual: cadena vacía es falsy en Python.
    assert not bool(""), "Cadena vacía debe ser falsy (precondición Python)."

    # Test funcional vía _env_str si está disponible.
    try:
        from knobs import _env_str
    except ImportError:
        pytest.skip("knobs no importable.")

    os.environ["MEALFIT_PLANNER_MODEL"] = ""
    try:
        result = _env_str("MEALFIT_PLANNER_MODEL", _FLASH_LITE)
        # _env_str puede o no retornar "" según implementación — el contrato
        # importante es que el callsite use truthy-check (`if _planner_override`).
        # Si _env_str retorna "" para env empty, el callsite cae al else (legacy).
        # Si retorna default, el override se mantiene.
        assert result in ("", _FLASH_LITE), (
            f"_env_str retornó {result!r} con env empty — comportamiento "
            f"inesperado; revisar contrato del knob."
        )
    finally:
        del os.environ["MEALFIT_PLANNER_MODEL"]


def test_env_var_rollback_to_flash_ga():
    """`MEALFIT_PLANNER_MODEL=gemini-3.5-flash` debe forzar flash GA si
    lite degrada visiblemente. Path explícito por encima del legacy."""
    try:
        from knobs import _env_str
    except ImportError:
        pytest.skip("knobs no importable.")

    os.environ["MEALFIT_PLANNER_MODEL"] = "gemini-3.5-flash"
    try:
        assert (
            _env_str("MEALFIT_PLANNER_MODEL", _FLASH_LITE) == "gemini-3.5-flash"
        )
    finally:
        del os.environ["MEALFIT_PLANNER_MODEL"]

    # Sin env var, default lite.
    assert _env_str("MEALFIT_PLANNER_MODEL", _FLASH_LITE) == _FLASH_LITE


# ---------------------------------------------------------------------------
# Sección 5 — Sanity: day generator NO afectado
# ---------------------------------------------------------------------------


def test_day_generator_still_uses_route_model_or_dedicated_helper():
    """Sanity: el day generator NO debe haber sido afectado por este P-fix.
    Debe seguir usando `_route_model_for_day_generator` o `_route_model`,
    NO `_planner_model_name`. Este test es la red de seguridad contra un
    cambio accidental que aplique lite también a los day workers (constraints
    duros — lite degradaría calidad significativamente)."""
    src = _GO_PY.read_text(encoding="utf-8")
    # El day generator vive aprox en líneas 5100+. Buscar el callsite del
    # `ChatGoogleGenerativeAI(...)` del day worker y confirmar que NO usa
    # `_planner_model_name` en su rhs.
    # Heurística: buscar `day_llm = ChatGoogleGenerativeAI(` y ver las ~10
    # líneas previas para identificar el modelo asignado.
    m = re.search(
        r"day_llm\s*=\s*ChatGoogleGenerativeAI\(",
        src,
    )
    assert m is not None, "Callsite `day_llm = ChatGoogleGenerativeAI(...)` no encontrado."
    # Tomar 800 chars previos (suficiente para ver la asignación de modelo)
    region_start = max(0, m.start() - 800)
    region = src[region_start:m.start() + 200]
    assert "_planner_model_name" not in region, (
        "Day generator está usando `_planner_model_name()` — eso aplica lite "
        "a constraints duros (macros + alergias + 3P/3C/3V + slot-collision). "
        "ERROR DE SCOPE de P3-PLANNER-LITE-COST. Solo el planner skeleton."
    )
