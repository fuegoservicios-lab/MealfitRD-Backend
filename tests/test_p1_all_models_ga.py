"""[P1-ALL-MODELS-GA · 2026-05-21] Eliminación total de modelos `*-preview` como
defaults en código de producción. Decisión del owner: reemplazar ambos
`gemini-3-flash-preview` y `gemini-3.1-pro-preview` por `gemini-3.5-flash` (GA).

Contexto:
  - P1-FLASH-MODEL-GA (mismo día) ya había swapped FLASH default.
  - Este P-fix completa el barrido: PRO, VISION, TOOLS (2 callsites),
    TOOLS_MEDICAL — todos a `gemini-3.5-flash`.

Razón:
  1. Modelos `*-preview` de Google pueden deprecarse sin SLA. Incidente real:
     `gemini-3.1-pro-preview` con CB stale 4.4 días el 2026-05-11.
  2. Preview models están sujetos a cuotas free-tier separadas
     (`quotaId: ...-FreeTier`) que persisten aunque el proyecto tenga billing
     habilitado. Bloqueo crítico observado el 2026-05-21.
  3. `gemini-3.5-flash` es GA estable, paid-tier directo cuando billing activo.

Trade-off: perfiles CLÍNICOS complejos ya no escalan a Pro reasoning. Si la
calidad clínica degrada visiblemente, rollback individual via env vars sin
redeploy: `MEALFIT_PRO_MODEL=gemini-3.1-pro-preview` (y demás).

Cobertura:
  - graph_orchestrator: `_plan_pro_model_name` default = `gemini-3.5-flash`
  - vision_agent: `_vision_model_name` default = `gemini-3.5-flash`
  - tools.py: `_tools_pref_agent_model_name` default = `gemini-3.5-flash`
  - tools.py: `_tools_modify_meal_model_name` default = `gemini-3.5-flash`
  - tools_medical: `_medical_tool_model_name` default = `gemini-3.5-flash`
  - Cero `_env_str/os.environ.get` con default `"*-preview"` en producción
  - Markers P1-ALL-MODELS-GA presentes en los 4 archivos productivos editados
"""
import os
import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).parent.parent
_GO_PY = _BACKEND / "graph_orchestrator.py"
_VISION_PY = _BACKEND / "vision_agent.py"
_TOOLS_PY = _BACKEND / "tools.py"
_TOOLS_MEDICAL_PY = _BACKEND / "tools_medical.py"

_GA_DEFAULT = "gemini-3.5-flash"
_PRO_PREVIEW = "gemini-3.1-pro-preview"
_FLASH_PREVIEW = "gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Sección 1 — Defaults de producción
# ---------------------------------------------------------------------------

def test_pro_default_is_gemini_3_5_flash():
    """`_plan_pro_model_name()` debe leer `MEALFIT_PRO_MODEL` con default
    `gemini-3.5-flash`. Pre-fix era `gemini-3.1-pro-preview` (preview)."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert f'_env_str("MEALFIT_PRO_MODEL", "{_GA_DEFAULT}")' in src, (
        f"Default Pro debe ser {_GA_DEFAULT!r}. Rollback via "
        f"`MEALFIT_PRO_MODEL=gemini-3.1-pro-preview` sin redeploy."
    )


def test_vision_default_is_gemini_3_5_flash():
    """`_vision_model_name()` debe leer `MEALFIT_VISION_MODEL` con default
    `gemini-3.5-flash`. Pre-fix era `gemini-3.1-pro-preview`."""
    src = _VISION_PY.read_text(encoding="utf-8")
    assert f'_env_str("MEALFIT_VISION_MODEL", "{_GA_DEFAULT}")' in src, (
        f"Default Vision debe ser {_GA_DEFAULT!r}. Rollback via "
        f"`MEALFIT_VISION_MODEL=gemini-3.1-pro-preview`."
    )


def test_tools_pref_agent_default_is_gemini_3_1_flash_lite():
    """`_tools_pref_agent_model_name()` debe leer
    `MEALFIT_TOOLS_PREF_AGENT_MODEL` con default `gemini-3.1-flash-lite`.

    [P3-FLASH-LITE-COST-CUT · 2026-05-21] Re-migrado de `gemini-3.5-flash`
    (P1-ALL-MODELS-GA) a `gemini-3.1-flash-lite`. El preference analyzer hace
    clasificación simple — output estructurado low-stakes; lite cubre el caso
    y reduce costo ~50%. Rollback: `MEALFIT_TOOLS_PREF_AGENT_MODEL=gemini-3.5-flash`.
    """
    src = _TOOLS_PY.read_text(encoding="utf-8")
    m = re.search(
        r'MEALFIT_TOOLS_PREF_AGENT_MODEL[\s\S]{0,400}',
        src,
    )
    assert m is not None, "MEALFIT_TOOLS_PREF_AGENT_MODEL no encontrado en tools.py"
    assert "gemini-3.1-flash-lite" in m.group(0), (
        f"Default debe contener 'gemini-3.1-flash-lite' (P3-FLASH-LITE-COST-CUT). "
        f"Encontrado:\n{m.group(0)[:400]}"
    )


def test_tools_modify_meal_default_is_gemini_3_5_flash():
    """`_tools_modify_meal_model_name()` debe leer
    `MEALFIT_TOOLS_MODIFY_MEAL_MODEL` con default `gemini-3.5-flash`."""
    src = _TOOLS_PY.read_text(encoding="utf-8")
    m = re.search(
        r'MEALFIT_TOOLS_MODIFY_MEAL_MODEL[\s\S]{0,200}',
        src,
    )
    assert m is not None
    assert _GA_DEFAULT in m.group(0)


def test_tools_medical_default_uses_knob_for_rollback():
    """`_medical_tool_model_name()` debe usar `MEALFIT_MEDICAL_TOOL_MODEL`
    como knob de override sin redeploy. El default actual es
    `gemini-3.1-flash-lite` (P3-COST-CUT-AUX · 2026-05-22) — antes era
    `gemini-3.5-flash` (P1-FLASH-MODEL-GA). El test parser-based del
    bundle Tier 1 ([`test_p3_cost_cut_aux.py::test_medical_tool_default_is_lite`])
    asserta el default actual; este solo verifica que el knob existe."""
    src = _TOOLS_MEDICAL_PY.read_text(encoding="utf-8")
    assert "MEALFIT_MEDICAL_TOOL_MODEL" in src, (
        "P1-FLASH-MODEL-GA / P3-COST-CUT-AUX regresión: knob removido. "
        "Sin él no hay rollback operacional sin redeploy."
    )


# ---------------------------------------------------------------------------
# Sección 2 — Cero defaults preview en producción (blanket scan)
# ---------------------------------------------------------------------------

_PROD_FILES = [
    "graph_orchestrator.py",
    "vision_agent.py",
    "tools.py",
    "tools_medical.py",
    "agent.py",
    "fact_extractor.py",
    "proactive_agent.py",
]


def test_no_preview_defaults_in_production_code():
    """Ningún archivo de producción debe tener un `_env_str("MEALFIT_X_MODEL",
    "<preview>")` o `os.environ.get("MEALFIT_X_MODEL", "<preview>")` como
    default activo. Pricing maps (`db_profiles.py`) son excepción intencional:
    deben listar pricing de modelos preview para que el operador pueda hacer
    rollback via env var sin perder tracking de costos.
    """
    violations = []
    for fname in _PROD_FILES:
        p = _BACKEND / fname
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        # _env_str("KNOB", "default")
        for m in re.finditer(
            r'_env_str\(\s*"(MEALFIT_[^"]+MODEL)"\s*,\s*"([^"]+)"\s*\)',
            src,
        ):
            knob, default = m.group(1), m.group(2)
            if "preview" in default:
                violations.append(f"{fname}: {knob}={default!r}")
        # os.environ.get("KNOB", "default")
        for m in re.finditer(
            r'os\.environ\.get\(\s*"(MEALFIT_[^"]+MODEL)"\s*,\s*"([^"]+)"',
            src,
        ):
            knob, default = m.group(1), m.group(2)
            if "preview" in default:
                violations.append(f"{fname}: {knob}={default!r}")

    assert not violations, (
        f"P1-ALL-MODELS-GA viola: {len(violations)} default(s) preview en producción:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# Sección 3 — Markers presentes
# ---------------------------------------------------------------------------

def test_marker_present_in_all_edited_files():
    """Marker `P1-ALL-MODELS-GA` debe estar en los archivos modificados como
    tooltip-anchor. Si alguien borra el comentario sin marker, este test cae."""
    expected_in = [_GO_PY, _VISION_PY, _TOOLS_PY]
    for p in expected_in:
        src = p.read_text(encoding="utf-8")
        assert "P1-ALL-MODELS-GA" in src, (
            f"Marker `P1-ALL-MODELS-GA` ausente en {p.name}. "
            f"Sin él, un revert futuro perdería el contexto del barrido."
        )


# ---------------------------------------------------------------------------
# Sección 4 — Env var override sigue funcionando (rollback path)
# ---------------------------------------------------------------------------

def test_env_var_rollback_pro_model():
    """`MEALFIT_PRO_MODEL=gemini-3.1-pro-preview` debe restaurar el modelo
    preview legacy si la calidad clínica del Flash demuestra ser insuficiente."""
    try:
        from knobs import _env_str
    except ImportError:
        pytest.skip("knobs module not importable.")

    os.environ["MEALFIT_PRO_MODEL"] = _PRO_PREVIEW
    try:
        assert _env_str("MEALFIT_PRO_MODEL", _GA_DEFAULT) == _PRO_PREVIEW
    finally:
        del os.environ["MEALFIT_PRO_MODEL"]

    # Sin env var, default GA
    assert _env_str("MEALFIT_PRO_MODEL", _GA_DEFAULT) == _GA_DEFAULT
