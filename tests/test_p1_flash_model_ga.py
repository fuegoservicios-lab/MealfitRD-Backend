"""[P1-FLASH-MODEL-GA · 2026-05-21] Swap del modelo Flash de
`gemini-3-flash-preview` (preview, sujeto a free-tier 20 RPD) a
`gemini-3.5-flash` (GA estable, paid-tier directo).

Bug productivo (2026-05-21):
  Aunque el user habilitó billing en Google Cloud, las llamadas seguían
  pegando `429 RESOURCE_EXHAUSTED` con `quotaId: ...-FreeTier` porque
  `gemini-3-flash-preview` (modelo preview) está sujeto a una cuota free-tier
  separada y de solo 20 RPD por proyecto/modelo. Con `gemini-3.5-flash` (GA),
  las cuotas son las del paid tier estándar cuando hay billing activo (sin
  sufijo `-FreeTier` en el quotaId).

Cambios anclados:
  - `_plan_flash_model_name()` default → `"gemini-3.5-flash"` (era preview)
  - `_route_model` usa la constante `_FLASH_MODEL_NAME` en lugar de hardcodear
    el string (los hardcodes anteriores no respetaban el cambio del knob)
  - El log del router ahora interpola dinámicamente el nombre del modelo

Cobertura:
  - Default del knob es `gemini-3.5-flash`
  - Knob env var override sigue funcionando (rollback path)
  - `_route_model` con force_fast=True devuelve `_FLASH_MODEL_NAME`
  - `_route_model` perfil FÁCIL devuelve `_FLASH_MODEL_NAME`
  - `_route_model` perfil CLÍNICO devuelve `_PRO_MODEL_NAME` (no se tocó)
  - Logs interpolan el modelo (parser-based, sin hardcode legacy)
  - Tooltip-anchor P1-FLASH-MODEL-GA presente para detectar reverts
"""
import os
import re
from pathlib import Path

import pytest


_GRAPH_ORCH = Path(__file__).parent.parent / "graph_orchestrator.py"
_TOOLS_MEDICAL = Path(__file__).parent.parent / "tools_medical.py"


# ---------------------------------------------------------------------------
# Sección 1 — Tests estructurales (parser-based)
# ---------------------------------------------------------------------------

def test_flash_model_default_is_gemini_3_5_flash():
    """El default literal del knob debe ser `gemini-3.5-flash` (GA), no
    `gemini-3-flash-preview` (preview). Si alguien revierte, este test cae
    antes de regresar al modelo preview sujeto a free-tier 20 RPD."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert 'return _env_str("MEALFIT_FLASH_MODEL", "gemini-3.5-flash")' in src, (
        "Default de _plan_flash_model_name debe ser 'gemini-3.5-flash'. "
        "Si necesitas usar preview, set MEALFIT_FLASH_MODEL env var sin tocar default."
    )


def test_route_model_force_fast_uses_constant_not_hardcode():
    """El branch `force_fast=True` debe devolver `_FLASH_MODEL_NAME`, NO una
    string literal `"gemini-3-flash-preview"`. Sin esto, el swap del knob
    no se refleja en este path."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    # El bloque del force_fast
    idx = src.find("if force_fast:")
    assert idx > 0, "_route_model force_fast branch not found"
    snippet = src[idx:idx + 200]
    assert "return _FLASH_MODEL_NAME" in snippet, (
        "force_fast branch debe usar _FLASH_MODEL_NAME, no literal."
    )
    assert '"gemini-3-flash-preview"' not in snippet, (
        "Hardcode `gemini-3-flash-preview` aún presente en force_fast — viola P1-FLASH-MODEL-GA."
    )


def test_route_model_facil_uses_constant_not_hardcode():
    """El branch FÁCIL (no clínico, no force_fast) debe devolver
    `_FLASH_MODEL_NAME` y el log debe interpolar el nombre dinámicamente."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    idx = src.find("Perfil FÁCIL detectado")
    assert idx > 0
    snippet = src[idx - 50:idx + 300]
    assert "_FLASH_MODEL_NAME" in snippet, (
        "FÁCIL branch debe usar _FLASH_MODEL_NAME."
    )
    # El log debe ser f-string con interpolación dinámica
    assert "{_FLASH_MODEL_NAME}" in snippet, (
        "Log del router FÁCIL debe interpolar _FLASH_MODEL_NAME, no usar string fijo."
    )


def test_route_model_clinical_pro_unchanged():
    """El branch CLÍNICO (PRO model) sigue usando `_PRO_MODEL_NAME` — solo
    cambiamos Flash en este P-fix, Pro queda intacto."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    idx = src.find("Perfil CLÍNICO complejo detectado")
    assert idx > 0
    snippet = src[idx - 50:idx + 300]
    assert "return _PRO_MODEL_NAME" in snippet, (
        "PRO branch debe usar _PRO_MODEL_NAME."
    )


def test_no_remaining_hardcoded_flash_preview_in_route_model():
    """Ningún `"gemini-3-flash-preview"` literal debe quedar en `_route_model`."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    route_idx = src.find("def _route_model(")
    assert route_idx > 0
    # Próxima función
    next_def_idx = src.find("\ndef ", route_idx + 1)
    assert next_def_idx > 0
    route_body = src[route_idx:next_def_idx]
    assert '"gemini-3-flash-preview"' not in route_body, (
        f"Hardcode `gemini-3-flash-preview` aún presente en _route_model. "
        f"Reemplazar con `_FLASH_MODEL_NAME`."
    )


def test_p1_flash_model_ga_marker_present():
    """El marker `P1-FLASH-MODEL-GA` debe estar en el comentario justificativo
    del knob. Si alguien revierte limpiando comentarios, este test cae."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert "P1-FLASH-MODEL-GA" in src
    # Debe explicar el porqué (free-tier vs paid-tier)
    idx = src.find("P1-FLASH-MODEL-GA")
    snippet = src[idx:idx + 1500]
    assert "free-tier" in snippet.lower() or "FreeTier" in snippet, (
        "Comentario justificativo debe mencionar el modo de fallo free-tier "
        "que motivó el swap."
    )


# ---------------------------------------------------------------------------
# Sección 2 — tools_medical.py (callsite que estaba hardcoded fuera del orquestador)
# ---------------------------------------------------------------------------

def test_tools_medical_uses_helper_not_hardcode():
    """[P1-FLASH-MODEL-GA] `tools_medical.py` antes tenía
    `model="gemini-3-flash-preview"` hardcoded en el `ChatGoogleGenerativeAI(...)`.
    Ahora debe usar el helper `_medical_tool_model_name()` que lee del knob
    `MEALFIT_MEDICAL_TOOL_MODEL` (default `gemini-3.5-flash`)."""
    src = _TOOLS_MEDICAL.read_text(encoding="utf-8")
    # Helper definido
    assert "def _medical_tool_model_name()" in src, (
        "Helper _medical_tool_model_name no definido en tools_medical.py."
    )
    # Default es GA
    assert 'os.environ.get("MEALFIT_MEDICAL_TOOL_MODEL", "gemini-3.5-flash")' in src, (
        "Default del knob MEALFIT_MEDICAL_TOOL_MODEL debe ser 'gemini-3.5-flash'."
    )
    # ChatGoogleGenerativeAI usa el helper, no hardcode
    assert "model=_medical_tool_model_name()" in src, (
        "ChatGoogleGenerativeAI debe construirse con model=_medical_tool_model_name(), "
        "no con string literal."
    )
    # NO debe quedar la string literal en el callsite del ChatGoogleGenerativeAI
    chat_idx = src.find("ChatGoogleGenerativeAI(")
    assert chat_idx > 0
    chat_block = src[chat_idx:chat_idx + 400]
    assert '"gemini-3-flash-preview"' not in chat_block, (
        "Hardcode literal `gemini-3-flash-preview` aún presente en el callsite "
        "ChatGoogleGenerativeAI de tools_medical.py."
    )


# ---------------------------------------------------------------------------
# Sección 3 — Test funcional: env var override (rollback path)
# ---------------------------------------------------------------------------

def test_env_var_override_rollback_to_preview():
    """Si producción necesita rollback al modelo preview por incompatibilidad
    detectada en `gemini-3.5-flash`: `MEALFIT_FLASH_MODEL=gemini-3-flash-preview`
    debe restaurar el comportamiento previo. Verificado parseando knobs."""
    try:
        from knobs import _env_str
    except ImportError:
        pytest.skip("knobs module not importable in this env.")

    os.environ["MEALFIT_FLASH_MODEL"] = "gemini-3-flash-preview"
    try:
        result = _env_str("MEALFIT_FLASH_MODEL", "gemini-3.5-flash")
        assert result == "gemini-3-flash-preview", (
            f"Env var override roto: esperado 'gemini-3-flash-preview', recibido {result!r}"
        )
    finally:
        del os.environ["MEALFIT_FLASH_MODEL"]

    # Sin env var, debe usar default GA
    assert _env_str("MEALFIT_FLASH_MODEL", "gemini-3.5-flash") == "gemini-3.5-flash"
