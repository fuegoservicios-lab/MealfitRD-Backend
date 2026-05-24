"""[P3-FLASH-LITE-COST-CUT · 2026-05-21] Swap selectivo de `gemini-3.5-flash`
a `gemini-3.1-flash-lite` en 2 callsites de baja-complejidad para reducir
costo Gemini (~50% per-call) sin tocar el path crítico de calidad.

Callsites afectados:
  1. `ai_helpers.generate_plan_title()` → knob `MEALFIT_PLAN_TITLE_MODEL`
     (pre-fix: `model="gemini-3.1-flash-lite"` HARDCODED, sin knob ni rollback).
  2. `tools._tools_pref_agent_model_name()` → knob `MEALFIT_TOOLS_PREF_AGENT_MODEL`
     (pre-fix: default `gemini-3.5-flash` desde P1-ALL-MODELS-GA).

Callsites NO afectados (riesgo demasiado alto):
  - Vision agent (`_vision_model_name`) — corre clasificación multimodal de
    fotos de comida. Flash-lite degrada reconocimiento visual → macros
    estimadas mal → adherencia del Diario Visual corrompida. Mantener flash.
  - Main 3-day generation (`_FLASH_MODEL_NAME` ruta FÁCIL) — constraints
    duros (macros + alergias + 3P/3C/3V + slot-collision + cultural). Lite
    aumentaría retry rate → costo NETO sube vs flash.
  - Self-critique correction step — re-genera días con sugerencias del
    crítico + constraints originales. Lite pierde constraint con frecuencia.
  - Chat agent + tools modify_meal — user-facing en chat, fail visible.

Pattern operacional (P3-PREVIEW-MODEL-KNOB):
  - Cada callsite tiene helper `_<feature>_model_name()` con default explícito.
  - SRE puede rollback individual via env var sin redeploy
    (`MEALFIT_PLAN_TITLE_MODEL=gemini-3.5-flash` o
    `MEALFIT_TOOLS_PREF_AGENT_MODEL=gemini-3.5-flash`).
  - Monitor retry rate + `pipeline_metrics.quality_score` post-swap; si
    cualquier callsite degrada >5pp en quality, rollback inmediato.

Cobertura del test:
  1. Default + knob explícito en ai_helpers.py.
  2. Default cambiado en tools.py.
  3. Hardcoded `model="gemini-3.1-flash-lite"` removido de ai_helpers.py
     (debe ir via helper).
  4. Markers P3-FLASH-LITE-COST-CUT presentes en ambos archivos.
  5. Env var override sigue funcionando (rollback path verificado).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).parent.parent
_AI_HELPERS_PY = _BACKEND / "ai_helpers.py"
_TOOLS_PY = _BACKEND / "tools.py"

_FLASH_LITE = "gemini-3.1-flash-lite"
_FLASH_GA = "gemini-3.5-flash"


# ---------------------------------------------------------------------------
# Sección 1 — Knob nuevo en ai_helpers (plan title)
# ---------------------------------------------------------------------------


def test_plan_title_helper_exists():
    """`_plan_title_model_name()` debe existir en ai_helpers.py + leer
    `MEALFIT_PLAN_TITLE_MODEL` vía `_env_str` (auto-registry en `_KNOBS_REGISTRY`).
    Patrón espejo de `_tools_pref_agent_model_name`."""
    src = _AI_HELPERS_PY.read_text(encoding="utf-8")
    assert "def _plan_title_model_name" in src, (
        "Helper `_plan_title_model_name()` no encontrado en ai_helpers.py. "
        "Pre-fix tenía `model='gemini-3.1-flash-lite'` HARDCODED — viola "
        "convención P3-PREVIEW-MODEL-KNOB y bloquea rollback sin redeploy."
    )
    assert "MEALFIT_PLAN_TITLE_MODEL" in src, (
        "Knob `MEALFIT_PLAN_TITLE_MODEL` ausente en ai_helpers.py."
    )


def test_plan_title_default_is_flash_lite():
    """Default del knob debe ser `gemini-3.1-flash-lite` (cero cambio de
    comportamiento — el callsite ya usaba lite hardcoded; el knob añade
    rollback path). Si Google deprecia lite, swap via env var sin redeploy."""
    src = _AI_HELPERS_PY.read_text(encoding="utf-8")
    assert f'_env_str("MEALFIT_PLAN_TITLE_MODEL", "{_FLASH_LITE}")' in src, (
        f"Default Plan Title debe ser {_FLASH_LITE!r}. Rollback via "
        f"`MEALFIT_PLAN_TITLE_MODEL=gemini-3.5-flash` sin redeploy."
    )


def test_plan_title_no_hardcoded_model_in_generate_plan_title():
    """El cuerpo de `generate_plan_title()` NO debe contener
    `model="gemini-3.1-flash-lite"` literal (debe usar `_plan_title_model_name()`).
    Si reaparece el hardcoded, el knob queda no-op.

    Nota: ai_helpers.py contiene 3 callsites ADICIONALES con `model="gemini-3.1-flash-lite"`
    hardcoded (`expand_recipe_agent`, `generate_llm_retrospective`,
    `extract_liked_flavor_profiles`) — son deuda P3-PREVIEW-MODEL-KNOB
    separada (tracked, no cubierta por este test). Este test se limita
    al callsite tocado por P3-FLASH-LITE-COST-CUT.
    """
    src = _AI_HELPERS_PY.read_text(encoding="utf-8")
    # Extraer cuerpo de generate_plan_title: desde `def generate_plan_title`
    # hasta el siguiente `def ` (start-of-line).
    m = re.search(
        r"def generate_plan_title\([\s\S]*?(?=\ndef |\Z)",
        src,
    )
    assert m is not None, "Función `generate_plan_title` no encontrada en ai_helpers.py."
    body = m.group(0)
    hardcoded_patterns = [
        f'model="{_FLASH_LITE}"',
        f"model='{_FLASH_LITE}'",
    ]
    for pat in hardcoded_patterns:
        assert pat not in body, (
            f"Hardcoded {pat!r} encontrado en `generate_plan_title`. Debe ir via "
            f"`model=_plan_title_model_name()` para que el knob aplique."
        )


# ---------------------------------------------------------------------------
# Sección 2 — Default re-migrado en tools.py (pref agent)
# ---------------------------------------------------------------------------


def test_tools_pref_agent_default_is_flash_lite():
    """`_tools_pref_agent_model_name()` default re-migrado a flash-lite.
    Pre-fix (P1-ALL-MODELS-GA): `gemini-3.5-flash`. El preference analyzer
    hace clasificación simple — output low-stakes."""
    src = _TOOLS_PY.read_text(encoding="utf-8")
    m = re.search(
        r'_env_str\(\s*"MEALFIT_TOOLS_PREF_AGENT_MODEL"\s*,\s*"([^"]+)"',
        src,
    )
    assert m is not None, (
        "Callsite `_env_str(\"MEALFIT_TOOLS_PREF_AGENT_MODEL\", ...)` no "
        "encontrado en tools.py."
    )
    assert m.group(1) == _FLASH_LITE, (
        f"Default debe ser {_FLASH_LITE!r}, encontrado {m.group(1)!r}. "
        f"Rollback: `MEALFIT_TOOLS_PREF_AGENT_MODEL=gemini-3.5-flash`."
    )


def test_tools_modify_meal_NOT_flipped():
    """Sanity: `_tools_modify_meal_model_name()` debe seguir en flash GA.
    Modify-meal es user-facing en chat (swap_meal tool) con constraints —
    flash-lite degradaría calidad visible. NO swap aquí."""
    src = _TOOLS_PY.read_text(encoding="utf-8")
    m = re.search(
        r'_env_str\(\s*"MEALFIT_TOOLS_MODIFY_MEAL_MODEL"\s*,\s*"([^"]+)"',
        src,
    )
    assert m is not None, "MEALFIT_TOOLS_MODIFY_MEAL_MODEL no encontrado."
    assert m.group(1) == _FLASH_GA, (
        f"modify_meal debe seguir en {_FLASH_GA!r} (user-facing, constraints). "
        f"Encontrado {m.group(1)!r}. Si decides bajarlo, documenta el trade-off."
    )


def test_vision_NOT_flipped():
    """Sanity: `_vision_model_name()` debe seguir en flash GA. Vision
    procesa fotos de comida multimodal — lite degradaría reconocimiento
    visual y corromperia macros del Diario Visual. NO swap aquí."""
    vision_py = _BACKEND / "vision_agent.py"
    src = vision_py.read_text(encoding="utf-8")
    m = re.search(
        r'_env_str\(\s*"MEALFIT_VISION_MODEL"\s*,\s*"([^"]+)"',
        src,
    )
    assert m is not None, "MEALFIT_VISION_MODEL no encontrado."
    assert m.group(1) == _FLASH_GA, (
        f"Vision debe seguir en {_FLASH_GA!r} (multimodal food classification). "
        f"Encontrado {m.group(1)!r}. Lite degrada reconocimiento visual."
    )


# ---------------------------------------------------------------------------
# Sección 3 — Markers presentes (cross-link con _LAST_KNOWN_PFIX)
# ---------------------------------------------------------------------------


def test_marker_present_in_edited_files():
    """Marker `P3-FLASH-LITE-COST-CUT` debe estar como tooltip-anchor en los
    archivos modificados. Si alguien borra el comentario sin marker, este
    test cae antes que `test_p2_hist_audit_14_marker_test_link.py`."""
    expected_in = [_AI_HELPERS_PY, _TOOLS_PY]
    for p in expected_in:
        src = p.read_text(encoding="utf-8")
        assert "P3-FLASH-LITE-COST-CUT" in src, (
            f"Marker `P3-FLASH-LITE-COST-CUT` ausente en {p.name}. "
            f"Sin él, un revert futuro perdería el contexto del swap."
        )


# ---------------------------------------------------------------------------
# Sección 4 — Env var override (rollback path operacional)
# ---------------------------------------------------------------------------


def test_env_var_rollback_plan_title():
    """`MEALFIT_PLAN_TITLE_MODEL=gemini-3.5-flash` debe restaurar flash GA si
    la calidad del título degrada visiblemente con lite."""
    try:
        from knobs import _env_str
    except ImportError:
        pytest.skip("knobs module not importable.")

    os.environ["MEALFIT_PLAN_TITLE_MODEL"] = _FLASH_GA
    try:
        assert _env_str("MEALFIT_PLAN_TITLE_MODEL", _FLASH_LITE) == _FLASH_GA
    finally:
        del os.environ["MEALFIT_PLAN_TITLE_MODEL"]

    # Sin env var, default lite
    assert _env_str("MEALFIT_PLAN_TITLE_MODEL", _FLASH_LITE) == _FLASH_LITE


def test_env_var_rollback_pref_agent():
    """`MEALFIT_TOOLS_PREF_AGENT_MODEL=gemini-3.5-flash` debe restaurar flash GA
    si el preference analyzer degrada (retry rate sube, quality drops)."""
    try:
        from knobs import _env_str
    except ImportError:
        pytest.skip("knobs module not importable.")

    os.environ["MEALFIT_TOOLS_PREF_AGENT_MODEL"] = _FLASH_GA
    try:
        assert _env_str("MEALFIT_TOOLS_PREF_AGENT_MODEL", _FLASH_LITE) == _FLASH_GA
    finally:
        del os.environ["MEALFIT_TOOLS_PREF_AGENT_MODEL"]


# ---------------------------------------------------------------------------
# Sección 5 — Helper retorna valor esperado in-process
# ---------------------------------------------------------------------------


def test_plan_title_helper_returns_default():
    """Sin env override, `_plan_title_model_name()` debe retornar el default
    `gemini-3.1-flash-lite` (acceptance test del helper, no solo source-scan)."""
    # Limpiar env var por si tests previos la dejaron
    os.environ.pop("MEALFIT_PLAN_TITLE_MODEL", None)
    try:
        from ai_helpers import _plan_title_model_name
    except ImportError:
        pytest.skip("ai_helpers no importable en este contexto de test.")
    assert _plan_title_model_name() == _FLASH_LITE


def test_pref_agent_helper_returns_default():
    """Sin env override, `_tools_pref_agent_model_name()` debe retornar
    `gemini-3.1-flash-lite`."""
    os.environ.pop("MEALFIT_TOOLS_PREF_AGENT_MODEL", None)
    try:
        from tools import _tools_pref_agent_model_name
    except ImportError:
        pytest.skip("tools no importable en este contexto de test.")
    assert _tools_pref_agent_model_name() == _FLASH_LITE
