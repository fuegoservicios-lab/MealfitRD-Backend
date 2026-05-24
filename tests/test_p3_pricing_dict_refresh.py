"""[P3-PRICING-DICT-REFRESH · 2026-05-21] Refresh del pricing dict de Gemini
en `db_profiles.py` para reflejar las tarifas reales del tier Estándar (paid)
de Google AI 2026-05-21.

Bug encontrado:
  El pricing dict tenía valores stale para `gemini-3.1-flash-lite`:
  pre-fix `$0.10/$0.40/$0.025` (micros 100_000/400_000/25_000), real
  `$0.25/$1.50/$0.025` (micros 250_000/1_500_000/25_000). El error:
  - Input: 2.5× sub-reportado
  - Output: 3.75× sub-reportado
  - Cached: correcto

Impacto en observabilidad:
  `llm_usage_events.cost_usd_micros` es la fuente de verdad para
  `/api/admin/cost-by-node`. Flash-lite es el modelo activo de:
  - Self-critique evaluator + judge + fact-checker (P1-FLASH-LITE-AUX-NODES)
  - Medical reviewer (`tools_medical._medical_tool_model_name`)
  - Plan title + preference analyzer (P3-FLASH-LITE-COST-CUT)
  - Skeleton planner (P3-PLANNER-LITE-COST)

  Con el pricing stale, todos esos nodos reportaban ~3× menos costo del real.
  El admin endpoint mostraba un panorama optimista falso — operador podía
  asumir que el ROI de los swaps a lite era mayor del real.

Backfill:
  Las filas históricas en `llm_usage_events` con `model LIKE 'gemini-3.1-flash-lite%'`
  tienen `cost_usd_micros` calculado con el pricing viejo. NO backfilleamos
  automáticamente — riesgo de double-correct si el operador ya backfilleó
  manualmente. SQL para backfill manual documentado en la memoria del P-fix.

Cobertura del test:
  1. Pricing dict tiene `gemini-3.1-flash-lite` con los micros correctos.
  2. `gemini-3.5-flash` mantiene los micros correctos (pre-existente, no
     debe degradarse durante el refresh).
  3. Marker `P3-PRICING-DICT-REFRESH` presente.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).parent.parent
_DB_PROFILES_PY = _BACKEND / "db_profiles.py"


def test_flash_lite_pricing_is_real():
    """`gemini-3.1-flash-lite` debe tener pricing real de Google AI 2026-05-21:
    input=250_000 (= $0.25/M), output=1_500_000 (= $1.50/M), cached=25_000
    (= $0.025/M)."""
    text = _DB_PROFILES_PY.read_text(encoding="utf-8")
    m = re.search(
        r'"gemini-3\.1-flash-lite"\s*:\s*\{\s*"input"\s*:\s*250_000\s*,\s*"output"\s*:\s*1_500_000\s*,\s*"cached"\s*:\s*25_000\s*\}',
        text,
    )
    assert m, (
        "Pricing de `gemini-3.1-flash-lite` debe ser "
        "{input: 250_000, output: 1_500_000, cached: 25_000} micros/M tokens. "
        "Si Google cambia, actualizar el dict + esta assertion."
    )


def test_flash_lite_pricing_not_stale():
    """Verificación negativa: el pricing viejo stale ($0.10/$0.40/$0.025)
    NO debe estar presente. Si reaparece, alguien revirtió el refresh."""
    text = _DB_PROFILES_PY.read_text(encoding="utf-8")
    stale_pattern = re.search(
        r'"gemini-3\.1-flash-lite"\s*:\s*\{\s*"input"\s*:\s*100_000\s*,\s*"output"\s*:\s*400_000',
        text,
    )
    assert stale_pattern is None, (
        "Pricing stale de flash-lite (input=100_000 / output=400_000) detectado. "
        "Debe ser 250_000 / 1_500_000 per P3-PRICING-DICT-REFRESH."
    )


def test_flash_pricing_preserved():
    """Sanity: `gemini-3.5-flash` mantiene pricing correcto (ya estaba bien
    pre-fix per Google AI doc 2026-05-19). $1.50/$9.00/$0.15 per M tokens."""
    text = _DB_PROFILES_PY.read_text(encoding="utf-8")
    m = re.search(
        r'"gemini-3\.5-flash"\s*:\s*\{\s*"input"\s*:\s*1_500_000\s*,\s*"output"\s*:\s*9_000_000\s*,\s*"cached"\s*:\s*150_000\s*\}',
        text,
    )
    assert m, (
        "Pricing de `gemini-3.5-flash` debe permanecer "
        "{input: 1_500_000, output: 9_000_000, cached: 150_000}. Si Google "
        "cambia, actualizar + esta assertion."
    )


def test_pro_pricing_preserved():
    """Sanity: `gemini-3.1-pro-preview` pricing intacto. User no compartió
    screenshot de Pro pricing — no tocamos este valor."""
    text = _DB_PROFILES_PY.read_text(encoding="utf-8")
    m = re.search(
        r'"gemini-3\.1-pro-preview"\s*:\s*\{\s*"input"\s*:\s*1_250_000\s*,\s*"output"\s*:\s*10_000_000',
        text,
    )
    assert m, "Pricing de Pro NO debe cambiar en este P-fix (out-of-scope)."


def test_marker_present_in_db_profiles():
    """Marker `P3-PRICING-DICT-REFRESH` debe estar en db_profiles.py como
    tooltip-anchor. Sin él, un revert futuro perdería contexto."""
    text = _DB_PROFILES_PY.read_text(encoding="utf-8")
    assert "P3-PRICING-DICT-REFRESH" in text, (
        "Marker `P3-PRICING-DICT-REFRESH` ausente en db_profiles.py."
    )


def test_compute_cost_with_real_pricing():
    """Functional: `compute_gemini_cost_micros` debe retornar costo real
    con el pricing nuevo. 1k input + 1k output con lite =
    1_000 × 250 + 1_000 × 1_500 = 250 + 1_500 = 1_750 micros."""
    try:
        from db_profiles import compute_gemini_cost_micros
    except ImportError:
        pytest.skip("db_profiles no importable en este contexto.")

    cost = compute_gemini_cost_micros(
        "gemini-3.1-flash-lite",
        input_tokens=1_000,
        output_tokens=1_000,
        cached_input_tokens=0,
    )
    expected = 1_750  # 1000 × 250/M + 1000 × 1500/M = 250 + 1500
    assert cost == expected, (
        f"compute_gemini_cost_micros retornó {cost}, esperado {expected}. "
        f"1k tokens input × $0.25/M + 1k output × $1.50/M = $0.00175 = 1750 micros."
    )
