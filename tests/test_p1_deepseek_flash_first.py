"""[P1-DEEPSEEK-FLASH-FIRST · 2026-06-28] GLM ELIMINADO (su tier gratis rate-limitea hasta ser inusable, 429 en 1 sola
llamada). Routing por costo SOLO DeepSeek: deepseek-v4-flash SUFICIENTE por default; deepseek-v4-pro SOLO cuando flash no
alcanza:
  - attempt 1 → [flash, pro]: flash primario; pro SOLO si el call de flash falla (reliability).
  - attempt > 1 (plan de flash rechazado → insuficiente) → [pro].
  - bariátrico → [pro] directo.
Minimiza costo/llamadas: el caso normal = 1 llamada flash.
"""
from __future__ import annotations

import graph_orchestrator as g
import llm_provider as lp

_BAR = {"medicalConditions": ["Cirugía Bariátrica (manga gástrica)"]}
_NON = {"medicalConditions": ["Diabetes tipo 2"]}


def _patch_flash(monkeypatch):
    # asegura flash≠pro para que el test sea significativo (en prod MEALFIT_FLASH_MODEL=deepseek-v4-flash)
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")


def test_attempt1_flash_primary_pro_fallback(monkeypatch):
    _patch_flash(monkeypatch)
    assert g._day_model_chain(_NON, 1) == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_retry_escalates_to_pro(monkeypatch):
    _patch_flash(monkeypatch)
    # plan de flash rechazado → reintento usa pro PRIMERO (flash no fue suficiente).
    # [P1-DAYGEN-RETRY-FLASH-NET · 2026-07-03] flash queda como RED de última instancia:
    # con el breaker de pro abierto, [pro] a secas mataba TODOS los workers del retry
    # → fallback matemático (gym baseline: 2/20 planes, uno maintenance sin condiciones).
    assert g._day_model_chain(_NON, 2) == ["deepseek-v4-pro", "deepseek-v4-flash"]


def test_bariatric_direct_pro(monkeypatch):
    _patch_flash(monkeypatch)
    assert g._day_model_chain(_BAR, 1) == ["deepseek-v4-pro"]


def test_chain_ends_in_pro():
    for fd in (_NON, _BAR):
        assert g._day_model_chain(fd, 1)[-1] == "deepseek-v4-pro"


def test_dedup_when_flash_equals_pro(monkeypatch):
    # si MEALFIT_FLASH_MODEL == pro (config previa), la cadena colapsa a [pro] (sin duplicado)
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")
    assert g._day_model_chain(_NON, 1) == ["deepseek-v4-pro"]


def test_retry_knob_off_keeps_flash(monkeypatch):
    _patch_flash(monkeypatch)
    monkeypatch.setattr(g, "DAY_GEN_RETRY_USE_PRO", False)
    # con el knob off, el reintento NO fuerza pro → vuelve a la cadena flash-first
    assert g._day_model_chain(_NON, 2) == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_glm_fully_removed():
    # GLM eliminado del provider y del orquestador
    assert not hasattr(lp, "GLM_FLASH")
    assert not hasattr(lp, "_is_glm_model")
    assert not hasattr(g, "GLM_DAYGEN_ENABLED")
    import pathlib
    lp_src = pathlib.Path(lp.__file__).read_text(encoding="utf-8")
    g_src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "glm" not in lp_src.lower()
    assert "z.ai" not in lp_src and "zhipu" not in lp_src.lower()
    # en el orquestador no debe quedar el model id de glm ni el knob
    assert "glm-4" not in g_src and "GLM_DAYGEN" not in g_src
    # la key real jamás en source
    assert "f98369bf" not in lp_src and "f98369bf" not in g_src


def test_provider_is_deepseek_only():
    ds = lp.ChatDeepSeek(model="deepseek-v4-flash", temperature=0.5)
    assert "deepseek.com" in str(ds.openai_api_base)


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-DEEPSEEK-FLASH-FIRST" in src
    assert "_day_model_chain" in src
