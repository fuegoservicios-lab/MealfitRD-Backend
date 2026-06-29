"""[P1-GLM-CASCADE · 2026-06-29] Cascada de modelos del day generator por COSTO (reduce el gasto en DeepSeek):
glm-4.7-flash (Zhipu, ~gratis) → deepseek-v4-flash → deepseek-v4-pro. Avanza al siguiente en CADA fallo (los 3 reintentos
de tenacity) o si el CB del modelo está abierto. Bariátrico → directo a deepseek-v4-pro (sin capas baratas). El revisor
médico NO usa la cascada (sigue en pro). Provider elegido por prefijo del model_id (glm* → z.ai; resto → DeepSeek).

Tests PUROS: _day_model_chain (composición) + provider routing (base_url por prefijo). La llamada LIVE a GLM se verifica
en el smoke del VPS (requiere GLM_API_KEY del .env).
"""
from __future__ import annotations

import graph_orchestrator as g
import llm_provider as lp

_BAR = {"medicalConditions": ["Cirugía Bariátrica (manga gástrica)"]}
_NON = {"medicalConditions": ["Diabetes tipo 2"]}


def test_chain_non_bariatric_is_cost_cascade():
    assert g._day_model_chain(_NON, 1) == ["glm-4.7-flash", "deepseek-v4-flash", "deepseek-v4-pro"]


def test_chain_bariatric_is_direct_pro():
    # bariátrico → directo a pro (perfil clínico más difícil; sin glm/flash)
    assert g._day_model_chain(_BAR, 1) == ["deepseek-v4-pro"]


def test_chain_skeleton_fidelity_retry_is_pro():
    chain = g._day_model_chain(_NON, 2, ["skeleton fidelity violation: proteína fuera del pool asignado"])
    assert chain == ["deepseek-v4-pro"]


def test_chain_glm_off_falls_to_deepseek(monkeypatch):
    monkeypatch.setattr(g, "GLM_DAYGEN_ENABLED", False)
    assert g._day_model_chain(_NON, 1) == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_chain_ends_in_pro_always():
    # la cadena SIEMPRE termina en pro (calidad como última instancia)
    for fd in (_NON, _BAR):
        assert g._day_model_chain(fd, 1)[-1] == "deepseek-v4-pro"


def test_chain_dedup_when_flash_equals_pro(monkeypatch):
    # si DEEPSEEK_FLASH coincidiera con el pro (config rara), no se duplica
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-flash")
    chain = g._day_model_chain(_NON, 1)
    assert len(chain) == len(set(chain))  # sin duplicados


def test_is_glm_model():
    assert lp._is_glm_model("glm-4.7-flash") is True
    assert lp._is_glm_model("GLM-4.5") is True
    assert lp._is_glm_model("deepseek-v4-pro") is False
    assert lp._is_glm_model("") is False
    assert lp._is_glm_model(None) is False


def test_provider_routing_by_prefix():
    glm = lp.ChatDeepSeek(model="glm-4.7-flash", temperature=0.5)
    ds = lp.ChatDeepSeek(model="deepseek-v4-flash", temperature=0.5)
    assert "z.ai" in str(glm.openai_api_base)
    assert "deepseek.com" in str(ds.openai_api_base)


def test_glm_base_url_default():
    assert lp._glm_base_url() == "https://api.z.ai/api/paas/v4"


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-GLM-CASCADE" in src
    assert "GLM_DAYGEN_ENABLED" in src and "_day_model_chain" in src
    assert g.GLM_DAYGEN_ENABLED is True
    # la cascada se construye con el chain dentro del day-gen, antes del invoke
    assert "_day_chain = _day_model_chain(" in src
    # GLM key NUNCA hardcodeada en el código (debe venir de env)
    lp_src = pathlib.Path(lp.__file__).read_text(encoding="utf-8")
    assert "f98369bf" not in src and "f98369bf" not in lp_src  # la key real jamás en source
    assert 'os.environ.get("GLM_API_KEY")' in lp_src
