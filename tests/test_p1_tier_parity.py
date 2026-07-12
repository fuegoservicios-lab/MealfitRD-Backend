"""[P1-TIER-PARITY · 2026-07-12] Todos los tiers tienen los mismos privilegios de features.

Decisión de producto del owner (verbatim): "quiero que los planes tengan todos
los mismos accesos (privilegios) por igual incluyendo el plan gratis, la única
diferencia debe de ser las cantidades de créditos".

Gates de features eliminados (4):
  - agent.py `_do_sentiment` — era plus/ultra/admin.
  - agent.py `_do_rag` — excluía gratis.
  - routers/chat.py `is_plus` ×2 (stream + non-stream) — la extracción de
    memoria a largo plazo excluía gratis (un usuario gratis con horas de chat
    quedaba sin user_facts, y Dreaming sin materia prima).

Lo ÚNICO que diferencia tiers: `auth._TIER_LIMITS` (créditos/mes). Guests
(sin cuenta) siguen sin RAG/memoria: no hay identidad estable que recordar.

NOTA (decisión separada, documentada): el ROUTER de modelos LLM
(llm_provider: gratis→flash, pagos→pro) se conserva — es ingeniería de costo
por mensaje no-medido (el chat no consume créditos), no un feature visible.
Flip disponible sin redeploy: MEALFIT_CHAT_AGENT_MODEL.
tooltip-anchor: P1-TIER-PARITY
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "agent.py"), encoding="utf-8") as f:
    _AG = f.read()
with open(os.path.join(_BACKEND, "routers", "chat.py"), encoding="utf-8") as f:
    _CH = f.read()


def test_sentiment_and_rag_for_all_tiers():
    assert "_do_sentiment = True" in _AG, \
        "el análisis de sentimiento es para todos los tiers"
    assert '_do_rag = bool(user_id) and user_id != "guest"' in _AG, \
        "el RAG es para todo usuario con cuenta (gratis incluido); guests fuera"
    # El gate viejo no debe volver por ningún path:
    assert 'plan_tier in ["plus", "ultra", "admin"]' not in _AG
    assert 'plan_tier in ["basic", "plus", "ultra", "admin"]' not in _AG


def test_long_term_memory_for_all_tiers():
    assert _CH.count('is_plus = bool(user_id and user_id != "guest")') >= 2, \
        "ambos paths (stream y non-stream) extraen memoria para cualquier cuenta"
    assert 'plan_tier in ["basic", "plus", "admin", "ultra"]' not in _CH, \
        "el gate por tier de la memoria no debe volver"


def test_credits_remain_the_only_tier_difference():
    with open(os.path.join(_BACKEND, "auth.py"), encoding="utf-8") as f:
        auth_src = f.read()
    # _TIER_LIMITS sigue siendo el diferenciador (y los 4 tiers existen).
    for tier in ("gratis", "basic", "plus", "ultra"):
        assert f'"{tier}":' in auth_src, f"_TIER_LIMITS debe conservar el tier {tier}"


def test_user_toggle_still_respected():
    # La paridad NO elimina el control del usuario sobre su memoria.
    assert _CH.count("long_term_memory_enabled") >= 2, \
        "el toggle user-controlled de memoria sobrevive a la paridad"
