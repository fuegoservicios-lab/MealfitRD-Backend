"""[P3-MICRO-FOOD-SUGGEST · 2026-06-15] Tool del agente que sugiere alimentos
del catálogo (master_ingredients) para cerrar un gap de micronutriente,
filtrando por alergias/rechazos/dieta del usuario. Conecta el panel de
micronutrientes del dashboard (tap → "¿cómo subo mi fibra?") con el coach.

Cobertura:
  - Estructural (parser, siempre corre): la tool está en `agent_tools`, en el
    doc de paridad, en ambas versiones de `build_tools_instructions`, y el
    mapeo de nutrientes incluye las claves esperadas.
  - Funcional (import-guarded, skip si tools.py no importa en el entorno):
    rankea por densidad, excluye alimentos por alergia/dieta, maneja techo vs
    piso, y rechaza nutrientes desconocidos.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_micro_food_suggest`
matchea este archivo. Tooltip-anchor: P3-MICRO-FOOD-SUGGEST.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_DOC = _BACKEND_ROOT / "docs" / "agent_tools_user_id_table.md"
_PROMPTS = _BACKEND_ROOT / "prompts" / "chat_agent.py"

_TOOL = "suggest_foods_for_nutrient"


# ============================ Estructural (parser) ============================

def test_tool_registered_in_agent_tools():
    src = _TOOLS_PY.read_text(encoding="utf-8")
    m = re.search(r"agent_tools\s*=\s*\[(.*?)\]", src, re.DOTALL)
    assert m is not None, "No se encontró la lista agent_tools en tools.py"
    assert _TOOL in m.group(1), (
        f"{_TOOL} no está en agent_tools. La tool no estará disponible para el agente."
    )


def test_tool_documented_for_parity():
    doc = _DOC.read_text(encoding="utf-8")
    assert f"`{_TOOL}`" in doc, (
        f"{_TOOL} no está en agent_tools_user_id_table.md → test_p2_chat_cleanup fallaría."
    )


def test_tool_in_both_instruction_builders():
    src = _PROMPTS.read_text(encoding="utf-8")
    # Debe mencionarse en build_tools_instructions Y build_tools_instructions_stream.
    assert src.count(f"`{_TOOL}`") >= 2, (
        f"{_TOOL} debe describirse en AMBAS versiones de build_tools_instructions "
        "(completa + stream) para que la LLM sepa cuándo llamarla."
    )


def test_nutrient_mapping_has_core_keys():
    src = _TOOLS_PY.read_text(encoding="utf-8")
    for key in ["fibra", "hierro", "calcio", "vitamina d", "potasio", "magnesio", "sodio"]:
        assert f'"{key}"' in src, f"El mapeo de nutrientes no incluye '{key}'."


def test_tooltip_anchor_present():
    src = _TOOLS_PY.read_text(encoding="utf-8")
    assert "P3-MICRO-FOOD-SUGGEST" in src, "tooltip-anchor P3-MICRO-FOOD-SUGGEST ausente de tools.py"


# ============================ Funcional (import-guarded) ============================

def _load_tools():
    try:
        import tools as tools_mod  # noqa
        import shopping_calculator  # noqa
        return tools_mod, shopping_calculator
    except Exception as e:  # entorno sin deps/DB
        pytest.skip(f"tools/shopping_calculator no importable: {e}")


def test_resolve_nutrient_es_en_and_unknown():
    tools_mod, _ = _load_tools()
    r = tools_mod._resolve_micro_nutrient
    assert r("fibra")[0] == "fiber_g_per_100g"
    assert r("Fiber")[0] == "fiber_g_per_100g"
    assert r("Vitamina D")[0] == "vitamin_d_mcg_per_100g"
    assert r("hierro")[0] == "iron_mg_per_100g"
    # techo
    assert r("sodio")[3] is True
    # piso
    assert r("fibra")[3] is False
    assert r("kriptonita") is None


def test_ranks_and_filters_floor_nutrient(monkeypatch):
    tools_mod, shopping_calculator = _load_tools()
    fake_catalog = [
        {"name": "Avena", "fiber_g_per_100g": 10.0},
        {"name": "Habichuelas", "fiber_g_per_100g": 8.0},
        {"name": "Pan con leche", "fiber_g_per_100g": 6.0},   # excluido por alergia 'leche'
        {"name": "Arroz blanco", "fiber_g_per_100g": 0.0},     # excluido: piso exige >0
    ]
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: fake_catalog)
    monkeypatch.setattr(
        tools_mod, "get_user_profile",
        lambda uid: {"health_profile": {"allergies": ["leche"], "dietType": "balanced"}},
    )
    out = tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient="fibra", top_n=5)
    assert "Avena" in out and "Habichuelas" in out
    assert "leche" not in out.lower(), "No debe sugerir 'Pan con leche' (alergia)"
    assert "Arroz" not in out, "No debe sugerir alimento con 0 de fibra"
    assert out.index("Avena") < out.index("Habichuelas"), "Debe rankear por densidad desc"


def test_ceiling_nutrient_ranks_lowest_first(monkeypatch):
    tools_mod, shopping_calculator = _load_tools()
    fake_catalog = [
        {"name": "Embutido salado", "sodium_mg_per_100g": 900.0},
        {"name": "Lechuga", "sodium_mg_per_100g": 10.0},
        {"name": "Tomate", "sodium_mg_per_100g": 5.0},
    ]
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: fake_catalog)
    monkeypatch.setattr(tools_mod, "get_user_profile", lambda uid: {"health_profile": {}})
    out = tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient="sodio", top_n=3)
    # Techo → más bajos primero (Tomate 5 < Lechuga 10 < Embutido 900)
    assert out.index("Tomate") < out.index("Lechuga") < out.index("Embutido salado")


def test_vegan_diet_excludes_animal_foods(monkeypatch):
    tools_mod, shopping_calculator = _load_tools()
    fake_catalog = [
        {"name": "Lentejas", "iron_mg_per_100g": 6.0},
        {"name": "Higado de res", "iron_mg_per_100g": 18.0},   # excluido por vegan
        {"name": "Huevo", "iron_mg_per_100g": 2.0},            # excluido por vegan
    ]
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: fake_catalog)
    monkeypatch.setattr(
        tools_mod, "get_user_profile",
        lambda uid: {"health_profile": {"dietType": "vegan"}},
    )
    out = tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient="hierro", top_n=5)
    assert "Lentejas" in out
    assert "Higado" not in out and "Huevo" not in out, "Dieta vegana debe excluir animales/huevo"


def test_unknown_nutrient_returns_supported_list(monkeypatch):
    tools_mod, _ = _load_tools()
    out = tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient="kriptonita")
    assert "No reconozco" in out and "fibra" in out
