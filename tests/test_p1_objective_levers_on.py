"""[P1-OBJECTIVE-LEVERS-ON · 2026-06-29] Test-ancla del batch de gaps P1 del audit-objetivo de plan-gen.

Implementa los 6 P1:
  P1-1  P1-MICRO-CLOSER-RAW-SYNC   — el micro-closer escala `ingredients_raw` (su efecto REFLEJA en panel/lista/macros).
  P1-2  MEALFIT_MICRONUTRIENT_CLOSER  flipped OFF→ON (único lever determinista que mueve micros al DRI).
  P1-3  MEALFIT_CARB_FLOOR            flipped OFF→ON (closer aditivo de carbos cuando el reconcile satura).
  P1-4  MEALFIT_UPDATE_MACRO_REBALANCE + MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER flipped OFF→ON en agent.py (swap/regen)
        y tools.py (chat-modify) — el motor de macros corre en TODAS las superficies de update.
  P1-5  P1-CREATIVITY-TRANSFORM-UPDATE — el bloque de transformación de staples llega a swap + chat-modify.
  P1-6  P1-DISH-RAW-STAPLE            — heurística ADVISORY de creatividad (staple sin transformar), canal SEPARADO
        del `low_quality_ratio` → NO toca el soft-gate (que sigue OFF).

Todos reversibles por env var (rollback sin redeploy). Parser-based donde tocar DB colgaría; behavioral donde es puro.
"""
import re
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_MEALOPS = (_BACKEND / "prompts" / "meal_operations.py").read_text(encoding="utf-8")


# ───────────────────────── P1-1: micro-closer escala ingredients_raw ─────────────────────────
def test_p1_1_micro_closer_writes_ingredients_raw():
    # el cuerpo del closer debe reescalar ingredients_raw por el MISMO factor (espejo del closer de proteína)
    assert "P1-MICRO-CLOSER-RAW-SYNC" in _GRAPH
    seg = _GRAPH[_GRAPH.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert 'meal["ingredients_raw"][idx] = _new_raw' in seg, "el closer debe escribir ingredients_raw"
    assert "_resc(_raw[idx], factor)" in seg, "debe reescalar el raw por el MISMO factor"


# ───────────────────────── P1-2 / P1-3: closers deterministas ON ─────────────────────────
def test_p1_2_micronutrient_closer_on_by_default():
    assert g.MICRONUTRIENT_CLOSER_ENABLED is True


def test_p1_3_carb_floor_on_by_default():
    assert g.CARB_FLOOR_ENABLED is True


# ───────────────────────── P1-4: motor de macros en superficies de update ─────────────────────────
def test_p1_4_update_macro_closers_on_in_agent_and_tools():
    for src, name in ((_AGENT, "agent.py"), (_TOOLS, "tools.py")):
        assert re.search(r'MEALFIT_UPDATE_MACRO_REBALANCE"\s*,\s*"true"', src), f"{name}: rebalance default ON"
        assert re.search(r'MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER"\s*,\s*"true"', src), f"{name}: protein-closer default ON"
    # SWAP_TARGET_FROM_SLOT NO se flipeó (modo de sobre-asignación conocido)
    assert 'os.environ.get("MEALFIT_SWAP_TARGET_FROM_SLOT", "false")' in _AGENT


# ───────────────────────── P1-5: creatividad en los prompts de update ─────────────────────────
def test_p1_5_creativity_transform_in_update_templates():
    assert _MEALOPS.count("P1-CREATIVITY-TRANSFORM-UPDATE") >= 2, "debe estar en SWAP y MODIFY"
    swap = re.search(r"SWAP_MEAL_PROMPT_TEMPLATE\s*=\s*\"\"\"(.*?)\"\"\"", _MEALOPS, re.DOTALL).group(1)
    mod = re.search(r"MODIFY_MEAL_PROMPT_TEMPLATE\s*=\s*\"\"\"(.*?)\"\"\"", _MEALOPS, re.DOTALL).group(1)
    for body, label in ((swap, "SWAP"), (mod, "MODIFY")):
        assert "P1-CREATIVITY-TRANSFORM-UPDATE" in body, label
        assert "panqueque" in body.lower() and "bollos de yuca" in body.lower(), label


# ───────────────────────── P1-6: raw-staple advisory, separado del gate ─────────────────────────
def test_p1_6_raw_staple_flags_bare_staple_not_composed():
    flag = g._meal_raw_staple_issue
    for name in ("Arroz Blanco", "Yuca Hervida", "Pollo a la Plancha", "Huevos Fritos"):
        raw, _why = flag({"name": name})
        assert raw is True, f"'{name}' debería marcarse como staple sin transformar"
    for name in ("Bollos de Yuca al Mojo", "Panqueques de Avena con Fresas",
                 "Pollo a la Plancha con Ensalada Criolla", "Mofongo de Plátano"):
        raw, _why = flag({"name": name})
        assert raw is False, f"'{name}' NO debería marcarse (es compuesto/acompañado)"


def test_p1_6_raw_staple_is_separate_from_low_quality_gate():
    # un plato staple-sin-transformar pero CON receta real: NO infla low_quality_ratio, SÍ cuenta en raw_staple
    meal = {"name": "Arroz Blanco",
            "ingredients": ["200 g de Arroz blanco", "5 g de Sal"],
            "recipe": ["Mise en place: lava y mide el arroz blanco con cuidado en una taza medidora",
                       "El Toque de Fuego: hierve el arroz blanco a fuego medio por 20 minutos hasta secar"]}
    rep = g.compute_dish_quality_report({"days": [{"day": 1, "meals": [meal]}]})
    assert rep["low_quality_meals"] == 0, "receta real → NO es placeholder"
    assert rep["low_quality_ratio"] == 0.0
    assert rep["raw_staple_meals"] == 1, "pero SÍ es un staple sin transformar"
    assert rep["raw_staple_ratio"] == 1.0
    # el soft-gate de dish-quality sigue OFF (la heurística es advisory, A/B-pending para promover a gate)
    assert g.DISH_QUALITY_SOFT_GATE_ENABLED is False
