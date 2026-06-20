"""[P3-MACRO-REBALANCE · 2026-06-19] Ancla el rebalanceador de macros post-cuantización. La cuantización
redondea cada ingrediente independiente → el error se acumula (carbos/grasas continuos casi perfectos ~0.9%
→ ~7%/5% tras redondear). El rebalanceador re-apunta las 3 macros al target + RE-CUANTIZA → recupera la
precisión sin perder cocinabilidad. A/B determinista (harness, corpus fijo): all4 53.7→87.0, proteína w10
78→96. Parser-based: la validación numérica vive en el harness (no CI: necesita master_ingredients + corpus)."""
import ast
import os
import re

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph_orchestrator.py")
with open(_SRC, encoding="utf-8") as _f:
    SRC = _f.read()
TREE = ast.parse(SRC)
_FUNCS = {n.name: n for n in ast.walk(TREE) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _body(name):
    return ast.get_source_segment(SRC, _FUNCS[name]) or ""


def test_rebalancer_exists_and_covers_three_macros():
    assert "_rebalance_day_macros_to_target" in _FUNCS, "se eliminó el rebalanceador de macros"
    body = _body("_rebalance_day_macros_to_target")
    # re-apunta las 3 macros (proteína incluida — sin ella, escalar carbo/grasa driftea la proteína)
    for mac in ('"protein"', '"carbs"', '"fats"'):
        assert f"_one({mac}" in body, f"el rebalanceador no re-apunta {mac}"
    # bidireccional: factor puede subir Y bajar (clamp con techo > 1.0)
    assert re.search(r"min\(2\.\d+,\s*desired_movable\s*/\s*movable\)", body), "el rebalanceo no es bidireccional (sin techo >1)"
    # re-cuantiza (mantiene cocinabilidad) + recompute honesto por delta
    assert "_quant(_resc(" in body, "el rebalanceador no re-cuantiza (perdería cocinabilidad)"
    assert "macros_from_ingredient_string" in body, "no recomputa macros honestamente desde el string cuantizado"


def test_rebalancer_protein_preserving_via_group_filter():
    """Solo toca ingredientes del grupo macro-dominante objetivo (no rompe otros macros arbitrariamente)."""
    body = _body("_rebalance_day_macros_to_target")
    assert "_ingredient_macro_group(str(ing), db) != macro_key" in body, \
        "el rebalanceador no filtra por grupo macro-dominante"


def test_rebalancer_wired_gated_default_on():
    engine = _body("_apply_macro_engine")
    assert "MACRO_REBALANCE_ENABLED" in engine and "_rebalance_day_macros_to_target" in engine, \
        "el rebalanceador no está cableado/gateado en el motor"
    assert "target_protein=_pg" in engine, "el motor no pasa el target de proteína al rebalanceador"
    # win grande validado → default ON; rollback por knob
    assert re.search(r'MACRO_REBALANCE_ENABLED\s*=\s*_env_bool\("MEALFIT_MACRO_REBALANCE",\s*True\)', SRC), \
        "el knob del rebalanceador no es default True (es un win validado)"


def test_runs_after_quantize_and_swap():
    """Debe correr DESPUÉS del swap (que ya corre tras la capa clínica/quantize) para re-apuntar el estado final."""
    engine = _body("_apply_macro_engine")
    i_swap = engine.find("CARB_TO_PROTEIN_SWAP_ENABLED")
    i_reb = engine.find("MACRO_REBALANCE_ENABLED")
    assert 0 < i_swap < i_reb, "el rebalanceador debe correr DESPUÉS del swap (estado post-quantize/swap)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
