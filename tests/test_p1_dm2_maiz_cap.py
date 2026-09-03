# [P1-DM2-MAIZ-CAP · 2026-08-08] Ancla del marker (la suite funcional completa vive en
# test_p1_dm2_glycemic_portion_cap.py::test_maiz_dulce_se_capea). Palanca 1 del 65% del
# benchmark (issue #9/#14): vegana_dm2 caía por 150-180g de maíz dulce — el token no estaba
# en _DM2_HIGH_GI_STARCH_TOKENS y el default del cap (150) excedía el criterio del reviewer.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_token_maiz_dulce_presente_y_cap_100():
    import graph_orchestrator as g
    assert "maiz dulce" in g._DM2_HIGH_GI_STARCH_TOKENS, (
        "sin el token, el maíz dulce escapa al cap y el reviewer rechaza el plan DM2 entero")
    assert g.DM2_HIGH_GI_CAP_G == 100, "default alineado al criterio del reviewer (~100g/comida)"
