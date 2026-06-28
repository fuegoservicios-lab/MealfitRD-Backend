"""[P1-BARIATRIC-CONDIMENT-CAP + P1-CLOSER-CALORIE-AWARE · 2026-06-27] Dos afinamientos tras el re-test corr=713b8e84
(plan bariátrico aprobado pero band 0.58 con kcal fuera + el revisor flageó condimentos en exceso):

FIX 3 (P1-BARIATRIC-CONDIMENT-CAP): cap por COUNT (número líder) de canela/aceite/linaza en unidades de cuchara
(cda/cdta) que el cap por-gramos NO atrapa (macros_from_ingredient_string no resuelve gramos de cucharadas). Cierra
"4.75 cdta de canela" (cumarina tóxica), "3.75 cdta de aceite", "3.25 cda de linaza".

FIX 2 (P1-CLOSER-CALORIE-AWARE): el re-cierre de proteína (FASE A) inyectaba +110g sin mirar el headroom calórico →
rompía la banda de kcal (band 0.58). Ahora `_close_protein_gap_for_meal` acepta slot_cal_target y no añade proteína
por encima del headroom calórico del slot (los callers sin slot_cal_target conservan el piso de 10g original).
"""
from __future__ import annotations

import graph_orchestrator as g

_FORM = {"medicalConditions": ["Cirugía Bariátrica"]}


# ---- FIX 3 ----
def test_condiment_count_cap_canela_aceite_linaza():
    days = [{"day": 0, "meals": [{"meal": "Merienda Nocturna", "name": "Yogurt con Canela",
        "ingredients": ["120 g de yogurt griego", "4.75 cucharaditas de canela en polvo",
                        "3.75 cdta de aceite de oliva", "3.25 cda de linaza molida (32g)"]}]}]
    n = g.cap_bariatric_portions(days, _FORM)
    ings = " | ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert n >= 3
    assert "4.75" not in ings and "3.75" not in ings and "3.25" not in ings  # capeados
    # [P1-BARIATRIC-CROSSDAY-CONDIMENT-CAP] el per-comida deja canela 1.5/linaza 2.0, PERO el cap cross-día (mismo día)
    # los baja más: canela ≤0.5 cdta/día, linaza ≤1 cda/día. Aceite NO tiene cap cross-día → queda en 2.
    assert "0.5" in ings  # canela → 0.5 (cross-día)
    assert "2 cdta de aceite" in ings or "2.0 cdta de aceite" in ings  # aceite per-comida (sin cross-día)
    assert "1 cda de linaza" in ings or "1.0 cda de linaza" in ings  # linaza → 1 cda (cross-día)


def test_condiment_cap_leaves_small_amounts_alone():
    # cantidades ≤ cap (canela 0.5 cross-día, aceite 1 cda per-comida) → intactas
    days = [{"day": 0, "meals": [{"meal": "Cena", "name": "Pollo",
        "ingredients": ["0.5 cdta de canela", "1 cda de aceite de oliva"]}]}]
    g.cap_bariatric_portions(days, _FORM)
    ings = " | ".join(days[0]["meals"][0]["ingredients"]).lower()
    assert "0.5 cdta de canela" in ings and "1 cda de aceite" in ings  # ≤ cap → intactos


def test_condiment_cap_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "BARIATRIC_CONDIMENT_CAP_ENABLED" in src and "P1-BARIATRIC-CONDIMENT-CAP" in src
    assert "_BARIATRIC_CONDIMENT_COUNT_CAPS" in src


# ---- FIX 2 ----
class _Info:
    def __init__(self, name, protein, carbs=2.0, fats=1.0, kcal=165.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, carbs, fats, kcal


_CANDS = [(0.25, "Pechuga de pollo", _Info("Pechuga de pollo", 31))]


def _meal(cals):
    return {"name": "Pollo con Ensalada", "protein": 5, "carbs": 10, "fats": 5, "cals": cals, "ingredients": ["pollo"]}


def test_calorie_aware_skips_when_no_headroom():
    """Si el slot ya está en/sobre su target de kcal → no añade proteína (evita inflar kcal/band)."""
    assert g._close_protein_gap_for_meal(_meal(500), 25, None, _CANDS, slot_cal_target=480) == 0


def test_calorie_aware_adds_when_headroom():
    assert g._close_protein_gap_for_meal(_meal(150), 25, None, _CANDS, slot_cal_target=480) > 0


def test_no_slot_cal_target_preserves_original_floor():
    """Callers sin slot_cal_target conservan el comportamiento original (piso 10g), aunque cals sean altas."""
    assert g._close_protein_gap_for_meal(_meal(500), 25, None, _CANDS) > 0


def test_calorie_aware_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-CLOSER-CALORIE-AWARE" in src and "slot_cal_target" in src
