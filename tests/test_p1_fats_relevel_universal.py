"""[P1-FATS-RELEVEL-UNIVERSAL · 2026-07-06] El re-trim de grasas (_trim_day_fats_to_target) SOLO corría
dentro del micro-recheck P2-2 (gated por `if _pe_adj`: SOLO si el micro-closer actuó). El forense de 20
planes reales mostró que el exceso de grasa (celda fats fuera en 39% de los días) vive en FUENTES DE GRASA
AÑADIDA del day-gen (aceite/queso/aguacate/semillas), INDEPENDIENTE de los micros → un día fats-out con
micros OK nunca se relevelea, y el refinador global EXIME esas fuentes. La celda fats ARRASTRA la de kcal
(Atwater 9 kcal/g).

Fix: `_relevel_fats_universal` corre `_trim_day_fats_to_target` INCONDICIONAL por día sobre banda, en los 3
sitios (S1 assemble + update helper + regen-day). Validado en 18 días reales fats-out: 13/18 celda fats
CERRADA, 16/18 all-4 mejora, 0 regresiones, kcal→~100% en cada uno (da7bb310 d2: fats 186%→105%, kcal
125%→100%, 2/4→4/4).
"""
from pathlib import Path
import graph_orchestrator as go

_GO = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
_PL = (Path(go.__file__).resolve().parent / "routers" / "plans.py").read_text(encoding="utf-8")


def _db():
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=[
        {"name": "Aceite de oliva", "aliases": ["aceite"], "kcal_per_100g": 884,
         "protein_g_per_100g": 0, "carbs_g_per_100g": 0, "fats_g_per_100g": 100},
        {"name": "Queso cheddar", "aliases": ["queso"], "kcal_per_100g": 403,
         "protein_g_per_100g": 25, "carbs_g_per_100g": 1.3, "fats_g_per_100g": 33},
        {"name": "Semillas de girasol", "aliases": ["girasol"], "kcal_per_100g": 584,
         "protein_g_per_100g": 20.8, "carbs_g_per_100g": 20, "fats_g_per_100g": 51.5},
        {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 165,
         "protein_g_per_100g": 31, "carbs_g_per_100g": 0, "fats_g_per_100g": 3.6},
    ])


def _days_fats_over():
    # grasa real ≈ aceite 40g×1.0 + queso 60g×0.33 + girasol 20g×0.515 + pollo 150g×0.036 ≈ 40+20+10+5 = 75g
    return [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "X",
             "ingredients": ["40g de aceite de oliva", "60g de queso cheddar",
                             "20g de semillas de girasol", "150g de pechuga de pollo"],
             "ingredients_raw": ["40g de aceite de oliva", "60g de queso cheddar",
                                 "20g de semillas de girasol", "150g de pechuga de pollo"],
             "protein": 61, "carbs": 5, "fats": 75, "cals": 939}]}]


def test_universal_trim_reduces_overtarget_fats():
    days = _days_fats_over()
    f0 = sum(go._meal_macro_num(m.get("fats")) for m in days[0]["meals"])
    n = go._relevel_fats_universal(days, 40.0, _db())   # target fats 40g, delivered ~75g
    assert n == 1, "el día sobre banda se relevelea"
    f1 = sum(go._meal_macro_num(m.get("fats")) for m in days[0]["meals"])
    assert f1 < f0, f"grasas recortadas hacia target: {f0} -> {f1}"
    assert f1 <= 40.0 * 1.20, f"grasas acercadas al target 40g: {f1}"


def test_protects_seed_carriers():
    days = _days_fats_over()
    go._relevel_fats_universal(days, 40.0, _db())
    girasol = [i for i in days[0]["meals"][0]["ingredients"] if "girasol" in i.lower()][0]
    # la semilla (portador de micros) NO se encoge: sigue en 20g.
    assert "20" in girasol, f"la semilla de micros se protege: {girasol}"


def test_noop_when_fats_in_band():
    days = _days_fats_over()
    # target fats alto (80g) → los 75g entregados están en banda → no toca.
    assert go._relevel_fats_universal(days, 80.0, _db()) == 0


def test_knob_off_skips(monkeypatch):
    monkeypatch.setattr(go, "FATS_RELEVEL_UNIVERSAL_ENABLED", False)
    assert go._relevel_fats_universal(_days_fats_over(), 40.0, _db()) == 0


def test_knob_defined_default_on():
    import re
    m = re.search(r'FATS_RELEVEL_UNIVERSAL_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_FATS_RELEVEL_UNIVERSAL"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"


def test_fail_safe_bad_target():
    assert go._relevel_fats_universal(_days_fats_over(), 0, _db()) == 0
    assert go._relevel_fats_universal(None, 40.0, _db()) == 0


# --- wiring en los 3 sitios (paridad SSOT: S1 + update helper + regen-day) ---

def test_wired_s1_after_refiner():
    i = _GO.index("refinador global no-op")
    blk = _GO[i:i + 700]
    assert "_relevel_fats_universal(days" in blk, "S1: relevel tras el refinador"


def test_wired_update_helper():
    i = _GO.index("def apply_update_macro_engine(")
    blk = _GO[i:i + 9000]
    assert "FATS_RELEVEL_UNIVERSAL_ENABLED" in blk and "_trim_day_fats_to_target(_meals" in blk


def test_wired_regen_day():
    assert "_trim_fats_rd(" in _PL and "_fru_rd" in _PL, "regen-day aplica el relevel de grasas"


def test_marker_anchored():
    assert "P1-FATS-RELEVEL-UNIVERSAL" in _GO
    assert "P1-FATS-RELEVEL-UNIVERSAL" in _PL
