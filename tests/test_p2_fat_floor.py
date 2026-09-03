"""[P2-FAT-FLOOR · 2026-09-03] La grasa era la única macro sin closer hacia ARRIBA: había piso de
carbos (`_close_carb_gap_for_day`), closers de proteína y un relevel de grasas que solo RECORTA.
Un «actualizar día» del dueño entregó 58 g de grasa contra 69 de target (kcal 2370/2500) con
proteína y carbos en banda, y el plan quedó marcado degradado. `_close_fat_gap_for_day` escala la
fuente de grasa existente más rica hacia el target (clamp, techo calórico, sin inventar compras) y
se invoca en assemble (paridad con el carb-floor) y en regen-day (tras el relevel).
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]


class _DB:
    """Nutrición de juguete: aceite grasa-dominante, pescado proteína-dominante, batata carbo."""
    TABLE = {
        "1 cdta de aceite de oliva": {"protein": 0.0, "carbs": 0.0, "fats": 4.5, "kcal": 40.0},
        "2 cdta de aceite de oliva": {"protein": 0.0, "carbs": 0.0, "fats": 9.0, "kcal": 81.0},
        "3 cdta de aceite de oliva": {"protein": 0.0, "carbs": 0.0, "fats": 13.5, "kcal": 121.0},
        "5 g de semillas de calabaza": {"protein": 1.5, "carbs": 0.5, "fats": 2.5, "kcal": 30.0},
        "12 g de semillas de calabaza": {"protein": 3.6, "carbs": 1.2, "fats": 6.0, "kcal": 72.0},
        "95 g de filete de pescado blanco": {"protein": 20.0, "carbs": 0.0, "fats": 1.5, "kcal": 95.0},
        "1 batata mediana": {"protein": 2.0, "carbs": 40.0, "fats": 0.2, "kcal": 170.0},
    }

    def macros_from_ingredient_string(self, s):
        return dict(self.TABLE.get(str(s), {})) or None


def _fake_rescale(orig, factor):
    # «1 cdta» ×2.4 → «2 cdta»/«3 cdta» — devuelve la línea con la cantidad escalada y redondeada
    m = re.match(r"(\d+) (cdta de aceite de oliva)", orig)
    if m:
        return f"{max(1, round(int(m.group(1)) * factor))} {m.group(2)}"
    m = re.match(r"(\d+) g de semillas de calabaza", orig)
    if m:
        return f"{max(1, round(int(m.group(1)) * factor))} g de semillas de calabaza"
    return orig


def _fake_quantize(s):
    return s, 1.0


def _day():
    return [
        {"name": "Pastelitos", "meal": "Almuerzo", "protein": 35, "carbs": 114, "fats": 11, "cals": 687,
         "ingredients": ["95 g de filete de pescado blanco", "1 batata mediana", "1 cdta de aceite de oliva"],
         "ingredients_raw": ["95 g de filete de pescado blanco", "1 batata mediana", "1 cdta de aceite de oliva"]},
        {"name": "Empanadita", "meal": "Merienda", "protein": 23, "carbs": 67, "fats": 9, "cals": 433,
         "ingredients": ["5 g de semillas de calabaza", "1 batata mediana"],
         "ingredients_raw": ["5 g de semillas de calabaza", "1 batata mediana"]},
    ]


def _patch(monkeypatch):
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "rescale_ingredient_string", _fake_rescale)
    monkeypatch.setattr(nutrition_db, "quantize_ingredient_string", _fake_quantize)
    monkeypatch.setattr(go, "_sync_one_raw_line", lambda m, idx, orig, f: m["ingredients_raw"].__setitem__(idx, m["ingredients"][idx]))
    monkeypatch.setattr(go, "FAT_FLOOR_ENABLED", True)


def test_sube_la_grasa_escalando_la_fuente_mas_rica_sin_inventar_compras(monkeypatch):
    _patch(monkeypatch)
    meals = _day()
    assert go._close_fat_gap_for_day(meals, target_fats=30.0, target_kcal=1300.0, db=_DB()) is True
    total = sum(m["fats"] for m in meals)
    assert total > 20 and total >= 30.0 * 0.9 - 1e-6   # 20 g → dentro de la banda inferior
    # se escaló el aceite (la fuente más rica) y la lista cruda quedó en sincronía
    assert meals[0]["ingredients"][2] != "1 cdta de aceite de oliva"
    assert meals[0]["ingredients_raw"][2] == meals[0]["ingredients"][2]
    assert len(meals[0]["ingredients"]) == 3 and len(meals[1]["ingredients"]) == 2   # ninguna línea nueva
    # macros honestas: kcal = 4P + 4C + 9F
    for m in meals:
        assert m["cals"] == round(4 * m["protein"] + 4 * m["carbs"] + 9 * m["fats"])


def test_no_actua_en_banda_ni_sin_palanca(monkeypatch):
    _patch(monkeypatch)
    meals = _day()
    assert go._close_fat_gap_for_day(meals, target_fats=21.0, target_kcal=1300.0, db=_DB()) is False   # 20 ≥ 21×0.9
    sin_grasa = [{"name": "x", "protein": 30, "carbs": 100, "fats": 2, "cals": 538,
                  "ingredients": ["95 g de filete de pescado blanco", "1 batata mediana"],
                  "ingredients_raw": ["95 g de filete de pescado blanco", "1 batata mediana"]}]
    assert go._close_fat_gap_for_day(sin_grasa, target_fats=40.0, target_kcal=1000.0, db=_DB()) is False
    assert sin_grasa[0]["ingredients"] == ["95 g de filete de pescado blanco", "1 batata mediana"]


def test_respeta_el_techo_calorico(monkeypatch):
    _patch(monkeypatch)
    meals = _day()  # 1120 kcal; con target 900 kcal el techo 1.12×900 = 1008 ya está superado
    assert go._close_fat_gap_for_day(meals, target_fats=60.0, target_kcal=900.0, db=_DB()) is False


def test_knob_apaga_el_closer(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(go, "FAT_FLOOR_ENABLED", False)
    meals = _day()
    assert go._close_fat_gap_for_day(meals, target_fats=60.0, target_kcal=2000.0, db=_DB()) is False


def test_cableado_en_assemble_y_en_regen_day():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'FAT_FLOOR_ENABLED = _env_bool("MEALFIT_FAT_FLOOR", True)' in src
    i_carb = src.index("if CARB_FLOOR_ENABLED and _db is not None:")
    i_fat = src.index("if FAT_FLOOR_ENABLED and _db is not None:")
    assert i_fat > i_carb, "el piso de grasa va DESPUÉS del de carbos en assemble"
    plans = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i_relevel = plans.index("[P2-REGEN-DAY-FATS-RELEVEL] no-op")
    i_floor = plans.index("MEALFIT_REGEN_DAY_FAT_FLOOR")
    assert i_floor > i_relevel, "en regen-day el piso va tras el relevel (que solo recorta)"
    assert "_close_fat_gap_for_day as _cfg_rd" in plans
    assert '_LAST_KNOWN_PFIX = "P2-FAT-FLOOR · 2026-09-03"' in (_BACKEND / "app.py").read_text(encoding="utf-8")
