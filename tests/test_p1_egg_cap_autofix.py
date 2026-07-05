"""[P1-EGG-CAP-AUTOFIX · 2026-07-05] Corrector determinista del sobreuso de huevo.

Causa #1 de retry COMPLETO medida 2026-07-05 (corr=95d76bb9: "huevo en 5 de 12 comidas,
máximo 3" disparó 2× el gate P3-VARIETY-HARD-GATE = 2 generaciones cobradas). El day-gen usa
huevo como relleno proteico por defecto.

Diseño: reescribe huevos SOLO donde son relleno (no en el NOMBRE del plato, no en platos
aglutinante tipo croqueta/panqueque) hacia proteínas verificadas slot-aware (merienda→yogurt
griego, desayuno→queso, fuertes→pollo/queso) con guards de alergia/dislike + anti-colisión
same-day re-escaneada por candidato. Protagonistas (revoltillo/tortilla) JAMÁS se tocan.
Marca `_protein_autofix_applied="huevo->X"` → el fidelity-discount la reconoce.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _meal(slot, name, ings):
    return {"meal": slot, "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": ["Prepara y sirve."]}


def _mk_days_egg5of12():
    """3 días × 4 comidas = 12; huevo en 5 (cap 3): 2 protagonistas + 3 rellenos."""
    return [
        {"day": 1, "meals": [
            _meal("Desayuno", "Revoltillo de Huevo con Mangú", ["2 huevos", "200 g de plátano"]),
            _meal("Almuerzo", "Pollo Guisado", ["150 g de pechuga de pollo", "150 g de arroz"]),
            _meal("Merienda", "Bowl Energético", ["1 huevo", "1 guineo"]),
            _meal("Cena", "Pescado con Batata", ["150 g de filete de pescado blanco", "100 g de batata"]),
        ]},
        {"day": 2, "meals": [
            _meal("Desayuno", "Tortilla de Claras", ["4 claras", "50 g de espinaca"]),
            _meal("Almuerzo", "Res Guisada", ["150 g de carne de res", "150 g de arroz"]),
            _meal("Merienda", "Yuca con Huevo Duro", ["1 huevo", "100 g de yuca"]),
            _meal("Cena", "Wrap de Pavo", ["120 g de pechuga de pavo", "1 tortilla integral"]),
        ]},
        {"day": 3, "meals": [
            _meal("Desayuno", "Avena con Guineo", ["40 g de avena", "1 guineo"]),
            _meal("Almuerzo", "Locrio de Pollo", ["150 g de pechuga de pollo", "150 g de arroz"]),
            _meal("Merienda", "Batido Verde", ["1 taza de espinaca", "2 huevos"]),
            _meal("Cena", "Chivo Guisado", ["150 g de chivo", "100 g de ñame"]),
        ]},
    ]


# ---------------------------------------------------------------------------

def test_knob_default_and_callsite_order():
    assert '_env_bool("MEALFIT_EGG_CAP_AUTOFIX", True)' in _GO
    i_egg = _GO.index("_egg_fixed = _egg_cap_autofix(days, form_data)")
    i_pr = _GO.index("_pr_fixed = _protein_repeat_autofix(days, form_data)")
    assert i_egg < i_pr, "el egg-fix corre ANTES del protein-repeat (le limpia repeticiones de huevo)"


def test_rewrites_fillers_down_to_cap(go):
    days = _mk_days_egg5of12()
    n = go._egg_cap_autofix(days, {}, db=object())
    assert n == 2, "5 con huevo, cap 3 → exceso 2 → reescribe exactamente 2 rellenos"
    # los protagonistas quedan intactos:
    assert "huevos" in " ".join(days[0]["meals"][0]["ingredients"])
    assert "claras" in " ".join(days[1]["meals"][0]["ingredients"])
    # los rellenos reescritos son meriendas (prioridad de slot) con marker:
    _rewritten = [m for d in days for m in d["meals"] if str(m.get("_protein_autofix_applied", "")).startswith("huevo->")]
    assert len(_rewritten) == 2
    for m in _rewritten:
        joined = " ".join(m["ingredients"]).lower()
        assert "huevo" not in joined and "clara" not in joined
        assert m["ingredients"] == m["ingredients_raw"]


def test_merienda_gets_yogurt_and_avoids_meat_collision(go):
    days = _mk_days_egg5of12()
    go._egg_cap_autofix(days, {}, db=object())
    for d in days:
        for m in d["meals"]:
            if str(m.get("_protein_autofix_applied", "")).startswith("huevo->"):
                joined = " ".join(m["ingredients"]).lower()
                assert "yogurt griego" in joined or "queso blanco" in joined, \
                    "merienda → yogurt/queso (no carne; los días ya tienen pollo/res)"


def test_binder_dishes_protected(go):
    days = _mk_days_egg5of12()
    # convertir un relleno en aglutinante: NO debe tocarse ni contar como candidato.
    days[0]["meals"][2]["name"] = "Croquetas de Yuca"
    n = go._egg_cap_autofix(days, {}, db=object())
    assert "1 huevo" in days[0]["meals"][2]["ingredients"], \
        "el huevo-aglutinante de la croqueta es funcional — jamás reemplazarlo"
    assert n <= 2


def test_under_cap_untouched_and_idempotent(go):
    days = _mk_days_egg5of12()
    # bajar a 3 con huevo (== cap) quitando 2 rellenos:
    days[0]["meals"][2]["ingredients"] = ["1 guineo"]
    days[1]["meals"][2]["ingredients"] = ["100 g de yuca"]
    assert go._egg_cap_autofix(days, {}, db=object()) == 0
    # idempotencia sobre el fixture completo:
    days2 = _mk_days_egg5of12()
    assert go._egg_cap_autofix(days2, {}, db=object()) == 2
    assert go._egg_cap_autofix(days2, {}, db=object()) == 0


def test_dislike_ladder_falls_through(go):
    days = _mk_days_egg5of12()
    n = go._egg_cap_autofix(days, {"dislikes": ["yogurt"]}, db=object())
    assert n >= 1
    for d in days:
        for m in d["meals"]:
            if str(m.get("_protein_autofix_applied", "")).startswith("huevo->"):
                assert "yogurt" not in " ".join(m["ingredients"]).lower()


def test_knob_off(go, monkeypatch):
    monkeypatch.setattr(go, "EGG_CAP_AUTOFIX_ENABLED", False)
    assert go._egg_cap_autofix(_mk_days_egg5of12(), {}, db=object()) == 0


def test_marker_feeds_fidelity_discount():
    """El marker huevo->X debe ser legible por el fidelity-discount (P1-FIDELITY-MUTATION-AWARE
    parsea `src` con split('->') y resuelve aliases de _MAIN_PROTEIN_ALIASES['huevo'])."""
    import graph_orchestrator as g
    assert "huevo" in g._MAIN_PROTEIN_ALIASES
    assert 'f"huevo->{_label}"' in _GO


def test_marker_anchored_in_source():
    assert "P1-EGG-CAP-AUTOFIX" in _GO
